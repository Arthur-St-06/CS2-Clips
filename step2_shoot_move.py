#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from demoparser2 import DemoParser  # type: ignore


# Defaults you approved
DEFAULT_TICKRATE = 64.0
SPEED_THRESHOLD = 35.0          # units/sec (2D)
DEATH_WINDOW_S = 1.5            # seconds after moving shot
INCIDENT_GAP_S = 0.25           # cluster moving shots close together (mostly for stats/weapon mode)

def unwrap_ticks(ret) -> pd.DataFrame:
    """
    demoparser2 sometimes returns [(name, df), ...] and sometimes a df-like.
    This normalizes to a DataFrame.
    """
    if isinstance(ret, pd.DataFrame):
        return ret
    if isinstance(ret, dict):
        for v in ret.values():
            if isinstance(v, pd.DataFrame):
                return v
        return pd.DataFrame(ret)
    if isinstance(ret, (list, tuple)) and len(ret) > 0:
        item = ret[0]
        if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], pd.DataFrame):
            return item[1]
    raise RuntimeError(f"Unexpected parse_ticks return type: {type(ret)}")


def unwrap_events(ret) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if isinstance(ret, dict):
        for k, v in ret.items():
            out[str(k)] = v if isinstance(v, pd.DataFrame) else pd.DataFrame(v)
        return out
    if isinstance(ret, (list, tuple)):
        for item in ret:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                name = str(item[0])
                df = item[1]
                out[name] = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        return out
    return out


def compute_speeds_from_positions(pos: pd.DataFrame, tickrate: float) -> pd.DataFrame:
    req = ["tick", "player_name", "player_steamid", "X", "Y"]
    missing = [c for c in req if c not in pos.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}. Got: {list(pos.columns)}")

    df = pos[["tick", "player_name", "player_steamid", "X", "Y"]].copy()
    df["tick"] = pd.to_numeric(df["tick"], errors="coerce")
    df = df[df["tick"].notna()].copy()
    df["tick"] = df["tick"].astype(np.int64)

    df["player_steamid"] = pd.to_numeric(df["player_steamid"], errors="coerce")
    df = df[df["player_steamid"].notna()].copy()
    df["player_steamid"] = df["player_steamid"].astype(np.int64)

    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")

    df = df.sort_values(["player_steamid", "tick"]).reset_index(drop=True)

    df["dX"] = df.groupby("player_steamid")["X"].diff()
    df["dY"] = df.groupby("player_steamid")["Y"].diff()

    df["dist2d_per_tick"] = np.sqrt(df["dX"] ** 2 + df["dY"] ** 2)
    df["speed_2d"] = (df["dist2d_per_tick"] * float(tickrate)).fillna(0.0)

    return df[["tick", "player_steamid", "player_name", "speed_2d"]]


def _mode(series: pd.Series) -> Optional[str]:
    if series is None or series.empty:
        return None
    vc = series.value_counts(dropna=True)
    if vc.empty:
        return None
    return str(vc.index[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True)
    ap.add_argument("--out", default="out_sprays")
    ap.add_argument("--player", default="", help="Optional: filter by player name substring")
    ap.add_argument("--tickrate", type=float, default=DEFAULT_TICKRATE)

    # keep defaults fixed but configurable if you ever want
    ap.add_argument("--speed-threshold", type=float, default=SPEED_THRESHOLD)
    ap.add_argument("--death-window-s", type=float, default=DEATH_WINDOW_S)
    ap.add_argument("--incident-gap-s", type=float, default=INCIDENT_GAP_S)

    ap.add_argument(
        "--exclude-weapons",
        default="knife|hegrenade|flashbang|smokegrenade|molotov|incgrenade|decoy|taser",
        help="Regex of weapons to ignore",
    )
    args = ap.parse_args()

    demo = Path(args.demo).expanduser().resolve()
    if not demo.exists():
        raise FileNotFoundError(demo)

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tickrate = float(args.tickrate)
    speed_thr = float(args.speed_threshold)
    death_window_ticks = int(round(float(args.death_window_s) * tickrate))
    gap_ticks = int(round(float(args.incident_gap_s) * tickrate))

    parser = DemoParser(str(demo))

    # --- Load events ---
    ev = unwrap_events(parser.parse_events(["weapon_fire", "player_death"]))
    wf = ev.get("weapon_fire", pd.DataFrame())
    deaths = ev.get("player_death", pd.DataFrame())

    out_path = out_dir / "shoot_move_candidates.parquet"

    if wf is None or wf.empty or deaths is None or deaths.empty:
        pd.DataFrame().to_parquet(out_path, index=False)
        print("No weapon_fire or player_death events found. Wrote empty shoot_move_candidates.parquet.")
        return

    # weapon_fire columns we expect
    wf_req = ["tick", "user_name", "user_steamid", "weapon"]
    wf_missing = [c for c in wf_req if c not in wf.columns]
    if wf_missing:
        raise RuntimeError(f"weapon_fire missing columns: {wf_missing}. Available: {list(wf.columns)}")

    shots = wf[wf_req].copy()
    shots["tick"] = pd.to_numeric(shots["tick"], errors="coerce")
    shots = shots[shots["tick"].notna()].copy()
    shots["tick"] = shots["tick"].astype(np.int64)

    shots["user_steamid"] = pd.to_numeric(shots["user_steamid"], errors="coerce")
    shots = shots[shots["user_steamid"].notna()].copy()
    shots["user_steamid"] = shots["user_steamid"].astype(np.int64)

    shots["user_name"] = shots["user_name"].astype(str)
    shots["weapon"] = shots["weapon"].astype(str)

    if args.exclude_weapons.strip():
        shots = shots[~shots["weapon"].str.contains(args.exclude_weapons, case=False, na=False)].copy()

    if args.player.strip():
        shots = shots[shots["user_name"].str.contains(args.player.strip(), case=False, na=False)].copy()

    if shots.empty:
        pd.DataFrame().to_parquet(out_path, index=False)
        print("No shots after filtering/excludes. Wrote empty shoot_move_candidates.parquet.")
        return

    shots = shots.sort_values(["user_steamid", "tick"]).reset_index(drop=True)
    shots["t_s"] = shots["tick"].astype(float) / tickrate

    # deaths columns vary; you’ve seen victim fields like assisted... etc.
    # We need tick + victim steamid + victim name
    # Common names in demoparser2: user_* is shooter for weapon_fire, but for player_death it’s victim_*.
    death_tick_col = "tick" if "tick" in deaths.columns else None
    victim_sid_col = None
    victim_name_col = None
    for c in ["user_steamid", "victim_steamid", "userid", "victim_userid"]:
        if c in deaths.columns:
            victim_sid_col = c
            break
    for c in ["user_name", "victim_name", "victim", "name"]:
        if c in deaths.columns:
            victim_name_col = c
            break

    if death_tick_col is None or victim_sid_col is None:
        raise RuntimeError(
            f"player_death missing required columns. Need tick + victim steamid. "
            f"Have cols: {list(deaths.columns)}"
        )

    d = deaths[[death_tick_col, victim_sid_col] + ([victim_name_col] if victim_name_col else [])].copy()
    d.rename(columns={death_tick_col: "death_tick", victim_sid_col: "victim_steamid"}, inplace=True)

    d["death_tick"] = pd.to_numeric(d["death_tick"], errors="coerce")
    d = d[d["death_tick"].notna()].copy()
    d["death_tick"] = d["death_tick"].astype(np.int64)

    d["victim_steamid"] = pd.to_numeric(d["victim_steamid"], errors="coerce")
    d = d[d["victim_steamid"].notna()].copy()
    d["victim_steamid"] = d["victim_steamid"].astype(np.int64)

    if victim_name_col:
        d["victim_name"] = d[victim_name_col].astype(str)
    else:
        d["victim_name"] = ""

    d = d.sort_values(["victim_steamid", "death_tick"]).reset_index(drop=True)
    d["death_s"] = d["death_tick"].astype(float) / tickrate

    # --- Compute speeds and join onto shots ---
    pos_ret = parser.parse_ticks(["tick", "player_name", "player_steamid", "X", "Y", "Z"])
    pos = unwrap_ticks(pos_ret)
    speeds = compute_speeds_from_positions(pos, tickrate=tickrate)

    joined = shots.merge(
        speeds,
        left_on=["tick", "user_steamid"],
        right_on=["tick", "player_steamid"],
        how="left",
        validate="many_to_one",
    )
    joined["speed_2d"] = pd.to_numeric(joined["speed_2d"], errors="coerce").fillna(0.0)

    moving_shots = joined[joined["speed_2d"] >= speed_thr].copy()
    if moving_shots.empty:
        pd.DataFrame().to_parquet(out_path, index=False)
        print("No moving shots above threshold. Wrote empty shoot_move_candidates.parquet.")
        return

    # --- Map each moving shot -> first death by same player within window ---
    # For each steamid, binary-search death ticks
    moving_shots = moving_shots.sort_values(["user_steamid", "tick"]).reset_index(drop=True)

    death_map = {}
    for sid, g in d.groupby("victim_steamid", sort=False):
        death_map[int(sid)] = g["death_tick"].to_numpy(dtype=np.int64, copy=False)

    death_tick_for_shot = np.full(len(moving_shots), -1, dtype=np.int64)

    sids = moving_shots["user_steamid"].to_numpy(dtype=np.int64, copy=False)
    stks = moving_shots["tick"].to_numpy(dtype=np.int64, copy=False)

    for i in range(len(moving_shots)):
        sid = int(sids[i])
        shot_tick = int(stks[i])
        arr = death_map.get(sid)
        if arr is None or arr.size == 0:
            continue
        j = int(np.searchsorted(arr, shot_tick, side="left"))
        if j >= arr.size:
            continue
        dt = int(arr[j])
        if dt - shot_tick <= death_window_ticks:
            death_tick_for_shot[i] = dt

    moving_shots["death_tick"] = death_tick_for_shot
    moving_shots = moving_shots[moving_shots["death_tick"] >= 0].copy()
    if moving_shots.empty:
        pd.DataFrame().to_parquet(out_path, index=False)
        print("No moving shots that were followed by death in window. Wrote empty shoot_move_candidates.parquet.")
        return

    moving_shots["time_to_death_after_shot_s"] = (moving_shots["death_tick"] - moving_shots["tick"]) / tickrate

    # --- Collapse into incidents (one per death tick per player) ---
    incidents = []
    for (sid, death_tick), g in moving_shots.groupby(["user_steamid", "death_tick"], dropna=False):
        g = g.sort_values("tick")

        # (optional) further cluster within the death window if there are disjoint bursts of moving shots
        # but since this is "shots -> same death", it’s already cohesive; we’ll keep one incident.
        start_tick = int(g["tick"].min())
        end_tick = int(g["tick"].max())

        # If shots are very spread, keep a tighter window based on last “cluster”
        # (prevents weird cases where you moved-shot early then died 1.5s later after other stuff)
        # We'll keep last cluster that ends closest to death.
        ticks = g["tick"].to_numpy(dtype=np.int64, copy=False)
        # find last cluster start index
        cluster_start = 0
        for k in range(1, len(ticks)):
            if ticks[k] - ticks[k - 1] > gap_ticks:
                cluster_start = k
        ticks_last = ticks[cluster_start:]
        start_tick = int(ticks_last.min())
        end_tick = int(ticks_last.max())

        g_last = g[g["tick"].between(start_tick, end_tick)].copy()

        player = str(g_last["user_name"].iloc[0])
        weapon = _mode(g_last["weapon"]) or ""
        avg_speed = float(g_last["speed_2d"].mean())
        max_speed = float(g_last["speed_2d"].max())
        shots_n = int(len(g_last))

        # time-to-death: use the earliest moving shot in the kept cluster
        ttd = float((int(death_tick) - start_tick) / tickrate)

        incidents.append({
            "player": player,
            "steamid": int(sid),
            "weapon": weapon,
            "start_tick": start_tick,
            "end_tick": end_tick,
            "start_s": start_tick / tickrate,
            "end_s": end_tick / tickrate,
            "death_tick": int(death_tick),
            "death_s": int(death_tick) / tickrate,
            "time_to_death_after_moving_shot_s": ttd,
            "shots_while_moving": shots_n,
            "avg_speed_2d": avg_speed,
            "max_speed_2d": max_speed,
        })

    cand = pd.DataFrame(incidents)

    # Sort: most causal first (smallest time-to-death), then higher speed, then more shots
    cand = cand.sort_values(
        ["time_to_death_after_moving_shot_s", "max_speed_2d", "shots_while_moving"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    cand.to_parquet(out_path, index=False)
    print(f"[write] {out_path} ({len(cand)} incidents)")

    print("\n=== sanity ===")
    print(f"speed_thr={speed_thr:.1f} units/s, death_window={death_window_ticks} ticks ({args.death_window_s}s)")
    print("incidents describe:")
    if not cand.empty:
        print(cand[["shots_while_moving", "avg_speed_2d", "max_speed_2d", "time_to_death_after_moving_shot_s"]].describe())
        print("\nhead:")
        print(cand.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

