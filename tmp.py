
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from demoparser2 import DemoParser  # type: ignore


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
    req = ["tick", "player_name", "player_steamid", "X", "Y", "Z"]
    missing = [c for c in req if c not in pos.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}. Got: {list(pos.columns)}")

    df = pos[req].copy()
    df["tick"] = pd.to_numeric(df["tick"], errors="coerce").astype("Int64")
    df = df[df["tick"].notna()].copy()
    df["tick"] = df["tick"].astype(np.int64)

    # keep steamid nullable-safe
    df["player_steamid"] = pd.to_numeric(df["player_steamid"], errors="coerce").astype("Int64")
    df = df[df["player_steamid"].notna()].copy()
    df["player_steamid"] = df["player_steamid"].astype(np.int64)

    # coords
    for c in ("X", "Y", "Z"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["player_steamid", "tick"]).reset_index(drop=True)

    df["dX"] = df.groupby("player_steamid")["X"].diff()
    df["dY"] = df.groupby("player_steamid")["Y"].diff()
    df["dZ"] = df.groupby("player_steamid")["Z"].diff()

    # 2D is usually what matters for "moving while shooting"
    df["dist2d_per_tick"] = np.sqrt(df["dX"] ** 2 + df["dY"] ** 2)
    df["dist3d_per_tick"] = np.sqrt(df["dX"] ** 2 + df["dY"] ** 2 + df["dZ"] ** 2)

    df["speed_2d"] = df["dist2d_per_tick"] * float(tickrate)
    df["speed_3d"] = df["dist3d_per_tick"] * float(tickrate)

    df["speed_2d"] = df["speed_2d"].fillna(0.0)
    df["speed_3d"] = df["speed_3d"].fillna(0.0)

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True, help="Path to .dem")
    ap.add_argument("--out", default="out_sprays", help="Output directory (same style as overspray)")
    ap.add_argument("--tickrate", type=float, default=64.0)

    # sane defaults (tune later by watching clips)
    ap.add_argument("--speed-threshold", type=float, default=35.0,
                    help="Flag shot if speed_2d >= this (units/sec). Start conservative.")
    ap.add_argument("--exclude-weapons", default="knife|hegrenade|flashbang|smokegrenade|molotov|incgrenade|decoy|taser",
                    help="Regex of weapons to ignore")

    ap.add_argument("--csv", default="", help="Optional CSV output path")
    ap.add_argument("--limit", type=int, default=10, help="Rows to print")
    args = ap.parse_args()

    demo = Path(args.demo).expanduser().resolve()
    if not demo.exists():
        raise FileNotFoundError(demo)

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tickrate = float(args.tickrate)
    speed_thr = float(args.speed_threshold)

    print(f"[load] {demo.name}")
    parser = DemoParser(str(demo))

    # --- weapon_fire (shots) ---
    ev = unwrap_events(parser.parse_events(["weapon_fire"]))
    wf = ev.get("weapon_fire", pd.DataFrame())
    if wf is None or wf.empty:
        out_path = out_dir / "shooting_while_moving.parquet"
        pd.DataFrame().to_parquet(out_path)
        print("No weapon_fire events found. Wrote empty shooting_while_moving.parquet.")
        return

    required = ["tick", "user_name", "user_steamid", "weapon"]
    missing = [c for c in required if c not in wf.columns]
    if missing:
        raise RuntimeError(f"weapon_fire missing columns: {missing}. Available: {list(wf.columns)}")

    shots = wf[required].copy()
    shots["tick"] = pd.to_numeric(shots["tick"], errors="coerce")
    shots = shots[shots["tick"].notna()].copy()
    shots["tick"] = shots["tick"].astype(np.int64)

    shots["user_steamid"] = pd.to_numeric(shots["user_steamid"], errors="coerce")
    shots = shots[shots["user_steamid"].notna()].copy()
    shots["user_steamid"] = shots["user_steamid"].astype(np.int64)

    shots["user_name"] = shots["user_name"].astype(str)
    shots["weapon"] = shots["weapon"].astype(str)

    # ignore grenades/knife/etc
    if args.exclude_weapons.strip():
        shots = shots[~shots["weapon"].str.contains(args.exclude_weapons, case=False, na=False)].copy()

    shots = shots.sort_values(["user_steamid", "tick"]).reset_index(drop=True)
    shots["shot_idx"] = np.arange(len(shots), dtype=np.int64)
    shots["t_s"] = shots["tick"].astype(float) / tickrate

    print(f"[shots] {len(shots)} (after excludes)")

    # --- positions -> speeds ---
    ret = parser.parse_ticks(["tick", "player_name", "player_steamid", "X", "Y", "Z"])
    pos = unwrap_ticks(ret)
    print(f"[ticks] rows={len(pos)} cols={list(pos.columns)}")

    speeds = compute_speeds_from_positions(pos, tickrate=tickrate)

    # Keep only what we need for joining
    speeds_small = speeds[["tick", "player_steamid", "player_name", "speed_2d", "speed_3d"]].copy()

    # --- exact join logic ---
    # We join weapon_fire.tick to the *same tick* speed computed from position deltas.
    joined = shots.merge(
        speeds_small,
        left_on=["tick", "user_steamid"],
        right_on=["tick", "player_steamid"],
        how="left",
        validate="many_to_one",
    )

    # if we didn't find a speed for some ticks (rare), treat as unknown/0 and keep row
    joined["speed_2d"] = pd.to_numeric(joined["speed_2d"], errors="coerce").fillna(0.0)
    joined["speed_3d"] = pd.to_numeric(joined["speed_3d"], errors="coerce").fillna(0.0)

    joined["shooting_while_moving"] = joined["speed_2d"] >= speed_thr

    out_path = out_dir / "shooting_while_moving.parquet"
    joined.to_parquet(out_path, index=False)
    print(f"[write] {out_path} ({len(joined)} rows)")

    if args.csv:
        csvp = Path(args.csv).expanduser().resolve()
        csvp.parent.mkdir(parents=True, exist_ok=True)
        joined.to_csv(csvp, index=False)
        print(f"[write] {csvp}")

    # --- sanity ---
    flagged = int(joined["shooting_while_moving"].sum())
    total = len(joined)
    pct = (flagged / total * 100.0) if total else 0.0

    print("\n=== sanity ===")
    print(f"threshold speed_2d >= {speed_thr:.1f}")
    print(f"flagged: {flagged}/{total} ({pct:.1f}%)")
    print("speed_2d describe:")
    print(joined["speed_2d"].describe())

    print("\nflagged by player (top 10):")
    byp = (
        joined.groupby("user_name", dropna=False)["shooting_while_moving"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    print((byp * 100).round(1).to_string() + "  %")

    print("\nhead:")
    cols = ["tick", "t_s", "user_name", "user_steamid", "weapon", "speed_2d", "shooting_while_moving"]
    print(joined[cols].head(args.limit).to_string(index=False))


if __name__ == "__main__":
    main()
