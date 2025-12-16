#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import importlib
import pandas as pd


def _load_demoparser(demo_path: str):
    """
    Try common demoparser2 entry points.
    Returns an instance that can be used to parse header/events.
    """
    dp = importlib.import_module("demoparser2")

    # Common class names across versions
    for cls_name in ("DemoParser", "Demo", "DemoFile", "Parser"):
        if hasattr(dp, cls_name):
            cls = getattr(dp, cls_name)
            try:
                return cls(demo_path)
            except TypeError:
                # some versions need keyword arg like path=...
                try:
                    return cls(path=demo_path)
                except Exception:
                    pass
            except Exception:
                pass

    raise RuntimeError("Could not construct a demoparser2 parser instance. Check your demoparser2 version.")


def _try_get_header(parser) -> dict:
    """
    Best-effort header extraction.
    """
    for attr in ("header", "demo_header", "info"):
        if hasattr(parser, attr):
            h = getattr(parser, attr)
            if isinstance(h, dict):
                return h
            # sometimes it's an object with fields
            try:
                return dict(h)
            except Exception:
                pass

    for fn in ("parse_header", "get_header", "read_header"):
        if hasattr(parser, fn):
            try:
                h = getattr(parser, fn)()
                if isinstance(h, dict):
                    return h
                try:
                    return dict(h)
                except Exception:
                    pass
            except Exception:
                pass

    return {}


def _try_parse_events(parser, event_names: list[str]) -> dict[str, pd.DataFrame]:
    """
    Best-effort event extraction. Returns dict[event_name] = DataFrame.
    """
    # Newer style: parse_events([...]) -> dict or DataFrame
    for fn in ("parse_events", "parse_event", "events", "get_events"):
        if hasattr(parser, fn):
            f = getattr(parser, fn)
            try:
                # Some APIs: parse_events(["a","b"]) -> dict
                out = f(event_names)
                if isinstance(out, dict):
                    return {k: (v if isinstance(v, pd.DataFrame) else pd.DataFrame(v)) for k, v in out.items()}
                # Some APIs: parse_event("player_death") -> df (single)
                if isinstance(out, pd.DataFrame):
                    # ambiguous: treat as the first requested
                    return {event_names[0]: out}
            except TypeError:
                # Maybe needs keyword args
                try:
                    out = f(events=event_names)
                    if isinstance(out, dict):
                        return {k: (v if isinstance(v, pd.DataFrame) else pd.DataFrame(v)) for k, v in out.items()}
                except Exception:
                    pass
            except Exception:
                pass

    # Fallback: try parsing one-by-one if a single-event method exists
    for single_fn in ("parse_event", "get_event"):
        if hasattr(parser, single_fn):
            f = getattr(parser, single_fn)
            res: dict[str, pd.DataFrame] = {}
            for name in event_names:
                try:
                    df = f(name)
                    if isinstance(df, pd.DataFrame):
                        res[name] = df
                    else:
                        res[name] = pd.DataFrame(df)
                except Exception:
                    res[name] = pd.DataFrame()
            return res

    return {name: pd.DataFrame() for name in event_names}


def _pick_tick_col(df: pd.DataFrame) -> str | None:
    for c in ("tick", "ticks", "game_tick", "server_tick"):
        if c in df.columns:
            return c
    return None


def _infer_map(header: dict) -> str:
    for k in ("map_name", "map", "level_name", "servername", "server_name"):
        v = header.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "Unknown"


def summarize_demo(demo_path: str, tickrate: float = 64.0, steamid: str | None = None) -> dict:
    p = Path(demo_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    mtime = datetime.fromtimestamp(p.stat().st_mtime)
    out: dict = {
        "path": str(p.resolve()),
        "filename": p.name,
        "played_at": mtime.isoformat(timespec="seconds"),
        "tickrate": float(tickrate),
    }

    parser = _load_demoparser(str(p))

    header = _try_get_header(parser)
    out["map"] = _infer_map(header)

    events = _try_parse_events(parser, ["player_death", "round_end"])
    deaths = events.get("player_death", pd.DataFrame())
    rounds = events.get("round_end", pd.DataFrame())

    # ---- duration ----
    max_tick = None
    for df in (deaths, rounds):
        if df is None or df.empty:
            continue
        tc = _pick_tick_col(df)
        if tc:
            v = int(df[tc].max())
            max_tick = v if max_tick is None else max(max_tick, v)

    if max_tick is not None:
        out["duration_s"] = round(max_tick / tickrate, 3)
        out["duration_min"] = round(out["duration_s"] / 60.0, 1)
    else:
        out["duration_s"] = None
        out["duration_min"] = None

    # ---- score/result (best-effort) ----
    out["score"] = "?"
    out["result"] = "?"

    if rounds is not None and not rounds.empty:
        # Common possibilities:
        # - columns like t_score/ct_score
        # - columns like team1_score/team2_score
        # - winner side + running score
        score_cols_sets = [
            ("t_score", "ct_score"),
            ("terrorist_score", "counterterrorist_score"),
            ("team1_score", "team2_score"),
            ("score_t", "score_ct"),
        ]
        found = False
        for a, b in score_cols_sets:
            if a in rounds.columns and b in rounds.columns:
                t = int(rounds[a].iloc[-1])
                ct = int(rounds[b].iloc[-1])
                out["score"] = f"{t}-{ct}"
                found = True
                break

        # If no explicit score columns, try count winners
        if not found:
            for wcol in ("winner", "winner_side", "winning_side", "team_winner"):
                if wcol in rounds.columns:
                    # Winner might be "T"/"CT", 2/3, or strings.
                    w = rounds[wcol].astype(str)
                    t_wins = int((w.str.upper().isin(["T", "2", "TEAM_T", "TERRORIST"])).sum())
                    ct_wins = int((w.str.upper().isin(["CT", "3", "TEAM_CT", "COUNTERTERRORIST"])).sum())
                    if t_wins + ct_wins > 0:
                        out["score"] = f"{t_wins}-{ct_wins}"
                        found = True
                    break

    # ---- player stats (K/D + HS%) ----
    if steamid and deaths is not None and not deaths.empty:
        # Try common column names
        atk_cols = [c for c in ("attacker_steamid", "attackerSteamId", "attacker_xuid", "attacker") if c in deaths.columns]
        vic_cols = [c for c in ("victim_steamid", "victimSteamId", "victim_xuid", "victim") if c in deaths.columns]

        if atk_cols and vic_cols:
            atk = deaths[atk_cols[0]].astype(str)
            vic = deaths[vic_cols[0]].astype(str)
            sid = str(steamid)

            k = int((atk == sid).sum())
            d = int((vic == sid).sum())
            out["kills"] = k
            out["deaths"] = d
            out["kd"] = round((k / d), 2) if d else float(k)

            hs_col = None
            for c in ("headshot", "is_headshot", "headShot"):
                if c in deaths.columns:
                    hs_col = c
                    break
            if hs_col:
                hs_k = int(((atk == sid) & (deaths[hs_col] == True)).sum())
                out["hs_pct"] = round(100.0 * hs_k / k, 1) if k else 0.0
            else:
                out["hs_pct"] = None

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True, help="Path to .dem")
    ap.add_argument("--tickrate", type=float, default=64.0)
    ap.add_argument("--steamid", default=None, help="Optional: your SteamID/XUID to compute K/D + HS%")
    args = ap.parse_args()

    info = summarize_demo(args.demo, tickrate=args.tickrate, steamid=args.steamid)

    # Pretty print
    print("\n=== Demo Summary ===")
    for k in (
        "filename", "map", "played_at", "duration_min", "score", "result",
        "kills", "deaths", "kd", "hs_pct", "path"
    ):
        if k in info:
            print(f"{k:>12}: {info[k]}")

    # Also write JSON next to demo
    out_json = Path(args.demo).with_suffix(".summary.json")
    import json
    out_json.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nWrote: {out_json}")


if __name__ == "__main__":
    main()

