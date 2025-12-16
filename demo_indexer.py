from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd
from demoparser2 import DemoParser  # type: ignore

def _local_dt_from_mtime(p: Path) -> datetime:
    return datetime.fromtimestamp(p.stat().st_mtime).astimezone()

def _unwrap_parse_events(ret):
    """
    Your code uses: wf = parser.parse_events(["weapon_fire"]); wf = wf[0][1]
    So parse_events returns something like [(name, df), ...] or similar.
    This returns a dict[name] = df.
    """
    out: dict[str, pd.DataFrame] = {}
    if isinstance(ret, dict):
        for k, v in ret.items():
            out[str(k)] = v if isinstance(v, pd.DataFrame) else pd.DataFrame(v)
        return out

    if isinstance(ret, list) or isinstance(ret, tuple):
        for item in ret:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                name = str(item[0])
                df = item[1]
                out[name] = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        # sometimes ret is like [(idx, df)] without event name; ignore that case
        return out

    return out

def _pick_tick_col(df: pd.DataFrame) -> str | None:
    for c in ("tick", "ticks", "game_tick", "server_tick"):
        if c in df.columns:
            return c
    return None

def summarize_demo(demo_path: Path, tickrate: float = 64.0) -> dict:
    demo_path = demo_path.resolve()
    parser = DemoParser(str(demo_path))

    # Map name (you already confirmed this works in your env)
    map_name = "Unknown"
    try:
        hdr = parser.parse_header()
        if isinstance(hdr, dict):
            map_name = hdr.get("map_name") or hdr.get("map") or map_name
    except Exception:
        pass

    played_at = _local_dt_from_mtime(demo_path)

    # Events
    score = "?"
    duration_min = None

    try:
        ev_raw = parser.parse_events(["round_end", "player_death"])
        ev = _unwrap_parse_events(ev_raw)
        rounds = ev.get("round_end", pd.DataFrame())
        deaths = ev.get("player_death", pd.DataFrame())

        # duration (max tick we see)
        max_tick = None
        for df in (rounds, deaths):
            if df is None or df.empty:
                continue
            tc = _pick_tick_col(df)
            if tc:
                v = int(df[tc].max())
                max_tick = v if max_tick is None else max(max_tick, v)
        if max_tick is not None:
            duration_min = round((max_tick / tickrate) / 60.0, 1)

        # score from round_end winner counts
        if rounds is not None and not rounds.empty:
            for wcol in ("winner", "winner_side", "winning_side", "team_winner"):
                if wcol in rounds.columns:
                    w = rounds[wcol].astype(str).str.upper()
                    t_wins = int(w.isin(["T", "2", "TEAM_T", "TERRORIST"]).sum())
                    ct_wins = int(w.isin(["CT", "3", "TEAM_CT", "COUNTERTERRORIST"]).sum())
                    if t_wins + ct_wins > 0:
                        score = f"{t_wins}-{ct_wins}"
                    break
    except Exception:
        pass

    return {
        "path": str(demo_path),
        "filename": demo_path.name,
        "map": map_name,
        "played_at": played_at,
        "score": score,
        "duration_min": duration_min,
    }

def index_demos(demo_dir: Path, tickrate: float = 64.0) -> pd.DataFrame:
    demo_dir = demo_dir.expanduser().resolve()
    demos = sorted(demo_dir.glob("*.dem"), key=lambda p: p.stat().st_mtime, reverse=True)
    rows = [summarize_demo(p, tickrate=tickrate) for p in demos]
    return pd.DataFrame(rows)
