from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from demoparser2 import DemoParser  # type: ignore
import json

# =========================
# Time helpers
# =========================

def _local_dt_from_mtime(p: Path) -> datetime:
    return datetime.fromtimestamp(p.stat().st_mtime).astimezone()


def played_at_from_dem_info_via_protoc(demo_path: Path) -> datetime | None:
    dem = demo_path.expanduser().resolve()

    info1 = Path(str(dem) + ".info")  # demo.dem -> demo.dem.info
    info2 = dem.with_suffix(dem.suffix + ".info")
    info = info1 if info1.exists() else info2 if info2.exists() else None
    if info is None:
        return None

    try:
        p = subprocess.run(
            ["protoc", "--decode_raw"],
            input=info.read_bytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except Exception:
        return None

    txt = p.stdout.decode("utf-8", errors="replace")
    m = re.search(r"(?m)^2:\s*(\d+)\s*$", txt)
    if not m:
        return None

    epoch = int(m.group(1))
    return datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone()


# =========================
# Demo parsing helpers
# =========================

def _unwrap_parse_events(ret) -> dict[str, pd.DataFrame]:
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


def _pick_tick_col(df: pd.DataFrame) -> str | None:
    for c in ("tick", "ticks", "game_tick", "server_tick"):
        if c in df.columns:
            return c
    return None


# =========================
# Demo indexing (map/time/score)
# =========================

def summarize_demo(demo_path: Path, tickrate: float = 64.0) -> dict:
    demo_path = demo_path.resolve()
    parser = DemoParser(str(demo_path))

    # Map name
    map_name = "Unknown"
    try:
        hdr = parser.parse_header()
        if isinstance(hdr, dict):
            map_name = hdr.get("map_name") or hdr.get("map") or map_name
    except Exception:
        pass

    # Played-at (REAL): prefer .dem.info (CS2 UI time), fallback to mtime
    played_at = played_at_from_dem_info_via_protoc(demo_path)
    played_at_source = "dem.info" if played_at is not None else "mtime"
    if played_at is None:
        played_at = _local_dt_from_mtime(demo_path)

    # Also keep file mtime for debugging
    file_mtime = _local_dt_from_mtime(demo_path)

    score = "?"
    duration_min = None

    try:
        ev_raw = parser.parse_events(["round_end", "player_death"])
        ev = _unwrap_parse_events(ev_raw)
        rounds = ev.get("round_end", pd.DataFrame())
        deaths = ev.get("player_death", pd.DataFrame())

        # duration: max tick over seen events
        max_tick = None
        for dfi in (rounds, deaths):
            if dfi is None or dfi.empty:
                continue
            tc = _pick_tick_col(dfi)
            if tc:
                v = int(dfi[tc].max())
                max_tick = v if max_tick is None else max(max_tick, v)

        if max_tick is not None:
            duration_min = round((max_tick / tickrate) / 60.0, 1)

        # score: count round winners if available
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
        "played_at_source": played_at_source,
        "file_mtime": file_mtime,
        "score": score,
        "duration_min": duration_min,
    }


def index_demos(demo_dir: Path, tickrate: float = 64.0) -> pd.DataFrame:
    demo_dir = demo_dir.expanduser().resolve()
    demos = sorted(demo_dir.glob("*.dem"), key=lambda p: p.stat().st_mtime, reverse=True)
    rows = [summarize_demo(p, tickrate=tickrate) for p in demos]
    df = pd.DataFrame(rows)
    if not df.empty and "played_at" in df.columns:
        df = df.sort_values("played_at", ascending=False)
    return df


# =========================
# Demo ↔ Video matching
# =========================

@dataclass
class MatchResult:
    video_path: Path
    delta_seconds: float
    used_anchor: str


def _local_tz():
    return datetime.now().astimezone().tzinfo


def _mtime_local(p: Path) -> datetime:
    return datetime.fromtimestamp(p.stat().st_mtime, tz=_local_tz())


def _list_videos(video_dir: Path) -> list[Path]:
    exts = {".mkv", ".mp4", ".mov"}
    vids = [p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    vids.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return vids


def _demo_candidate_times(demo_path: Path) -> list[tuple[str, datetime]]:
    cands: list[tuple[str, datetime]] = []
    played_at = played_at_from_dem_info_via_protoc(demo_path)
    if played_at is not None:
        cands.append(("deminfo", played_at))
    cands.append(("mtime", _mtime_local(demo_path)))
    return cands


def match_demo_to_video(
    demo_path: Path,
    video_dir: Path,
    *,
    max_window_hours: float = 12.0,
) -> MatchResult | None:
    demo_path = demo_path.resolve()
    video_dir = video_dir.resolve()

    vids = _list_videos(video_dir)
    if not vids:
        return None

    candidates = _demo_candidate_times(demo_path)

    best: tuple[Path, float, str] | None = None
    for anchor_name, demo_dt in candidates:
        for v in vids:
            v_dt = _mtime_local(v)
            abs_delta = abs((v_dt - demo_dt).total_seconds())
            if best is None or abs_delta < best[1]:
                best = (v, abs_delta, anchor_name)

    if best is None:
        return None

    v, abs_delta, used_anchor = best
    if abs_delta > max_window_hours * 3600:
        return None

    return MatchResult(video_path=v, delta_seconds=abs_delta, used_anchor=used_anchor)


# =========================
# Reset / Pipeline
# =========================

def reset_analysis(out_dir: Path):
    files = [
        # overspray
        out_dir / "coaching.txt",
        out_dir / "bursts.parquet",
        out_dir / "overspray_candidates.parquet",
        # movement
        out_dir / "shoot_move_candidates.parquet",
        out_dir / "coaching_move.txt",
        out_dir / "shooting_while_moving.parquet",
        out_dir / "shooting_while_moving.csv",
    ]
    for f in files:
        try:
            if f.exists():
                f.unlink()
        except Exception:
            pass

    for d in (out_dir / "clips", out_dir / "clips_move"):
        if d.exists():
            for mp4 in d.glob("*.mp4"):
                try:
                    mp4.unlink()
                except Exception:
                    pass


def run_pipeline(
    demo: Path,
    video: Path,
    out: Path,
    *,
    player_filter: str,
    tickrate: float,
    top: int,
    pre: float,
    post: float,
    video_anchor_s: float | None = None
):
    # Overspray
    cmd1 = ["python3", "step1_bursts.py", "--demo", str(demo), "--out", str(out)]
    if player_filter.strip():
        cmd1 += ["--player", player_filter.strip()]
    subprocess.run(cmd1, check=True)

    cmd2 = [
        "python3", "step2_overspray.py",
        "--demo", str(demo),
        "--out", str(out),
        "--tickrate", str(float(tickrate)),
    ]
    subprocess.run(cmd2, check=True)

    # Movement (death-linked incidents)
    cmd2m = [
        "python3", "step2_shoot_move.py",
        "--demo", str(demo),
        "--out", str(out),
        "--tickrate", str(float(tickrate)),
    ]
    if player_filter.strip():
        cmd2m += ["--player", player_filter.strip()]
    subprocess.run(cmd2m, check=True)

    # Clip movement incidents
    cmd3m = [
        "python3", "step3_make_clips_movement.py",  # ensure file name matches your repo
        "--out", str(out),
        "--demo", str(demo),
        "--video", str(video),
        "--tickrate", str(float(tickrate)),
        "--top", str(int(top)),
        "--pre", str(float(pre)),
        "--post", str(float(post)),
    ]
    if player_filter.strip():
        cmd3m += ["--prefer-player", player_filter.strip()]
    if video_anchor_s is not None:
        cmd3m += ["--video-anchor-s", str(float(video_anchor_s))]

    subprocess.run(cmd3m, check=True)

    # Clip overspray incidents
    cmd3 = [
        "python3", "step3_make_clips.py",
        "--out", str(out),
        "--demo", str(demo),
        "--video", str(video),
        "--tickrate", str(float(tickrate)),
        "--top", str(int(top)),
        "--pre", str(float(pre)),
        "--post", str(float(post)),
    ]
    if player_filter.strip():
        cmd3 += ["--prefer-player", player_filter.strip()]
    if video_anchor_s is not None:
        cmd3 += ["--video-anchor-s", str(float(video_anchor_s))]

    subprocess.run(cmd3, check=True)


# =========================
# UI helpers
# =========================

def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def _count_parquet_rows(p: Path) -> int:
    try:
        df = pd.read_parquet(p)
        return len(df)
    except Exception:
        return 0


def _show_mistake_panel(
    *,
    title: str,
    coaching_path: Path,
    candidates_path: Path,
    clips_dir: Path,
):
    st.markdown(f"### {title}")

    cand_n = _count_parquet_rows(candidates_path) if candidates_path.exists() else 0
    clip_n = len(list(clips_dir.glob("*.mp4"))) if clips_dir.exists() else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Candidates", cand_n)
    c2.metric("Clips", clip_n)
    c3.metric("Status", "Ready" if clip_n > 0 else ("Detected" if cand_n > 0 else "None"))

    if coaching_path.exists():
        st.markdown("**Coaching**")
        st.code(_read_text(coaching_path), language="text")
    else:
        st.info("No coaching file yet. Run analysis.")

    with st.expander("Show clips", expanded=False):
        clips = sorted(clips_dir.glob("*.mp4"))
        if not clips:
            st.info("No clips yet.")
        else:
            for c in clips:
                st.write(f"**{c.name}**")
                st.video(str(c))


def _fmt_duration_min(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "? min"
    try:
        return f"{float(v):.1f} min"
    except Exception:
        return "? min"


def _fmt_played_at(v) -> str:
    try:
        if hasattr(v, "to_pydatetime"):
            v = v.to_pydatetime()
        if isinstance(v, datetime):
            return v.astimezone().replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
        return str(v)
    except Exception:
        return "Unknown"

# =========================
# Cache
# =========================

def _sync_cache_path(out_dir: Path) -> Path:
    return out_dir / "sync_offset.json"

def _load_cached_video_anchor(out_dir: Path) -> float | None:
    p = _sync_cache_path(out_dir)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        v = data.get("video_anchor_s")
        return float(v) if v is not None else None
    except Exception:
        return None

# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="CS2 Coach MVP", layout="wide")

st.markdown(
    """
    <style>
      /* Keep MAIN content safely below the Streamlit header */
      .block-container { padding-top: 2.25rem !important; }

      /* Make SIDEBAR match the same top padding */
      section[data-testid="stSidebar"] .block-container {
        padding-top: 2.25rem !important;
      }

      /* Don’t kill the title spacing (it causes clipping) */
      h1 { margin-top: 0.5rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("CS2 Coach - MVP")

with st.sidebar:
    st.header("Inputs")

    st.caption("CS2 demos folder (Steam replays). Example:")
    st.code("~/.steam/steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays", language="text")

    demo_dir = st.text_input(
        "CS2 demos folder",
        value="~/.steam/steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays",
    )
    video_dir = st.text_input("OBS recordings folder", value="~/Videos")
    workspace_dir = st.text_input("Workspace/output folder", value="~/projects/goClips/out_mvp")

    st.divider()
    st.header("Pipeline settings")
    player_filter = st.text_input("Player filter (name substring)", value="")
    # Not sure if should add it to the UI TODO
    # tickrate = st.number_input("Tickrate", min_value=16.0, max_value=256.0, value=64.0, step=1.0)
    tickrate = 64
    top = st.slider("Clips to export (top)", 1, 20, 5)
    pre = st.number_input("Clip pre-roll (sec)", min_value=0.0, max_value=15.0, value=3.0, step=0.5)
    post = st.number_input("Clip post-roll (sec)", min_value=0.0, max_value=15.0, value=2.0, step=0.5)

    # Not sure if should add it to the UI TODO
    # st.divider()
    # st.header("Auto-match demo ↔ video")
    # max_window_h = st.slider("Max match window (hours)", 1, 24, 12)
    max_window_h = 12

demo_dir_p = Path(demo_dir).expanduser()
video_dir_p = Path(video_dir).expanduser()
workspace_p = Path(workspace_dir).expanduser()
workspace_p.mkdir(parents=True, exist_ok=True)


@st.cache_data(show_spinner=False)
def load_index(demo_dir_str: str, tickrate_val: float) -> pd.DataFrame:
    return index_demos(Path(demo_dir_str), tickrate=tickrate_val)


try:
    df = load_index(str(demo_dir_p), float(tickrate))
except Exception as e:
    st.error(f"Failed to index demos: {e}")
    st.stop()

if df.empty:
    st.info("No .dem files found in that folder.")
    st.stop()

selected_path = st.selectbox(
    "Select a game",
    options=df["path"].tolist(),
    format_func=lambda p: (
        f"{df.loc[df['path'] == p, 'map'].iloc[0]} - "
        f"{_fmt_played_at(df.loc[df['path'] == p, 'played_at'].iloc[0])} - "
        f"{_fmt_duration_min(df.loc[df['path'] == p, 'duration_min'].iloc[0])}"
    ),
)
demo_path = Path(selected_path)

match = None
if video_dir_p.exists():
    match = match_demo_to_video(
        demo_path,
        video_dir_p,
        max_window_hours=float(max_window_h),
    )

if match is None:
    st.warning("No close recording found (or videos folder missing).")
else:
    st.write(f"Matched recording: `{match.video_path.name}`")
    show = st.checkbox("Preview recording", value=False, key="show_preview_video")
    if show:
        st.video(str(match.video_path))

st.divider()

out_dir = workspace_p / demo_path.stem
out_dir.mkdir(parents=True, exist_ok=True)

video_anchor_s: float | None = None

if match is not None:
    # Fast path: reuse cache
    video_anchor_s = _load_cached_video_anchor(out_dir)

    # Slow path: compute once (detector also writes cache)
    if video_anchor_s is None:
        cache_json = _sync_cache_path(out_dir)
        cmd = [
            "python3", "timer_flip_detector.py",
            "--video", str(match.video_path),
            "--cache-json", str(cache_json),
            # keep args here if you override defaults; otherwise you can omit them
        ]
        p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True)
        video_anchor_s = float(p.stdout.strip())

    st.caption(f"Auto-sync anchor: {video_anchor_s:.3f}s (cached in { _sync_cache_path(out_dir).name })")

st.markdown("## Pipeline output")

# Run/Reset row (replaces the two checkboxes + bottom run section)
left, mid, right = st.columns([2, 1, 6], gap="small")

with left:
    if st.button("Run analysis for this demo", type="primary", disabled=(match is None)):
        try:
            # Always start fresh
            reset_analysis(out_dir)

            with st.spinner("Running overspray + movement pipelines..."):
                run_pipeline(
                    demo_path,
                    match.video_path,  # type: ignore[arg-type]
                    out_dir,
                    player_filter=player_filter,
                    tickrate=float(tickrate),
                    top=int(top),
                    pre=float(pre),
                    post=float(post),
                    video_anchor_s=video_anchor_s
                )

            st.success("Done. New coaching + clips generated.")
            st.rerun()
        except subprocess.CalledProcessError as e:
            st.error(f"Pipeline failed (exit {e.returncode}). Check your terminal output.")
        except Exception as e:
            st.error(str(e))

with right:
    if st.button("Reset analysis (delete outputs)"):
        reset_analysis(out_dir)
        st.success("Cleared outputs for this demo.")
        st.rerun()

# Paths for tabs
coaching_overspray = out_dir / "coaching.txt"
overspray_candidates = out_dir / "overspray_candidates.parquet"
clips_overspray = out_dir / "clips"

coaching_move = out_dir / "coaching_move.txt"
move_candidates = out_dir / "shoot_move_candidates.parquet"
clips_move = out_dir / "clips_move"

st.markdown("## Mistakes")
tab1, tab2 = st.tabs(["Overspray", "Shooting while moving"])

with tab1:
    _show_mistake_panel(
        title="Over-spraying instead of resetting",
        coaching_path=coaching_overspray,
        candidates_path=overspray_candidates,
        clips_dir=clips_overspray,
    )

with tab2:
    _show_mistake_panel(
        title="Shooting while moving (no full stop / no counter-strafe)",
        coaching_path=coaching_move,
        candidates_path=move_candidates,
        clips_dir=clips_move,
    )
