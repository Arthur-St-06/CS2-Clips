from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from demoparser2 import DemoParser  # type: ignore


# =========================
# Time helpers
# =========================

def _local_dt_from_mtime(p: Path) -> datetime:
    return datetime.fromtimestamp(p.stat().st_mtime).astimezone()


def played_at_from_dem_info_via_protoc(demo_path: Path) -> datetime | None:
    """
    Extracts match start time shown by CS2 UI from <demo>.dem.info.
    We observed: top-level protobuf field 2 is epoch seconds UTC.
      e.g. `2: 1765821408` -> 2025-12-15 09:56 local (PST)
    """
    dem = demo_path.expanduser().resolve()

    # demo.dem -> demo.dem.info
    info1 = Path(str(dem) + ".info")
    # some tools may use demo.dem + ".info" already; this covers both patterns
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

    # We want the top-level line: 2: <epoch>
    m = re.search(r"(?m)^2:\s*(\d+)\s*$", txt)
    if not m:
        return None

    epoch = int(m.group(1))
    # sanity range (adjust if needed, but this keeps garbage out)
    if not (1_600_000_000 <= epoch <= 2_200_000_000):
        return None

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
        for df in (rounds, deaths):
            if df is None or df.empty:
                continue
            tc = _pick_tick_col(df)
            if tc:
                v = int(df[tc].max())
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
    # sort by played_at (real) desc, fallback is mtime anyway
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
    """
    Candidates for matching:
    - played_at from dem.info (best)
    - file mtime (fallback)
    We also allow timezone shift brute-force because video mtime can be local vs weird.
    """
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
        out_dir / "coaching.txt",
        out_dir / "bursts.parquet",
        out_dir / "overspray_candidates.parquet",
    ]
    for f in files:
        try:
            if f.exists():
                f.unlink()
        except Exception:
            pass

    clips_dir = out_dir / "clips"
    if clips_dir.exists():
        for mp4 in clips_dir.glob("*.mp4"):
            try:
                mp4.unlink()
            except Exception:
                pass


def run_pipeline(demo: Path, video: Path, out: Path, *, player_filter: str, tickrate: float, top: int, pre: float, post: float):
    cmd1 = ["python3", "step1_bursts.py", "--demo", str(demo), "--out", str(out)]
    if player_filter.strip():
        cmd1 += ["--player", player_filter.strip()]
    subprocess.run(cmd1, check=True)

    cmd2 = ["python3", "step2_overspray.py", "--demo", str(demo), "--out", str(out), "--tickrate", str(float(tickrate))]
    subprocess.run(cmd2, check=True)

    # IMPORTANT: pass --demo so step3 can read .dem.info and write correct match time
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
    subprocess.run(cmd3, check=True)


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="CS2 Coach MVP", layout="wide")
st.title("CS2 Coach — MVP")

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
    tickrate = st.number_input("Tickrate", min_value=16.0, max_value=256.0, value=64.0, step=1.0)
    top = st.slider("Clips to export (top)", 1, 20, 5)
    pre = st.number_input("Clip pre-roll (sec)", min_value=0.0, max_value=15.0, value=3.0, step=0.5)
    post = st.number_input("Clip post-roll (sec)", min_value=0.0, max_value=15.0, value=2.0, step=0.5)

    st.divider()
    st.header("Auto-match demo ↔ video")
    max_window_h = st.slider("Max match window (hours)", 1, 24, 12)

    st.divider()
    if st.button("Clear Streamlit cache (fix stale times)"):
        st.cache_data.clear()
        st.rerun()


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

st.subheader("Games")
df_show = df.copy()
df_show["played_at"] = df_show["played_at"].astype(str)
df_show["file_mtime"] = df_show["file_mtime"].astype(str)
df_show = df_show[["filename", "map", "played_at", "played_at_source", "score", "duration_min", "file_mtime", "path"]]
st.dataframe(df_show, use_container_width=True, hide_index=True)

st.subheader("Select a game")
selected_path = st.selectbox(
    "Demo",
    options=df["path"].tolist(),
    format_func=lambda p: (
        f"{Path(p).name} — "
        f"{df.loc[df['path']==p,'map'].iloc[0]} — "
        f"{df.loc[df['path']==p,'score'].iloc[0]} — "
        f"{df.loc[df['path']==p,'played_at'].iloc[0]}"
    ),
)
demo_path = Path(selected_path)

match = None
if video_dir_p.exists():
    match = match_demo_to_video(
        demo_path,
        video_dir_p,
        max_window_hours=float(max_window_h)
    )

row = df[df["path"] == str(demo_path)].iloc[0]
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### Demo")
    st.write(f"**Map:** {row['map']}")
    st.write(f"**Played at:** {row['played_at']} ({row.get('played_at_source','?')})")
    st.write(f"**Score:** {row['score']}")
    st.write(f"**Duration:** {row['duration_min']} min")
    st.caption(f"File mtime (debug): {row['file_mtime']}")

with col2:
    st.markdown("### Matched recording")
    if match is None:
        st.warning("No close recording found (or videos folder missing).")
    else:
        st.write(f"**Video:** `{match.video_path.name}`")
        st.write(f"**Δ:** {match.delta_seconds:.1f}s")
        st.write(f"**Anchor used:** {match.used_anchor}")
        st.video(str(match.video_path))

st.divider()

out_dir = workspace_p / demo_path.stem
clips_dir = out_dir / "clips"
clips_dir.mkdir(parents=True, exist_ok=True)
coaching_txt = out_dir / "coaching.txt"

st.markdown("## Pipeline output")

cA, cB, cC = st.columns([1, 1, 2], gap="small")
with cA:
    show_existing = st.checkbox("Show existing outputs", value=False)
with cB:
    always_fresh = st.checkbox("Always start fresh on Run", value=True)
with cC:
    if st.button("Reset analysis (delete outputs)"):
        reset_analysis(out_dir)
        st.success("Cleared coaching.txt / parquet files / clips for this demo.")
        st.rerun()

if show_existing and coaching_txt.exists():
    st.markdown("### Coaching summary")
    st.code(coaching_txt.read_text(encoding="utf-8"), language="text")
elif show_existing:
    st.info("No coaching.txt yet. Run analysis below.")
else:
    st.info("Existing outputs hidden. Run analysis to generate fresh outputs (or enable 'Show existing outputs').")

st.markdown("### Clips")
if show_existing:
    clips = sorted(clips_dir.glob("*.mp4"))
    if not clips:
        st.info("No clips yet.")
    else:
        for c in clips:
            st.write(f"**{c.name}**")
            st.video(str(c))

st.divider()
st.markdown("## Run analysis")

can_run = match is not None
if not can_run:
    st.warning("Need a matched video to run automatically. (Or set a bigger match window.)")
else:
    if st.button("Run analysis for this demo", type="primary"):
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            if always_fresh:
                reset_analysis(out_dir)

            with st.spinner("Running step1 → step2 → step3..."):
                run_pipeline(
                    demo_path,
                    match.video_path,
                    out_dir,
                    player_filter=player_filter,
                    tickrate=float(tickrate),
                    top=int(top),
                    pre=float(pre),
                    post=float(post),
                )

            st.success("Done. New coaching summary + clips generated.")
            st.rerun()
        except subprocess.CalledProcessError as e:
            st.error(f"Pipeline failed (exit {e.returncode}). Check your terminal output.")
        except Exception as e:
            st.error(str(e))
