from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from demoparser2 import DemoParser  # type: ignore

from concurrent.futures import ThreadPoolExecutor, as_completed

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

    map_name = "Unknown"
    try:
        hdr = parser.parse_header()
        if isinstance(hdr, dict):
            map_name = hdr.get("map_name") or hdr.get("map") or map_name
    except Exception:
        pass

    played_at = played_at_from_dem_info_via_protoc(demo_path)
    played_at_source = "dem.info" if played_at is not None else "mtime"
    if played_at is None:
        played_at = _local_dt_from_mtime(demo_path)

    file_mtime = _local_dt_from_mtime(demo_path)

    score = "?"
    duration_min = None

    try:
        ev_raw = parser.parse_events(["round_end", "player_death"])
        ev = _unwrap_parse_events(ev_raw)
        rounds = ev.get("round_end", pd.DataFrame())
        deaths = ev.get("player_death", pd.DataFrame())

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
        out_dir / "coaching.txt",
        out_dir / "bursts.parquet",
        out_dir / "overspray_candidates.parquet",
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
    video_anchor_s: float | None = None,
):
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

    cmd2m = [
        "python3", "step2_shoot_move.py",
        "--demo", str(demo),
        "--out", str(out),
        "--tickrate", str(float(tickrate)),
    ]
    if player_filter.strip():
        cmd2m += ["--player", player_filter.strip()]
    subprocess.run(cmd2m, check=True)

    cmd3m = [
        "python3", "step3_make_clips_movement.py",
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


# =========================
# Cache (video anchor)
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


def _get_or_compute_video_anchor(video_path: Path, out_dir: Path) -> float:
    cached = _load_cached_video_anchor(out_dir)
    if cached is not None:
        return cached

    cache_json = _sync_cache_path(out_dir)
    cmd = [
        "python3", "timer_flip_detector.py",
        "--video", str(video_path),
        "--cache-json", str(cache_json),
    ]
    p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True)
    return float(p.stdout.strip())


# =========================
# Batch helpers
# =========================

def _paths_for_demo_out(out_dir: Path) -> dict[str, Path]:
    return {
        "coaching_overspray": out_dir / "coaching.txt",
        "overspray_candidates": out_dir / "overspray_candidates.parquet",
        "clips_overspray": out_dir / "clips",
        "coaching_move": out_dir / "coaching_move.txt",
        "move_candidates": out_dir / "shoot_move_candidates.parquet",
        "clips_move": out_dir / "clips_move",
    }


def _is_already_analyzed(out_dir: Path) -> bool:
    p = _paths_for_demo_out(out_dir)
    # consider analyzed if we have any candidates or any clips from either pipeline
    has_cands = (p["overspray_candidates"].exists() and _count_parquet_rows(p["overspray_candidates"]) > 0) or (
        p["move_candidates"].exists() and _count_parquet_rows(p["move_candidates"]) > 0
    )
    has_clips = (p["clips_overspray"].exists() and any(p["clips_overspray"].glob("*.mp4"))) or (
        p["clips_move"].exists() and any(p["clips_move"].glob("*.mp4"))
    )
    # also treat "ran but no mistakes" as analyzed if files exist
    has_any_outputs = (
        p["overspray_candidates"].exists()
        or p["move_candidates"].exists()
        or p["coaching_overspray"].exists()
        or p["coaching_move"].exists()
    )
    return bool(has_clips or has_cands or has_any_outputs)


def _summarize_out_dir(out_dir: Path) -> dict:
    p = _paths_for_demo_out(out_dir)
    over_n = _count_parquet_rows(p["overspray_candidates"]) if p["overspray_candidates"].exists() else 0
    move_n = _count_parquet_rows(p["move_candidates"]) if p["move_candidates"].exists() else 0
    over_clips = len(list(p["clips_overspray"].glob("*.mp4"))) if p["clips_overspray"].exists() else 0
    move_clips = len(list(p["clips_move"].glob("*.mp4"))) if p["clips_move"].exists() else 0
    return {
        "overspray_candidates": int(over_n),
        "move_candidates": int(move_n),
        "overspray_clips": int(over_clips),
        "move_clips": int(move_clips),
    }


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="CS2 Coach MVP", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.25rem !important; }
      section[data-testid="stSidebar"] .block-container { padding-top: 2.25rem !important; }
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
    tickrate = 64
    top = st.slider("Clips to export (top)", 1, 20, 5)
    pre = st.number_input("Clip pre-roll (sec)", min_value=0.0, max_value=15.0, value=3.0, step=0.5)
    post = st.number_input("Clip post-roll (sec)", min_value=0.0, max_value=15.0, value=2.0, step=0.5)
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


tab_single, tab_batch = st.tabs(["Single demo", "Batch summary (matched videos only)"])


# =========================
# Single demo tab
# =========================
with tab_single:
    selected_path = st.selectbox(
        "Select a game",
        options=df["path"].tolist(),
        format_func=lambda p: (
            f"{df.loc[df['path'] == p, 'map'].iloc[0]} - "
            f"{_fmt_played_at(df.loc[df['path'] == p, 'played_at'].iloc[0])} - "
            f"{_fmt_duration_min(df.loc[df['path'] == p, 'duration_min'].iloc[0])}"
        ),
        key="single_select_demo",
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
        show = st.checkbox("Preview recording", value=False, key="show_preview_video_single")
        if show:
            st.video(str(match.video_path))

    st.divider()

    out_dir = workspace_p / demo_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    video_anchor_s: float | None = None
    if match is not None:
        try:
            video_anchor_s = _get_or_compute_video_anchor(match.video_path, out_dir)
            st.caption(f"Auto-sync anchor: {video_anchor_s:.3f}s (cached in {_sync_cache_path(out_dir).name})")
        except Exception as e:
            st.warning(f"Could not compute cached anchor yet: {e}")

    st.markdown("## Pipeline output")

    left, mid, right = st.columns([2, 1, 6], gap="small")

    with left:
        if st.button("Run analysis for this demo", type="primary", disabled=(match is None), key="run_single"):
            try:
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
                        video_anchor_s=video_anchor_s,
                    )

                st.success("Done. New coaching + clips generated.")
                st.rerun()
            except subprocess.CalledProcessError as e:
                st.error(f"Pipeline failed (exit {e.returncode}). Check your terminal output.")
            except Exception as e:
                st.error(str(e))

    with right:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear sync cache", key="clear_sync_single"):
                try:
                    _sync_cache_path(out_dir).unlink(missing_ok=True)
                except Exception:
                    pass
                st.success("Cleared sync cache for this demo.")
                st.rerun()

        with c2:
            if st.button("Reset analysis (delete outputs)", key="reset_single"):
                reset_analysis(out_dir)
                st.success("Cleared outputs for this demo.")
                st.rerun()

    p = _paths_for_demo_out(out_dir)
    coaching_overspray = p["coaching_overspray"]
    overspray_candidates = p["overspray_candidates"]
    clips_overspray = p["clips_overspray"]
    coaching_move = p["coaching_move"]
    move_candidates = p["move_candidates"]
    clips_move = p["clips_move"]

    st.markdown("## Mistakes")
    t1, t2 = st.tabs(["Overspray", "Shooting while moving"])

    with t1:
        _show_mistake_panel(
            title="Over-spraying instead of resetting",
            coaching_path=coaching_overspray,
            candidates_path=overspray_candidates,
            clips_dir=clips_overspray,
        )

    with t2:
        _show_mistake_panel(
            title="Shooting while moving (no full stop / no counter-strafe)",
            coaching_path=coaching_move,
            candidates_path=move_candidates,
            clips_dir=clips_move,
        )

# =========================
# Batch summary tab (Option B) — PARALLEL (up to 4 demos)
# =========================
with tab_batch:
    run_batch = st.button("Run batch analysis", type="primary")

    # Analyze last n demos
    n = len(df)
    workers = 4
    skip_done = False
    reset_before = True

    if "batch_rows" not in st.session_state:
        st.session_state["batch_rows"] = []

    def _analyze_one_demo(demo_path: Path, row_dict: dict) -> dict:
        out_dir = workspace_p / demo_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) match video (Option B: skip if no match)
        match = None
        if video_dir_p.exists():
            match = match_demo_to_video(
                demo_path,
                video_dir_p,
                max_window_hours=float(max_window_h),
            )

        if match is None:
            return {
                "demo": demo_path.name,
                "map": row_dict.get("map", ""),
                "played_at": _fmt_played_at(row_dict.get("played_at")),
                "status": "SKIP (no matched video)",
                "video": "",
                "overspray": 0,
                "shoot_move": 0,
                "overspray_clips": 0,
                "move_clips": 0,
            }

        # 2) skip already analyzed (optional)
        if skip_done and _is_already_analyzed(out_dir):
            s = _summarize_out_dir(out_dir)
            return {
                "demo": demo_path.name,
                "map": row_dict.get("map", ""),
                "played_at": _fmt_played_at(row_dict.get("played_at")),
                "status": "SKIP (already analyzed)",
                "video": match.video_path.name,
                "overspray": s["overspray_candidates"],
                "shoot_move": s["move_candidates"],
                "overspray_clips": s["overspray_clips"],
                "move_clips": s["move_clips"],
            }

        # 3) run analysis
        if reset_before:
            reset_analysis(out_dir)

        # cached anchor (computed once per demo/video if missing)
        video_anchor_s = _get_or_compute_video_anchor(match.video_path, out_dir)

        run_pipeline(
            demo_path,
            match.video_path,
            out_dir,
            player_filter=player_filter,
            tickrate=float(tickrate),
            top=int(top),
            pre=float(pre),
            post=float(post),
            video_anchor_s=video_anchor_s,
        )

        s = _summarize_out_dir(out_dir)
        return {
            "demo": demo_path.name,
            "map": row_dict.get("map", ""),
            "played_at": _fmt_played_at(row_dict.get("played_at")),
            "status": "OK",
            "video": match.video_path.name,
            "overspray": s["overspray_candidates"],
            "shoot_move": s["move_candidates"],
            "overspray_clips": s["overspray_clips"],
            "move_clips": s["move_clips"],
        }

    if run_batch:
        st.session_state["batch_rows"] = []

        demos = df.head(int(n)).copy().reset_index(drop=True)
        total = len(demos)

        progress = st.progress(0.0)
        status = st.empty()

        rows: list[dict] = []

        # Submit all jobs
        with ThreadPoolExecutor(max_workers=int(workers)) as ex:
            futures = []
            for _, r in demos.iterrows():
                demo_path = Path(str(r["path"]))
                row_dict = r.to_dict()
                futures.append(ex.submit(_analyze_one_demo, demo_path, row_dict))

            # Collect as they complete (UI updates here only)
            done = 0
            for fut in as_completed(futures):
                done += 1
                try:
                    rows.append(fut.result())
                except subprocess.CalledProcessError as e:
                    # We don't know which demo if the exception happens before returning; keep generic
                    rows.append({
                        "demo": "?",
                        "map": "",
                        "played_at": "",
                        "status": f"FAIL (exit {e.returncode})",
                        "video": "",
                        "overspray": 0,
                        "shoot_move": 0,
                        "overspray_clips": 0,
                        "move_clips": 0,
                    })
                except Exception as e:
                    rows.append({
                        "demo": "?",
                        "map": "",
                        "played_at": "",
                        "status": f"FAIL ({type(e).__name__})",
                        "video": "",
                        "overspray": 0,
                        "shoot_move": 0,
                        "overspray_clips": 0,
                        "move_clips": 0,
                    })

                status.write(f"Completed {done}/{total} (parallel workers: {workers})")
                progress.progress(done / max(1, total))

        # Sort newest first for nicer viewing
        def _sort_key(x: dict) -> str:
            # played_at is already formatted string; keep as-is (good enough)
            return str(x.get("played_at", ""))

        st.session_state["batch_rows"] = sorted(rows, key=_sort_key, reverse=True)

        status.empty()
        progress.empty()

    rows = st.session_state.get("batch_rows", [])
    if rows:
        res = pd.DataFrame(rows)

        ok = int((res["status"] == "OK").sum())
        skip_nm = int(res["status"].astype(str).str.startswith("SKIP (no matched").sum())
        skip_an = int(res["status"].astype(str).str.startswith("SKIP (already").sum())
        fail = int(res["status"].astype(str).str.startswith("FAIL").sum())

        total_over = int(res["overspray"].sum())
        total_move = int(res["shoot_move"].sum())
        total_clips = int(res["overspray_clips"].sum() + res["move_clips"].sum())

        c5, c6, c7 = st.columns(3)
        c5.metric("Total overspray candidates", total_over)
        c6.metric("Total shoot-move candidates", total_move)
        c7.metric("Total clips produced", total_clips)

        st.markdown("### Results")
        st.dataframe(
            res.loc[:, ["map", "played_at", "overspray", "shoot_move"]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Drilldown")
        ok_demos = res[res["status"].isin(["OK", "SKIP (already analyzed)"])].copy()
        if ok_demos.empty:
            st.info("No analyzable demos yet.")

        else:
            # Use full demo paths so we can format like single demo
            ok_demo_names = set(ok_demos["demo"].tolist())

            drill_paths = [p for p in df["path"].tolist() if Path(p).name in ok_demo_names]

            sel_path = st.selectbox(
                "Select a demo to view outputs",
                options=drill_paths,
                format_func=lambda p: (
                    f"{df.loc[df['path'] == p, 'map'].iloc[0]} - "
                    f"{_fmt_played_at(df.loc[df['path'] == p, 'played_at'].iloc[0])} - "
                    f"{_fmt_duration_min(df.loc[df['path'] == p, 'duration_min'].iloc[0])}"
                ),
                key="batch_drill_demo",
            )

            demo_path = Path(sel_path)
            out_dir = workspace_p / demo_path.stem
            p = _paths_for_demo_out(out_dir)

            st.write(f"Outputs folder: `{out_dir}`")

            t1, t2 = st.tabs(["Overspray", "Shooting while moving"])
            with t1:
                _show_mistake_panel(
                    title="Over-spraying instead of resetting",
                    coaching_path=p["coaching_overspray"],
                    candidates_path=p["overspray_candidates"],
                    clips_dir=p["clips_overspray"],
                )
            with t2:
                _show_mistake_panel(
                    title="Shooting while moving (no full stop / no counter-strafe)",
                    coaching_path=p["coaching_move"],
                    candidates_path=p["move_candidates"],
                    clips_dir=p["clips_move"],
                )
    else:
        st.info("Run a batch to generate an aggregate summary.")
