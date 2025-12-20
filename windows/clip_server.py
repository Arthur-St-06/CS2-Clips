import os
import time
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# =========================
# Shared auth
# =========================
TOKEN = os.environ.get("CLIP_TOKEN", "token")

# =========================
# Static config (edit these)
# =========================
CS2_EXE = Path(os.environ.get(
    "CS2_EXE",
    r"D:\SteamLibrary\steamapps\common\Counter-Strike Global Offensive\game\bin\win64\cs2.exe"
))
HLAE_EXE = Path(os.environ.get("HLAE_EXE", r"C:\Program Files (x86)\HLAE\HLAE.exe"))
HOOK_DLL = Path(os.environ.get("HOOK_DLL", r"C:\Program Files (x86)\HLAE\x64\AfxHookSource2.dll"))
OUT_DIR = Path(os.environ.get("HLAE_OUT_DIR", r"D:\hlae_out"))
FFMPEG_EXE = os.environ.get("FFMPEG_EXE", "ffmpeg")

TICKRATE = float(os.environ.get("TICKRATE", "64"))
FPS = int(os.environ.get("FPS", "30"))
WARMUP_S = float(os.environ.get("WARMUP_S", "2.0"))
EXTRA_LAUNCH = os.environ.get("EXTRA_LAUNCH", "-steam -insecure -novid -nojoy -console")
CLEANUP_EXTRA_TAKES = os.environ.get("CLEANUP_EXTRA_TAKES", "1") == "1"

# Safety limits
MIN_DURATION_S = float(os.environ.get("MIN_DURATION_S", "1.0"))
MAX_DURATION_S = float(os.environ.get("MAX_DURATION_S", "60.0"))
MIN_START_S = float(os.environ.get("MIN_START_S", "0.0"))
MAX_START_S = float(os.environ.get("MAX_START_S", "99999.0"))

# Streaming/monitoring tuning
TAKE_POLL_S = float(os.environ.get("TAKE_POLL_S", "0.25"))
WAIT_FRAMES_TIMEOUT_S = float(os.environ.get("WAIT_FRAMES_TIMEOUT_S", "30.0"))
WAIT_STABLE_TIMEOUT_S = float(os.environ.get("WAIT_STABLE_TIMEOUT_S", "120.0"))
STABLE_WINDOW_S = float(os.environ.get("STABLE_WINDOW_S", "1.25"))

# Demo registry (safe)
DEMO_MAP: Dict[str, Path] = {
    "demo392": Path(r"D:\SteamLibrary\steamapps\common\Counter-Strike Global Offensive\game\csgo\replays\match730_003792077342759714824_1663050728_392.dem"),
}
DEMO_DIR = os.environ.get("DEMO_DIR")
if DEMO_DIR:
    ddir = Path(DEMO_DIR).resolve()
    if ddir.exists():
        for p in ddir.glob("*.dem"):
            DEMO_MAP[p.stem] = p


# =========================
# Clip generator core
# =========================
@dataclass
class Job:
    username: str
    cs2_exe: Path
    demo_path: Path
    hlae_exe: Path
    hook_dll: Path

    tickrate: int = 64
    warmup_s: float = 2.0

    out_dir: Path = Path(r"D:\hlae_out")
    cfg_name: str = "auto_record.cfg"
    extra_launch: str = "-steam -insecure -novid -nojoy -console"

    fps: int = 30
    ffmpeg_exe: str = "ffmpeg"
    cleanup_extra_takes: bool = True


def cs2_cfg_dir_from_cs2_exe(cs2_exe: Path) -> Path:
    game_dir = cs2_exe.parent.parent.parent
    return game_dir / "csgo" / "cfg"


def _list_take_dirs(out_dir: Path) -> list[Path]:
    if not out_dir.exists():
        return []
    return sorted(
        [p for p in out_dir.iterdir() if p.is_dir() and p.name.lower().startswith("take")],
        key=lambda p: p.stat().st_mtime,
    )


def _count_tga_frames(take_dir: Path) -> int:
    return sum(1 for _ in take_dir.glob("*.tga"))


def _wait_for_frames(take_dir: Path, timeout_s: float = WAIT_FRAMES_TIMEOUT_S) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if any(take_dir.glob("*.tga")):
            return
        time.sleep(TAKE_POLL_S)
    raise RuntimeError(f"No .tga frames appeared in {take_dir} within {timeout_s:.0f}s")


def _latest_tga_mtime_and_count(take_dir: Path) -> Tuple[int, float]:
    cnt = 0
    latest = 0.0
    for p in take_dir.glob("*.tga"):
        cnt += 1
        try:
            mt = p.stat().st_mtime
        except OSError:
            continue
        if mt > latest:
            latest = mt
    return cnt, latest


def _wait_for_frames_stable(
    take_dir: Path,
    stable_window_s: float = STABLE_WINDOW_S,
    timeout_s: float = WAIT_STABLE_TIMEOUT_S,
) -> None:
    """
    Why: ffmpeg must not read while HLAE is still writing frames.
    Strategy: wait until (frame_count, latest_frame_mtime) stops changing for stable_window_s.
    """
    _wait_for_frames(take_dir, timeout_s=min(timeout_s, WAIT_FRAMES_TIMEOUT_S))

    t0 = time.time()
    last_cnt, last_mt = _latest_tga_mtime_and_count(take_dir)
    stable_since = time.time()

    while time.time() - t0 < timeout_s:
        time.sleep(TAKE_POLL_S)

        cnt, mt = _latest_tga_mtime_and_count(take_dir)

        if cnt == last_cnt and mt == last_mt and cnt > 0:
            if time.time() - stable_since >= stable_window_s:
                return
        else:
            last_cnt, last_mt = cnt, mt
            stable_since = time.time()

    raise RuntimeError(f"Frames in {take_dir} did not stabilize within {timeout_s:.0f}s")


def convert_take_to_mp4(job: Job, take_dir: Path) -> Path:
    _wait_for_frames_stable(take_dir)
    out_mp4 = take_dir / "out.mp4"

    cmd = [
        job.ffmpeg_exe, "-y",
        "-framerate", str(job.fps),
        "-start_number", "0",
        "-i", "%05d.tga",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "30",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    subprocess.run(cmd, cwd=take_dir, check=True)
    return out_mp4


def _safe_filename(name: str) -> str:
    name = name.replace("\\", "/").split("/")[-1]
    safe = "".join(c for c in name if c.isalnum() or c in ("-", "_", ".", " ")).strip()
    return safe or "clip.mp4"


def write_auto_record_cfg_multi(
    username: str,
    cfg_path: Path,
    segments: List[Tuple[int, int]],
    out_dir: Path,
    fps: int,
    tickrate: int,
    warmup_s: float,
) -> None:
    if not segments:
        raise RuntimeError("No segments provided")

    out_dir.mkdir(parents=True, exist_ok=True)
    segments_sorted = sorted(segments, key=lambda x: x[0])

    warmup_ticks = max(0, int(round(warmup_s * tickrate)))
    first_start = segments_sorted[0][0]
    jump_tick = max(1, first_start - warmup_ticks)

    out_path_cfg = out_dir.as_posix()

    seg_cmds: List[str] = []

    for i, (s, e) in enumerate(segments_sorted):
        seg_cmds.append(f'mirv_cmd addAtTick {s} "spec_player {username};"')
        seg_cmds.append(f'mirv_cmd addAtTick {s} "host_framerate {fps}; mirv_streams record start"')
        seg_cmds.append(f'mirv_cmd addAtTick {e} "mirv_streams record end; host_framerate 0"')

        if i + 1 < len(segments_sorted):
            next_start, _ = segments_sorted[i + 1]
            next_jump = max(1, next_start - warmup_ticks)
            seg_cmds.append(f'mirv_cmd addAtTick {e + 1} "demo_gototick {next_jump}"')
            seg_cmds.append(f'mirv_cmd addAtTick {max(2, next_jump + 1)} "spec_player {username};"')

    last_end = segments_sorted[-1][1]
    quit_tick = last_end + max(10, int(round(0.5 * tickrate)))

    cfg = f"""
echo "=== auto_record MULTI (CS2): segments={len(segments_sorted)} first_jump={jump_tick} warmup_ticks={warmup_ticks} ==="
mirv_cmd clear

mirv_streams record end

mirv_streams record screen enabled 1
mirv_streams record startMovieWav 0

mirv_streams record name "{out_path_cfg}"
mirv_streams record fps {fps}

mirv_streams record screen settings afxClassic

host_timescale 1
mirv_snd_timescale 1

mirv_cmd addAtTick 1 "demo_gototick {jump_tick}"
mirv_cmd addAtTick {max(2, jump_tick + 1)} "spec_player {username}; demoui"

{chr(10).join(seg_cmds)}

mirv_cmd addAtTick {quit_tick} "quit"
"""
    cfg_path.write_text(cfg.strip() + "\n", encoding="utf-8")


def _start_hlae(job: Job) -> subprocess.Popen:
    cmdline = f'{job.extra_launch} +playdemo "{job.demo_path}" +exec {job.cfg_name}'
    cmd = [
        str(job.hlae_exe),
        "-customLoader",
        "-noGui",
        "-autoStart",
        "-hookDllPath", str(job.hook_dll),
        "-programPath", str(job.cs2_exe),
        "-cmdLine", cmdline,
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _collect_proc_output(proc: subprocess.Popen, buf: List[str]) -> None:
    try:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            buf.append(line)
    except Exception:
        pass


def _wait_next_new_take(
    out_dir: Path,
    known: set[str],
    seen_in_this_run: set[str],
    timeout_s: float,
) -> Path:
    """
    Why: we want to process takes as soon as they appear.
    Chooses the oldest (by mtime) among new unseen take dirs.
    """
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        takes = _list_take_dirs(out_dir)
        candidates = [t for t in takes if t.name not in known and t.name not in seen_in_this_run]
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime)
            return candidates[0]
        time.sleep(TAKE_POLL_S)
    raise RuntimeError(f"Timed out waiting for next take folder in {out_dir}")


def _upload_to_ubuntu(ubuntu_upload_url: str, mp4_path: Path) -> dict:
    with mp4_path.open("rb") as f:
        r = requests.post(
            ubuntu_upload_url,
            files={"file": (mp4_path.name, f, "video/mp4")},
            headers={"X-Token": TOKEN},
            timeout=600,
        )
    r.raise_for_status()
    return r.json()


def launch_cs2_hlae_convert_and_upload_streaming(
    job: Job,
    demo_id: str,
    ubuntu_upload_url: str,
    clips: List[dict],
) -> List[dict]:
    """
    NEW BEHAVIOR:
    - While HLAE is still running and creating take folders, we:
      (1) detect each new take folder as it appears
      (2) wait for its frames to stop changing
      (3) ffmpeg it to mp4
      (4) rename + upload immediately
    The HTTP response still returns at the end, but Ubuntu starts receiving clips ASAP.
    """
    for p in [job.cs2_exe, job.demo_path, job.hlae_exe, job.hook_dll]:
        if not p.exists():
            raise FileNotFoundError(str(p))

    subprocess.run([job.ffmpeg_exe, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Build segment ticks and sort by start tick
    segments: List[Tuple[int, int]] = []
    for c in clips:
        start_s = float(c["start_s"])
        duration_s = float(c["duration_s"])
        start_tick = int(round(start_s * job.tickrate))
        dur_ticks = int(round(duration_s * job.tickrate))
        end_tick = start_tick + max(1, dur_ticks)
        segments.append((start_tick, end_tick))

    seg_sorted = sorted(list(enumerate(segments)), key=lambda x: x[1][0])
    sorted_indices = [i for i, _ in seg_sorted]
    segments_sorted = [seg for _, seg in seg_sorted]

    cfg_dir = cs2_cfg_dir_from_cs2_exe(job.cs2_exe)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / job.cfg_name

    write_auto_record_cfg_multi(
        username=job.username,
        cfg_path=cfg_path,
        segments=segments_sorted,
        out_dir=job.out_dir,
        fps=job.fps,
        tickrate=job.tickrate,
        warmup_s=job.warmup_s,
    )

    takes_before = {p.name for p in _list_take_dirs(job.out_dir)}
    seen_in_this_run: set[str] = set()
    chosen_takes: List[Path] = []
    results_by_req_index: Dict[int, dict] = {}

    proc = _start_hlae(job)
    out_buf: List[str] = []
    t_out = threading.Thread(target=_collect_proc_output, args=(proc, out_buf), daemon=True)
    t_out.start()

    expected = len(segments_sorted)

    # A reasonable upper bound timeout per take: warmup + duration + some slack + gaps (we skip gaps)
    per_take_timeout = max(60.0, WAIT_STABLE_TIMEOUT_S + 30.0)

    try:
        for out_i in range(expected):
            take_dir = _wait_next_new_take(
                out_dir=job.out_dir,
                known=takes_before,
                seen_in_this_run=seen_in_this_run,
                timeout_s=per_take_timeout,
            )
            seen_in_this_run.add(take_dir.name)
            chosen_takes.append(take_dir)

            # Convert as soon as recording for this take finishes (frames stabilize)
            mp4_path = convert_take_to_mp4(job, take_dir)

            # Map this take (in appearance order) to the clip segment order
            clip = clips[sorted_indices[out_i]]
            start_s = float(clip["start_s"])
            duration_s = float(clip["duration_s"])
            req_index = int(clip["req_index"])

            new_name = _safe_filename(f"{demo_id}_{job.username}_{req_index}_{start_s:.3f}s_{duration_s:.3f}s.mp4")
            final_mp4 = mp4_path.with_name(new_name)
            try:
                if final_mp4.exists():
                    final_mp4.unlink()
                mp4_path.rename(final_mp4)
            except OSError:
                final_mp4 = mp4_path

            # Upload immediately (this is the key change)
            ubuntu_resp = _upload_to_ubuntu(ubuntu_upload_url, final_mp4)

            results_by_req_index[req_index] = {
                "req_index": req_index,
                "start_s": start_s,
                "duration_s": duration_s,
                "sent_mp4": str(final_mp4),
                "ubuntu_response": ubuntu_resp,
            }

        # Wait for HLAE to end (should quit from cfg)
        rc = proc.wait(timeout=120)
        if rc != 0:
            out = "".join(out_buf[-4000:])  # keep tail if huge
            raise RuntimeError(f"HLAE exited with code {rc}\n\n{out}")

    finally:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass

    # Optional cleanup: remove extra empty takes that appeared
    if job.cleanup_extra_takes:
        try:
            takes_after = _list_take_dirs(job.out_dir)
            new_takes = [p for p in takes_after if p.name not in takes_before]
            for t in new_takes:
                if t in chosen_takes:
                    continue
                if _count_tga_frames(t) == 0:
                    try:
                        for f in t.iterdir():
                            f.unlink()
                        t.rmdir()
                    except OSError:
                        pass
        except Exception:
            pass

    ordered_results = [results_by_req_index[i] for i in sorted(results_by_req_index.keys())]
    return ordered_results


# =========================
# Windows API
# =========================
app = FastAPI()


def _require_token(x_token: Optional[str]) -> None:
    if x_token != TOKEN:
        raise HTTPException(status_code=401, detail="Bad token")


def _validate_float(name: str, v: Any, lo: float, hi: float) -> float:
    try:
        f = float(v)
    except Exception:
        raise HTTPException(status_code=400, detail=f"{name} must be a number")
    if not (lo <= f <= hi):
        raise HTTPException(status_code=400, detail=f"{name} out of range [{lo}, {hi}]")
    return f


def _validate_username(u: Any) -> str:
    if not isinstance(u, str) or not u.strip():
        raise HTTPException(status_code=400, detail="username required")
    safe = "".join(c for c in u.strip() if c.isalnum() or c in ("_", "-", " ", ".", "'"))
    if not safe:
        raise HTTPException(status_code=400, detail="bad username")
    return safe


def _normalize_clips(payload: dict) -> List[dict]:
    if isinstance(payload.get("clips"), list):
        raw = payload["clips"]
        if not raw:
            raise HTTPException(status_code=400, detail="clips must be a non-empty list")
        clips: List[dict] = []
        for i, c in enumerate(raw):
            if not isinstance(c, dict):
                raise HTTPException(status_code=400, detail=f"clips[{i}] must be an object")
            s = _validate_float(f"clips[{i}].start_s", c.get("start_s"), MIN_START_S, MAX_START_S)
            d = _validate_float(f"clips[{i}].duration_s", c.get("duration_s"), MIN_DURATION_S, MAX_DURATION_S)
            clips.append({"start_s": s, "duration_s": d, "req_index": i})
        return clips

    s = _validate_float("start_s", payload.get("start_s"), MIN_START_S, MAX_START_S)
    d = _validate_float("duration_s", payload.get("duration_s"), MIN_DURATION_S, MAX_DURATION_S)
    return [{"start_s": s, "duration_s": d, "req_index": 0}]


@app.get("/demos")
def list_demos(x_token: Optional[str] = Header(default=None)):
    _require_token(x_token)
    return {"ok": True, "demo_ids": sorted(DEMO_MAP.keys())}


@app.post("/clip")
def make_clip(payload: dict, x_token: Optional[str] = Header(default=None)):
    _require_token(x_token)

    demo_id = payload.get("demo_id")
    ubuntu_upload_url = payload.get("ubuntu_upload_url")
    if not isinstance(demo_id, str) or demo_id not in DEMO_MAP:
        raise HTTPException(status_code=400, detail="demo_id invalid or unknown")
    if not isinstance(ubuntu_upload_url, str) or not ubuntu_upload_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="ubuntu_upload_url invalid")

    username = _validate_username(payload.get("username"))
    clips = _normalize_clips(payload)

    req_tag = f"{int(time.time())}_{os.getpid()}"
    cfg_name = f"auto_record_multi_{req_tag}.cfg"

    job = Job(
        username=username,
        cs2_exe=CS2_EXE,
        demo_path=DEMO_MAP[demo_id],
        hlae_exe=HLAE_EXE,
        hook_dll=HOOK_DLL,
        out_dir=OUT_DIR,
        warmup_s=WARMUP_S,
        fps=FPS,
        ffmpeg_exe=FFMPEG_EXE,
        cleanup_extra_takes=CLEANUP_EXTRA_TAKES,
        tickrate=int(TICKRATE),
        extra_launch=EXTRA_LAUNCH,
        cfg_name=cfg_name,
    )

    # NEW: convert + upload in a streaming manner (uploads happen ASAP)
    ordered_results = launch_cs2_hlae_convert_and_upload_streaming(
        job=job,
        demo_id=demo_id,
        ubuntu_upload_url=ubuntu_upload_url,
        clips=clips,
    )

    return JSONResponse({
        "ok": True,
        "demo_id": demo_id,
        "username": username,
        "clips_count": len(ordered_results),
        "results": ordered_results,
        "mode": "single_cs2_run_multi_segments_skip_gaps_stream_upload",
    })


def main():
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("WIN_PORT", "8788")))


if __name__ == "__main__":
    main()
