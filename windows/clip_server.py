import os
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

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

# Demo registry: Ubuntu requests a demo_id, Windows maps it to a path.
# Why: prevents arbitrary path exfiltration.
DEMO_MAP: Dict[str, Path] = {
    # Example:
    "demo392": Path(r"D:\SteamLibrary\steamapps\common\Counter-Strike Global Offensive\game\csgo\replays\match730_003792077342759714824_1663050728_392.dem"),
}
# Optional: allow adding more via env var "DEMO_DIR" (serve newest by id)
DEMO_DIR = os.environ.get("DEMO_DIR")
if DEMO_DIR:
    ddir = Path(DEMO_DIR).resolve()
    if ddir.exists():
        for p in ddir.glob("*.dem"):
            # use filename stem as id
            DEMO_MAP[p.stem] = p

@dataclass
class Job:
    username: str
    cs2_exe: Path
    demo_path: Path
    hlae_exe: Path
    hook_dll: Path

    tickrate: int = 64
    start_s: float = 40.0
    duration_s: float = 60.0

    warmup_s: float = 1.5
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

def _wait_for_frames(take_dir: Path, timeout_s: float = 30.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if any(take_dir.glob("*.tga")):
            return
        time.sleep(0.25)
    raise RuntimeError(f"No .tga frames appeared in {take_dir} within {timeout_s:.0f}s")

def _pick_best_take(candidates: list[Path]) -> Path:
    if not candidates:
        raise RuntimeError("No take folders found for this run.")
    return max(candidates, key=lambda p: (_count_tga_frames(p), p.stat().st_mtime))

def write_auto_record_cfg(
    username: str,
    cfg_path: Path,
    start_tick: int,
    duration_ticks: int,
    out_dir: Path,
    fps: int,
    tickrate: int,
    warmup_s: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    warmup_ticks = max(0, int(round(warmup_s * tickrate)))
    start_after_jump_tick = start_tick + warmup_ticks
    end_tick = start_after_jump_tick + duration_ticks
    out_path_cfg = out_dir.as_posix()

    cfg = f"""
echo "=== auto_record (CS2): start_tick={start_tick} warmup_ticks={warmup_ticks} start_after_jump_tick={start_after_jump_tick} end_tick={end_tick} ==="
mirv_cmd clear
mirv_streams record end
mirv_streams record screen enabled 1
mirv_streams record startMovieWav 0
mirv_streams record name "{out_path_cfg}"
mirv_streams record fps {fps}
mirv_streams record screen settings afxClassic
host_timescale 1
mirv_snd_timescale 1
mirv_cmd addAtTick 1 "demo_gototick {start_tick}"
mirv_cmd addAtTick {start_after_jump_tick} "spec_player {username}; demoui"
mirv_cmd addAtTick {start_after_jump_tick} "host_framerate {fps}; mirv_streams record start"
mirv_cmd addAtTick {end_tick} "mirv_streams record end; host_framerate 0; echo \\"=== auto_record done ===\\""
"""
    cfg_path.write_text(cfg.strip() + "\n", encoding="utf-8")

def convert_take_to_mp4(job: Job, take_dir: Path) -> Path:
    _wait_for_frames(take_dir)
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

def launch_cs2_hlae_and_convert(job: Job) -> Path:
    for p in [job.cs2_exe, job.demo_path, job.hlae_exe, job.hook_dll]:
        if not p.exists():
            raise FileNotFoundError(str(p))

    subprocess.run([job.ffmpeg_exe, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    start_tick = int(round(job.start_s * job.tickrate))
    duration_ticks = int(round(job.duration_s * job.tickrate))

    cfg_dir = cs2_cfg_dir_from_cs2_exe(job.cs2_exe)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / job.cfg_name

    write_auto_record_cfg(
        username=job.username,
        cfg_path=cfg_path,
        start_tick=start_tick,
        duration_ticks=duration_ticks,
        out_dir=job.out_dir,
        fps=job.fps,
        tickrate=job.tickrate,
        warmup_s=job.warmup_s,
    )

    takes_before = {p.name for p in _list_take_dirs(job.out_dir)}

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

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.communicate()[0] or ""
    if proc.returncode not in (0, None):
        raise RuntimeError(f"HLAE exited with code {proc.returncode}\n\n{out}")

    takes_after = _list_take_dirs(job.out_dir)
    new_takes = [p for p in takes_after if p.name not in takes_before]
    best_take = _pick_best_take(new_takes if new_takes else takes_after)
    mp4 = convert_take_to_mp4(job, best_take)

    if job.cleanup_extra_takes:
        for t in new_takes:
            if t == best_take:
                continue
            if _count_tga_frames(t) == 0:
                try:
                    for f in t.iterdir():
                        f.unlink()
                    t.rmdir()
                except OSError:
                    pass

    return mp4

# =========================
# Windows API
# =========================
app = FastAPI()

def _require_token(x_token: str | None) -> None:
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
    # Why: avoid shell-ish characters. spec_player uses it directly.
    safe = "".join(c for c in u.strip() if c.isalnum() or c in ("_", "-", " ", ".", "'"))
    if not safe:
        raise HTTPException(status_code=400, detail="bad username")
    return safe

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

@app.get("/demos")
def list_demos(x_token: str | None = Header(default=None)):
    _require_token(x_token)
    return {"ok": True, "demo_ids": sorted(DEMO_MAP.keys())}

@app.post("/clip")
def make_clip(payload: dict, x_token: str | None = Header(default=None)):
    _require_token(x_token)

    demo_id = payload.get("demo_id")
    ubuntu_upload_url = payload.get("ubuntu_upload_url")
    if not isinstance(demo_id, str) or demo_id not in DEMO_MAP:
        raise HTTPException(status_code=400, detail="demo_id invalid or unknown")
    if not isinstance(ubuntu_upload_url, str) or not ubuntu_upload_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="ubuntu_upload_url invalid")

    username = _validate_username(payload.get("username"))
    start_s = _validate_float("start_s", payload.get("start_s"), MIN_START_S, MAX_START_S)
    duration_s = _validate_float("duration_s", payload.get("duration_s"), MIN_DURATION_S, MAX_DURATION_S)

    job = Job(
        username=username,
        cs2_exe=CS2_EXE,
        demo_path=DEMO_MAP[demo_id],
        hlae_exe=HLAE_EXE,
        hook_dll=HOOK_DLL,
        out_dir=OUT_DIR,
        start_s=start_s,
        duration_s=duration_s,
        warmup_s=WARMUP_S,
        fps=FPS,
        ffmpeg_exe=FFMPEG_EXE,
        cleanup_extra_takes=CLEANUP_EXTRA_TAKES,
        tickrate=int(TICKRATE),
        extra_launch=EXTRA_LAUNCH,
    )

    mp4 = launch_cs2_hlae_and_convert(job)
    ubuntu_resp = _upload_to_ubuntu(ubuntu_upload_url, mp4)

    return JSONResponse({
        "ok": True,
        "demo_id": demo_id,
        "username": username,
        "start_s": start_s,
        "duration_s": duration_s,
        "sent_mp4": str(mp4),
        "ubuntu_response": ubuntu_resp,
    })

def main():
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("WIN_PORT", "8788")))

if __name__ == "__main__":
    main()
