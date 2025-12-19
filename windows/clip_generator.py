import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Job:
    cs2_exe: Path
    demo_path: Path
    hlae_exe: Path
    hook_dll: Path

    tickrate: int = 64
    start_s: float = 40.0
    duration_s: float = 5.0

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
    # Pick the take with the most frames (audio-only takes will have 0 frames)
    return max(candidates, key=lambda p: (_count_tga_frames(p), p.stat().st_mtime))


def write_auto_record_cfg(
    cfg_path: Path,
    start_tick: int,
    duration_ticks: int,
    out_dir: Path,
    fps: int,
    tickrate: int,
    warmup_s: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Why: demo_gototick returns before the map/render is fully “settled” on some machines.
    # Waiting warmup_s seconds (converted to ticks) avoids capturing loading frames.
    warmup_ticks = max(0, int(round(warmup_s * tickrate)))
    start_after_jump_tick = start_tick + warmup_ticks
    end_tick = start_after_jump_tick + duration_ticks

    out_path_cfg = out_dir.as_posix()

    cfg = f"""
echo "=== auto_record (CS2): start_tick={start_tick} warmup_ticks={warmup_ticks} start_after_jump_tick={start_after_jump_tick} end_tick={end_tick} ==="
mirv_cmd clear

// Safety: ensure nothing is currently recording.
mirv_streams record end

// Enable screen capture (frames)
mirv_streams record screen enabled 1

// Disable audio capture (avoid extra audio-only takes)
mirv_streams record startMovieWav 0

// Output directory + FPS
mirv_streams record name "{out_path_cfg}"
mirv_streams record fps {fps}

// TGA image sequence preset
mirv_streams record screen settings afxClassic

host_timescale 1
mirv_snd_timescale 1

// Jump as soon as demo starts ticking:
mirv_cmd addAtTick 1 "demo_gototick {start_tick}"

// Set POV + open UI after jump:
mirv_cmd addAtTick {start_tick} "spec_player Remag; demoui"

// Start recording AFTER warmup:
mirv_cmd addAtTick {start_after_jump_tick} "host_framerate {fps}; mirv_streams record start"

// Stop recording:
mirv_cmd addAtTick {end_tick} "mirv_streams record end; host_framerate 0; echo \\"=== auto_record done ===\\""
"""
    cfg_path.write_text(cfg.strip() + "\n", encoding="utf-8")


def convert_take_to_mp4(job: Job, take_dir: Path) -> Path:
    _wait_for_frames(take_dir)

    out_mp4 = take_dir / "out.mp4"

    cmd = [
        job.ffmpeg_exe,
        "-y",
        "-framerate", str(job.fps),
        "-start_number", "0",
        "-i", "%05d.tga",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
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

    print(out.strip() or "HLAE finished.")

    takes_after = _list_take_dirs(job.out_dir)
    new_takes = [p for p in takes_after if p.name not in takes_before]

    best_take = _pick_best_take(new_takes if new_takes else takes_after)
    print(f"[convert] using take: {best_take} (frames={_count_tga_frames(best_take)})")

    mp4 = convert_take_to_mp4(job, best_take)
    print(f"[convert] wrote {mp4}")

    if job.cleanup_extra_takes:
        for t in new_takes:
            if t == best_take:
                continue
            if _count_tga_frames(t) == 0:
                try:
                    for f in t.iterdir():
                        f.unlink()
                    t.rmdir()
                    print(f"[cleanup] removed extra take: {t}")
                except OSError:
                    pass

    return mp4

if __name__ == "__main__":
    job = Job(
        cs2_exe=Path(r"D:\SteamLibrary\steamapps\common\Counter-Strike Global Offensive\game\bin\win64\cs2.exe"),
        demo_path=Path(r"D:\SteamLibrary\steamapps\common\Counter-Strike Global Offensive\game\csgo\replays\match730_003792077342759714824_1663050728_392.dem"),
        hlae_exe=Path(r"C:\Program Files (x86)\HLAE\HLAE.exe"),
        hook_dll=Path(r"C:\Program Files (x86)\HLAE\x64\AfxHookSource2.dll"),
        out_dir=Path(r"D:\hlae_out"),
        start_s=40.0,
        duration_s=5.0,
        warmup_s=1.0,  # <-- 1 second wait before recording starts
        fps=30,
        ffmpeg_exe="ffmpeg",
        cleanup_extra_takes=True,
    )

    launch_cs2_hlae_and_convert(job)
