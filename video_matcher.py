from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone, timedelta
import re

@dataclass
class MatchResult:
    video_path: Path
    delta_seconds: float
    used_anchor: str

def _local_tz():
    return datetime.now().astimezone().tzinfo

def _mtime_local(p: Path) -> datetime:
    return datetime.fromtimestamp(p.stat().st_mtime, tz=_local_tz())

def _extract_unix_epoch_from_name(name: str) -> int | None:
    # Find plausible 10-digit unix epochs (~2015..2035) in filename
    for m in re.finditer(r"\b(\d{10})\b", name):
        v = int(m.group(1))
        if 1_420_000_000 <= v <= 2_080_000_000:
            return v
    return None

def _demo_candidate_times(demo_path: Path) -> list[tuple[str, datetime]]:
    tz = _local_tz()
    cands: list[tuple[str, datetime]] = [("mtime", _mtime_local(demo_path))]

    epoch = _extract_unix_epoch_from_name(demo_path.name)
    if epoch is not None:
        dt = datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone(tz)
        cands.append(("filename_epoch", dt))

    return cands

def _list_videos(video_dir: Path) -> list[Path]:
    exts = {".mkv", ".mp4", ".mov"}
    vids = [p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    vids.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return vids

def match_demo_to_video(
    demo_path: Path,
    video_dir: Path,
    *,
    max_window_hours: float = 12.0,
    allow_timezone_shift_hours: int = 12,
) -> MatchResult | None:
    demo_path = demo_path.resolve()
    video_dir = video_dir.resolve()

    vids = _list_videos(video_dir)
    if not vids:
        return None

    candidates = _demo_candidate_times(demo_path)

    best: tuple[Path, float, str] | None = None
    for anchor_name, demo_dt in candidates:
        for shift_h in range(-allow_timezone_shift_hours, allow_timezone_shift_hours + 1):
            shifted = demo_dt + timedelta(hours=shift_h)

            for v in vids:
                v_dt = _mtime_local(v)
                abs_delta = abs((v_dt - shifted).total_seconds())
                if best is None or abs_delta < best[1]:
                    best = (v, abs_delta, f"{anchor_name}{'' if shift_h==0 else f'_shift{shift_h:+d}h'}")

    if best is None:
        return None

    v, abs_delta, used_anchor = best
    if abs_delta > max_window_hours * 3600:
        return None

    return MatchResult(video_path=v, delta_seconds=abs_delta, used_anchor=used_anchor)
