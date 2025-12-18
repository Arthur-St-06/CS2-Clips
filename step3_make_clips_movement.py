
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import re
from datetime import datetime, timezone

import pandas as pd

from timer_flip_detector import detect_first_red_to_white_flip


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def get_demo_played_at_from_dem_info(demo_path: Path) -> tuple[datetime, datetime] | None:
    demo_path = demo_path.expanduser().resolve()

    info1 = Path(str(demo_path) + ".info")
    info2 = demo_path.with_suffix(demo_path.suffix + ".info")
    info_path = info1 if info1.exists() else info2 if info2.exists() else None
    if info_path is None:
        return None

    try:
        p = subprocess.run(
            ["protoc", "--decode_raw"],
            input=info_path.read_bytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    txt = p.stdout.decode("utf-8", errors="replace")
    m = re.search(r"(?m)^2:\s*(\d+)\s*$", txt)
    if not m:
        return None

    epoch = int(m.group(1))
    dt_utc = datetime.fromtimestamp(epoch, tz=timezone.utc)
    dt_local = dt_utc.astimezone()
    return dt_utc, dt_local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out_sprays")
    ap.add_argument("--demo", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--pre", type=float, default=3.0)
    ap.add_argument("--post", type=float, default=2.0)
    ap.add_argument("--prefer-player", default="")

    # sync
    ap.add_argument("--tickrate", type=float, default=64.0)
    ap.add_argument("--anchor-tick", type=int, default=1441)
    ap.add_argument("--video-anchor-s", type=float, default=None)
    ap.add_argument("--offset", type=float, default=None)

    ap.add_argument("--auto-sync-start", type=float, default=0.0)
    ap.add_argument("--auto-sync-max", type=float, default=180.0)
    ap.add_argument("--auto-downsample", type=int, default=2)

    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    demo_path = Path(args.demo).expanduser().resolve()
    video_path = str(Path(args.video).expanduser().resolve())

    df_path = out / "shoot_move_candidates.parquet"
    df = pd.read_parquet(df_path).copy() if df_path.exists() else pd.DataFrame()

    played_at = get_demo_played_at_from_dem_info(demo_path)
    if played_at is None:
        played_at_utc_s = "Unknown (missing .dem.info or protoc)"
        played_at_local_s = "Unknown"
    else:
        dt_utc, dt_local = played_at
        played_at_utc_s = dt_utc.isoformat()
        played_at_local_s = dt_local.isoformat()

    if df.empty or len(df.columns) == 0:
        print("No movement candidates found — nothing to clip.")
        summary = f"""Shooting while moving (no full stop / no counter-strafe)
Match time (local): {played_at_local_s}
Match time (UTC):   {played_at_utc_s}

No movement-related deaths were detected (moving shots + death within 1.5s).
"""
        (out / "coaching_move.txt").write_text(summary, encoding="utf-8")
        print(f"✅ Movement coaching written to: {out/'coaching_move.txt'}")
        return

    if args.prefer_player:
        df = df[df["player"].astype(str).str.contains(args.prefer_player, case=False, na=False)].copy()

    # movement ranking
    df = df.sort_values(
        ["time_to_death_after_moving_shot_s", "max_speed_2d", "shots_while_moving"],
        ascending=[True, False, False]
    ).head(args.top)

    # offset
    if args.offset is not None:
        offset = float(args.offset)
        demo_anchor_s = None
        video_anchor_s = None
    else:
        demo_anchor_s = args.anchor_tick / args.tickrate
        if args.video_anchor_s is None:
            print("[auto-sync] video_anchor_s not provided; detecting timer flip...")
            video_anchor_s = detect_first_red_to_white_flip(
                video_path,
                start_sec=args.auto_sync_start,
                max_sec=args.auto_sync_max,
                downsample=args.auto_downsample,
            )
            print(f"[auto-sync] Detected video_anchor_s = {video_anchor_s:.3f}s")
        else:
            video_anchor_s = float(args.video_anchor_s)
        offset = video_anchor_s - demo_anchor_s

    print("[sync]")
    print(f"  tickrate: {args.tickrate}")
    print(f"  offset (video = demo + offset): {offset:.6f}s")

    clips_dir = out / "clips_move"
    clips_dir.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(df.itertuples(index=False), start=1):
        start = max(0.0, float(r.start_s) + offset - args.pre)
        end = float(r.death_s) + offset + args.post

        safe_player = str(r.player).replace(" ", "_")
        weapon = str(getattr(r, "weapon", "")).replace("weapon_", "")
        out_name = (
            f"{i:02d}_{safe_player}_{weapon}"
            f"_spd{float(r.max_speed_2d):.0f}"
            f"_shots{int(r.shots_while_moving)}.mp4"
        )
        out_path = clips_dir / out_name

        cmd = [
          "ffmpeg", "-hide_banner", "-loglevel", "error",
          "-ss", f"{start:.3f}",
          "-to", f"{end:.3f}",
          "-i", video_path,
          "-c", "copy",
          "-movflags", "+faststart",
          str(out_path),
        ]
        run(cmd)

    total = len(pd.read_parquet(out / "shoot_move_candidates.parquet"))
    shown = len(df)
    avg_shots = df["shots_while_moving"].mean() if shown else 0
    avg_spd = df["max_speed_2d"].mean() if shown else 0
    avg_ttd = df["time_to_death_after_moving_shot_s"].mean() if shown else 0

    summary = f"""Detected {total} candidate cases where you fired while moving and died soon after.

Top {shown} exported clips:
- Avg shots while moving (per incident): {avg_shots:.1f}
- Avg max speed during incident: {avg_spd:.0f} units/s
- Avg time-to-death after moving shot: {avg_ttd:.2f}s

Try instead:
- Counter-strafe: release movement and tap the opposite key to stop instantly before shooting.
- Take 1–3 accurate bullets after a full stop (especially with rifles).
- If you’re caught moving, prioritize getting to cover first, then re-peek with a planned stop.
"""
    (out / "coaching_move.txt").write_text(summary, encoding="utf-8")

    print(f"\n✅ Movement clips saved in: {clips_dir}")
    print(f"✅ Movement coaching: {out/'coaching_move.txt'}")


if __name__ == "__main__":
    main()
