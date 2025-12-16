#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import pandas as pd

from timer_flip_detector import detect_first_red_to_white_flip

def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out_sprays")
    ap.add_argument("--video", required=True, help="OBS recording (mkv/mp4)")
    ap.add_argument("--top", type=int, default=5, help="how many clips to export")
    ap.add_argument("--pre", type=float, default=3.0, help="seconds before burst start")
    ap.add_argument("--post", type=float, default=2.0, help="seconds after death")
    ap.add_argument("--prefer-player", default="", help="optional: filter clips to player substring")

    # --- Sync parameters ---
    ap.add_argument("--tickrate", type=float, default=64.0, help="Demo tickrate for tick->seconds conversion")
    ap.add_argument("--anchor-tick", type=int, default=1441,
                    help="Demo tick for anchor event (freeze-end tick where players unfreeze)")
    ap.add_argument("--video-anchor-s", type=float, default=None,
                    help="Timestamp (seconds) in the VIDEO for the same anchor. If omitted, auto-detect from HUD timer.")
    ap.add_argument("--offset", type=float, default=None,
                    help="Direct offset: video = demo + offset. If set, other sync params are ignored.")

    # Auto-detect tuning (no need for user normally; keep optional)
    ap.add_argument("--auto-sync-start", type=float, default=0.0, help="Auto-detect: start scanning at this second")
    ap.add_argument("--auto-sync-max", type=float, default=180.0, help="Auto-detect: scan duration (seconds)")
    ap.add_argument("--auto-downsample", type=int, default=2, help="Auto-detect: process every Nth frame")

    args = ap.parse_args()

    out = Path(args.out)
    df = pd.read_parquet(out / "overspray_candidates.parquet").copy()

    if df.empty or len(df.columns) == 0:
        print("No overspray candidates found — nothing to clip.")
        summary = """Over-spraying instead of resetting
No overspray mistakes were detected in this match.

Nice job — either you didn’t over-spray, or the situations didn’t lead to deaths.
"""
        (out / "coaching.txt").write_text(summary, encoding="utf-8")
        print(f"✅ Coaching summary written to: {out/'coaching.txt'}")
        return

    if args.prefer_player:
        df = df[df["player"].str.contains(args.prefer_player, case=False, na=False)].copy()

    df = df.sort_values(
        ["bullets", "duration_s", "time_to_death_after_burst_s"],
        ascending=[False, False, True]
    ).head(args.top)

    video_path = str(Path(args.video).expanduser().resolve())

    # --- Compute offset ---
    if args.offset is not None:
        offset = float(args.offset)
        demo_anchor_s = None
        video_anchor_s = None
    else:
        demo_anchor_s = args.anchor_tick / args.tickrate

        if args.video_anchor_s is None:
            video_anchor_s = detect_first_red_to_white_flip(
                video_path,
                start_sec=args.auto_sync_start,
                max_sec=args.auto_sync_max,
                downsample=args.auto_downsample,

                # no GUI prompt:
                interactive_roi=False,

                # your measured position:
                roi_left_pct=0.516,
                roi_top_pct=0.027,

                # size of the timer box (tweak only if needed)
                roi_w_pct=0.08,
                roi_h_pct=0.06,
            )
        else:
            video_anchor_s = float(args.video_anchor_s)

        offset = video_anchor_s - demo_anchor_s

    print("[sync]")
    print(f"  tickrate: {args.tickrate}")
    if demo_anchor_s is not None:
        print(f"  anchor_tick: {args.anchor_tick} -> demo_anchor_s: {demo_anchor_s:.6f}s")
        if video_anchor_s is not None:
            print(f"  video_anchor_s: {video_anchor_s:.6f}s")
    print(f"  offset (video = demo + offset): {offset:.6f}s")

    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(df.itertuples(index=False), start=1):
        start = max(0.0, float(r.start_s) + offset - args.pre)
        end = float(r.death_s) + offset + args.post

        safe_player = str(r.player).replace(" ", "_")
        weapon = str(r.weapon).replace("weapon_", "")
        out_name = f"{i:02d}_{safe_player}_{weapon}_bul{int(r.bullets)}.mp4"
        out_path = clips_dir / out_name

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-i", video_path,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "160k",
            str(out_path)
        ]
        run(cmd)

    total = len(pd.read_parquet(out / "overspray_candidates.parquet"))
    shown = len(df)
    avg_bul = df["bullets"].mean() if shown else 0
    avg_dur = df["duration_s"].mean() if shown else 0
    avg_ttd = df["time_to_death_after_burst_s"].mean() if shown else 0

    summary = f"""Over-spraying instead of resetting
Detected {total} candidate cases where you kept spraying, got no kill, and died soon after.

Top {shown} exported clips:
- Avg bullets: {avg_bul:.1f}
- Avg spray duration: {avg_dur:.2f}s
- Avg time-to-death after burst: {avg_ttd:.2f}s

Try instead:
- Burst 3–5 bullets, then reset.
- Strafe/reposition after first contact instead of committing to a long spray.
- If you whiff, stop shooting briefly to regain accuracy, then re-peek intentionally.
"""
    (out / "coaching.txt").write_text(summary, encoding="utf-8")
    print(f"\n✅ Clips saved in: {clips_dir}")
    print(f"✅ Coaching summary: {out/'coaching.txt'}")

if __name__ == "__main__":
    main()
