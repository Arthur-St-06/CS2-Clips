#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <demo.dem> <video.mkv/mp4> <anchor_tick> <video_anchor_s> [out_dir]"
  echo "Example: $0 match.dem \"~/Videos/2025-12-15 09-57-29.mkv\" 512 14.72 out_sprays"
  exit 1
fi

DEMO="$1"
VIDEO="$2"
PLAYER="$3"
OUT="${4:-out_sprays}"

echo "[1/3] Bursts..."
python3 step1_bursts.py --demo "$DEMO" --out "$OUT" --player "$PLAYER"

echo "[2/3] Overspray detection..."
python3 step2_overspray.py --demo "$DEMO" --out "$OUT"

echo "[3/3] Clip export..."
python3 step3_make_clips.py \
  --out "$OUT" \
  --video "$VIDEO" \

echo "✅ Done. Check: $OUT/clips/ and $OUT/coaching.txt"

