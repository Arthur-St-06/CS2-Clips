# timer_flip_detector.py
from __future__ import annotations
import cv2
import numpy as np

def _compute_stats(bgr_roi: np.ndarray):
    """Return (red_ratio, white_ratio, mean_v) in HSV space."""
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Red hue wraps: [0..10] or [170..180], with decent saturation/value.
    red_mask = (((h <= 10) | (h >= 170)) & (s >= 80) & (v >= 80))
    red_ratio = float(np.mean(red_mask))

    # White-ish: low saturation, high value.
    white_mask = ((s <= 40) & (v >= 170))
    white_ratio = float(np.mean(white_mask))

    mean_v = float(np.mean(v))
    return red_ratio, white_ratio, mean_v


def _roi_from_percent(
    frame: np.ndarray,
    *,
    left_pct: float,
    top_pct: float,
    roi_w_pct: float = 0.10,
    roi_h_pct: float = 0.06,
) -> tuple[int, int, int, int]:
    """
    Build an ROI around a (left_pct, top_pct) anchor point.
    Interprets (left_pct, top_pct) as the CENTER of the ROI.
    """
    H, W = frame.shape[:2]
    cx = int(round(W * left_pct))
    cy = int(round(H * top_pct))
    w = int(round(W * roi_w_pct))
    h = int(round(H * roi_h_pct))

    w = max(4, w)
    h = max(4, h)

    x = cx - w // 2
    y = cy - h // 2

    # clamp to frame
    x = max(0, min(x, W - w))
    y = max(0, min(y, H - h))

    return (x, y, w, h)


def detect_first_red_to_white_flip(
    video_path: str,
    *,
    start_sec: float = 0.0,
    max_sec: float = 180.0,
    downsample: int = 2,
    min_red_ratio: float = 0.08,
    min_white_ratio: float = 0.12,
    stable_frames: int = 3,

    # ROI options:
    interactive_roi: bool = False,  # default off now
    roi: tuple[int, int, int, int] | None = None,

    # Percent-based ROI (your use case):
    roi_left_pct: float | None = None,   # e.g. 0.48
    roi_top_pct: float | None = None,    # e.g. 0.035
    roi_w_pct: float = 0.10,
    roi_h_pct: float = 0.06,
) -> float:
    """
    Detect first sustained red->white transition in a HUD timer ROI.

    Returns: timestamp in seconds (float) in the VIDEO.

    ROI priority:
      1) explicit `roi=(x,y,w,h)`
      2) percent-based roi_left_pct/roi_top_pct (+ roi_w_pct/roi_h_pct)
      3) interactive ROI selection (only if interactive_roi=True)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 60.0

    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ok, frame0 = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read first frame (check start_sec / video path).")

    # Decide ROI (one-time)
    if roi is None:
        if roi_left_pct is not None and roi_top_pct is not None:
            roi = _roi_from_percent(
                frame0,
                left_pct=float(roi_left_pct),
                top_pct=float(roi_top_pct),
                roi_w_pct=float(roi_w_pct),
                roi_h_pct=float(roi_h_pct),
            )
        elif interactive_roi:
            cv2.namedWindow("Select TIMER ROI (press ENTER when done)", cv2.WINDOW_NORMAL)
            roi_sel = cv2.selectROI(
                "Select TIMER ROI (press ENTER when done)",
                frame0,
                fromCenter=False,
                showCrosshair=True,
            )
            cv2.destroyAllWindows()
            x, y, w, h = map(int, roi_sel)
            if w <= 0 or h <= 0:
                cap.release()
                raise RuntimeError("ROI selection cancelled or invalid.")
            roi = (x, y, w, h)
        else:
            cap.release()
            raise RuntimeError(
                "ROI not set. Provide `roi=(x,y,w,h)` or "
                "`roi_left_pct`+`roi_top_pct`, or enable interactive_roi."
            )

    # Rewind to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    seen_red = False
    white_streak = 0
    max_frames = int(max_sec * fps)

    frame_idx = start_frame
    processed = 0

    while processed < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if (frame_idx - start_frame) % max(1, downsample) != 0:
            frame_idx += 1
            continue

        x, y, w, h = roi
        roi_bgr = frame[y:y + h, x:x + w]
        if roi_bgr.size == 0:
            cap.release()
            raise RuntimeError("ROI out of bounds; adjust ROI percents/size.")

        red_ratio, white_ratio, _mean_v = _compute_stats(roi_bgr)

        if not seen_red:
            if red_ratio >= min_red_ratio and white_ratio < min_white_ratio:
                seen_red = True
        else:
            if white_ratio >= min_white_ratio and red_ratio < min_red_ratio:
                white_streak += 1
            else:
                white_streak = 0

            if white_streak >= stable_frames:
                t_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                t_sec = (t_msec / 1000.0) if t_msec and t_msec > 0 else (frame_idx / fps)
                cap.release()
                return float(t_sec)

        frame_idx += 1
        processed += 1

    cap.release()
    raise RuntimeError(
        "No red->white flip detected in the scanned range. "
        "Try increasing max_sec, or adjust ROI percents/size, "
        "or lower min_white_ratio slightly."
    )
