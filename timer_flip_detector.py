from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
import argparse

# ---------------------------
# Utilities
# ---------------------------

def _hex_to_bgr(hex_color: str) -> np.ndarray:
    s = hex_color.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Bad hex color: {hex_color}")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return np.array([b, g, r], dtype=np.uint8)

def _roi_from_percent(
    frame: np.ndarray,
    *,
    left_pct: float,
    top_pct: float,
    roi_w_pct: float,
    roi_h_pct: float,
) -> tuple[int, int, int, int]:
    """
    ROI defined by percentages, (left_pct, top_pct) is CENTER of ROI.
    """
    H, W = frame.shape[:2]
    cx = int(round(W * left_pct))
    cy = int(round(H * top_pct))
    w = int(round(W * roi_w_pct))
    h = int(round(H * roi_h_pct))

    w = max(4, w)
    h = max(4, h)

    x = max(0, min(cx - w // 2, W - w))
    y = max(0, min(cy - h // 2, H - h))
    return (x, y, w, h)

# ---------------------------
# Core color stats (ROI-average)
# ---------------------------

def _compute_stats(
    roi_bgr: np.ndarray,
    *,
    red_hex: str,
    red_dist: float,
    white_s_max: int,
    white_v_min: int,
    red_s_min: int,
    red_v_min: int,
):
    """
    Compute (red_ratio, white_ratio, dbg) using ROI-AVERAGE masks.

    - red_ratio: fraction of ROI pixels close to target red in HSV distance
    - white_ratio: fraction of ROI pixels with low saturation and high value
    - dbg: includes center pixel + reddest pixel (for debugging only)
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # --- White mask (ROI-average)
    white_mask = (s <= white_s_max) & (v >= white_v_min)
    white_ratio = float(np.mean(white_mask))

    # --- Target red in HSV
    target_bgr = _hex_to_bgr(red_hex).reshape(1, 1, 3)
    target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0, 0].astype(np.int16)
    th, ts, tv = int(target_hsv[0]), int(target_hsv[1]), int(target_hsv[2])

    hh = h.astype(np.int16)
    ss = s.astype(np.int16)
    vv = v.astype(np.int16)

    # Circular hue distance
    dh = np.abs(hh - th)
    dh = np.minimum(dh, 180 - dh)

    # Weighted HSV distance
    # (Hue matters most; S/V matter but less)
    dist = (2.0 * dh) + (0.6 * np.abs(ss - ts)) + (0.25 * np.abs(vv - tv))

    # Red mask (ROI-average): close enough + not too dark/desaturated
    red_mask = (dist <= red_dist) & (ss >= red_s_min) & (vv >= red_v_min)
    red_ratio = float(np.mean(red_mask))

    # Debug pixels:
    H, W = roi_bgr.shape[:2]
    cx, cy = W // 2, H // 2
    center_bgr = roi_bgr[cy, cx].tolist()
    center_hsv = hsv[cy, cx].tolist()

    # "reddest pixel" (min distance) just for inspection
    min_idx = np.unravel_index(np.argmin(dist), dist.shape)
    py, px = int(min_idx[0]), int(min_idx[1])

    dbg = {
        "center_px": (cx, cy),
        "center_bgr": center_bgr,
        "center_hsv": center_hsv,
        "reddest_px": (px, py),
        "reddest_bgr": roi_bgr[py, px].tolist(),
        "reddest_hsv": hsv[py, px].tolist(),
        "min_dist": float(dist[py, px]),
    }

    return red_ratio, white_ratio, dbg

# ---------------------------
# Main detection
# ---------------------------

def detect_first_red_to_white_flip(
    video_path: str,
    *,
    start_sec: float = 0.0,
    max_sec: float = 180.0,
    downsample: int = 2,
    stable_frames: int = 5,

    # ROI defaults (NOW optional)
    roi_left_pct: float = 0.50,
    roi_top_pct: float = 0.018,
    roi_w_pct: float = 0.03,
    roi_h_pct: float = 0.02,

    # Color detection defaults
    red_hex: str = "#ee3d3a",
    red_dist: float = 45.0,
    min_red_ratio: float = 0.05,
    min_white_ratio: float = 0.12,

    # Masks
    white_s_max: int = 40,
    white_v_min: int = 170,
    red_s_min: int = 25,
    red_v_min: int = 45,

    # Debug
    debug_show: bool = False,
    debug_every: int = 15,
) -> float:
    """
    Detect first sustained red -> white HUD timer flip.
    Returns video timestamp in seconds.
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
        raise RuntimeError("Could not read first frame")

    roi = _roi_from_percent(
        frame0,
        left_pct=roi_left_pct,
        top_pct=roi_top_pct,
        roi_w_pct=roi_w_pct,
        roi_h_pct=roi_h_pct,
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    seen_red = False
    white_streak = 0
    processed = 0
    max_frames = int(max_sec * fps)
    frame_idx = start_frame

    while processed < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if (frame_idx - start_frame) % max(1, downsample) != 0:
            frame_idx += 1
            continue

        x, y, w, h = roi
        roi_bgr = frame[y:y+h, x:x+w]
        if roi_bgr.size == 0:
            cap.release()
            raise RuntimeError("ROI out of bounds")

        red_ratio, white_ratio, dbg = _compute_stats(
            roi_bgr,
            red_hex=red_hex,
            red_dist=red_dist,
            white_s_max=white_s_max,
            white_v_min=white_v_min,
            red_s_min=red_s_min,
            red_v_min=red_v_min,
        )

        # --- State machine (same as before, but driven by ROI-average ratios)
        if not seen_red:
            # Arm once we truly see red
            if red_ratio >= min_red_ratio:
                seen_red = True
        else:
            # After red, look for sustained white while red drops
            if white_ratio >= min_white_ratio and red_ratio < min_red_ratio:
                white_streak += 1
            else:
                white_streak = 0

            if white_streak >= stable_frames:
                t_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                t_sec = (t_msec / 1000.0) if t_msec and t_msec > 0 else (frame_idx / fps)
                cap.release()
                if debug_show:
                    cv2.destroyAllWindows()
                return float(t_sec)

        # --- Debug overlay
        if debug_show and (processed % max(1, debug_every) == 0):
            vis = frame.copy()
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Mark center sample pixel (what you likely care about)
            cx, cy = dbg["center_px"]
            cpx = x + cx
            cpy = y + cy
            cv2.circle(vis, (cpx, cpy), 5, (255, 255, 0), -1)  # cyan-ish

            # Mark reddest pixel (for inspection)
            rx, ry = dbg["reddest_px"]
            rpx = x + rx
            rpy = y + ry
            cv2.circle(vis, (rpx, rpy), 5, (0, 255, 255), -1)  # yellow

            txt1 = f"t={frame_idx/fps:.2f}s red_ratio={red_ratio:.3f} white_ratio={white_ratio:.3f} seen_red={seen_red} streak={white_streak}"
            txt2 = f"center BGR={dbg['center_bgr']} HSV={dbg['center_hsv']} | reddest dist={dbg['min_dist']:.1f}"
            cv2.putText(vis, txt1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
            cv2.putText(vis, txt2, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

            cv2.imshow("timer_flip_detector (q to quit)", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                raise RuntimeError("Aborted by user")

        frame_idx += 1
        processed += 1

    cap.release()
    if debug_show:
        cv2.destroyAllWindows()

    raise RuntimeError("No red→white flip detected")
