"""Color naming and diameter estimation for detected pipes."""
from __future__ import annotations

import cv2
import numpy as np


def estimate_diameter_mm(w_px: int, h_px: int, px_per_mm: float | None) -> float | None:
    """Estimate diameter from a bounding box.

    The longer side is used because a tilted pipe end looks like a narrow
    oval whose long side is closest to the true diameter. Returns None when no calibration is set.
    """
    if not px_per_mm or px_per_mm <= 0:
        return None
    return round(max(w_px, h_px) / px_per_mm, 2)


def dominant_color_name(crop_bgr: np.ndarray) -> str:
    """Name the dominant color of a pipe crop using HSV (OpenCV hue 0-179).

    Only the central 50% of the crop is used, to avoid background pixels.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"

    h, w = crop_bgr.shape[:2]
    cy, cx = h // 2, w // 2
    dh, dw = max(h // 4, 1), max(w // 4, 1)
    center = crop_bgr[cy - dh: cy + dh, cx - dw: cx + dw]
    if center.size == 0:
        center = crop_bgr

    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hue = float(np.median(hsv[:, 0]))
    sat = float(np.median(hsv[:, 1]))
    val = float(np.median(hsv[:, 2]))

    if val < 55:
        return "black"
    if sat < 40:
        return "white" if val > 140 else "gray"
    if hue < 10 or hue >= 170:
        return "red"
    if hue < 22:
        return "orange"
    if hue < 35:
        return "yellow"
    if hue < 85:
        return "green"
    if hue < 130:
        return "blue"
    if hue < 160:
        return "purple"
    return "pink"


def size_group(diameter_mm: float | None, step: float = 0.5) -> str:
    """Round a diameter to the nearest step so pipes can be grouped by size."""
    if diameter_mm is None:
        return "not calibrated"
    rounded = round(diameter_mm / step) * step
    return f"{rounded:g} mm"
