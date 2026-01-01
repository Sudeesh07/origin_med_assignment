from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class EllipseMeasurement:
    center: tuple[float, float]
    major_axis: float
    minor_axis: float
    angle_deg: float
    bpd_points: tuple[tuple[int, int], tuple[int, int]]
    cfd_points: tuple[tuple[int, int], tuple[int, int]]
    contour_area: float


def largest_contour(binary_mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def fit_cranium_ellipse(mask: np.ndarray) -> Optional[EllipseMeasurement]:
    """
    Fits an ellipse to the largest contour and returns endpoints for BPD (major axis)
    and CFD (minor axis). The mask should be a binary uint8 array.
    """
    contour = largest_contour(mask)
    if contour is None or len(contour) < 5:
        return None

    (cx, cy), (major_len, minor_len), angle = cv2.fitEllipse(contour)
    # Ensure major_len is the longer axis
    if major_len < minor_len:
        major_len, minor_len = minor_len, major_len
        angle = (angle + 90.0) % 180.0

    theta = np.deg2rad(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Major axis endpoints (BPD)
    dx_major = (major_len / 2) * cos_t
    dy_major = (major_len / 2) * sin_t
    p1_major = (int(round(cx - dx_major)), int(round(cy - dy_major)))
    p2_major = (int(round(cx + dx_major)), int(round(cy + dy_major)))

    # Minor axis endpoints (CFD)
    dx_minor = (minor_len / 2) * -sin_t
    dy_minor = (minor_len / 2) * cos_t
    p1_minor = (int(round(cx - dx_minor)), int(round(cy - dy_minor)))
    p2_minor = (int(round(cx + dx_minor)), int(round(cy + dy_minor)))

    return EllipseMeasurement(
        center=(cx, cy),
        major_axis=major_len,
        minor_axis=minor_len,
        angle_deg=angle,
        bpd_points=(p1_major, p2_major),
        cfd_points=(p1_minor, p2_minor),
        contour_area=cv2.contourArea(contour),
    )


def mask_from_logits(logits: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Converts model logits (H, W) to uint8 binary mask."""
    probs = 1 / (1 + np.exp(-logits))
    return (probs >= threshold).astype("uint8") * 255

