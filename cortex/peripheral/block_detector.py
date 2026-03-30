"""
Fast image-based colored-block detection via HSV segmentation.

Runs at ~0.4ms per frame (2300+ Hz) — orders of magnitude faster than VLM
captioning (~70s).  Used by the planner for scan + approach navigation.

Returns bearing (degrees from image center) and estimated distance (metres
from apparent pixel size) for the largest matching contour.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# HSV ranges tuned for MuJoCo-rendered blocks.
# Each color maps to a list of (lower, upper) HSV bound tuples.
_HSV_RANGES: Dict[str, list] = {
    "red":     [((0, 120, 70), (10, 255, 255)),
                ((170, 120, 70), (180, 255, 255))],
    "green":   [((35, 80, 60), (85, 255, 255))],
    "blue":    [((100, 120, 60), (130, 255, 255))],
    "yellow":  [((22, 120, 100), (35, 255, 255))],
    "orange":  [((10, 200, 100), (22, 255, 255))],
    "purple":  [((125, 60, 50), (141, 255, 255))],
    "cyan":    [((80, 80, 175), (100, 255, 255))],
    "teal":    [((80, 80, 50), (100, 255, 175))],
    "magenta": [((146, 100, 80), (158, 255, 255))],
    "pink":    [((158, 40, 150), (175, 255, 255)),
                ((0, 30, 150), (10, 150, 255))],
    "white":   [((0, 0, 200), (180, 40, 255))],
    "brown":   [((8, 100, 40), (20, 255, 180))],
    "coral":   [((0, 80, 150), (12, 200, 255)),
                ((170, 80, 150), (180, 200, 255))],
}

MIN_CONTOUR_AREA = 80
MAX_CONTOUR_AREA = 60000

HFOV_DEG = 90.0
KNOWN_BLOCK_SIZE_M = 0.8
FOCAL_LENGTH_PX = 320.0


@dataclass
class BlockDetection:
    color: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area: int
    bearing_deg: float
    est_distance_m: float


def has_color(color: str) -> bool:
    """Check if we have HSV ranges defined for a color."""
    return color.lower() in _HSV_RANGES


def detect_block(
    frame: np.ndarray,
    color: str,
    min_area: int = MIN_CONTOUR_AREA,
) -> Optional[BlockDetection]:
    """Detect the largest block of the given color in the frame.

    Returns None if no block found or color not in _HSV_RANGES.
    """
    color_lower = color.lower()
    ranges = _HSV_RANGES.get(color_lower)
    if ranges is None:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    h, w = frame.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = max(contours, key=cv2.contourArea)
    area = int(cv2.contourArea(best))
    if area < min_area:
        return None

    x, y, bw, bh = cv2.boundingRect(best)
    cx = x + bw / 2.0
    pixel_offset = cx - w / 2.0
    # Negate: left-of-center (pixel_offset<0) → positive bearing (turn left)
    bearing = -(pixel_offset / w) * HFOV_DEG

    if area > MAX_CONTOUR_AREA:
        # Block fills most of the frame — robot is right on top of it
        logger.info(
            "[DETECT] %s block: bbox=(%d,%d,%d,%d) area=%d bearing=%.1f° — VERY CLOSE",
            color, x, y, bw, bh, area, bearing,
        )
        return BlockDetection(
            color=color_lower,
            bbox=(x, y, bw, bh),
            area=area,
            bearing_deg=bearing,
            est_distance_m=0.3,
        )

    apparent_h = max(bh, 1)
    est_dist = (KNOWN_BLOCK_SIZE_M * FOCAL_LENGTH_PX) / apparent_h

    logger.info(
        "[DETECT] %s block: bbox=(%d,%d,%d,%d) area=%d bearing=%.1f° dist=%.1fm",
        color, x, y, bw, bh, area, bearing, est_dist,
    )
    return BlockDetection(
        color=color_lower,
        bbox=(x, y, bw, bh),
        area=area,
        bearing_deg=bearing,
        est_distance_m=est_dist,
    )
