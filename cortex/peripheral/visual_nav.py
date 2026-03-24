"""
Visual Target Navigation
========================
Generalized "go to <something I can see>" behaviour.

The robot uses its head camera + VLM **once** to detect direction and
rough distance to a target, computes an estimated world XY, then hands
off to the motor controller's dead-reckoning course correction to walk
there.  No repeated VLM calls during the walk.
"""

from __future__ import annotations

import logging
import math
import re
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Bearing / distance mappings from VLM qualitative output ──────────

_LATERAL_BEARING_DEG: Dict[str, float] = {
    "far_left":     40.0,
    "left":         30.0,
    "slight_left":  15.0,
    "center":        0.0,
    "slight_right": -15.0,
    "right":        -30.0,
    "far_right":    -40.0,
}

_RANGE_DISTANCE_M: Dict[str, float] = {
    "close":  1.5,
    "medium": 3.0,
    "far":    5.0,
}

# ── Detection ────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = (
    "Do you see {description} on the ground or floor in this image? "
    "If yes, describe its position: is it to the left, center, or right "
    "of the image? Is it close (near the bottom), medium, or far (near "
    "the top)? Answer ONLY with one of: "
    "'not visible' OR '<lateral> <range>' where lateral is one of "
    "far_left/left/slight_left/center/slight_right/right/far_right "
    "and range is close/medium/far. Example: 'left far' or 'center close'."
)

_LATERAL_WORDS = set(_LATERAL_BEARING_DEG.keys())
_RANGE_WORDS = set(_RANGE_DISTANCE_M.keys())


def detect_visual_target(
    description: str,
    get_image_fn: Callable,
    vlm_caption_fn: Callable,
) -> Optional[Tuple[str, str]]:
    """Ask the VLM whether *description* is visible and where.

    Returns ``(lateral, range_estimate)`` or *None* if the target is not
    visible or the response is unparseable.
    """
    image = get_image_fn()
    if image is None:
        logger.warning("[VISUAL_NAV] No camera frame available")
        return None

    prompt = _PROMPT_TEMPLATE.format(description=description)
    t0 = time.perf_counter()
    raw = vlm_caption_fn(image, prompt)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "[VISUAL_NAV] VLM detect (%.0fms): description='%s' raw='%s'",
        elapsed, description[:40], str(raw)[:120],
    )

    if not raw:
        return None
    return _parse_vlm_position(raw)


def _parse_vlm_position(text: str) -> Optional[Tuple[str, str]]:
    """Extract (lateral, range) from free-form VLM text."""
    low = text.lower().strip()

    if "not visible" in low or "don't see" in low or "do not see" in low or "cannot see" in low:
        return None

    # Try to find known keywords
    found_lat: Optional[str] = None
    found_rng: Optional[str] = None

    for word in _LATERAL_WORDS:
        if word in low:
            if found_lat is None or len(word) > len(found_lat):
                found_lat = word
    for word in _RANGE_WORDS:
        if word in low:
            found_rng = word

    # Fallback heuristics for free-form answers
    if found_lat is None:
        if "left" in low:
            found_lat = "left"
        elif "right" in low:
            found_lat = "right"
        elif "center" in low or "middle" in low or "straight" in low:
            found_lat = "center"

    if found_rng is None:
        if "close" in low or "near" in low or "bottom" in low:
            found_rng = "close"
        elif "far" in low or "distant" in low or "top" in low:
            found_rng = "far"
        else:
            found_rng = "medium"

    if found_lat is None:
        found_lat = "center"

    return (found_lat, found_rng)


# ── World-coordinate estimation ──────────────────────────────────────

def estimate_target_xy(
    lateral: str,
    range_est: str,
    robot_pos: Tuple[float, float],
    robot_yaw: float,
) -> Tuple[float, float]:
    """Map qualitative VLM output to an approximate world XY."""
    bearing_offset = math.radians(_LATERAL_BEARING_DEG.get(lateral, 0.0))
    dist = _RANGE_DISTANCE_M.get(range_est, 3.0)
    angle = robot_yaw + bearing_offset
    tx = robot_pos[0] + dist * math.cos(angle)
    ty = robot_pos[1] + dist * math.sin(angle)
    return (tx, ty)


# ── Full navigation behaviour ────────────────────────────────────────

_SEARCH_TURNS_DEG = [30.0, -60.0, 60.0, -30.0]


def navigate_to_visual_target(
    description: str,
    get_image_fn: Callable,
    vlm_caption_fn: Callable,
    motor,
    sim_state_fn: Callable,
) -> str:
    """Use VLM once to find direction, then dead-reckon to the target.

    1. VLM detects bearing + rough distance (one call, ~1-2 s).
    2. ``move_to_point`` walks there using proportional yaw correction
       from odometry — no further VLM calls during the walk.
    """
    # ── 1. Initial search (VLM) ──────────────────────────────────────
    detection = detect_visual_target(description, get_image_fn, vlm_caption_fn)

    if detection is None:
        for turn_deg in _SEARCH_TURNS_DEG:
            logger.info("[VISUAL_NAV] Target not visible, turning %.0f°", turn_deg)
            motor.turn_degrees(turn_deg)
            time.sleep(0.5)
            detection = detect_visual_target(description, get_image_fn, vlm_caption_fn)
            if detection is not None:
                break

    if detection is None:
        return f"I cannot see '{description}' from any direction."

    lateral, range_est = detection
    pos, yaw = sim_state_fn()
    target_x, target_y = estimate_target_xy(
        lateral, range_est, (pos[0], pos[1]), yaw,
    )
    logger.info(
        "[VISUAL_NAV] VLM estimate: lateral=%s range=%s -> target=(%.1f, %.1f)",
        lateral, range_est, target_x, target_y,
    )

    # ── 2. Walk using dead-reckoning course correction ───────────────
    result = motor.move_to_point(
        target_x=target_x,
        target_y=target_y,
        speed=0.3,
        tolerance=0.4,
    )
    return f"Visual navigation to '{description}': {result}"
