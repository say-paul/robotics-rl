"""
VLM-enhanced terrain detector for stair/ladder/ramp identification.

Combines:
  1. Fast edge-based detection (Canny + Hough) for stair-like patterns
  2. VLM fallback for complex terrain classification
  3. AutonomousTerrainNavigator that selects locomotion policy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TerrainType(Enum):
    FLAT = auto()
    STAIRS_UP = auto()
    STAIRS_DOWN = auto()
    RAMP_UP = auto()
    RAMP_DOWN = auto()
    OBSTACLE = auto()
    UNKNOWN = auto()


@dataclass
class TerrainAssessment:
    terrain: TerrainType
    confidence: float
    description: str


def detect_stairs_fast(frame: np.ndarray, min_lines: int = 4) -> Optional[TerrainAssessment]:
    """Fast edge-based stair detection using Canny + Hough lines.

    Looks for multiple roughly-horizontal lines at regular vertical spacing,
    characteristic of stair edges.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    roi = gray[h // 3 : h, :]
    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=w // 4, maxLineGap=20)

    if lines is None:
        return None

    horizontal = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 15 or angle > 165:
            horizontal.append((y1 + y2) / 2)

    if len(horizontal) < min_lines:
        return None

    ys = sorted(set(round(y / 10) * 10 for y in horizontal))
    if len(ys) < min_lines:
        return None

    spacings = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    if not spacings:
        return None

    mean_sp = np.mean(spacings)
    std_sp = np.std(spacings)
    regularity = 1.0 - min(1.0, std_sp / (mean_sp + 1e-6))

    if regularity < 0.4:
        return None

    conf = min(1.0, 0.3 + 0.1 * len(ys) + 0.3 * regularity)

    center_y = np.mean(horizontal)
    roi_h = roi.shape[0]
    terrain = TerrainType.STAIRS_UP if center_y > roi_h * 0.4 else TerrainType.STAIRS_DOWN

    return TerrainAssessment(
        terrain=terrain,
        confidence=conf,
        description=f"Detected {len(ys)} stair edges (regularity={regularity:.2f})",
    )


def detect_terrain_vlm(
    frame: np.ndarray,
    vlm_caption_fn: Callable,
) -> TerrainAssessment:
    """Use VLM to classify terrain type from camera image."""
    prompt = (
        "Look at the terrain ahead. Classify it as one of: "
        "flat ground, stairs going up, stairs going down, ramp going up, "
        "ramp going down, obstacle. "
        "Reply with ONLY the terrain type and a brief description."
    )
    result = vlm_caption_fn(frame, prompt)
    result_lower = result.lower()

    if "stair" in result_lower and "up" in result_lower:
        terrain = TerrainType.STAIRS_UP
    elif "stair" in result_lower and "down" in result_lower:
        terrain = TerrainType.STAIRS_DOWN
    elif "ramp" in result_lower and "up" in result_lower:
        terrain = TerrainType.RAMP_UP
    elif "ramp" in result_lower and "down" in result_lower:
        terrain = TerrainType.RAMP_DOWN
    elif "obstacle" in result_lower:
        terrain = TerrainType.OBSTACLE
    elif "flat" in result_lower:
        terrain = TerrainType.FLAT
    else:
        terrain = TerrainType.UNKNOWN

    return TerrainAssessment(
        terrain=terrain,
        confidence=0.7,
        description=result[:200],
    )


class AutonomousTerrainNavigator:
    """Selects and triggers locomotion policies based on terrain assessment."""

    def __init__(self, motor, get_image_fn, vlm_caption_fn=None):
        self._motor = motor
        self._get_image = get_image_fn
        self._vlm_caption = vlm_caption_fn
        self._stair_policy = None

    def load_stair_policy(self, policy_path: str) -> bool:
        """Load a trained RL stair-climbing policy."""
        try:
            from stable_baselines3 import PPO
            self._stair_policy = PPO.load(policy_path)
            logger.info("Stair policy loaded from %s", policy_path)
            return True
        except Exception as e:
            logger.warning("Failed to load stair policy: %s", e)
            return False

    def assess_and_act(self) -> str:
        """Assess terrain and take appropriate action."""
        frame = self._get_image()
        if frame is None:
            return "Camera not available."

        assessment = detect_stairs_fast(frame)

        if assessment is None and self._vlm_caption:
            assessment = detect_terrain_vlm(frame, self._vlm_caption)

        if assessment is None:
            return "Terrain unclear — proceeding with caution on flat ground."

        logger.info(
            "[TERRAIN] %s (conf=%.2f): %s",
            assessment.terrain.name, assessment.confidence, assessment.description,
        )

        if assessment.terrain == TerrainType.FLAT:
            return f"Flat ground detected. {assessment.description}"

        if assessment.terrain in (TerrainType.STAIRS_UP, TerrainType.STAIRS_DOWN):
            direction = "up" if assessment.terrain == TerrainType.STAIRS_UP else "down"
            if self._stair_policy:
                return f"Stairs ({direction}) detected — switching to RL policy."
            return (
                f"Stairs ({direction}) detected (conf={assessment.confidence:.0%}). "
                f"{assessment.description}. RL policy not loaded — cannot climb."
            )

        return f"Terrain: {assessment.terrain.name}. {assessment.description}"
