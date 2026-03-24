"""
Vision-Language-Action (VLA) Module
====================================
Translates natural language instructions into navigation goals
and velocity command sequences.

Architecture:
  1. Language parser extracts intent + target from free-form text
  2. Scene graph provides known landmark positions
  3. Navigation controller drives the WBC toward the goal

Designed to be swappable: replace the rule-based parser with a
learned VLA model (e.g., RT-2, OpenVLA) by overriding `parse()`.
"""

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Scene Landmarks ─────────────────────────────────────────────────────────

SCENE_LANDMARKS = {
    "water_bottle_stage": {
        "backstage":    (-5.0,  0.0),
        "stairs":       (-1.5,  0.0),
        "stage":        ( 2.0,  0.0),
        "podium":       ( 3.5,  0.0),
        "table":        ( 1.5, -2.0),
        "water bottle": ( 1.5, -2.0),
        "center stage": ( 2.0,  0.0),
        "audience":     ( 7.0,  0.0),
    },
    "flat_ground": {
        "origin":  (0.0, 0.0),
        "north":   (5.0, 0.0),
        "south":   (-5.0, 0.0),
        "east":    (0.0, -5.0),
        "west":    (0.0, 5.0),
    },
}


# ── VLA Action Types ────────────────────────────────────────────────────────

class VLAActionType(Enum):
    NAVIGATE_TO = auto()    # walk to a (x, y) target
    WALK_DISTANCE = auto()  # walk forward N metres
    TURN_ANGLE = auto()     # turn by N degrees
    STOP = auto()
    UNKNOWN = auto()


@dataclass
class VLAAction:
    action_type: VLAActionType
    target_xy: Optional[Tuple[float, float]] = None
    distance: float = 0.0
    angle_deg: float = 0.0
    description: str = ""
    walk_speed: float = 0.3
    turn_speed: float = 0.5


# ── Language Parser ─────────────────────────────────────────────────────────

# Patterns for distance commands: "walk forward 3 meters", "go 2m ahead"
_DIST_RE = re.compile(
    r"(?:walk|go|move)\s+(?:forward\s+)?(\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)",
    re.IGNORECASE,
)

# Patterns for turn commands: "turn around", "turn 90 degrees left"
_TURN_RE = re.compile(
    r"turn\s+(?:around|(\d+)\s*(?:degrees?|deg)?\s*(left|right)?)",
    re.IGNORECASE,
)


def parse_instruction(text: str, scene_name: str = "flat_ground") -> VLAAction:
    """Parse a natural language instruction into a VLAAction."""
    text_lower = text.strip().lower()

    # Stop
    if text_lower in ("stop", "halt", "freeze"):
        return VLAAction(action_type=VLAActionType.STOP, description="stop")

    # Distance-based walk
    m = _DIST_RE.search(text_lower)
    if m:
        dist = float(m.group(1))
        return VLAAction(
            action_type=VLAActionType.WALK_DISTANCE,
            distance=dist,
            description=f"walk forward {dist}m",
        )

    # Turn commands
    m = _TURN_RE.search(text_lower)
    if m:
        if "around" in text_lower:
            return VLAAction(
                action_type=VLAActionType.TURN_ANGLE,
                angle_deg=180.0,
                description="turn around",
            )
        angle = float(m.group(1)) if m.group(1) else 90.0
        direction = m.group(2) or "left"
        if direction == "right":
            angle = -angle
        return VLAAction(
            action_type=VLAActionType.TURN_ANGLE,
            angle_deg=angle,
            description=f"turn {abs(angle)}° {'left' if angle > 0 else 'right'}",
        )

    # Simple "turn left" / "turn right" without angle
    if "turn" in text_lower:
        if "right" in text_lower:
            return VLAAction(action_type=VLAActionType.TURN_ANGLE, angle_deg=-90.0, description="turn 90° right")
        if "left" in text_lower:
            return VLAAction(action_type=VLAActionType.TURN_ANGLE, angle_deg=90.0, description="turn 90° left")

    # Navigate to landmark
    landmarks = SCENE_LANDMARKS.get(scene_name, {})
    for name, pos in landmarks.items():
        if name in text_lower:
            return VLAAction(
                action_type=VLAActionType.NAVIGATE_TO,
                target_xy=pos,
                description=f"navigate to {name} at ({pos[0]:.1f}, {pos[1]:.1f})",
            )

    # Fuzzy intent matching for common phrases
    if any(w in text_lower for w in ["come", "approach", "walk to", "go to", "head to", "move to"]):
        for name, pos in landmarks.items():
            # Check if any word from the landmark name appears
            name_words = name.split()
            if any(w in text_lower for w in name_words):
                return VLAAction(
                    action_type=VLAActionType.NAVIGATE_TO,
                    target_xy=pos,
                    description=f"navigate to {name}",
                )

    # Generic walk forward
    if any(w in text_lower for w in ["walk", "go", "move", "advance", "run"]):
        speed = 0.3
        if "fast" in text_lower or "run" in text_lower:
            speed = 0.6
        if "slow" in text_lower:
            speed = 0.15
        return VLAAction(
            action_type=VLAActionType.WALK_DISTANCE,
            distance=2.0,
            walk_speed=speed,
            description=f"walk forward 2m at speed {speed}",
        )

    return VLAAction(action_type=VLAActionType.UNKNOWN, description=text)


# ── Navigation Controller ──────────────────────────────────────────────────

class NavigationController:
    """Drives the robot toward a goal by producing velocity commands each tick.

    Uses proportional control on heading error and distance to target.
    """

    ARRIVE_THRESHOLD = 0.5   # metres
    HEADING_THRESHOLD = 0.15  # radians (~8°)

    def __init__(self):
        self.active = False
        self._action: Optional[VLAAction] = None
        self._start_pos: Optional[np.ndarray] = None
        self._start_yaw: float = 0.0
        self._accumulated_yaw: float = 0.0
        self._last_yaw: float = 0.0
        self._ticks: int = 0

    def start(self, action: VLAAction, current_pos: np.ndarray, current_yaw: float) -> None:
        self.active = True
        self._action = action
        self._start_pos = current_pos[:2].copy()
        self._start_yaw = current_yaw
        self._accumulated_yaw = 0.0
        self._last_yaw = current_yaw
        self._ticks = 0
        logger.info("Nav started: %s", action.description)

    def tick(self, current_pos: np.ndarray, current_yaw: float) -> Tuple[float, float, float]:
        """Returns (vx, vy, yaw_rate). Returns (0,0,0) and sets active=False when done."""
        if not self.active or self._action is None:
            return 0.0, 0.0, 0.0

        self._ticks += 1
        if self._ticks > 5000:
            logger.warning("Nav timeout — stopping")
            self.cancel()
            return 0.0, 0.0, 0.0

        a = self._action

        if a.action_type == VLAActionType.NAVIGATE_TO:
            return self._navigate_to(current_pos, current_yaw, a)
        elif a.action_type == VLAActionType.WALK_DISTANCE:
            return self._walk_distance(current_pos, a)
        elif a.action_type == VLAActionType.TURN_ANGLE:
            return self._turn_angle(current_yaw, a)
        elif a.action_type == VLAActionType.STOP:
            self.cancel()
            return 0.0, 0.0, 0.0

        self.cancel()
        return 0.0, 0.0, 0.0

    def _navigate_to(
        self, pos: np.ndarray, yaw: float, a: VLAAction,
    ) -> Tuple[float, float, float]:
        tx, ty = a.target_xy
        dx = tx - pos[0]
        dy = ty - pos[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < self.ARRIVE_THRESHOLD:
            logger.info("Nav arrived at target (%.1f, %.1f)", tx, ty)
            self.cancel()
            return 0.0, 0.0, 0.0

        # Desired heading toward target
        desired_yaw = math.atan2(dy, dx)
        heading_err = self._wrap_angle(desired_yaw - yaw)

        # If heading error is large, turn first; otherwise walk + correct
        if abs(heading_err) > self.HEADING_THRESHOLD:
            yaw_rate = np.clip(heading_err * 2.0, -0.8, 0.8)
            vx = a.walk_speed * 0.2 if abs(heading_err) < 0.5 else 0.0
        else:
            vx = min(a.walk_speed, dist * 0.5)
            yaw_rate = np.clip(heading_err * 1.5, -0.5, 0.5)

        return float(vx), 0.0, float(yaw_rate)

    def _walk_distance(self, pos: np.ndarray, a: VLAAction) -> Tuple[float, float, float]:
        walked = np.linalg.norm(pos[:2] - self._start_pos)
        if walked >= a.distance:
            logger.info("Nav walked %.2fm (target %.2fm)", walked, a.distance)
            self.cancel()
            return 0.0, 0.0, 0.0
        return a.walk_speed, 0.0, 0.0

    def _turn_angle(self, yaw: float, a: VLAAction) -> Tuple[float, float, float]:
        # Track accumulated yaw change
        dyaw = self._wrap_angle(yaw - self._last_yaw)
        self._accumulated_yaw += dyaw
        self._last_yaw = yaw

        target_rad = math.radians(a.angle_deg)
        remaining = target_rad - self._accumulated_yaw

        if abs(remaining) < 0.05:
            logger.info("Nav turn complete (%.1f°)", math.degrees(self._accumulated_yaw))
            self.cancel()
            return 0.0, 0.0, 0.0

        rate = np.clip(remaining * 2.0, -a.turn_speed, a.turn_speed)
        return 0.0, 0.0, float(rate)

    def cancel(self) -> None:
        self.active = False
        self._action = None

    @staticmethod
    def _wrap_angle(a: float) -> float:
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a
