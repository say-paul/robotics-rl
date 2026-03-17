"""Waypoint-following controller that outputs velocity commands.

Converts a target waypoint into (vx, vy, vyaw) commands for the
low-level RL locomotion policy.  Pure Python, no RL needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .mission import Waypoint


@dataclass
class FollowerConfig:
    """Tuning parameters for the path follower."""
    max_vx: float = 0.5
    max_vy: float = 0.2
    max_vyaw: float = 0.8
    heading_kp: float = 2.0       # proportional gain for yaw correction
    distance_kp: float = 1.0      # proportional gain for forward speed
    slowdown_radius: float = 1.0  # start decelerating within this distance


def wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class PathFollower:
    """PID-like controller: waypoint → velocity commands."""

    def __init__(self, config: Optional[FollowerConfig] = None):
        self.cfg = config or FollowerConfig()

    def compute_command(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        target: Waypoint,
    ) -> Tuple[float, float, float]:
        """Compute (vx, vy, vyaw) to drive toward the target waypoint.

        Returns commands in the robot's local frame.
        """
        cfg = self.cfg
        dx = target.x - robot_x
        dy = target.y - robot_y
        dist = math.sqrt(dx * dx + dy * dy)

        bearing = math.atan2(dy, dx)
        heading_err = wrap_angle(bearing - robot_yaw)

        speed = min(target.speed, cfg.distance_kp * dist)
        if dist < cfg.slowdown_radius:
            speed *= dist / cfg.slowdown_radius

        vyaw = cfg.heading_kp * heading_err
        vyaw = max(-cfg.max_vyaw, min(cfg.max_vyaw, vyaw))

        cos_h, sin_h = math.cos(heading_err), math.sin(heading_err)
        vx = speed * cos_h
        vy = speed * sin_h

        vx = max(-cfg.max_vx, min(cfg.max_vx, vx))
        vy = max(-cfg.max_vy, min(cfg.max_vy, vy))

        return vx, vy, vyaw

    def has_arrived(
        self,
        robot_x: float,
        robot_y: float,
        target: Waypoint,
    ) -> bool:
        """Check if the robot is within the arrival radius."""
        return target.distance_to(robot_x, robot_y) < target.arrival_radius
