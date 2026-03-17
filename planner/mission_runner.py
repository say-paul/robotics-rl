"""State machine that executes a Mission using the PathFollower.

Orchestrates navigation through a sequence of waypoints, switching
behaviors and triggering skills on arrival.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from .mission import Mission, Waypoint
from .path_follower import PathFollower

log = logging.getLogger(__name__)


class MissionState(enum.Enum):
    IDLE = "idle"
    HOLDING = "holding"
    NAVIGATING = "navigating"
    EXECUTING_SKILL = "executing_skill"
    ARRIVED = "arrived"
    FAILED = "failed"


@dataclass
class MissionCommand:
    """Output from a single mission step."""
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0
    behavior: str = "stand"


class MissionRunner:
    """Executes a mission waypoint by waypoint."""

    def __init__(self, mission: Mission, follower: Optional[PathFollower] = None):
        self.mission = mission
        self.follower = follower or PathFollower()
        self.state = MissionState.IDLE
        self._wp_idx = 0
        self._hold_timer = 0.0
        self._skill_timer = 0.0
        self._prev_behavior: Optional[str] = None

    @property
    def current_waypoint(self) -> Optional[Waypoint]:
        if self._wp_idx < len(self.mission.waypoints):
            return self.mission.waypoints[self._wp_idx]
        return None

    @property
    def current_behavior(self) -> str:
        wp = self.current_waypoint
        return wp.behavior if wp else "stand"

    @property
    def behavior_changed(self) -> bool:
        """True if the behavior changed since the last step."""
        return self.current_behavior != self._prev_behavior

    @property
    def progress(self) -> str:
        total = len(self.mission.waypoints)
        return f"{self._wp_idx}/{total} ({self.state.value})"

    def start(self) -> None:
        """Begin mission execution."""
        self._wp_idx = 0
        self._prev_behavior = None
        wp = self.current_waypoint
        if wp is None:
            self.state = MissionState.ARRIVED
            return

        if wp.hold_duration > 0:
            self.state = MissionState.HOLDING
            self._hold_timer = wp.hold_duration
        else:
            self.state = MissionState.NAVIGATING
        log.info("Mission '%s' started -> %s (%s)", self.mission.name, wp.id, wp.behavior)

    def step(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        dt: float = 0.02,
    ) -> MissionCommand:
        """Advance one step. Returns velocity commands and the active behavior.

        Call this at the policy rate (50 Hz).
        """
        if self.state in (MissionState.IDLE, MissionState.ARRIVED, MissionState.FAILED):
            self._prev_behavior = "stand"
            return MissionCommand(behavior="stand")

        wp = self.current_waypoint
        if wp is None:
            self.state = MissionState.ARRIVED
            log.info("Mission '%s' complete", self.mission.name)
            self._prev_behavior = "stand"
            return MissionCommand(behavior="stand")

        self._prev_behavior = wp.behavior

        # Hold phase (e.g. standing to stabilize)
        if self.state == MissionState.HOLDING:
            self._hold_timer -= dt
            if self._hold_timer <= 0:
                log.info("Hold at '%s' complete, advancing", wp.id)
                self._advance()
                wp = self.current_waypoint
                if wp is None:
                    return MissionCommand(behavior="stand")
                return MissionCommand(behavior=wp.behavior)
            return MissionCommand(behavior=wp.behavior)

        # Skill execution phase
        if self.state == MissionState.EXECUTING_SKILL:
            self._skill_timer -= dt
            if self._skill_timer <= 0:
                log.info("Skill at %s complete, advancing", wp.id)
                self._advance()
            return MissionCommand(behavior=wp.behavior)

        # Navigation phase
        if self.follower.has_arrived(robot_x, robot_y, wp):
            log.info("Arrived at waypoint '%s'", wp.id)
            if wp.on_arrive:
                log.info("Triggering skill: %s", wp.on_arrive)
                self.state = MissionState.EXECUTING_SKILL
                self._skill_timer = 5.0
                return MissionCommand(behavior=wp.behavior)
            self._advance()
            wp = self.current_waypoint
            if wp is None:
                return MissionCommand(behavior="stand")
            return MissionCommand(behavior=wp.behavior)

        vx, vy, vyaw = self.follower.compute_command(robot_x, robot_y, robot_yaw, wp)
        return MissionCommand(vx=vx, vy=vy, vyaw=vyaw, behavior=wp.behavior)

    def _advance(self) -> None:
        """Move to the next waypoint."""
        self._wp_idx += 1
        if self._wp_idx >= len(self.mission.waypoints):
            self.state = MissionState.ARRIVED
            log.info("Mission '%s' complete", self.mission.name)
        else:
            wp = self.current_waypoint
            if wp.hold_duration > 0:
                self.state = MissionState.HOLDING
                self._hold_timer = wp.hold_duration
                log.info("Holding at '%s' for %.1fs (%s)", wp.id, wp.hold_duration, wp.behavior)
            else:
                self.state = MissionState.NAVIGATING
                log.info("Navigating to '%s' (%s)", wp.id, wp.behavior)
