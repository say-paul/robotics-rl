"""
Motor Cortex
============
High-level motor API for the G1 humanoid robot.

Provides four blocking movement primitives that the planner (or any
higher layer) calls directly:

    move(distance, speed, direction)
        Walk a given distance along the current heading.
        *speed* defaults to 0.3 (normalised), *direction* defaults to
        the current heading.  If a cardinal string is given
        (e.g. "north") the robot turns first.

    turn_degrees(degrees)
        Rotate by *degrees* relative to current orientation (positive = left).

    turn_relative(direction)
        Turn 90° left or right.

    turn_magnetic(heading)
        Face a compass heading string ("north", "se", "west" …).

    halt()
        Immediately zero all velocities.

All blocking methods use the StepOdometry for feedback (step counting +
IMU heading).  The low-level WBC tick loop runs independently at 50 Hz.
"""

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from lerobot.robots.unitree_g1.g1_utils import (
    G1_29_JointIndex,
    REMOTE_AXES,
    default_remote_input,
)

logger = logging.getLogger(__name__)

DEFAULT_SPEED = 0.3        # normalised forward velocity [-1, 1]
DEFAULT_TURN_RATE = 0.35   # normalised yaw rate for turns (lower = more precise)
TURN_TOLERANCE_DEG = 2.0   # acceptable heading error after a turn


class ControlMode(Enum):
    IDLE = auto()
    WBC_WALKING = auto()
    STABILISING = auto()


@dataclass
class VelocityCommand:
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0


class MotorCortex:
    """Bridges high-level movement commands to the WBC locomotion controller."""

    def __init__(self, controller_type: str = "groot"):
        self.controller_type = controller_type
        self._controller = None
        self._controller_loaded = False

        self.mode = ControlMode.IDLE
        self.velocity_cmd = VelocityCommand()

        self._stabilise_steps = 0
        self._stabilise_target = 500

        self._odometry = None
        self._sim_state_fn: Optional[Callable] = None

        logger.info("MotorCortex created (controller=%s)", controller_type)

    # ── Wiring (called by orchestrator) ──────────────────────────────────

    def wire(
        self,
        odometry=None,
        sim_state_fn: Optional[Callable] = None,
    ) -> None:
        """Connect sensors needed by the blocking movement API."""
        self._odometry = odometry
        self._sim_state_fn = sim_state_fn

    # ── Controller Management ────────────────────────────────────────────

    def load_controller(self) -> None:
        if self._controller_loaded:
            return

        logger.info("Loading WBC controller: %s ...", self.controller_type)
        from lerobot.robots.unitree_g1.g1_utils import make_locomotion_controller

        name_map = {
            "groot": "GrootLocomotionController",
            "holosoma": "HolosomaLocomotionController",
        }
        cls_name = name_map.get(self.controller_type)
        if cls_name is None:
            raise ValueError(f"Unknown controller '{self.controller_type}'.")

        self._controller = make_locomotion_controller(cls_name)
        self._controller_loaded = True

        self._ctrl_kp = getattr(self._controller, 'kp', None)
        self._ctrl_kd = getattr(self._controller, 'kd', None)
        logger.info("WBC controller loaded: %s", type(self._controller).__name__)

    @property
    def control_dt(self) -> float:
        if self._controller is not None:
            return self._controller.control_dt
        return 0.02

    # ── WBC tick (called every sim control step) ─────────────────────────

    def tick(self, lowstate) -> Dict[str, float]:
        if not self._controller_loaded or self._controller is None:
            return {}
        if self.mode == ControlMode.IDLE:
            return {}

        action = default_remote_input()
        action["remote.ly"] = float(np.clip(self.velocity_cmd.vx, -1, 1))
        action["remote.lx"] = float(np.clip(self.velocity_cmd.vy, -1, 1))
        action["remote.rx"] = float(np.clip(-self.velocity_cmd.yaw_rate, -1, 1))

        target = self._controller.run_step(action, lowstate)

        if self.mode == ControlMode.STABILISING:
            self._stabilise_steps += 1
            if self._stabilise_steps >= self._stabilise_target:
                self.mode = ControlMode.WBC_WALKING
                logger.info("Stabilisation complete — WBC walking mode active")

        return target

    # ══════════════════════════════════════════════════════════════════════
    #  HIGH-LEVEL BLOCKING MOVEMENT API
    # ══════════════════════════════════════════════════════════════════════

    def move(
        self,
        distance: float = 1.0,
        speed: float = DEFAULT_SPEED,
        direction: Optional[str] = None,
    ) -> str:
        """Walk *distance* metres at *speed*.

        Parameters
        ----------
        distance : metres to walk (positive = forward, negative = backward).
        speed    : normalised velocity [0.05 .. 1.0].  Default 0.3.
        direction: optional cardinal string ("north", "sw", …).
                   If given the robot turns to that heading first.
                   If omitted, walks along current heading.

        Returns a status string.
        """
        results = []

        if direction:
            turn_result = self.turn_magnetic(direction)
            results.append(turn_result)

        speed = float(np.clip(abs(speed), 0.05, 1.0))
        if distance < 0:
            speed = -speed

        target_dist = abs(distance)
        pos0 = self._get_pos()

        if self._odometry:
            self._odometry.reset()
            heading0 = self._odometry.heading_rad
            self.set_velocity(speed, 0.0, 0.0)
            timeout = target_dist / max(abs(speed), 0.05) + 8.0
            t0 = time.time()
            drift_total = 0.0

            while time.time() - t0 < timeout:
                if self._odometry.total_distance >= target_dist * 0.95:
                    break
                # Heading-hold autopilot: P-controller corrects yaw drift
                drift = _angle_diff(self._odometry.heading_rad, heading0)
                yaw_corr = float(np.clip(-1.5 * drift, -0.4, 0.4))
                drift_total += abs(drift)
                self.set_velocity(speed, 0.0, yaw_corr)
                time.sleep(0.05)

            self.halt()
            imu_dist = self._odometry.total_distance
            imu_heading = self._odometry.heading_deg
            step_count = self._odometry.step_count
            stride = self._odometry.stride_length
            drift_deg = math.degrees(_angle_diff(self._odometry.heading_rad, heading0))

            pos_final = self._get_pos()
            qpos_dist = float(np.linalg.norm(pos_final[:2] - pos0[:2]))
            error = imu_dist - qpos_dist

            results.append(
                f"IMU: {imu_dist:.2f}m ({step_count} steps, stride {stride:.3f}m), "
                f"heading {imu_heading:.0f}°. "
                f"qpos: {qpos_dist:.2f}m at ({pos_final[0]:.2f}, {pos_final[1]:.2f}). "
                f"Error: {error:+.2f}m."
            )
            logger.info("[MOTOR:MOVE] drift=%.1f° during walk", drift_deg)
        else:
            self.set_velocity(speed, 0.0, 0.0)
            timeout = target_dist / max(abs(speed), 0.05) + 8.0
            t0 = time.time()
            while time.time() - t0 < timeout:
                pos = self._get_pos()
                if np.linalg.norm(pos[:2] - pos0[:2]) >= target_dist * 0.9:
                    break
                time.sleep(0.05)
            self.halt()
            pos_final = self._get_pos()
            qpos_dist = float(np.linalg.norm(pos_final[:2] - pos0[:2]))
            yaw_deg = math.degrees(self._get_yaw())
            results.append(
                f"Walked {qpos_dist:.2f}m (no IMU). "
                f"Position: ({pos_final[0]:.2f}, {pos_final[1]:.2f}), "
                f"heading {yaw_deg:.0f}°."
            )

        logger.info("[MOTOR:MOVE] distance=%.1f speed=%.2f dir=%s drift=auto -> %s",
                     distance, speed, direction, results[-1][:80])
        return " ".join(results)

    def turn_degrees(self, degrees: float) -> str:
        """Rotate *degrees* relative to current orientation.

        Positive = counter-clockwise (left), negative = clockwise (right).
        Uses IMU heading when available; falls back to qpos yaw.
        """
        if self._odometry:
            imu_yaw0 = self._odometry.heading_rad
            target_yaw = imu_yaw0 + math.radians(degrees)
        else:
            imu_yaw0 = self._get_yaw()
            target_yaw = imu_yaw0 + math.radians(degrees)

        rate = DEFAULT_TURN_RATE if degrees > 0 else -DEFAULT_TURN_RATE
        self.set_velocity(0.0, 0.0, rate)

        timeout = abs(degrees) / 25.0 + 3.0
        t0 = time.time()
        while time.time() - t0 < timeout:
            current_yaw = self._odometry.heading_rad if self._odometry else self._get_yaw()
            if abs(_angle_diff(current_yaw, target_yaw)) < math.radians(TURN_TOLERANCE_DEG):
                break
            time.sleep(0.05)

        self.halt()

        if self._odometry:
            imu_turned = math.degrees(_angle_diff(self._odometry.heading_rad, imu_yaw0))
            imu_heading = self._odometry.heading_deg
            cardinal = self._odometry.heading_cardinal
            qpos_yaw = math.degrees(self._get_yaw())
            result = (
                f"Turned {imu_turned:.1f}° (requested {degrees:.0f}°). "
                f"IMU heading: {imu_heading:.0f}° ({cardinal}). "
                f"qpos yaw: {qpos_yaw:.1f}°."
            )
        else:
            heading = self._heading_str()
            result = f"Turned {degrees:.0f}°. Heading: {heading}."

        logger.info("[MOTOR:TURN_DEG] degrees=%.1f -> %s", degrees, result[:80])
        return result

    def turn_relative(self, direction: str = "left") -> str:
        """Turn 90° left or right."""
        deg = 90.0 if direction.lower().startswith("l") else -90.0
        return self.turn_degrees(deg)

    def turn_magnetic(self, heading: str) -> str:
        """Turn to face a compass direction ("north", "se", "west", …).

        Uses IMU to compute the shortest rotation to the target heading.
        """
        from cortex.peripheral.odometry import StepOdometry, CARDINAL_HEADINGS

        target_deg = StepOdometry.heading_for_cardinal(heading)
        if target_deg is None:
            return f"Unknown direction: '{heading}'. Use: {', '.join(CARDINAL_HEADINGS.keys())}"

        if self._odometry:
            needed = self._odometry.turn_needed(target_deg)
        else:
            current_compass = (90.0 - math.degrees(self._get_yaw())) % 360.0
            needed = (target_deg - current_compass + 180) % 360 - 180

        if abs(needed) < TURN_TOLERANCE_DEG:
            return f"Already facing {heading} ({target_deg:.0f}°)."

        return self.turn_degrees(needed)

    def halt(self) -> str:
        """Immediately stop all motion."""
        self.velocity_cmd = VelocityCommand()
        return "All motion stopped."

    def move_to_point(
        self,
        target_x: float,
        target_y: float,
        speed: float = DEFAULT_SPEED,
        tolerance: float = 0.4,
        re_detect_fn: Optional[Callable] = None,
        re_detect_interval_m: float = 2.0,
    ) -> str:
        """Walk toward (target_x, target_y) with closed-loop course correction.

        Proportional yaw correction keeps the robot aimed at the target while
        walking.  If *re_detect_fn* is provided it is called every
        *re_detect_interval_m* metres; it should return an updated (x, y) tuple
        or None to keep the current target.
        """
        KP_YAW = 1.5
        MAX_YAW_CORRECTION = 0.5
        LARGE_ERROR_RAD = 0.8          # ~45 deg — stop-and-reorient threshold
        POLL_DT = 0.1

        pos = self._get_pos()
        yaw = self._get_yaw()
        dx = target_x - pos[0]
        dy = target_y - pos[1]
        initial_dist = math.sqrt(dx * dx + dy * dy)

        if initial_dist < tolerance:
            return f"Already at target ({target_x:.1f}, {target_y:.1f})."

        timeout = max(initial_dist / max(abs(speed), 0.05) + 15.0, 30.0)
        t0 = time.time()

        prev_pos = pos[:2].copy()
        dist_since_detect = 0.0

        # Initial turn toward target
        target_bearing = math.atan2(dy, dx)
        turn_needed = _angle_diff(target_bearing, yaw)
        if abs(turn_needed) > math.radians(TURN_TOLERANCE_DEG):
            self.turn_degrees(math.degrees(turn_needed))

        self.set_velocity(speed, 0.0, 0.0)

        while time.time() - t0 < timeout:
            time.sleep(POLL_DT)
            pos = self._get_pos()
            yaw = self._get_yaw()

            dx = target_x - pos[0]
            dy = target_y - pos[1]
            remaining = math.sqrt(dx * dx + dy * dy)

            if remaining < tolerance:
                self.halt()
                return (
                    f"Arrived at ({pos[0]:.1f}, {pos[1]:.1f}), "
                    f"{initial_dist:.1f}m walked."
                )

            # Distance accumulation for re-detection trigger
            step_d = float(np.linalg.norm(pos[:2] - prev_pos))
            dist_since_detect += step_d
            prev_pos = pos[:2].copy()

            # Periodic visual re-detection
            if re_detect_fn and dist_since_detect >= re_detect_interval_m:
                self.halt()
                time.sleep(0.3)
                try:
                    new_target = re_detect_fn()
                except Exception as e:
                    logger.warning("[MOVE_TO_POINT] re_detect error: %s", e)
                    new_target = None
                if new_target is not None:
                    target_x, target_y = new_target[0], new_target[1]
                    dx = target_x - pos[0]
                    dy = target_y - pos[1]
                    logger.info(
                        "[MOVE_TO_POINT] re-detect updated target to (%.1f, %.1f)",
                        target_x, target_y,
                    )
                dist_since_detect = 0.0
                self.set_velocity(speed, 0.0, 0.0)

            # Course correction
            target_bearing = math.atan2(dy, dx)
            bearing_error = _angle_diff(target_bearing, yaw)

            if abs(bearing_error) > LARGE_ERROR_RAD:
                self.halt()
                self.turn_degrees(math.degrees(bearing_error))
                self.set_velocity(speed, 0.0, 0.0)
            else:
                yaw_corr = float(np.clip(KP_YAW * bearing_error,
                                         -MAX_YAW_CORRECTION, MAX_YAW_CORRECTION))
                self.set_velocity(speed, 0.0, yaw_corr)

        self.halt()
        pos = self._get_pos()
        dist_walked = math.sqrt(
            (pos[0] - (target_x - dx)) ** 2 + (pos[1] - (target_y - dy)) ** 2
        )
        return (
            f"Timed out after {timeout:.0f}s. "
            f"Position: ({pos[0]:.1f}, {pos[1]:.1f}), "
            f"remaining ~{math.sqrt(dx*dx + dy*dy):.1f}m."
        )

    # ══════════════════════════════════════════════════════════════════════
    #  LOW-LEVEL VELOCITY API (used by tick loop and blocking methods)
    # ══════════════════════════════════════════════════════════════════════

    def set_velocity(self, vx: float = 0.0, vy: float = 0.0, yaw: float = 0.0) -> None:
        self.velocity_cmd = VelocityCommand(vx=vx, vy=vy, yaw_rate=yaw)

    def stop(self) -> None:
        self.velocity_cmd = VelocityCommand()

    def start_stabilising(self) -> None:
        self.mode = ControlMode.STABILISING
        self._stabilise_steps = 0
        self.velocity_cmd = VelocityCommand()
        if self._controller:
            self._controller.reset()
        logger.info("Stabilisation started (%d steps)", self._stabilise_target)

    def start_walking(self) -> None:
        self.mode = ControlMode.WBC_WALKING
        logger.info("Walking mode started")

    def go_idle(self) -> None:
        self.mode = ControlMode.IDLE
        self.velocity_cmd = VelocityCommand()

    def set_height(self, height: float) -> None:
        if hasattr(self._controller, "groot_height_cmd"):
            self._controller.groot_height_cmd = np.clip(height, 0.5, 1.0)

    # ── Queries ──────────────────────────────────────────────────────────

    @property
    def is_stabilised(self) -> bool:
        return self.mode == ControlMode.WBC_WALKING

    @property
    def is_active(self) -> bool:
        return self.mode != ControlMode.IDLE

    @property
    def controller_kp(self):
        return self._ctrl_kp if self._controller_loaded else None

    @property
    def controller_kd(self):
        return self._ctrl_kd if self._controller_loaded else None

    # ── Internal helpers ─────────────────────────────────────────────────

    def _get_pos(self) -> np.ndarray:
        if self._sim_state_fn:
            pos, _ = self._sim_state_fn()
            return np.asarray(pos)
        return np.zeros(3)

    def _get_yaw(self) -> float:
        if self._sim_state_fn:
            _, yaw = self._sim_state_fn()
            return float(yaw)
        return 0.0

    def _pos_yaw(self) -> Tuple[np.ndarray, float]:
        return self._get_pos(), self._get_yaw()

    def _heading_str(self) -> str:
        if self._odometry:
            return f"{self._odometry.heading_deg:.0f}°"
        return f"{math.degrees(self._get_yaw()):.0f}° (yaw)"


def _angle_diff(a: float, b: float) -> float:
    """Signed angle difference in radians, result in [-pi, pi]."""
    d = (a - b) % (2 * math.pi)
    if d > math.pi:
        d -= 2 * math.pi
    return d
