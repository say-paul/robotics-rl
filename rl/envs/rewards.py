"""Modular reward components for humanoid locomotion.

Each function computes a single scalar reward term from raw simulation state.
The environment combines them with weights from ``RewardConfig``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def velocity_tracking(
    body_vel_local: NDArray, cmd_vx: float
) -> float:
    """Exponential reward for matching commanded forward speed."""
    return float(np.exp(-4.0 * (body_vel_local[0] - cmd_vx) ** 2))


def lateral_velocity(
    body_vel_local: NDArray, cmd_vy: float
) -> float:
    """Exponential reward for matching commanded lateral speed."""
    return float(np.exp(-4.0 * (body_vel_local[1] - cmd_vy) ** 2))


def yaw_rate(
    angular_vel_z: float, cmd_vyaw: float
) -> float:
    """Exponential reward for matching commanded yaw rate."""
    return float(np.exp(-4.0 * (angular_vel_z - cmd_vyaw) ** 2))


def upright(gravity_projected: NDArray) -> float:
    """Reward for keeping the body vertical.

    ``gravity_projected`` is the gravity vector in the body frame; the z
    component equals ``-cos(tilt_angle)`` when the robot is upright.
    """
    return float(-gravity_projected[2])


def height(pelvis_z: float, target_z: float) -> float:
    """Gaussian reward centred on the target pelvis height."""
    return float(np.exp(-40.0 * (pelvis_z - target_z) ** 2))


def energy(
    torques: NDArray, joint_velocities: NDArray
) -> float:
    """Total instantaneous mechanical power (always >= 0)."""
    return float(np.sum(np.abs(torques * joint_velocities)))


def action_smoothness(
    action: NDArray, prev_action: NDArray
) -> float:
    """Squared difference between consecutive actions."""
    return float(np.sum((action - prev_action) ** 2))


def joint_limits(
    joint_pos: NDArray,
    joint_range_low: NDArray,
    joint_range_high: NDArray,
    margin: float = 0.1,
) -> float:
    """Penalty that grows as joints approach their limits.

    Returns the sum of squared violations inside *margin* radians of each
    limit.  Zero when all joints are well within range.
    """
    low_violation = np.clip(joint_range_low + margin - joint_pos, 0.0, None)
    high_violation = np.clip(joint_pos - (joint_range_high - margin), 0.0, None)
    return float(np.sum(low_violation ** 2 + high_violation ** 2))


def alive() -> float:
    """Constant survival bonus."""
    return 1.0


def feet_contact(
    left_contact: bool, right_contact: bool, phase_sin: float
) -> float:
    """Reward for alternating foot contacts synchronized with the gait phase.

    ``phase_sin`` is ``sin(phase)`` of the gait oscillator. Positive phase
    expects the left foot on the ground (right swinging) and vice versa.
    """
    if phase_sin > 0:
        return float(left_contact and not right_contact)
    elif phase_sin < 0:
        return float(right_contact and not left_contact)
    return 0.5 * float(left_contact or right_contact)
