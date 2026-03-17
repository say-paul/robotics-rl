"""Modular reward components for humanoid locomotion.

Each function computes a single scalar reward term from raw simulation state.
The environment combines them with weights from ``RewardConfig``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def velocity_tracking(
    body_vel_local: NDArray, cmd_vx: float, scale: float = 4.0
) -> float:
    """Exponential reward for matching commanded forward speed.

    Higher *scale* makes the Gaussian narrower → stricter tracking.
    scale=4  → 70% reward at 0.3 m/s error (lenient)
    scale=16 → 24% reward at 0.3 m/s error (strict)
    """
    return float(np.exp(-scale * (body_vel_local[0] - cmd_vx) ** 2))


def lateral_velocity(
    body_vel_local: NDArray, cmd_vy: float, scale: float = 4.0
) -> float:
    """Exponential reward for matching commanded lateral speed."""
    return float(np.exp(-scale * (body_vel_local[1] - cmd_vy) ** 2))


def yaw_rate(
    angular_vel_z: float, cmd_vyaw: float, scale: float = 4.0
) -> float:
    """Exponential reward for matching commanded yaw rate."""
    return float(np.exp(-scale * (angular_vel_z - cmd_vyaw) ** 2))


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


def action_acceleration(
    action: NDArray, prev_action: NDArray, prev_prev_action: NDArray
) -> float:
    """Squared second derivative of actions (jerk penalty).

    Penalises oscillation and abrupt direction changes while allowing
    steady movements.  ``a_t - 2*a_{t-1} + a_{t-2}`` is the discrete
    second difference.
    """
    accel = action - 2.0 * prev_action + prev_prev_action
    return float(np.sum(accel ** 2))


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


# ---------------------------------------------------------------------------
# Human-posture rewards
# ---------------------------------------------------------------------------

def default_pose_tracking(
    joint_pos: NDArray, default_pos: NDArray, mask: NDArray,
    scale: float = 2.0,
) -> float:
    """Gaussian reward for staying near the default pose.

    *mask* is a per-joint weight (0 = ignore, 1 = enforce).  Use it to free
    the legs during walking while keeping arms/waist constrained.
    scale=2.0 gives wider reward basin so partial credit for being close.
    """
    diff = (joint_pos - default_pos) * mask
    return float(np.exp(-scale * np.sum(diff ** 2)))


def waist_stability(joint_pos: NDArray) -> float:
    """Summed squared deviation of waist joints (indices 12-14) from zero."""
    waist = joint_pos[12:15]
    return float(np.sum(waist ** 2))


def arm_pose_penalty(joint_pos: NDArray, default_pos: NDArray) -> float:
    """Summed squared deviation of arm joints (indices 15-28) from default."""
    arm_diff = joint_pos[15:] - default_pos[15:]
    return float(np.sum(arm_diff ** 2))


def knee_symmetry(joint_pos: NDArray) -> float:
    """Summed squared difference between left and right leg joints (0-5 vs 6-11)."""
    return float(np.sum((joint_pos[0:6] - joint_pos[6:12]) ** 2))


def angular_velocity_penalty(gyro: NDArray) -> float:
    """Summed squared body angular velocity -- penalises wobble."""
    return float(np.sum(gyro ** 2))


def linear_velocity_penalty(linvel_local: NDArray) -> float:
    """Summed squared XY body velocity -- penalises drift during standing."""
    return float(linvel_local[0] ** 2 + linvel_local[1] ** 2)


def action_magnitude(action: NDArray) -> float:
    """Summed squared action -- prefers minimal corrections."""
    return float(np.sum(action ** 2))


def torque_penalty(torques: NDArray) -> float:
    """Summed squared torques -- encourages energy-efficient actuation."""
    return float(np.sum(torques ** 2))


def foot_clearance(
    foot_z: float, is_swing: bool, target: float = 0.05,
) -> float:
    """Gaussian reward for lifting the swing foot to *target* metres."""
    if not is_swing:
        return 0.0
    return float(np.exp(-100.0 * max(0.0, target - foot_z) ** 2))


def feet_air_time(
    contact_duration: float, min_duration: float = 0.2,
) -> float:
    """Reward for stance contact lasting at least *min_duration* seconds."""
    return float(min(contact_duration / min_duration, 1.0))


# ---------------------------------------------------------------------------
# Trajectory-based rewards (v2)
# ---------------------------------------------------------------------------

def displacement_tracking(
    actual_disp: NDArray, expected_disp: NDArray, scale: float = 10.0
) -> float:
    """Gaussian reward for matching expected XY displacement over a window.

    *actual_disp* and *expected_disp* are 2D vectors (world-frame XY).
    Penalises both wrong direction and wrong magnitude simultaneously.
    """
    diff = actual_disp - expected_disp
    return float(np.exp(-scale * np.dot(diff, diff)))


def heading_tracking(
    actual_yaw_change: float, expected_yaw_change: float, scale: float = 10.0
) -> float:
    """Gaussian reward for matching expected heading change over a window."""
    err = actual_yaw_change - expected_yaw_change
    # Wrap to [-pi, pi]
    err = (err + np.pi) % (2 * np.pi) - np.pi
    return float(np.exp(-scale * err * err))


def pace_tracking(
    actual_dist: float,
    expected_dist: float,
    tolerance: float = 0.15,
    scale: float = 10.0,
) -> float:
    """Reward for covering the expected distance on time.

    Full reward when progress ratio is within *tolerance* of 1.0.
    Outside the band, drops via Gaussian on the excess error.
    When expected_dist is near zero (standing), returns 1.0.
    """
    if expected_dist < 0.01:
        return 1.0
    ratio = actual_dist / expected_dist
    excess = max(0.0, abs(ratio - 1.0) - tolerance)
    return float(np.exp(-scale * excess * excess))


def posture_composite(
    gravity_proj: NDArray,
    pelvis_z: float,
    target_z: float,
    upper_body_deviation: float,
    height_scale: float = 40.0,
    upper_scale: float = 4.0,
) -> float:
    """Combined posture quality: upright * height * upper-body stillness.

    Returns a product of three [0,1] factors so ALL must be satisfied.
    *upper_body_deviation* is sum-of-squares of waist + arm deviations.
    """
    upright_factor = max(0.0, float(-gravity_proj[2]))
    height_factor = float(np.exp(-height_scale * (pelvis_z - target_z) ** 2))
    upper_factor = float(np.exp(-upper_scale * upper_body_deviation))
    return upright_factor * height_factor * upper_factor
