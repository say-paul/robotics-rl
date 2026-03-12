"""
G1 29-DOF joint indices, PD gains, and standing pose.

Joint layout (matching MuJoCo actuator order and SDK G1JointIndex):
  0-5   left leg    (hip pitch/roll/yaw, knee, ankle pitch/roll)
  6-11  right leg   (hip pitch/roll/yaw, knee, ankle pitch/roll)
  12-14 waist       (yaw, roll, pitch)
  15-21 left arm    (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)
  22-28 right arm   (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)

Sign conventions (from MuJoCo model joint limits):
  hip_pitch   positive = forward flexion
  hip_roll    left: positive = abduction; right: NEGATIVE = abduction (mirrored)
  knee        positive = flexion (bending)
  ankle_pitch negative = plantarflexion, positive = dorsiflexion
  shoulder_pitch  positive direction moves arm backward/downward
"""

NUM_MOTORS = 29


class J:
    """Joint indices."""
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5

    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11

    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14

    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21

    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28


# PD gains per joint (from Unitree SDK low-level example)
KP_DEFAULT = [
    60, 60, 60, 100, 40, 40,       # left leg
    60, 60, 60, 100, 40, 40,       # right leg
    60, 40, 40,                     # waist
    40, 40, 40, 40, 40, 40, 40,    # left arm
    40, 40, 40, 40, 40, 40, 40,    # right arm
]

KD_DEFAULT = [
    1, 1, 1, 2, 1, 1,     # left leg
    1, 1, 1, 2, 1, 1,     # right leg
    1, 1, 1,               # waist
    1, 1, 1, 1, 1, 1, 1,  # left arm
    1, 1, 1, 1, 1, 1, 1,  # right arm
]

# Natural standing pose (all zeros = legs straight at 0.793m height)
STAND_POSE = [0.0] * NUM_MOTORS

# Default joint angles for RL policies (mujoco_playground "knees_bent" keyframe).
# Policies output deltas relative to this pose.
DEFAULT_POSITIONS = [
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
     0.0,   0.0, 0.073,                       # waist
     0.2,   0.2, 0.0, 0.6, 0.0, 0.0, 0.0,   # left arm
     0.2,  -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,   # right arm
]

# RL policy action scale (action output * ACTION_SCALE + DEFAULT_POSITIONS = target).
ACTION_SCALE = 0.5
