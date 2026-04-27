"""DDS IDL types for robot state and joint commands.

Uses cyclonedds-python's @idl dataclass approach.  Types are intentionally
simple (flat arrays + timestamp) so that the same definitions work for both
Isaac Sim bridges and real Unitree robots (with unitree_compat remapping
handled in the engine, not the message types).

Requires: pip install cyclonedds
"""

from dataclasses import dataclass

try:
    from cyclonedds.idl import IdlStruct
    from cyclonedds.idl.types import sequence
except ImportError:
    raise ImportError(
        "cyclonedds is required for DDS transport. "
        "Install with:  pip install cyclonedds"
    )


@dataclass
class RobotStateDDS(IdlStruct, typename="rdp.RobotState"):
    """Published by the robot / simulator, subscribed by the policy runner.

    Layout mirrors RobotState (MuJoCo convention):
        qpos = [root_pos(3), root_quat(4), joint_angles(n)]
        qvel = [root_lin_vel(3), root_ang_vel(3), joint_velocities(n)]
    """
    qpos: sequence[float]
    qvel: sequence[float]
    timestamp_ns: int


@dataclass
class JointCommandDDS(IdlStruct, typename="rdp.JointCommand"):
    """Published by the policy runner, subscribed by the robot / simulator.

    Contains position targets and PD gains.  The receiver (Isaac Sim's
    ArticulationController or Unitree's motor controller) applies PD
    control locally.
    """
    target_positions: sequence[float]
    kps: sequence[float]
    kds: sequence[float]
    timestamp_ns: int
