"""Simulation engine abstraction.

Provides RobotState as the common data interface between any simulation
backend (MuJoCo, DDS/real-robot, Isaac Sim) and the policy runner, plus
SimulationEngine as the abstract loop each backend implements.
"""

import dataclasses
from abc import ABC, abstractmethod

import numpy as np


@dataclasses.dataclass
class RobotState:
    """Snapshot of robot state — the universal interface between engine and policy.

    Layout follows MuJoCo convention so that index math is consistent
    across all backends:

        qpos = [root_pos(3), root_quat(4), joint_angles(n)]
        qvel = [root_lin_vel(3), root_ang_vel(3), joint_velocities(n)]

    For MuJoCo the arrays may alias data.qpos / data.qvel directly
    (zero-copy).  For DDS they are standalone numpy arrays populated
    from the network.
    """

    qpos: np.ndarray
    qvel: np.ndarray
    time: float = 0.0


class SimulationEngine(ABC):
    """Abstract simulation / control loop.

    Concrete engines implement physics stepping (MuJoCo), DDS I/O
    (real robot / remote sim), or shadow mode, calling the policy
    runner at the configured rate.
    """

    @abstractmethod
    def run(self, runner, *, controller=None, **kwargs) -> None:
        """Run the main loop until stopped.

        Parameters
        ----------
        runner : PolicyRunner
            The policy brain that produces joint targets.
        controller : ControllerSource or None
            Optional human-input controller.
        """
