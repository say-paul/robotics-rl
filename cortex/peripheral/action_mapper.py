"""
Action Mapper
=============
Translates raw VLA model outputs into G1-compatible commands.

The pi0.5 base model outputs 32-dim action vectors per timestep.
These are mapped to either:
  - Velocity commands (vx, vy, yaw_rate) for the WBC locomotion controller
  - Direct joint targets for the 29 G1 joints (when fine-tuned for G1)

The mapping mode is selected based on the model's training embodiment.
For a base (non-G1-finetuned) model, we extract locomotion intent from
the first few action dimensions and map to velocity commands.
"""

import logging
from enum import Enum, auto
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

NUM_G1_JOINTS = 29


class MappingMode(Enum):
    VELOCITY = auto()       # Extract velocity commands from action vector
    JOINT_TARGETS = auto()  # Direct joint position targets


class ActionMapper:
    """Maps VLA action vectors to G1-compatible commands.

    For base pi0.5 (not fine-tuned for G1), the action space is the
    training embodiment's joint space. We interpret the action vector's
    direction and magnitude as locomotion velocity commands.

    For a G1-fine-tuned model, actions map directly to joint targets.
    """

    def __init__(
        self,
        mode: MappingMode = MappingMode.VELOCITY,
        action_dim: int = 32,
        velocity_scale: float = 0.5,
    ):
        self.mode = mode
        self.action_dim = action_dim
        self.velocity_scale = velocity_scale

        # Running statistics for adaptive scaling
        self._action_history: list = []
        self._max_abs = np.ones(action_dim) * 0.01

        logger.info("ActionMapper: mode=%s  action_dim=%d", mode.name, action_dim)

    def map_action(self, action: np.ndarray) -> Dict:
        """Map a single action vector to G1 commands.

        Returns:
            dict with either:
                {"vx": float, "vy": float, "yaw_rate": float}  (VELOCITY mode)
                {"joint_targets": dict[str, float]}             (JOINT_TARGETS mode)
        """
        self._update_stats(action)

        if self.mode == MappingMode.VELOCITY:
            return self._to_velocity(action)
        else:
            return self._to_joint_targets(action)

    def _to_velocity(self, action: np.ndarray) -> Dict:
        """Extract velocity commands from action vector.

        Strategy: Interpret the dominant action direction as locomotion.
        First 3 action dims → (forward, lateral, rotation) after normalization.
        """
        if len(action) < 3:
            return {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}

        # Normalize by running max
        norm_a = action[:3] / (self._max_abs[:3] + 1e-8)

        vx = float(np.clip(norm_a[0] * self.velocity_scale, -0.6, 0.6))
        vy = float(np.clip(norm_a[1] * self.velocity_scale, -0.3, 0.3))
        yaw_rate = float(np.clip(norm_a[2] * self.velocity_scale, -0.8, 0.8))

        return {"vx": vx, "vy": vy, "yaw_rate": yaw_rate}

    def _to_joint_targets(self, action: np.ndarray) -> Dict:
        """Map action vector directly to G1 joint targets.

        Assumes the model was fine-tuned with G1's 29-joint action space.
        """
        from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex

        targets = {}
        n = min(len(action), NUM_G1_JOINTS)
        for joint in G1_29_JointIndex:
            if joint.value < n:
                targets[f"{joint.name}.q"] = float(action[joint.value])

        return {"joint_targets": targets}

    def _update_stats(self, action: np.ndarray) -> None:
        """Track running max for adaptive normalization."""
        n = min(len(action), self.action_dim)
        abs_a = np.abs(action[:n])
        self._max_abs[:n] = np.maximum(self._max_abs[:n], abs_a)

    def map_chunk_to_velocities(
        self, chunk: np.ndarray, dt: float = 0.02,
    ) -> list:
        """Map an entire action chunk to a list of velocity commands.

        Args:
            chunk: (T, action_dim) array
            dt: time between consecutive actions

        Returns:
            List of (vx, vy, yaw_rate) tuples
        """
        velocities = []
        for t in range(chunk.shape[0]):
            cmd = self._to_velocity(chunk[t])
            velocities.append((cmd["vx"], cmd["vy"], cmd["yaw_rate"]))
        return velocities
