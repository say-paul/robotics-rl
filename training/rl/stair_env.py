"""
Gymnasium environment for G1 stair-climbing RL training.

Wraps the MuJoCo simulation with:
  - 82-dim observation space (joint pos/vel, IMU, foot z, base state)
  - 29-dim action space (target joint angle offsets)
  - Multi-component reward: forward progress, height, stability, foot contact, energy
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

from config import SCENE_DIR, NUM_JOINTS, SIM_DT

logger = logging.getLogger(__name__)

_SIM_SUBSTEPS = 2
_MAX_EPISODE_STEPS = 1000
_ACTION_SCALE = 0.3


class StairClimbEnv(gym.Env):
    """MuJoCo G1 stair-climbing environment for RL training."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = _MAX_EPISODE_STEPS,
    ):
        super().__init__()
        self.render_mode = render_mode
        self._max_steps = max_episode_steps

        xml_path = SCENE_DIR / "stair_training.xml"
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = SIM_DT

        self._body_jnt_ids = []
        for i in range(self.model.njnt):
            name = self.model.joint(i).name
            if any(part in name for part in [
                "hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"
            ]):
                self._body_jnt_ids.append(i)
        self._body_jnt_ids = np.array(self._body_jnt_ids)
        assert len(self._body_jnt_ids) == NUM_JOINTS

        self.torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )
        self._left_foot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"
        )
        self._right_foot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"
        )

        self._kp = np.array([
            150, 150, 150, 300, 40, 40,
            150, 150, 150, 300, 40, 40,
            250, 250, 250,
            50, 50, 80, 80, 40, 40, 40,
            50, 50, 80, 80, 40, 40, 40,
        ], dtype=np.float64)
        self._kd = np.array([
            2, 2, 2, 4, 2, 2,
            2, 2, 2, 4, 2, 2,
            5, 5, 5,
            3, 3, 3, 3, 1.5, 1.5, 1.5,
            3, 3, 3, 3, 1.5, 1.5, 1.5,
        ], dtype=np.float64)
        self._torque_limit = np.array([
            88, 88, 88, 139, 50, 50,
            88, 88, 88, 139, 50, 50,
            88, 50, 50,
            25, 25, 25, 25, 25, 5, 5,
            25, 25, 25, 25, 25, 5, 5,
        ], dtype=np.float64)

        # obs = joint_q(29) + joint_dq(29) + base_pos(3) + base_quat(4) +
        #        base_linvel(3) + base_angvel(3) + foot_z(2) + imu_rpy(3) +
        #        target_dir(3) + height_above_ground(1) + time_frac(1) + prev_action(1) = 82
        obs_dim = 29 + 29 + 3 + 4 + 3 + 3 + 2 + 3 + 3 + 1 + 1 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(NUM_JOINTS,), dtype=np.float32
        )

        self._step_count = 0
        self._prev_x = 0.0
        self._prev_action_norm = 0.0
        self._renderer = None
        self._idle_steps = 20

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[0] = -2.0
        self.data.qpos[2] = 0.75
        mujoco.mj_forward(self.model, self.data)

        for _ in range(self._idle_steps):
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        self._prev_x = float(self.data.qpos[0])
        self._prev_action_norm = 0.0
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0)

        cur_q = np.zeros(NUM_JOINTS)
        cur_dq = np.zeros(NUM_JOINTS)
        for i, jid in enumerate(self._body_jnt_ids):
            jnt = self.model.joint(jid)
            cur_q[i] = self.data.qpos[jnt.qposadr[0]]
            cur_dq[i] = self.data.qvel[jnt.dofadr[0]]

        targets = cur_q + action * _ACTION_SCALE
        torques = self._kp * (targets - cur_q) + self._kd * (0 - cur_dq)
        torques = np.clip(torques, -self._torque_limit, self._torque_limit)

        self.data.ctrl[:NUM_JOINTS] = torques
        for _ in range(_SIM_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._get_obs()
        reward, info = self._compute_reward(action)
        terminated = self._check_terminated()
        truncated = self._step_count >= self._max_steps

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        joint_q = np.zeros(NUM_JOINTS, dtype=np.float32)
        joint_dq = np.zeros(NUM_JOINTS, dtype=np.float32)
        for i, jid in enumerate(self._body_jnt_ids):
            jnt = self.model.joint(jid)
            joint_q[i] = self.data.qpos[jnt.qposadr[0]]
            joint_dq[i] = self.data.qvel[jnt.dofadr[0]]

        base_pos = self.data.qpos[:3].astype(np.float32)
        base_quat = self.data.qpos[3:7].astype(np.float32)

        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_BODY, self.torso_id, vel6, 0
        )
        base_linvel = vel6[3:6].astype(np.float32)
        base_angvel = vel6[0:3].astype(np.float32)

        lf_z = np.float32(self.data.xpos[self._left_foot_id][2])
        rf_z = np.float32(self.data.xpos[self._right_foot_id][2])

        qw = base_quat
        rot = Rotation.from_quat([qw[1], qw[2], qw[3], qw[0]])
        rpy = rot.as_euler("xyz").astype(np.float32)

        goal_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        height = np.float32(base_pos[2])
        time_frac = np.float32(self._step_count / self._max_steps)
        prev_act = np.float32(self._prev_action_norm)

        return np.concatenate([
            joint_q, joint_dq,
            base_pos, base_quat,
            base_linvel, base_angvel,
            [lf_z, rf_z],
            rpy,
            goal_dir,
            [height, time_frac, prev_act],
        ])

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict]:
        x = float(self.data.qpos[0])
        z = float(self.data.qpos[2])

        r_forward = (x - self._prev_x) * 5.0
        self._prev_x = x

        r_height = max(0.0, z - 0.5) * 2.0

        qw = self.data.qpos[3:7]
        rot = Rotation.from_quat([qw[1], qw[2], qw[3], qw[0]])
        rpy = rot.as_euler("xyz")
        roll, pitch = rpy[0], rpy[1]
        r_stable = -2.0 * (abs(roll) + abs(pitch))

        lf_z = self.data.xpos[self._left_foot_id][2]
        rf_z = self.data.xpos[self._right_foot_id][2]
        on_ground = (lf_z < 0.05) + (rf_z < 0.05)
        r_contact = 0.5 * on_ground

        act_norm = float(np.linalg.norm(action))
        r_energy = max(-2.0, -0.01 * act_norm)
        self._prev_action_norm = act_norm

        r_alive = 0.1

        total = r_forward + r_height + r_stable + r_contact + r_energy + r_alive
        info = {
            "r_forward": r_forward, "r_height": r_height,
            "r_stable": r_stable, "r_contact": r_contact,
            "r_energy": r_energy, "r_alive": r_alive,
        }
        return total, info

    def _check_terminated(self) -> bool:
        z = float(self.data.qpos[2])
        if z < 0.3:
            return True
        qw = self.data.qpos[3:7]
        rot = Rotation.from_quat([qw[1], qw[2], qw[3], qw[0]])
        rpy = rot.as_euler("xyz")
        if abs(rpy[0]) > 1.2 or abs(rpy[1]) > 1.2:
            return True
        return False

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="global_view")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
