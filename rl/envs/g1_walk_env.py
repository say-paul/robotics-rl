"""Gymnasium environment for Unitree G1 bipedal walking.

Loads the MuJoCo model directly (no DDS) and exposes a 103-dim observation /
29-dim action interface that is wire-compatible with the existing
``OnnxPolicyAction`` deployment path.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from actions.joints import (
    ACTION_SCALE,
    DEFAULT_POSITIONS,
    KD_DEFAULT,
    KP_DEFAULT,
    NUM_MOTORS,
)

from ..configs.training_config import EnvConfig, RewardConfig
from . import rewards

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Body ids resolved once per model load (see _cache_ids)
_FLOOR_BODY_NAME = "world"
_PELVIS_BODY_NAME = "pelvis"
_LEFT_FOOT_BODY_NAME = "left_ankle_roll_link"
_RIGHT_FOOT_BODY_NAME = "right_ankle_roll_link"

OBS_DIM = 103
ACT_DIM = NUM_MOTORS  # 29


class G1WalkEnv(gym.Env):
    """MuJoCo-based walking environment for the Unitree G1.

    Observation (103):
        linvel_local(3), gyro(3), gravity(3), command(3),
        joint_pos - default(29), joint_vel(29), last_action(29), phase(4)

    Action (29):
        Normalised joint-position offsets in [-1, 1].
        Target = DEFAULT_POSITIONS + action * ACTION_SCALE
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        env_cfg: Optional[EnvConfig] = None,
        reward_cfg: Optional[RewardConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._env_cfg = env_cfg or EnvConfig()
        self._reward_cfg = reward_cfg or RewardConfig()
        self.render_mode = render_mode

        scene = _PROJECT_ROOT / self._env_cfg.scene_path
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.model.opt.timestep = self._env_cfg.sim_dt
        self.data = mujoco.MjData(self.model)

        self._substeps = round(self._env_cfg.policy_dt / self._env_cfg.sim_dt)

        # Spaces
        obs_hi = np.full(OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_hi, obs_hi, dtype=np.float32)
        self.action_space = spaces.Box(
            -np.ones(ACT_DIM, dtype=np.float32),
            np.ones(ACT_DIM, dtype=np.float32),
            dtype=np.float32,
        )

        # PD gains & torque limits
        self._kp = np.array(KP_DEFAULT, dtype=np.float64)
        self._kd = np.array(KD_DEFAULT, dtype=np.float64)
        self._default_pos = np.array(DEFAULT_POSITIONS, dtype=np.float64)
        self._torque_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self._torque_hi = self.model.actuator_ctrlrange[:, 1].copy()

        # Joint ranges for limit penalty (actuated joints start at index 1)
        self._jnt_range_lo = self.model.jnt_range[1:, 0].copy()
        self._jnt_range_hi = self.model.jnt_range[1:, 1].copy()

        # Cache body ids
        self._pelvis_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, _PELVIS_BODY_NAME
        )
        self._left_foot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, _LEFT_FOOT_BODY_NAME
        )
        self._right_foot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, _RIGHT_FOOT_BODY_NAME
        )
        self._floor_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, _FLOOR_BODY_NAME
        )

        # Initial qpos snapshot (taken after mj_resetData)
        mujoco.mj_resetData(self.model, self.data)
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()

        # Per-episode state
        self._step_count = 0
        self._last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._command = np.zeros(3, dtype=np.float32)
        self._phase = np.zeros(2, dtype=np.float64)
        self._phase_dt = 2.0 * math.pi * self._env_cfg.policy_dt * self._env_cfg.phase_freq

        # Optional renderer
        self._renderer: Optional[mujoco.Renderer] = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray, Dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random

        mujoco.mj_resetData(self.model, self.data)

        # Randomise base position
        self.data.qpos[:3] = self._init_qpos[:3]
        self.data.qpos[0] += rng.uniform(-self._env_cfg.pos_noise, self._env_cfg.pos_noise)
        self.data.qpos[1] += rng.uniform(-self._env_cfg.pos_noise, self._env_cfg.pos_noise)

        # Randomise yaw via quaternion (qpos[3:7] is w,x,y,z)
        yaw = rng.uniform(-self._env_cfg.yaw_noise, self._env_cfg.yaw_noise)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        self.data.qpos[3] = cy
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = sy

        # Randomise joint positions around default
        noise = rng.uniform(
            -self._env_cfg.joint_noise,
            self._env_cfg.joint_noise,
            size=NUM_MOTORS,
        )
        self.data.qpos[7:] = self._default_pos + noise

        # Small velocity noise
        self.data.qvel[6:] = rng.uniform(
            -self._env_cfg.vel_noise, self._env_cfg.vel_noise, size=NUM_MOTORS
        )

        mujoco.mj_forward(self.model, self.data)

        # Sample command
        self._command[0] = rng.uniform(*self._env_cfg.cmd_vx_range)
        self._command[1] = rng.uniform(*self._env_cfg.cmd_vy_range)
        self._command[2] = rng.uniform(*self._env_cfg.cmd_vyaw_range)

        self._last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._phase[:] = [0.0, math.pi]
        self._step_count = 0

        return self._get_obs(), self._get_info()

    def step(
        self, action: NDArray
    ) -> Tuple[NDArray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        q_target = self._default_pos + action * ACTION_SCALE

        for _ in range(self._substeps):
            self._apply_pd_control(q_target)
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._get_obs()
        reward, reward_info = self._compute_reward(action)
        terminated = self._is_fallen()
        truncated = self._step_count >= self._env_cfg.episode_length

        # Advance phase oscillator
        self._phase += self._phase_dt
        self._phase = np.fmod(self._phase + math.pi, 2 * math.pi) - math.pi

        self._last_action = action.astype(np.float32)

        info = self._get_info()
        info["reward_components"] = reward_info

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[NDArray]:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_pd_control(self, q_target: NDArray) -> None:
        q = self.data.qpos[7:]
        dq = self.data.qvel[6:]
        tau = self._kp * (q_target - q) - self._kd * dq
        self.data.ctrl[:] = np.clip(tau, self._torque_lo, self._torque_hi)

    def _get_obs(self) -> NDArray:
        """Build the 103-dim observation matching OnnxPolicyAction convention."""
        quat = self.data.qpos[3:7].copy()  # w, x, y, z
        w, x, y, z = quat

        # Body-frame gravity projection
        gravity = np.array([
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            -(w * w - x * x - y * y + z * z),
        ], dtype=np.float64)

        # Body-frame linear velocity: rotate world linvel by R^T
        rot_mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot_mat, quat)
        rot = rot_mat.reshape(3, 3)
        world_linvel = self.data.qvel[:3]
        linvel_local = rot.T @ world_linvel

        # Gyroscope (body angular velocity)
        gyro = self.data.qvel[3:6].copy()

        # Joint state
        joint_pos = self.data.qpos[7:].copy()
        joint_vel = self.data.qvel[6:].copy()

        # Phase
        cos_phase = np.cos(self._phase)
        sin_phase = np.sin(self._phase)

        obs = np.concatenate([
            linvel_local,                                    # 3
            gyro,                                            # 3
            gravity,                                         # 3
            self._command,                                   # 3
            joint_pos - self._default_pos,                   # 29
            joint_vel,                                       # 29
            self._last_action,                               # 29
            cos_phase, sin_phase,                            # 4
        ]).astype(np.float32)

        assert obs.shape == (OBS_DIM,), f"obs shape {obs.shape} != {OBS_DIM}"
        return obs

    def _compute_reward(
        self, action: NDArray
    ) -> Tuple[float, Dict[str, float]]:
        cfg = self._reward_cfg

        quat = self.data.qpos[3:7]
        w, x, y, z = quat
        gravity_proj = np.array([
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            -(w * w - x * x - y * y + z * z),
        ])

        rot_mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot_mat, quat)
        rot = rot_mat.reshape(3, 3)
        linvel_local = rot.T @ self.data.qvel[:3]
        ang_vel_z = self.data.qvel[5]

        pelvis_z = self.data.qpos[2]
        joint_pos = self.data.qpos[7:]
        joint_vel = self.data.qvel[6:]
        torques = self.data.ctrl[:NUM_MOTORS]

        left_contact, right_contact = self._foot_contacts()

        components: Dict[str, float] = {}

        components["velocity_tracking"] = cfg.velocity_tracking * rewards.velocity_tracking(
            linvel_local, self._command[0]
        )
        components["lateral_velocity"] = cfg.lateral_velocity * rewards.lateral_velocity(
            linvel_local, self._command[1]
        )
        components["yaw_rate"] = cfg.yaw_rate * rewards.yaw_rate(
            ang_vel_z, self._command[2]
        )
        components["upright"] = cfg.upright * rewards.upright(gravity_proj)
        components["height"] = cfg.height * rewards.height(pelvis_z, cfg.target_height)
        components["energy"] = cfg.energy * rewards.energy(torques, joint_vel)
        components["action_smoothness"] = cfg.action_smoothness * rewards.action_smoothness(
            action, self._last_action
        )
        components["joint_limits"] = cfg.joint_limits * rewards.joint_limits(
            joint_pos, self._jnt_range_lo, self._jnt_range_hi
        )
        components["alive"] = cfg.alive * rewards.alive()
        components["feet_contact"] = cfg.feet_contact * rewards.feet_contact(
            left_contact, right_contact, math.sin(self._phase[0])
        )

        total = sum(components.values())
        return float(total), components

    def _foot_contacts(self) -> Tuple[bool, bool]:
        """Check if left/right foot bodies are in contact with the floor."""
        left = False
        right = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            b1 = self.model.geom_bodyid[g1]
            b2 = self.model.geom_bodyid[g2]
            pair = {b1, b2}
            if self._floor_body_id not in pair:
                continue
            if self._left_foot_id in pair:
                left = True
            if self._right_foot_id in pair:
                right = True
            if left and right:
                break
        return left, right

    def _is_fallen(self) -> bool:
        pelvis_z = self.data.qpos[2]
        if pelvis_z < self._env_cfg.min_height:
            return True
        quat = self.data.qpos[3:7]
        w, x, y, z = quat
        cos_tilt = w * w - x * x - y * y + z * z  # dot(body_z, world_z)
        return float(cos_tilt) < math.cos(self._env_cfg.max_tilt)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step": self._step_count,
            "pelvis_z": float(self.data.qpos[2]),
            "command": self._command.copy(),
            "forward_vel": float(self.data.qvel[0]),
        }
