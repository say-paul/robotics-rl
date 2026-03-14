"""Centralized configuration for environment, reward, and PPO hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


NUM_MOTORS = 29

# Pre-built joint masks (0 = free, 1 = enforce default pose)
_MASK_ALL = np.ones(NUM_MOTORS, dtype=np.float64)
_MASK_UPPER = np.zeros(NUM_MOTORS, dtype=np.float64)
_MASK_UPPER[12:] = 1.0  # waist + arms


@dataclass
class EnvConfig:
    """MuJoCo environment parameters."""

    scene_path: str = "unitree_robots/g1/scene_29dof.xml"
    policy_dt: float = 0.02          # 50 Hz policy rate
    sim_dt: float = 0.005            # 200 Hz physics
    episode_length: int = 1000       # steps at policy_dt  →  20 s

    # Command sampling ranges  [min, max]
    cmd_vx_range: Tuple[float, float] = (0.0, 1.0)
    cmd_vy_range: Tuple[float, float] = (-0.3, 0.3)
    cmd_vyaw_range: Tuple[float, float] = (-0.5, 0.5)

    # Reset noise
    pos_noise: float = 0.05          # metres xy
    yaw_noise: float = 0.1           # radians
    joint_noise: float = 0.05        # radians around default
    vel_noise: float = 0.05          # rad/s

    # Termination thresholds
    min_height: float = 0.3          # pelvis z below this → fall
    max_tilt: float = 1.05           # ~60 deg from vertical (cos < 0.5)

    # Phase oscillator
    phase_freq: float = 1.25         # Hz

    # Domain randomization
    domain_rand: bool = False
    friction_range: Tuple[float, float] = (0.5, 2.0)
    mass_range: Tuple[float, float] = (0.8, 1.2)
    push_interval: Tuple[float, float] = (2.0, 5.0)
    push_magnitude: float = 30.0     # Newtons
    gyro_noise: float = 0.0          # std-dev added to gyro obs
    joint_pos_noise: float = 0.0     # std-dev added to joint pos obs


@dataclass
class RewardConfig:
    """Reward component weights.

    Weights enforce strict human-like posture: arms at sides, waist stable,
    symmetric stance, smooth minimal actions.
    """

    # Velocity tracking
    velocity_tracking: float = 1.5
    lateral_velocity: float = 0.8
    yaw_rate: float = 0.5

    # Core stability
    upright: float = 1.0
    height: float = 10.0
    alive: float = 0.15

    # Human-posture enforcement
    default_pose_tracking: float = 0.0
    waist_stability: float = 0.0
    arm_pose_penalty: float = 0.0
    knee_symmetry: float = 0.0

    # Anti-jitter / smoothness
    angular_vel_penalty: float = 0.0
    linear_vel_penalty: float = 0.0
    action_smoothness: float = -0.05
    action_magnitude: float = 0.0
    torque_penalty: float = 0.0

    # Gait quality
    foot_clearance: float = 0.0
    feet_air_time: float = 0.0
    feet_contact: float = 0.2

    # Efficiency
    energy: float = -0.005
    joint_limits: float = -5.0

    # Physical targets
    target_height: float = 0.68      # pelvis z for knees-bent stance

    # Joint mask for default_pose_tracking (set per stage)
    pose_mask: NDArray = field(default_factory=lambda: _MASK_ALL.copy())


@dataclass
class PPOConfig:
    """Stable-Baselines3 PPO hyperparameters."""

    learning_rate: float = 3e-4
    n_steps: int = 4096
    batch_size: int = 256
    n_epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    log_std_init: float = -1.0
    net_arch: List[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "silu"         # tanh | silu | elu
    n_envs: int = 16
    total_timesteps: int = 50_000_000

    # Callbacks
    eval_freq: int = 100_000         # steps between evaluations
    eval_episodes: int = 20
    checkpoint_freq: int = 500_000
    log_dir: str = "rl/logs"
    checkpoint_dir: str = "rl/checkpoints"


@dataclass
class TrainingConfig:
    """Top-level config aggregating all sub-configs."""

    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    seed: int = 42

    @classmethod
    def stage_stand(cls) -> TrainingConfig:
        """Curriculum stage 1: rock-solid human-like standing."""
        cfg = cls()
        cfg.env.episode_length = 2000           # 40 s
        cfg.env.cmd_vx_range = (0.0, 0.0)
        cfg.env.cmd_vy_range = (0.0, 0.0)
        cfg.env.cmd_vyaw_range = (0.0, 0.0)

        r = cfg.reward
        r.velocity_tracking = 0.0
        r.lateral_velocity = 0.0
        r.yaw_rate = 0.0

        # Positive incentives -- balanced so no single term dominates.
        # Max positive ≈ 20/step when perfectly still at default pose.
        r.upright = 3.0
        r.height = 8.0                         # was 15; reduced to prevent domination
        r.alive = 1.0
        r.default_pose_tracking = 8.0          # was 5; with wider Gaussian (scale=2)

        r.pose_mask = _MASK_ALL.copy()

        # Phase-1 penalties: moderate so the policy can first learn to stand.
        # Phase-2 (stage_stand_phase2) tightens these to eliminate jitter.
        r.action_magnitude = -0.3
        r.action_smoothness = -0.3
        r.angular_vel_penalty = -1.0
        r.linear_vel_penalty = -2.0
        r.torque_penalty = -5e-5
        r.waist_stability = -2.0
        r.arm_pose_penalty = -1.5
        r.knee_symmetry = -1.0
        r.energy = -0.01
        r.joint_limits = -5.0
        r.feet_contact = 0.0
        r.foot_clearance = 0.0
        r.feet_air_time = 0.0

        cfg.ppo.total_timesteps = 20_000_000
        return cfg

    @classmethod
    def stage_stand_phase2(cls) -> TrainingConfig:
        """Phase 2: tighten penalties on an already-standing policy.

        The policy from phase-1 can stand; now we force near-zero actions
        so it stands rock-solid without jitter.
        """
        cfg = cls.stage_stand()
        r = cfg.reward
        r.action_magnitude = -1.5
        r.action_smoothness = -0.8
        r.angular_vel_penalty = -3.0
        r.linear_vel_penalty = -5.0
        r.torque_penalty = -1e-4
        r.waist_stability = -3.0
        r.arm_pose_penalty = -2.0
        r.knee_symmetry = -1.5
        r.energy = -0.01
        cfg.ppo.log_std_init = -2.0
        cfg.ppo.total_timesteps = 10_000_000
        return cfg

    @classmethod
    def stage_slow_walk(cls) -> TrainingConfig:
        """Curriculum stage 2: smooth forward walking with human gait."""
        cfg = cls()
        cfg.env.episode_length = 3000           # 60 s
        cfg.env.cmd_vx_range = (0.0, 0.3)
        cfg.env.cmd_vy_range = (0.0, 0.0)
        cfg.env.cmd_vyaw_range = (0.0, 0.0)

        r = cfg.reward
        r.velocity_tracking = 2.0
        r.lateral_velocity = 0.0
        r.yaw_rate = 0.0
        r.upright = 2.0
        r.height = 10.0
        r.alive = 0.15
        r.default_pose_tracking = 0.5
        r.pose_mask = _MASK_UPPER.copy()
        r.waist_stability = -2.0
        r.arm_pose_penalty = -0.3
        r.knee_symmetry = 0.0
        r.angular_vel_penalty = -0.3
        r.linear_vel_penalty = 0.0
        r.action_smoothness = -0.15
        r.action_magnitude = -0.01
        r.torque_penalty = -1e-5
        r.energy = -0.005
        r.joint_limits = -5.0
        r.feet_contact = 0.3
        r.foot_clearance = 0.5
        r.feet_air_time = 0.3

        cfg.ppo.total_timesteps = 30_000_000
        return cfg

    @classmethod
    def stage_full_walk(cls) -> TrainingConfig:
        """Curriculum stage 3: full omnidirectional walking."""
        cfg = cls()
        cfg.env.episode_length = 5000           # 100 s
        cfg.env.domain_rand = True
        cfg.env.gyro_noise = 0.05
        cfg.env.joint_pos_noise = 0.02

        r = cfg.reward
        r.velocity_tracking = 2.0
        r.lateral_velocity = 1.0
        r.yaw_rate = 0.5
        r.upright = 1.5
        r.height = 10.0
        r.alive = 0.15
        r.default_pose_tracking = 0.3
        r.pose_mask = _MASK_UPPER.copy()
        r.waist_stability = -1.5
        r.arm_pose_penalty = -0.3
        r.knee_symmetry = 0.0
        r.angular_vel_penalty = -0.2
        r.linear_vel_penalty = 0.0
        r.action_smoothness = -0.1
        r.action_magnitude = -0.01
        r.torque_penalty = -1e-5
        r.energy = -0.005
        r.joint_limits = -5.0
        r.feet_contact = 0.3
        r.foot_clearance = 0.5
        r.feet_air_time = 0.3

        cfg.ppo.total_timesteps = 50_000_000
        return cfg

    STAGES = ("stand", "stand_phase2", "slow_walk", "full_walk")

    @classmethod
    def from_stage(cls, stage: str | None) -> TrainingConfig:
        """Look up a curriculum stage by name, defaulting to full_walk."""
        if stage is None:
            return cls()
        _factories = {
            "stand": cls.stage_stand,
            "stand_phase2": cls.stage_stand_phase2,
            "slow_walk": cls.stage_slow_walk,
            "full_walk": cls.stage_full_walk,
        }
        factory = _factories.get(stage)
        if factory is None:
            raise ValueError(
                f"Unknown stage {stage!r}. Choose from: {list(_factories)}"
            )
        return factory()
