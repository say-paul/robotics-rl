"""Centralized configuration for environment, reward, and PPO hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


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


@dataclass
class RewardConfig:
    """Reward component weights.

    Weights calibrated from Unitree's official unitree_rl_gym G1 config.
    Height penalty is intentionally strong to prevent crouching/collapse.
    """

    velocity_tracking: float = 1.5
    lateral_velocity: float = 0.8
    yaw_rate: float = 0.5
    upright: float = 1.0
    height: float = 10.0
    energy: float = -0.005
    action_smoothness: float = -0.05
    joint_limits: float = -5.0
    alive: float = 0.15
    feet_contact: float = 0.2

    target_height: float = 0.68      # pelvis z for knees-bent stance


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
    log_std_init: float = -0.2
    net_arch: List[int] = field(default_factory=lambda: [256, 256, 256])
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
        """Curriculum stage 1: learn to stand and balance."""
        cfg = cls()
        cfg.env.cmd_vx_range = (0.0, 0.0)
        cfg.env.cmd_vy_range = (0.0, 0.0)
        cfg.env.cmd_vyaw_range = (0.0, 0.0)
        cfg.reward.velocity_tracking = 0.0
        cfg.reward.lateral_velocity = 0.0
        cfg.reward.yaw_rate = 0.0
        cfg.reward.upright = 3.0
        cfg.reward.height = 15.0
        cfg.reward.alive = 1.0
        cfg.reward.energy = -0.01
        cfg.reward.action_smoothness = -0.1
        cfg.reward.joint_limits = -5.0
        cfg.ppo.total_timesteps = 5_000_000
        return cfg

    @classmethod
    def stage_slow_walk(cls) -> TrainingConfig:
        """Curriculum stage 2: slow forward walking."""
        cfg = cls()
        cfg.env.cmd_vx_range = (0.0, 0.3)
        cfg.env.cmd_vy_range = (0.0, 0.0)
        cfg.env.cmd_vyaw_range = (0.0, 0.0)
        cfg.reward.height = 0.2
        cfg.ppo.total_timesteps = 20_000_000
        return cfg

    @classmethod
    def stage_full_walk(cls) -> TrainingConfig:
        """Curriculum stage 3: full omnidirectional walking."""
        return cls()  # default config is the full-walk config

    STAGES = ("stand", "slow_walk", "full_walk")

    @classmethod
    def from_stage(cls, stage: str | None) -> TrainingConfig:
        """Look up a curriculum stage by name, defaulting to full_walk."""
        if stage is None:
            return cls()
        _factories = {
            "stand": cls.stage_stand,
            "slow_walk": cls.stage_slow_walk,
            "full_walk": cls.stage_full_walk,
        }
        factory = _factories.get(stage)
        if factory is None:
            raise ValueError(
                f"Unknown stage {stage!r}. Choose from: {list(_factories)}"
            )
        return factory()
