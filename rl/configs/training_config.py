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

    # Command resampling: 0 = fixed per episode, >0 = resample every N steps
    cmd_resample_interval: int = 0

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
    action_acceleration: float = 0.0    # jerk penalty (second derivative of actions)
    action_magnitude: float = 0.0
    torque_penalty: float = 0.0

    # Gait quality
    foot_clearance: float = 0.0
    feet_air_time: float = 0.0
    feet_contact: float = 0.2

    # Efficiency
    energy: float = -0.005
    joint_limits: float = -5.0

    # Velocity tracking Gaussian tightness (higher = stricter)
    velocity_tracking_scale: float = 4.0

    # Trajectory-based rewards (v2) -- set displacement_tracking_weight > 0 to enable
    displacement_tracking_weight: float = 0.0
    heading_tracking_weight: float = 0.0
    pace_tracking_weight: float = 0.0
    posture_composite_weight: float = 0.0
    displacement_window: int = 50     # sliding window in steps (50 * 0.02 = 1 second)
    displacement_scale: float = 10.0  # Gaussian tightness for displacement/heading
    pace_tolerance: float = 0.15      # 15% tolerance band for pace reward

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
    gamma: float = 0.999
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
    def stage_walk(cls) -> TrainingConfig:
        """Omnidirectional walking at standard gait speed.

        Strict rewards for:
          - Precise velocity command tracking (forward, lateral, yaw)
          - Smooth human-like gait with proper foot contact timing
          - Stable upper body (arms at sides, waist locked)
          - No excessive actions or jitter

        Speed limited to ~0.5 m/s forward (standard walk, no running).
        Commands resample every 10s so the policy learns to respond to
        changing directions (keyboard joystick use-case).
        """
        cfg = cls()
        cfg.env.episode_length = 4000           # 80 s
        cfg.env.cmd_vx_range = (-0.3, 0.6)     # slow backward + natural forward walk
        cfg.env.cmd_vy_range = (-0.3, 0.3)     # moderate strafing
        cfg.env.cmd_vyaw_range = (-1.0, 1.0)   # full turning
        cfg.env.cmd_resample_interval = 500     # resample every 10s
        cfg.env.domain_rand = True
        cfg.env.friction_range = (0.6, 1.5)
        cfg.env.mass_range = (0.9, 1.1)
        cfg.env.push_interval = (4.0, 8.0)
        cfg.env.push_magnitude = 20.0
        cfg.env.gyro_noise = 0.02
        cfg.env.joint_pos_noise = 0.01

        r = cfg.reward
        # -- Command following: primary objective --
        r.velocity_tracking = 6.0               # match forward speed precisely
        r.lateral_velocity = 5.0                # match lateral speed precisely
        r.yaw_rate = 5.0                        # match turn rate precisely
        r.velocity_tracking_scale = 16.0        # TIGHT Gaussian: 24% at 0.3 m/s error

        # -- Core stability --
        r.upright = 4.0                         # stronger upright incentive
        r.height = 6.0
        r.alive = 1.0

        # -- Upper body: lock arms/waist, free legs --
        r.default_pose_tracking = 4.0           # doubled: enforce upper body pose
        r.pose_mask = _MASK_UPPER.copy()
        r.waist_stability = -6.0                # doubled: no waist twist
        r.arm_pose_penalty = -5.0               # doubled: arms at sides
        r.knee_symmetry = 0.0                   # legs need asymmetry for gait

        # -- Smooth walking, no jitter --
        r.angular_vel_penalty = -2.0            # 4x: penalise body wobble hard
        r.linear_vel_penalty = 0.0              # DON'T penalise walking velocity
        r.action_smoothness = -0.5              # stronger: smooth transitions
        r.action_magnitude = -0.05              # slightly larger to discourage wild swings
        r.torque_penalty = -1e-4                # doubled

        # -- Gait quality: proper alternating footsteps --
        r.foot_clearance = 1.5
        r.feet_contact = 1.5
        r.feet_air_time = 1.0

        # -- Efficiency --
        r.energy = -0.005
        r.joint_limits = -5.0

        cfg.ppo.learning_rate = 5e-5
        cfg.ppo.ent_coef = 0.003                # lower entropy to keep std tight
        cfg.ppo.total_timesteps = 30_000_000
        cfg.ppo.n_envs = 16
        cfg.ppo.log_std_init = -1.5
        cfg.ppo.gamma = 0.999
        return cfg

    @classmethod
    def stage_walk_v2(cls) -> TrainingConfig:
        """Trajectory-based walking with 7 simplified reward terms.

        Replaces instantaneous velocity matching with displacement tracking
        over a 1-second sliding window. Rewards WHERE the robot went,
        WHETHER it arrived on time, and HOW its posture looked.
        """
        cfg = cls()
        cfg.env.episode_length = 4000           # 80 s
        cfg.env.cmd_vx_range = (-0.3, 0.6)
        cfg.env.cmd_vy_range = (-0.3, 0.3)
        cfg.env.cmd_vyaw_range = (-1.0, 1.0)
        cfg.env.cmd_resample_interval = 500     # resample every 10s
        cfg.env.domain_rand = True
        cfg.env.friction_range = (0.6, 1.5)
        cfg.env.mass_range = (0.9, 1.1)
        cfg.env.push_interval = (4.0, 8.0)
        cfg.env.push_magnitude = 20.0
        cfg.env.gyro_noise = 0.02
        cfg.env.joint_pos_noise = 0.01

        r = cfg.reward

        # -- Trajectory rewards (the core 3) --
        r.displacement_tracking_weight = 8.0    # did the robot go where it should?
        r.heading_tracking_weight = 6.0         # is the robot facing correctly?
        r.pace_tracking_weight = 5.0            # did it arrive on time? (15% tolerance)
        r.displacement_window = 50              # 1-second sliding window
        r.displacement_scale = 10.0
        r.pace_tolerance = 0.15

        # -- Posture (single composite term) --
        r.posture_composite_weight = 5.0        # upright * height * upper-body

        # -- Disable legacy instantaneous velocity tracking --
        r.velocity_tracking = 0.0
        r.lateral_velocity = 0.0
        r.yaw_rate = 0.0
        r.upright = 0.0
        r.height = 0.0
        r.default_pose_tracking = 0.0
        r.waist_stability = 0.0
        r.arm_pose_penalty = 0.0
        r.knee_symmetry = 0.0
        r.angular_vel_penalty = 0.0
        r.linear_vel_penalty = 0.0
        r.action_magnitude = 0.0
        r.torque_penalty = 0.0

        # -- Alive bonus --
        r.alive = 2.0

        # -- Smoothness --
        r.action_smoothness = -0.5

        # -- Gait quality --
        r.feet_contact = 2.0
        r.foot_clearance = 0.0
        r.feet_air_time = 0.0

        # -- Efficiency --
        r.energy = -0.003
        r.joint_limits = -3.0

        # -- PPO hyperparameters --
        cfg.ppo.learning_rate = 3e-5
        cfg.ppo.ent_coef = 0.0                  # ZERO entropy: prevent std explosion
        cfg.ppo.total_timesteps = 30_000_000
        cfg.ppo.n_envs = 16
        cfg.ppo.log_std_init = -1.0             # std ceiling = 0.37
        cfg.ppo.gamma = 0.998
        cfg.ppo.eval_freq = 500_000             # less frequent evals (long episodes)
        cfg.ppo.eval_episodes = 5               # fewer eval episodes
        return cfg

    @classmethod
    def stage_walk_v3(cls) -> TrainingConfig:
        """Refined walking: fine-tune from walk_v2 best model.

        Fixes: high clip_fraction/KL by lowering LR and n_epochs.
        Adds foot_clearance, feet_air_time, action_acceleration (jerk)
        to reduce shuffling and upper-body jitter.
        """
        cfg = cls.stage_walk_v2()

        # -- Fix over-aggressive updates --
        cfg.ppo.learning_rate = 1e-5
        cfg.ppo.n_epochs = 3
        cfg.ppo.total_timesteps = 20_000_000

        # -- Gait quality: teach proper foot lifting --
        r = cfg.reward
        r.foot_clearance = 1.5
        r.feet_air_time = 1.0

        # -- Stronger smoothness + jerk penalty --
        r.action_smoothness = -2.0
        r.action_acceleration = -1.0

        # -- Stronger posture to match trajectory reward scale --
        r.posture_composite_weight = 8.0

        # -- Stronger efficiency --
        r.energy = -0.005

        # -- Gentler pushes during refinement, faster command resampling --
        cfg.env.push_magnitude = 15.0
        cfg.env.cmd_resample_interval = 200

        # -- n_envs = 8 to avoid OOM --
        cfg.ppo.n_envs = 8

        return cfg

    @classmethod
    def stage_slow_walk(cls) -> TrainingConfig:
        """Legacy: forward-only walking. Use stage_walk instead."""
        return cls.stage_walk()

    @classmethod
    def stage_full_walk(cls) -> TrainingConfig:
        """Legacy: omnidirectional walking. Use stage_walk instead."""
        return cls.stage_walk()

    STAGES = ("stand", "stand_phase2", "walk", "walk_v2", "walk_v3", "slow_walk", "full_walk")

    @classmethod
    def from_stage(cls, stage: str | None) -> TrainingConfig:
        """Look up a curriculum stage by name, defaulting to full_walk."""
        if stage is None:
            return cls()
        _factories = {
            "stand": cls.stage_stand,
            "stand_phase2": cls.stage_stand_phase2,
            "walk": cls.stage_walk,
            "walk_v2": cls.stage_walk_v2,
            "walk_v3": cls.stage_walk_v3,
            "slow_walk": cls.stage_slow_walk,
            "full_walk": cls.stage_full_walk,
        }
        factory = _factories.get(stage)
        if factory is None:
            raise ValueError(
                f"Unknown stage {stage!r}. Choose from: {list(_factories)}"
            )
        return factory()
