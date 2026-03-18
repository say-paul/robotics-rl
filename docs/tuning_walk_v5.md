# walk_v5 Stage: Parameter Guide and Tuning

This doc explains every parameter in **stage_walk_v5** (and inherited from walk_v3 / walk_v2), when to tune them, and how to use a **reference pose** (e.g. stand pose) for comparison or mimic fine-tuning.

---

## 1. Environment (EnvConfig)

| Parameter | Meaning | walk_v5 value | Tuning |
|-----------|--------|----------------|--------|
| **episode_length** | Max steps per episode (policy_dt = 0.02 s). | 4000 (80 s) | Increase for longer trials; decrease if episodes rarely complete. |
| **cmd_vx_range** | Forward velocity command [min, max] m/s. | (-0.3, 0.6) | Widen for more speed range; narrow for safer fine-tuning. |
| **cmd_vy_range** | Lateral velocity command m/s. | (-0.3, 0.3) | Same idea as vx. |
| **cmd_vyaw_range** | Yaw rate command rad/s. | (-1.0, 1.0) | Reduce if turning causes instability. |
| **cmd_resample_interval** | Resample velocity command every N steps (0 = once per episode). | 200 (4 s) | Smaller = more direction changes; larger = longer straight segments. |
| **tune_file** | Path to YAML for live tuning (e.g. `tune.yaml`). | "tune.yaml" | Set to "" to disable live tuning. |
| **pos_noise, yaw_noise, joint_noise, vel_noise** | Reset randomization. | from walk_v3 | Slightly increase for robustness; decrease for more deterministic evals. |
| **min_height, max_tilt** | Fall termination: pelvis z and body tilt. | 0.3 m, ~60° | Rarely need to change. |
| **domain_rand** | Enable friction/mass/push randomization. | True | Turn False for more stable, less diverse training. |
| **friction_range, mass_range, push_interval, push_magnitude** | Domain randomization ranges. | from walk_v3 | Reduce push_magnitude if the policy regresses under disturbance. |
| **phase_freq** | Gait phase oscillator Hz. | 1.25 | Match desired step frequency. |

---

## 2. Reward Weights (RewardConfig)

Reward per step = sum of (weight × term). Positive weights = reward; negative = penalty.

### 2.1 Trajectory (command following in x, y, heading)

| Parameter | Meaning | walk_v5 | Tuning |
|-----------|--------|---------|--------|
| **displacement_tracking_weight** | Reward for moving in the commanded xy direction over a 1 s window. | 9.0 | **Primary task.** Increase if the policy ignores commands; keep above smoothness penalties. |
| **heading_tracking_weight** | Reward for matching commanded yaw change over the window. | 6.0 | Same: increase if turning is poor. |
| **pace_tracking_weight** | Reward for covering the expected distance in time (pace). | 5.0 | Increase if the robot is too slow/fast relative to command. |
| **displacement_scale** | Gaussian tightness for displacement/heading (higher = stricter). | 10.0 | Lower (e.g. 6) for more tolerance. |
| **pace_tolerance** | Acceptable pace error band (e.g. 0.2 = 20%). | 0.2 | Increase if pace reward is too harsh during adaptation. |
| **displacement_window** | Sliding window length in steps (50 × 0.02 s = 1 s). | 50 | Larger = smoother but slower to react. |

### 2.2 Posture

| Parameter | Meaning | walk_v5 | Tuning |
|-----------|--------|---------|--------|
| **posture_composite_weight** | Single term: upright × height × upper-body stillness. | 8.0 | Increase for more upright/stable posture; decrease if it overpowers walking. |
| **target_height** | Target pelvis z (m) for height term inside posture. | 0.68 | Match desired standing/walking height. |

### 2.3 Smoothness (penalties)

| Parameter | Meaning | walk_v5 | Tuning |
|-----------|--------|---------|--------|
| **action_smoothness** | Penalty on squared difference between consecutive actions. | -1.2 | **If training deteriorates:** make less negative (e.g. -0.8). If too jerky: more negative (e.g. -1.5). Never let smoothness dominate task reward. |
| **action_acceleration** | Penalty on action “jerk” (second difference). | -0.8 | Same: soften (e.g. -0.4) if policy collapses to minimal motion. |

### 2.4 Gait quality

| Parameter | Meaning | walk_v5 | Tuning |
|-----------|--------|---------|--------|
| **foot_clearance** | Reward for lifting swing foot to target height. | 2.0 | Increase if shuffling; decrease if tripping. |
| **feet_air_time** | Reward for stance lasting at least ~0.2 s. | 1.0 | Adjust for desired stance duration. |
| **feet_contact** | Reward for alternating contacts in phase with gait. | 2.0 | Increase for clearer alternation. |

### 2.5 Efficiency and survival

| Parameter | Meaning | walk_v5 | Tuning |
|-----------|--------|---------|--------|
| **energy** | Penalty on |torque × velocity|. | -0.004 | Slightly more negative for efficiency; less for stability. |
| **joint_limits** | Penalty near joint limits. | -5.0 | Keep to avoid limit hits. |
| **alive** | Constant bonus per step. | 2.0 | Small; helps survival. |

### 2.6 Reference pose (stand / custom pose)

| Parameter | Meaning | Tuning |
|-----------|--------|--------|
| **reference_pose** | Optional length-29 array of joint angles (rad). If set, **default_pose_tracking** and upper-body part of **posture_composite** reward matching this pose instead of the control default. | Use `STAND_POSE` (all zeros) to reward “straight stand”, or a custom pose to mimic a specific posture. See “Comparing / mimicking a stand pose” below. |
| **default_pose_tracking** | Weight for reward toward reference/default pose (with **pose_mask**). | walk_v5 inherits 0 from walk_v3; enable (e.g. 2.0–5.0) when using **reference_pose** for pose mimic. |
| **pose_mask** | Per-joint weight (0 = ignore, 1 = enforce). | e.g. upper body only (waist + arms) so legs are free for walking. |

---

## 3. PPO (PPOConfig)

| Parameter | Meaning | walk_v5 | Tuning |
|-----------|--------|---------|--------|
| **learning_rate** | Optimizer step size. | 3e-6 | **If training deteriorates:** try 2e-6 or 1e-6. If too slow: 5e-6. |
| **clip_range** | PPO clip for policy updates. | 0.1 | Smaller (0.08) for very conservative fine-tuning; 0.12–0.15 if learning is too slow. |
| **n_epochs** | PPO epochs per rollout. | 3 | Keep low when fine-tuning to avoid forgetting. |
| **n_steps** | Steps per env before update. | 4096 | Increase for more stable gradients; decrease for faster updates. |
| **batch_size** | Minibatch size. | 256 | Rarely change. |
| **gamma** | Discount factor. | 0.998 | Slightly lower (0.995) for shorter horizon. |
| **log_std_init** | Initial log std of policy (exploration). | -2.0 | Lower = less exploration (good for fine-tuning). |
| **ent_coef** | Entropy bonus. | 0.0 from v3 | Keep 0 or very small to avoid std explosion. |
| **n_envs** | Parallel envs. | 16 | Reduce if OOM; increase for throughput. |

---

## 4. When training gets worse over time (e.g. PPO_11 with pretrained)

- **Lower learning_rate** (e.g. 2e-6 or 1e-6).
- **Reduce smoothness penalty magnitude** so task reward dominates: e.g. `action_smoothness: -0.8`, `action_acceleration: -0.4`.
- **Slightly increase trajectory weights** so command following is clearly the main objective.
- **Tighten clip_range** to 0.08 for smaller policy updates.
- **Resume from last good checkpoint** (e.g. `best_model.zip`) instead of training from scratch or from an ONNX that was already good.

---

## 5. Comparing against / mimicking a stand pose

- **Observation:** The policy always gets **joint_pos - default_pos** (control default = knees-bent pose). So the policy outputs **deltas** from that default.
- **Reward:** You can define a **reference pose** (e.g. stand pose or any keyframe) and reward the robot for matching it. That gives you:
  - A **comparison** in reward space (how close to the reference pose).
  - **Fine-tuning to mimic** that pose by turning on pose-tracking reward toward that reference.

Options:

1. **Use built-in stand pose**  
   In `actions/joints.py`, `STAND_POSE` is all zeros (straight legs, 0.793 m height). You can set `reference_pose = STAND_POSE` in config and enable `default_pose_tracking` (and optionally use `pose_mask` to enforce only upper body or only legs) so the policy is rewarded for matching the stand pose where you choose.

2. **Custom reference pose**  
   Define a length-29 array of joint angles (rad) in the same order as the G1 model (see `actions/joints.py` for indices). Set that as `reference_pose` in the reward config. With `default_pose_tracking > 0` and a suitable `pose_mask`, the policy will be rewarded for mimicking that pose (e.g. “arms up”, “one leg forward”, etc.).

3. **Evaluation only**  
   You can log or plot `joint_pos` vs your reference pose during eval without changing rewards; the reference_pose in config is for **rewarding** the policy during training so it learns to mimic that pose.

### Using the built-in stand-mimic stage

- **Stage `stand_mimic`** trains the robot to hold the straight stand pose (`STAND_POSE` = all zeros). Zero velocity commands; reward = matching that pose.
  ```bash
  python3 -m rl.train --stage stand_mimic --device cpu
  ```
- To **add a custom reference pose to walk_v5** (e.g. mimic a pose while walking), set it in code after loading the stage:
  ```python
  from rl.configs.training_config import TrainingConfig
  from actions.joints import J, NUM_MOTORS
  import numpy as np

  cfg = TrainingConfig.stage_walk_v5()
  # Example: custom pose (rad); order as in actions/joints.py
  my_pose = np.zeros(NUM_MOTORS)
  my_pose[J.WaistPitch] = 0.1   # slight forward lean
  # my_pose[...] = ...
  cfg.reward.reference_pose = my_pose.astype(np.float64)
  cfg.reward.default_pose_tracking = 2.0   # enable mimic reward
  cfg.reward.pose_mask = ...   # e.g. upper body only: _MASK_UPPER
  ```
  Then run training with this config (e.g. by passing it into `train(cfg, ...)` in a small script or by adding a new stage in `training_config.py` that sets `reference_pose` and `default_pose_tracking`).
