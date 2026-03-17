# Training Guide

## Prerequisites

```bash
pip install stable-baselines3 gymnasium mujoco torch numpy
```

## Curriculum Stages

| Stage | Ep Length | Timesteps | Focus | Base |
|-------|-----------|-----------|-------|------|
| `stand` | 2000 (40s) | 20M | Upright posture, zero movement | scratch or `g1_joystick.onnx` |
| `stand_phase2` | 2000 (40s) | 10M | Rock-solid stillness, strict penalties | `stand` |
| `walk` | 4000 (80s) | 20M | Instantaneous velocity tracking | `g1_joystick.onnx` |
| `walk_v2` | 4000 (80s) | 20M | Trajectory-based displacement tracking | `walk` |
| `walk_v3` | 4000 (80s) | 20M | Refined smoothness + jerk penalty | `walk_v2` |

## Quick Start

```bash
# Stage 1: Stand
python3 -m rl.train --stage stand

# Stage 2: Walk (fine-tune from stand or joystick ONNX)
python3 -m rl.train --stage walk --pretrained policies/g1_joystick.onnx

# Stage 3+: Fine-tune from previous checkpoint
python3 -m rl.train --stage walk_v2 --resume rl/checkpoints/best_model.zip
python3 -m rl.train --stage walk_v3 --resume rl/checkpoints/best_model.zip
```

## Common Overrides

```bash
python3 -m rl.train --stage stand --episode-length 3000
python3 -m rl.train --stage stand --total-timesteps 10000000
python3 -m rl.train --stage stand --n-envs 8       # less RAM
python3 -m rl.train --stage stand --device xpu      # GPU
python3 -m rl.train --stage stand --activation tanh
```

## Overnight Run

```bash
nohup python3 -m rl.train --stage walk_v3 \
    --resume rl/checkpoints/best_model.zip \
    --total-timesteps 20000000 \
    > /tmp/train_overnight.log 2>&1 &
echo "PID: $!"
```

Monitor: `tail -f /tmp/train_overnight.log`

## Monitoring with TensorBoard

```bash
tensorboard --logdir=rl/logs --bind_all --port 6006
```

Key metrics:
- `train/std` -- exploration noise (clamped by StdClampCallback, should stay stable)
- `train/clip_fraction` -- target < 0.2
- `train/approx_kl` -- target < 0.03
- `eval/mean_ep_length` -- should approach episode ceiling

Note: `rollout/ep_rew_mean` and `eval/` metrics appear only after the first complete episodes, which may take several iterations for long episodes.

## Outputs

- `rl/checkpoints/best_model.zip` -- best eval model (auto-saved)
- `rl/checkpoints/final_model.zip` -- end-of-training model
- `rl/checkpoints/*.vecnorm.pkl` -- observation normalization stats
- `rl/logs/` -- TensorBoard logs
