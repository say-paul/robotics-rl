# Training Guide

## Prerequisites

```bash
pip install stable-baselines3 gymnasium mujoco torch numpy
```

## Curriculum Stages

| Stage | Ep Length | Timesteps | Focus |
|-------|-----------|-----------|-------|
| `stand` | 2000 (40s) | 20M | Upright posture, zero movement |
| `slow_walk` | 3000 (60s) | 30M | Forward walk 0-0.3 m/s |
| `full_walk` | 5000 (100s) | 50M | Omnidirectional + domain rand |

## Quick Start

```bash
# Stage 1: Stand (run first)
python -m rl.train --stage stand

# Stage 2: Slow walk (fine-tune from stand)
python -m rl.train --stage slow_walk --resume rl/checkpoints/best_model.zip

# Stage 3: Full walk (fine-tune from slow_walk)
python -m rl.train --stage full_walk --resume rl/checkpoints/best_model.zip
```

## Common Overrides

```bash
# Custom episode length
python -m rl.train --stage stand --episode-length 3000

# Custom timesteps
python -m rl.train --stage stand --total-timesteps 10000000

# Fewer parallel envs (less RAM)
python -m rl.train --stage stand --n-envs 8

# Use GPU
python -m rl.train --stage stand --device xpu

# Use specific activation function
python -m rl.train --stage stand --activation tanh
```

## Fine-tune from Pretrained ONNX

```bash
python -m rl.train --pretrained policies/g1_joystick.onnx --stage stand
```

## Overnight Run (all stages)

```bash
nohup bash -c '
  python -m rl.train --stage stand && \
  python -m rl.train --stage slow_walk --resume rl/checkpoints/best_model.zip && \
  python -m rl.train --stage full_walk --resume rl/checkpoints/best_model.zip
' > /tmp/train_overnight.log 2>&1 &
echo "PID: $!"
```

Monitor: `tail -f /tmp/train_overnight.log`

## Outputs

- `rl/checkpoints/best_model.zip` -- best eval model (auto-saved)
- `rl/checkpoints/final_model.zip` -- end-of-training model
- `rl/checkpoints/*.vecnorm.pkl` -- observation normalization stats
- `rl/logs/` -- TensorBoard logs (`tensorboard --logdir rl/logs`)
