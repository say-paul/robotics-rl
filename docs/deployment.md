# Deployment Guide

## 1. Export to ONNX

```bash
python -m rl.export_onnx \
  --model rl/checkpoints/best_model.zip \
  --stage stand \
  --output policies/g1_stand.onnx
```

This bakes observation normalization into the ONNX graph -- no separate vecnorm file needed at runtime.

## 2. Evaluate Before Deploying

```bash
python -m rl.evaluate \
  --model rl/checkpoints/best_model.zip \
  --stage stand \
  --episodes 50
```

Key metrics to check:
- `fall_rate` < 0.05
- `mean_length` near episode ceiling (2000/3000/5000)

## 3. Run in Simulator

```bash
python main.py
```

Then open http://localhost:5000 in your browser:
1. Select your `.onnx` policy from the dropdown
2. Click "Start Policy"
3. Gradually lower the harness slider

## 4. Copy to Robot

```bash
scp policies/g1_stand.onnx unitree@<robot-ip>:/path/to/policies/
```

The ONNX model expects 103-dim observations and outputs 29-dim actions at 50 Hz.
