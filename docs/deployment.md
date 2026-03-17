# Deployment Guide

## 1. Export to ONNX

```bash
python3 -m rl.export_onnx \
  --model rl/checkpoints/best_model.zip \
  --stage walk_v3 \
  --output policies/g1_walk_v5.onnx
```

This bakes observation normalization into the ONNX graph -- no separate vecnorm file needed at runtime.

## 2. Evaluate Before Deploying

```bash
python3 -m rl.evaluate \
  --model rl/checkpoints/best_model.zip \
  --stage walk_v3 \
  --episodes 50
```

Key metrics: `fall_rate` < 0.05, `mean_length` near episode ceiling.

## 3. Run in Simulator

### Option A: Native display

```bash
python3 vnc/sim_viewer.py --policy policies/g1_walk_v5.onnx --device GPU
```

### Option B: VNC (headless server)

#### Set up the VNC container (one-time)

```bash
podman run -d --name g1-vnc --replace --network=host g1-vnc
```

Open `http://<host-ip>:6080/vnc.html` in a browser.

#### Launch

```bash
DISPLAY=:99 python3 vnc/sim_viewer.py --policy policies/g1_walk_v5.onnx --device GPU
```

### Option C: Browser UI (Flask, EGL rendering)

```bash
MUJOCO_GL=egl python3 main.py --port 8888
```

Then open `http://localhost:8888`. Select a `.onnx` policy from the dropdown and click "Start Policy".

## 4. Run a Mission

```bash
python3 vnc/sim_viewer.py --mission missions/stand_walk_50m.yaml --device GPU
```

Press **R** to reset, **M** to start. See the [README](../README.md) for full details.

## 5. Copy to Robot

```bash
scp policies/g1_walk_v5.onnx unitree@<robot-ip>:/path/to/policies/
```

The ONNX model expects 103-dim observations and outputs 29-dim actions at 50 Hz.

## 6. Tear Down

```bash
pkill -f sim_viewer                       # stop viewer
podman stop g1-vnc && podman rm g1-vnc    # stop VNC container
```
