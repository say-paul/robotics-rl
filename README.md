# G1 Humanoid Robot -- RL Locomotion & Navigation

PPO-trained walking for the Unitree G1, simulated in MuJoCo, deployed via ONNX.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  High-Level: Mission Planner (planner/)         │
│  YAML waypoints → PID path follower → vx/vy/yaw│
├─────────────────────────────────────────────────┤
│  Low-Level: RL Policy (rl/)                     │
│  103-dim obs → MLP [512,256,128] → 29 joints   │
├─────────────────────────────────────────────────┤
│  Physics: MuJoCo (unitree_robots/g1/)           │
│  200Hz sim, 50Hz policy, PD joint control       │
└─────────────────────────────────────────────────┘
```

## Dependencies

### Core (simulation + training)

```bash
pip install mujoco gymnasium stable-baselines3 torch numpy pyyaml
```

### Inference accelerators (optional)

| Backend | Install | Use case |
|---------|---------|----------|
| CPU (default) | `pip install onnxruntime` | Development, any machine |
| Intel GPU (Arc) | `pip install openvino onnxruntime-openvino` | Low-latency on Intel dGPU |
| Intel NPU | Same as GPU + Intel NPU driver/firmware | Ultra-low-power on-device |

### Remote viewing (VNC)

| Component | Purpose |
|-----------|---------|
| `Xvnc` | Virtual X display with built-in VNC |
| `noVNC` + `websockify` | Browser-accessible VNC client |
| `podman` or `docker` | Container for the VNC stack |

The VNC container image is built from `vnc/Dockerfile` (if present) or assembled manually. See [docs/deployment.md](docs/deployment.md) for setup.

### Real-robot deployment

| Package | Purpose |
|---------|---------|
| `unitree_sdk2py` | DDS low-level motor commands (500 Hz) |
| `onnxruntime` | Policy inference on robot's onboard compute |

## Quick Start

### Option A: Native display (GPU/NPU machine with monitor)

```bash
python3 vnc/sim_viewer.py --policy policies/g1_walk_v5.onnx --device GPU
```

`--device` accepts `CPU` (default), `GPU` (OpenVINO), or `NPU`.

### Option B: Remote via VNC (headless server)

```bash
# 1. Start VNC container
podman run -d --name g1-vnc --replace --network=host g1-vnc

# 2. Launch viewer on the virtual display
DISPLAY=:99 python3 vnc/sim_viewer.py --policy policies/g1_walk_v5.onnx --device GPU

# 3. Open in browser
#    http://<host-ip>:6080/vnc.html
```

### Viewer keyboard controls

| Key | Action |
|-----|--------|
| P | Toggle policy ON/OFF |
| M | Start/stop mission (requires `--mission`) |
| R | Reset robot to ground |
| W/S | Forward / backward |
| A/D | Turn left / right |
| Q/E | Strafe left / right |
| 0 | Zero all commands |
| 7/8 | Raise / lower harness |
| 9 | Release harness |
| H | Full harness support |

## Mission Planning

Missions chain multiple policies together through YAML waypoint files.
The planner and RL policy are fully decoupled -- the planner decides
*where* to go, the policy decides *how* to walk.

```
Mission YAML → MissionRunner (state machine)
                    ↓
              PathFollower (PID)
                    ↓
              vx, vy, vyaw commands
                    ↓
              RL Policy → joint torques
```

### Example: stand 3s then walk 50m

```yaml
mission:
  name: "stand_then_walk_50m"
  policies:
    stand: "policies/g1_stand_v5.onnx"
    walk:  "policies/g1_walk_v5.onnx"
  waypoints:
    - id: "stabilize"
      pose: {x: 0, y: 0, yaw: 0}
      behavior: "stand"
      hold_duration: 3.0

    - id: "destination"
      pose: {x: 50, y: 0, yaw: 0}
      behavior: "walk"
      speed: 0.4
      arrival_radius: 0.5
```

### Run a mission

```bash
# VNC (headless)
DISPLAY=:99 python3 vnc/sim_viewer.py \
    --mission missions/stand_walk_50m.yaml --device GPU

# Native display
python3 vnc/sim_viewer.py \
    --mission missions/stand_walk_50m.yaml --device GPU
```

1. Press **R** to reset the robot.
2. Press **M** to start the mission.
3. The robot stands, switches to walking, navigates to the waypoint, then returns to stand.
4. Press **M** to abort or **R** to reset at any time.

### Tear down

```bash
pkill -f sim_viewer            # stop the viewer
podman stop g1-vnc && podman rm g1-vnc   # stop VNC (if used)
```

### Waypoint fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | string | required | Unique waypoint name |
| `pose.x/y/yaw` | float | 0.0 | Position (m) and heading (rad) |
| `behavior` | string | `"walk"` | Key in the `policies` map |
| `speed` | float | 0.4 | Target speed (m/s) |
| `arrival_radius` | float | 0.3 | Distance to trigger arrival (m) |
| `hold_duration` | float | 0.0 | Seconds to hold before advancing |
| `on_arrive` | string | null | Action to execute on arrival |

See `missions/` for more examples.

## Detailed Guides

| Guide | Contents |
|-------|----------|
| [Training](docs/training.md) | Curriculum stages, hyperparameters, overnight runs |
| [Deployment](docs/deployment.md) | ONNX export, evaluation, VNC setup, real-robot copy |
| [Inference](docs/inference.md) | CPU / GPU / NPU inference code, observation format |

## Project Structure

```
g1_sim/
├── actions/            # Joint definitions (joints.py) + real-robot DDS actions
├── missions/           # Waypoint YAML mission files
├── planner/            # High-level navigation
│   ├── mission.py      # Mission/Waypoint dataclasses + YAML loader
│   ├── path_follower.py# PID waypoint → velocity controller
│   └── mission_runner.py# State machine for mission execution
├── policies/           # Exported ONNX models
├── rl/                 # Reinforcement learning
│   ├── configs/        # TrainingConfig, RewardConfig, PPOConfig
│   ├── envs/           # G1WalkEnv (Gymnasium), reward functions
│   ├── train.py        # PPO training entry point
│   └── export_onnx.py  # Model → ONNX export with baked normalization
├── unitree_robots/g1/  # MuJoCo 29-DOF robot model
├── vnc/                # VNC viewer + server scripts
│   └── sim_viewer.py   # Native MuJoCo viewer with keyboard + mission control
├── web/                # Flask browser UI (alternative to VNC viewer)
├── main.py             # Browser UI entry point (EGL rendering)
└── docs/               # Training, deployment, inference guides
```
