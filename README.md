# Robotics-RL: Unitree G1 Humanoid Simulation & Control

Whole-body control framework for the **Unitree G1 29-DoF humanoid robot** in MuJoCo simulation, with VLM-driven natural language command interface.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User (Browser)                           │
│                   WebSocket chat + camera feed                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    FastAPI Server (server/app.py)                │
│              WebSocket chatbot  ·  Camera stream                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                 Orchestrator (orchestrator.py)                   │
│          Main simulation loop  ·  Component wiring              │
│                                                                 │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ Prefrontal       │  │ Peripheral       │  │ Motor Cortex  │  │
│  │ Cortex (brain)   │  │ Cortex           │  │ (controller)  │  │
│  │                  │  │                  │  │               │  │
│  │ Command queue    │  │ VLM Engine       │  │ Groot WBC     │  │
│  │ Intent routing   │  │ Planner (VLM     │  │ Velocity cmd  │  │
│  │                  │  │   tool calling)  │  │ PD control    │  │
│  │                  │  │ Odometry (IMU)   │  │               │  │
│  │                  │  │ Scene context    │  │               │  │
│  │                  │  │ Visual nav       │  │               │  │
│  └─────────────────┘  └──────────────────┘  └───────┬───────┘  │
│                                                      │          │
│  ┌───────────────────────────────────────────────────▼───────┐  │
│  │              MuJoCo Simulation (scene/simulation.py)      │  │
│  │         Physics @ 250 Hz  ·  WBC control @ 50 Hz         │  │
│  │        100 sensors: IMU, joint encoders, foot contact     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Command Flow

```
User message (e.g. "walk 3 meters")
        │
        ▼
┌─ Safety Intercept ──────────────────────┐
│  "stop" / "halt" / "release" → instant  │
└──────────┬──────────────────────────────┘
           │ (not safety)
           ▼
┌─ VLM Tool-Call Planner ─────────────────┐
│  SmolVLM 500M on Intel iGPU (OpenVINO)  │
│                                          │
│  User msg + few-shot examples            │
│         ▼                                │
│  JSON plan:                              │
│  {"tasks": [                             │
│    {"tool": "move", "args":              │
│      {"distance": 3.0}}                  │
│  ]}                                      │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─ Tool Execution ────────────────────────┐
│  move() / turn_degrees() / look() /     │
│  get_sensor_info() / goto_landmark() /  │
│  goto_visual_target() / ...             │
│                                          │
│  IMU-based distance tracking             │
│  qpos reported for error comparison      │
└──────────┬──────────────────────────────┘
           │
           ▼
     Response to user
```

## Sensor Stack (INS - Inertial Navigation System)

| Sensor | Source | Update rate |
|--------|--------|-------------|
| Heading (yaw) | IMU framequat → Euler decomposition | 250 Hz |
| Roll / Pitch | IMU framequat → Euler decomposition | 250 Hz |
| Angular velocity | IMU gyroscope (3-axis, rad/s) | 250 Hz |
| Linear acceleration | IMU accelerometer (3-axis, m/s²) | 250 Hz |
| Body velocity | Frame velocity sensor (3-axis, m/s) | 250 Hz |
| Foot contact | 8 force sensors (4 per foot, N) | 250 Hz |
| Step odometry | Heel-strike detection × stride length | Event-driven |
| Joint positions | 29 joint encoders (rad) | 250 Hz |
| Joint velocities | 29 joint velocity sensors (rad/s) | 250 Hz |
| Joint torques | 29 torque sensors (Nm) | 250 Hz |

The robot uses IMU + step counting for its internal navigation (no qpos dependency). MuJoCo `qpos` is reported alongside for ground-truth comparison.

## Available Tools (VLM-callable)

| Tool | Args | Description |
|------|------|-------------|
| `move` | `distance`, `speed`, `direction` | Walk N metres; optional cardinal direction |
| `turn_degrees` | `degrees` | Relative turn (+left, -right) using IMU |
| `turn_magnetic` | `heading` | Absolute compass heading ("north", "east", ...) |
| `goto_landmark` | `name` | Navigate to a known scene landmark |
| `goto_visual_target` | `description` | Camera + VLM guided navigation to a described object |
| `look` | `prompt` | Head-camera VLM captioning |
| `look_around` | — | 360° scan with captions |
| `get_position` | — | IMU heading + qpos ground truth |
| `get_sensor_info` | — | Full IMU/INS sensor dump |
| `describe_scene` | — | Scene layout from memory |
| `halt` | — | Emergency stop |

## Installation

### Prerequisites

- Python 3.12+
- Intel GPU with OpenVINO support (tested on Intel Core Ultra, Panther Lake iGPU)
  - Or NVIDIA GPU with CUDA
- MuJoCo 3.x
- lerobo 

### Setup

```bash
# Clone the repository
git clone <repo-url> /home/robotics-rl
cd /home/robotics-rl

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Install LeRobot (required — provides Groot WBC locomotion controller)

LeRobot is a core runtime dependency. Full installation guide:
https://huggingface.co/docs/lerobot/installation

```bash
# From PyPI (quickest)
pip install lerobot

# Or from source (for development / latest features)
git clone https://github.com/huggingface/lerobot.git /home/lerobot
cd /home/lerobot && pip install -e .
cd /home/robotics-rl
```

> **Note:** LeRobot requires Python 3.12+ and `ffmpeg` with `libsvtav1` encoder.
> If using conda: `conda install ffmpeg -c conda-forge`
> Build errors may require: `sudo apt-get install cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev`

The Groot WBC model auto-downloads from HuggingFace on first run.

### Download VLM model

```bash
# SmolVLM 500M (auto-downloads on first run, or manually):
huggingface-cli download HuggingFaceTB/SmolVLM-500M-Instruct --local-dir models/smolvlm_500m
```

### Pre-export OpenVINO model (recommended for Intel GPU)

First run auto-exports, or manually:

```bash
python -c "
from cortex.peripheral.vlm_engine import VLMEngine
engine = VLMEngine(model_name='smolvlm500m')
engine.load()
print('OpenVINO model exported to models/smolvlm_500m_ov/')
"
```

## Running

### Basic simulation with chatbot

```bash
python orchestrator.py \
  --scene water_bottle_stage \
  --controller groot \
  --vlm-model smolvlm500m \
  --headless \
  --chatbot-port 9000
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--scene` | `water_bottle_stage` | MuJoCo scene (`flat_ground`, `water_bottle_stage`) |
| `--controller` | `groot` | WBC controller (`groot`, `holosoma`) |
| `--vlm-model` | `smolvlm500m` | VLM model for perception/planning |
| `--headless` | `False` | Run without MuJoCo viewer |
| `--chatbot-port` | `9000` | WebSocket chatbot port |
| `--planner-mode` | `auto` | Command routing mode |
| `--max-duration` | `0` | Max sim time in seconds (0 = unlimited) |

### Interacting

Open `http://<host>:9000` in a browser for the chatbot UI. Example commands:

```
release              → Drop the stabilisation harness
walk 3 meters        → Walk forward 3m
turn -10 degrees     → Turn 10° clockwise
face north           → Turn to compass north
go to stairs         → Navigate to stairs landmark
what do you see      → Camera + VLM scene description
sensor info          → Full IMU/INS readout
move to the red block → Visual target navigation
stop                 → Emergency halt
```

## Project Structure

```
robotics-rl/
├── orchestrator.py          # Main entry point and simulation loop
├── config.py                # Global configuration (paths, timing, hardware)
├── requirements.txt         # Python dependencies
├── setup.sh                 # Environment activation helper
│
├── cortex/                  # Robot brain modules
│   ├── motor/               # Whole-body control
│   │   └── controller.py    #   MotorCortex: Groot WBC, velocity commands, IMU turns
│   ├── perefrontal/         # High-level cognition
│   │   ├── brain.py         #   PrefrontalCortex: command queue, intent routing
│   │   └── vla.py           #   Vision-Language-Action interface
│   └── peripheral/          # Perception and planning
│       ├── vlm_engine.py    #   SmolVLM on OpenVINO GPU (captioning + text gen)
│       ├── planner.py       #   Command execution via VLM tool calling
│       ├── model_task_chain.py  # VLM prompt → JSON plan → tool dispatch
│       ├── odometry.py      #   IMU heading + step counting (INS)
│       ├── scene_context.py #   Landmark map from MuJoCo XML
│       ├── visual_nav.py    #   Camera-guided navigation
│       ├── device_manager.py#   Intel GPU / OpenVINO / CUDA detection
│       ├── intent_matcher.py#   Embedding-based intent matching (optional)
│       └── action_mapper.py #   Action type → tool dispatch
│
├── scene/                   # MuJoCo environments
│   ├── simulation.py        #   G1Simulation: physics, harness, rendering
│   ├── flat_ground.xml      #   Empty flat terrain
│   ├── water_bottle_stage.xml # Stage with stairs, podium, markers
│   ├── g1_29dof_no_hand.xml #   Robot MJCF model
│   ├── meshes/              #   G1 robot STL mesh files
│   └── textures/            #   Scene textures
│
├── server/                  # Web interface
│   └── app.py               #   FastAPI: WebSocket chat, camera stream
│
├── models/                  # VLM model weights (gitignored)
│   ├── smolvlm_500m/        #   SmolVLM 500M (HuggingFace)
│   ├── smolvlm_500m_ov/     #   SmolVLM 500M (OpenVINO IR, pre-exported)
│   ├── smolvlm_256m/        #   SmolVLM 256M
│   ├── moondream2/           #   Moondream 2 (alternative VLM)
│   └── qwen2vl_2b/          #   Qwen2-VL 2B (alternative VLM)
│
├── training/                # Training utilities
│   ├── generate_dataset.py  #   Dataset generation from simulation
│   ├── motion_library.py    #   Motion clip loading (CSV)
│   ├── playback.py          #   Motion replay
│   └── train_lerobot.sh     #   LeRobot training pipeline
│
└── datasets/                # Training datasets (gitignored)
    └── g1_wb_stairs/        #   Stair walking dataset
```

## Hardware Tested

- **CPU**: Intel Core Ultra 5 338H
- **GPU**: Intel Graphics (Panther Lake iGPU) via OpenVINO
- **VLM inference**: ~1-5s per caption on iGPU (SmolVLM 500M, 5 patches @ 768px)
- **Tool-call planning**: ~4-6s per command via VLM text generation
- **Simulation**: 250 Hz physics, 50 Hz WBC control

## License

See individual model licenses in `models/*/` directories.
