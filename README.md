# Robot Deployment Package (RDP)

YAML-driven robot deployment for Unitree G1 with GR00T VLA and Whole Body Control.

Define your robot in a single YAML file (hardware, models, gains, controller bindings) and launch it in MuJoCo simulation with one command.

## Supported Configurations

| Config | WBC Type | Joints | Policy |
|--------|----------|--------|--------|
| `g1_groot_wbc_unified.yaml` | Decoupled | 15 lower-body + passive arms | Single ONNX |
| `g1_sonic_wbc.yaml` | SONIC | 29 full-body | Encoder-decoder ONNX |

## Quick Start

```bash
pip install -r requirements.txt

# Launch SONIC WBC in MuJoCo (uses default scene from YAML)
python scripts/launch_robot.py --robot configs/robots/g1_sonic_wbc.yaml

# Launch with keyboard controller
python scripts/launch_robot.py --robot configs/robots/g1_sonic_wbc.yaml --controller keyboard

# Launch decoupled WBC with custom scene
python scripts/launch_robot.py \
    --robot configs/robots/g1_groot_wbc_unified.yaml \
    --world /path/to/scene.xml

# Dry run (print resolved config, don't launch)
python scripts/launch_robot.py --robot configs/robots/g1_sonic_wbc.yaml --dry-run
```

## In-Simulation Controls

The robot starts hanging on a virtual harness with the policy disabled.

| Key | Action |
|-----|--------|
| P | Toggle policy on/off |
| 7 / H | Raise harness |
| 8 / J | Lower harness |
| 9 / K | Release harness (5s fade) |

With `--controller keyboard` (SONIC only):

| Key | Action |
|-----|--------|
| Arrow keys | Move forward/back/left/right |
| 1-9 | Switch locomotion style (walk, run, stealth, injured, zombie, ...) |
| 0 | Stop (IDLE) |

When the keyboard controller is active, harness keys move to H/J/K.

## Project Structure

```
rdp-demo/
├── configs/robots/           Robot YAML definitions
│   ├── g1_groot_wbc_unified.yaml   Decoupled WBC (15-DOF)
│   └── g1_sonic_wbc.yaml           SONIC WBC (29-DOF)
├── scripts/
│   ├── launch_robot.py       CLI orchestrator
│   ├── mujoco_engine.py      MuJoCo simulation loop (policy-agnostic)
│   ├── policy_runners.py     PolicyRunner classes + KeyboardController
│   └── download_groot_model.py
├── models/                   ONNX models (gitignored, download separately)
├── requirements.txt
└── pyproject.toml
```

## Architecture

The launcher reads the robot YAML, detects the WBC type, builds the appropriate `PolicyRunner`, and hands it to the MuJoCo engine:

```
launch_robot.py  →  reads YAML  →  builds PolicyRunner  →  mujoco_engine.run_simulation()
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                    DecoupledPolicyRunner    SonicPolicyRunner
                    (1 ONNX, 15 actions)    (encoder+decoder, 29 actions)
```

The engine is completely agnostic to the policy. It handles the MuJoCo scene, harness, PD control loop, and viewer. The runner handles ONNX loading, observation building, inference, and action mapping.

## Robot YAML Format

Each robot YAML is self-contained with all parameters inlined:

- **variables** — path substitution (`${VAR}`)
- **metadata** — name, version, description
- **hardware** — URDF, MuJoCo scene, sensors, actuators, joint names
- **models** — ONNX model paths, I/O shapes, dtypes
- **execution** — dataflow graph, frequencies, threading
- **configuration** — WBC params (gains, scales, joint mappings, controller bindings)
- **deployment_profiles** — overrides for sim vs real

## Requirements

- Python >= 3.10
- mujoco >= 3.0
- onnxruntime
- numpy
- PyYAML
