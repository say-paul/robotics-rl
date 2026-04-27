# Robot Deployment Package (RDP)

YAML-driven robot deployment for Unitree G1 with GR00T VLA and Whole Body Control.

Define your robot in a single YAML file (hardware, models, gains, controller bindings) and run it in MuJoCo, Isaac Sim, or on real hardware — same config, same models, different `--world` or `--deploy` flag.

## Supported Configurations

| Config | WBC Type | Joints | Policy |
|--------|----------|--------|--------|
| `g1_groot_wbc_unified.yaml` | Decoupled | 15 lower-body + passive arms | Single ONNX |
| `g1_sonic_wbc.yaml` | SONIC | 29 full-body | Encoder-decoder ONNX |

## Install

```bash
# Simulation only (MuJoCo + ONNX)
./install.sh

# Full robot setup (CycloneDDS, Unitree SDK2, GR00T WBC)
./setup_robot.sh
```

## Quick Start

### Local MuJoCo Simulation

```bash
# Uses default scene from robot YAML
python scripts/launch_robot.py --robot configs/robots/g1_sonic_wbc.yaml

# With keyboard controller
python scripts/launch_robot.py --robot configs/robots/g1_sonic_wbc.yaml --controller keyboard

# Pick a specific simulator world from YAML
python scripts/launch_robot.py --robot configs/robots/g1_sonic_wbc.yaml --world mujoco

# Custom scene file
python scripts/launch_robot.py --robot configs/robots/g1_sonic_wbc.yaml --world /path/to/scene.xml

# Dry run (print resolved config, don't launch)
python scripts/launch_robot.py --robot configs/robots/g1_sonic_wbc.yaml --dry-run
```

### Deploy on Real Robot (DDS)

```bash
# Real robot — multicast discovery on local network
python scripts/launch_robot.py --robot configs/robots/g1_sonic_wbc.yaml --deploy

# With keyboard controller for manual override
python scripts/launch_robot.py --robot configs/robots/g1_sonic_wbc.yaml --deploy --controller keyboard
```

The policy runs headless, communicating with the robot over Cyclone DDS.
The robot's onboard firmware handles PD control.

### Remote Simulation (Isaac Sim, O3DE, etc.)

Run the simulator on a remote machine, and the policy locally:

**On the remote machine** (inside Isaac Sim's Python env):
```bash
./python.sh isaac_sim/dds_bridge.py \
    --usd /path/to/g1_scene.usd \
    --robot-prim /World/G1 \
    --domain-id 0
```

> **`--robot-prim`** is the USD scene-graph path to the robot articulation
> inside the Isaac Sim stage. The robot USD assets (e.g.
> `g1_29dof_rev_1_0.usd`) are included in the GR00T WBC repo, which is
> downloaded by `./install.sh`. Import one into an Isaac Sim stage with a
> ground plane and lighting, then pass the prim path where it was placed.

**On your local machine** (runs the policy):
```bash
python scripts/launch_robot.py \
    --robot configs/robots/g1_sonic_wbc.yaml \
    --deploy 192.168.1.100
```

The `--deploy HOST` flag configures Cyclone DDS unicast discovery to reach
the remote machine directly (works across subnets and VPNs).

DDS topic names and domain ID come from the `interfaces` section of the robot YAML:
```yaml
interfaces:
  - name: "dds_control"
    protocol: "dds"
    domain_id: 0
    topics:
      state: "rt/g1/state"
      command: "rt/g1/command"
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
| 0 | Stop (IDLE) |
| 1 | Slow walk |
| 2 | Walk |
| 3 | Run |
| 4-6 | Other styles (stealth, zombie, happy dance) |

When the keyboard controller is active, harness keys move to H/J/K.

## Project Structure

```
rdp-demo/
├── configs/robots/             Robot YAML definitions
│   ├── g1_groot_wbc_unified.yaml   Decoupled WBC (15-DOF)
│   └── g1_sonic_wbc.yaml           SONIC WBC (29-DOF)
├── scripts/
│   ├── launch_robot.py         CLI entry point (--world / --deploy / --shadow-url)
│   ├── model_runner.py         YAML graph builder, SignalBus, PolicyRunner, Scheduler
│   ├── engines/
│   │   ├── __init__.py         RobotState dataclass + SimulationEngine ABC
│   │   ├── mujoco_engine.py    Local MuJoCo simulation (harness, PD, viewer)
│   │   ├── dds_engine.py       DDS transport (real robot / remote sim)
│   │   └── dds_types.py        Cyclone DDS message definitions
│   └── handlers/
│       ├── sonic_wbc.py        SONIC WBC handler (joint remap, history, gains)
│       ├── sonic_encoder.py    Observation encoder handler
│       ├── sonic_decoder.py    Policy decoder handler
│       └── sonic_planner.py    Local motion planner handler
├── isaac_sim/
│   └── dds_bridge.py           Standalone Isaac Sim DDS bridge (remote machine)
├── models/                     ONNX models (gitignored, download separately)
├── install.sh                  Simulation-only install
├── setup_robot.sh              Full robot setup (CycloneDDS, Unitree SDK, WBC)
├── requirements.txt
└── pyproject.toml
```

## Architecture

```
launch_robot.py
  --world mujoco    →  MuJoCoEngine  →  local viewer + PD control
  --world isaac_sim →  (placeholder)
  --deploy          →  DDSEngine     →  headless, DDS I/O to real robot
  --deploy HOST     →  DDSEngine     →  headless, DDS unicast to remote sim
```

The engine abstraction (`RobotState`) decouples the policy from the physics backend:

```
                    ┌─────────────────────────────┐
                    │       GraphPolicyRunner      │
                    │  (WBC handler + ONNX graph)  │
                    └──────────┬──────────────────┘
                               │ RobotState
              ┌────────────────┼────────────────┐
              │                │                │
        MuJoCoEngine     DDSEngine         (future)
        (local sim)    (real robot /     Isaac Sim local,
                       remote sim)       Genesis, O3DE
```

All engines pass the same `RobotState(qpos, qvel, time)` to the policy.
The WBC handler, ONNX models, and controller work identically regardless of backend.

## Robot YAML Format

Each robot YAML is self-contained with all parameters inlined:

- **variables** — path substitution (`${VAR}`)
- **metadata** — name, version, description
- **hardware** — URDF, world scenes per simulator, sensors, actuators
- **models** — ONNX model paths, runtime, precision
- **execution** — dataflow DAG, node frequencies, handlers
- **configuration** — WBC params (gains, scales, joint mappings, controller bindings)
- **interfaces** — DDS, ZMQ, gRPC, MQTT endpoints
- **deployment_profiles** — overrides for sim vs real

## Requirements

- Python >= 3.10
- mujoco >= 3.0
- onnxruntime
- numpy
- PyYAML
- cyclonedds (only for `--deploy` mode)
