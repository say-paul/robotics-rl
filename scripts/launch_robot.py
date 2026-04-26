#!/usr/bin/env python3
"""
Unified Robot Launcher

Usage:
  # MuJoCo simulation (world type auto-detected from .xml)
  python scripts/launch_robot.py \\
      --robot configs/robots/g1_groot_wbc_unified.yaml \\
      --world /path/to/scene_29dof.xml

  # Shadow a real robot (digital twin)
  python scripts/launch_robot.py \\
      --robot configs/robots/g1_groot_wbc_unified.yaml \\
      --shadow-url 192.168.123.164 --shadow-port 5558

  # Use default world from robot YAML
  python scripts/launch_robot.py \\
      --robot configs/robots/g1_groot_wbc_unified.yaml

  # Dry run — print resolved config, don't launch
  python scripts/launch_robot.py \\
      --robot configs/robots/g1_groot_wbc_unified.yaml \\
      --dry-run
"""

import argparse
import json
import os
import sys
import signal
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# YAML loading with ${VAR} substitution
# ---------------------------------------------------------------------------

def substitute_variables(obj, variables):
    """Recursively replace ${VAR_NAME} placeholders in a config tree."""
    if isinstance(obj, dict):
        return {k: substitute_variables(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute_variables(item, variables) for item in obj]
    if isinstance(obj, str):
        result = obj
        for name, value in variables.items():
            result = result.replace(f"${{{name}}}", str(value))
        return result
    return obj


def load_robot_config(yaml_path):
    """Load a robot YAML and resolve all ${VAR} references."""
    path = Path(yaml_path)
    if not path.exists():
        print(f"Error: robot config not found: {path}")
        sys.exit(1)

    with open(path) as f:
        raw = yaml.safe_load(f)

    variables = raw.get("variables", {})
    config = substitute_variables(raw, variables)
    return config


# ---------------------------------------------------------------------------
# World type detection
# ---------------------------------------------------------------------------

WORLD_TYPE_MAP = {
    ".xml": "mujoco",
    ".mjcf": "mujoco",
    ".usd": "isaac_sim",
    ".usda": "isaac_sim",
    ".usdc": "isaac_sim",
}


def detect_world_type(world_path):
    """Infer simulation backend from file extension."""
    ext = Path(world_path).suffix.lower()
    wtype = WORLD_TYPE_MAP.get(ext)
    if wtype is None:
        print(f"Error: unknown world file type '{ext}' for {world_path}")
        print(f"  Supported: {', '.join(WORLD_TYPE_MAP.keys())}")
        sys.exit(1)
    return wtype


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def _require(d, key, context="config"):
    """Get a required key from dict, exit with clear error if missing."""
    if key not in d:
        print(f"Error: missing required key '{key}' in {context}")
        sys.exit(1)
    return d[key]


def validate_model_shapes(config):
    """
    Check that model output shapes match downstream input shapes
    across the execution dataflow declared in the YAML.
    """
    models = {m["name"]: m for m in config.get("models", [])}
    dataflow = (config.get("execution", {}).get("dataflow") or [])
    if not dataflow:
        return

    def get_output_shape(model_name, field_name=None):
        m = models.get(model_name, {})
        for o in m.get("output", []):
            if field_name is None or o["name"] == field_name:
                return o.get("shape")
        return None

    def get_input_shape(model_name, field_name=None):
        m = models.get(model_name, {})
        for i in m.get("input", []):
            if field_name is None or i["name"] == field_name:
                return i.get("shape")
        return None

    print("\nValidating model I/O shapes...")
    errors = []

    for model_name, m in models.items():
        for out in m.get("output", []):
            out_name = out["name"]
            out_shape = out.get("shape")
            if out_shape is None:
                continue

            for downstream_name, dm in models.items():
                if downstream_name == model_name:
                    continue
                for inp in dm.get("input", []):
                    if inp["name"] == out_name:
                        in_shape = inp.get("shape")
                        if in_shape is not None and in_shape != out_shape:
                            errors.append(
                                f"  {model_name}.output.{out_name} {out_shape} != "
                                f"{downstream_name}.input.{inp['name']} {in_shape}"
                            )

    if errors:
        print("  Shape mismatches found:")
        for e in errors:
            print(e)
        sys.exit(1)
    else:
        print("  All model I/O shapes consistent.")


# ---------------------------------------------------------------------------
# Backend launchers
# ---------------------------------------------------------------------------

def _detect_wbc_type(config):
    """Return 'sonic' or 'decoupled' based on configuration.wbc.type."""
    wbc = config.get("configuration", {}).get("wbc", {})
    return wbc.get("type", "decoupled")


def _build_decoupled_config(config):
    """Build engine_config dict for the decoupled WBC path."""
    cfg = _require(config, "configuration")
    wbc = _require(cfg, "wbc", "configuration")

    required_wbc_keys = [
        "num_actions", "num_obs", "obs_history_len", "single_obs_dim",
        "default_angles", "kps", "kds", "arm_kp", "arm_kd",
        "action_scale", "dof_pos_scale", "dof_vel_scale", "ang_vel_scale",
        "cmd_scale", "cmd_init", "height_cmd", "rpy_cmd",
    ]
    for key in required_wbc_keys:
        _require(wbc, key, "configuration.wbc")

    engine_config = {
        **wbc,
        "simulation_dt": _require(cfg, "simulation_dt", "configuration"),
        "control_decimation": _require(cfg, "control_decimation", "configuration"),
    }

    models = config.get("models", [])
    policy_path = None
    for m in models:
        if m.get("type") == "locomotion_policy":
            paths = m.get("model_paths", {})
            policy_path = paths.get("balance") or paths.get("walk")
            break

    if policy_path is None:
        print("Error: no locomotion_policy model found in robot config")
        sys.exit(1)

    return engine_config, policy_path


def _build_sonic_config(config):
    """Build engine_config dict for the SONIC WBC path."""
    cfg = _require(config, "configuration")
    wbc = _require(cfg, "wbc", "configuration")

    required_wbc_keys = [
        "num_actions", "default_angles", "kps", "kds",
        "action_scales", "encoder_obs_dim", "encoder_output_dim",
        "policy_obs_dim", "history_frames", "history_step",
        "isaaclab_to_mujoco",
    ]
    for key in required_wbc_keys:
        _require(wbc, key, "configuration.wbc")

    engine_config = {
        **wbc,
        "simulation_dt": _require(cfg, "simulation_dt", "configuration"),
        "control_decimation": _require(cfg, "control_decimation", "configuration"),
    }

    models = config.get("models", [])
    encoder_path = None
    decoder_path = None
    for m in models:
        if m.get("type") == "observation_encoder":
            encoder_path = m.get("model_path")
        elif m.get("type") == "action_policy":
            decoder_path = m.get("model_path")

    if encoder_path is None:
        print("Error: no observation_encoder model found in robot config")
        sys.exit(1)
    if decoder_path is None:
        print("Error: no action_policy model found in robot config")
        sys.exit(1)

    return engine_config, encoder_path, decoder_path


def launch_mujoco(config, world_path, *, controller_type=None):
    """Launch the MuJoCo simulation with the appropriate policy runner."""
    sys.path.insert(0, str(Path(__file__).parent))

    if not Path(world_path).exists():
        print(f"Error: scene file not found: {world_path}")
        sys.exit(1)

    wbc_type = _detect_wbc_type(config)
    print(f"WBC type: {wbc_type}")

    cfg = _require(config, "configuration")
    sim_dt = float(_require(cfg, "simulation_dt", "configuration"))
    decimation = int(_require(cfg, "control_decimation", "configuration"))
    harness_cfg = cfg.get("wbc", {}).get("harness", {})

    from policy_runners import DecoupledPolicyRunner, SonicPolicyRunner, KeyboardController
    from mujoco_engine import run_simulation

    # Build controller if requested
    controller = None
    if controller_type == "keyboard":
        ctrl_cfg = cfg.get("controller")
        if ctrl_cfg is None:
            print("Error: --controller keyboard requested but no controller "
                  "section found in robot YAML (configuration.controller)")
            sys.exit(1)
        controller = KeyboardController(ctrl_cfg)
        print(f"Controller: keyboard (use arrow keys + number keys)")

    if wbc_type == "sonic":
        engine_config, encoder_path, decoder_path = _build_sonic_config(config)

        for label, p in [("encoder", encoder_path), ("decoder", decoder_path)]:
            if not Path(p).exists():
                print(f"Error: {label} ONNX not found: {p}")
                print(f"  (SONIC ONNX files are stored via Git LFS — run 'git lfs pull' inside GR00T-WholeBodyControl)")
                sys.exit(1)

        runner = SonicPolicyRunner(encoder_path, decoder_path, engine_config)

    else:
        engine_config, policy_path = _build_decoupled_config(config)

        if not Path(policy_path).exists():
            print(f"Error: policy file not found: {policy_path}")
            sys.exit(1)

        runner = DecoupledPolicyRunner(policy_path, engine_config)

    run_simulation(
        world_path, runner,
        sim_dt=sim_dt,
        decimation=decimation,
        harness_cfg=harness_cfg,
        controller=controller,
    )


def launch_isaac_sim(config, world_path, *, controller_type=None):
    """Placeholder for Isaac Sim / USD backend."""
    print(f"Isaac Sim backend not yet implemented.")
    print(f"  Robot: {config['metadata']['name']}")
    print(f"  World: {world_path}")
    sys.exit(1)


def launch_shadow(config, shadow_url, shadow_port):
    """Placeholder for shadow / digital-twin mode."""
    print(f"Shadow mode not yet implemented.")
    print(f"  Robot:  {config['metadata']['name']}")
    print(f"  Source: tcp://{shadow_url}:{shadow_port}")
    sys.exit(1)


BACKENDS = {
    "mujoco": launch_mujoco,
    "isaac_sim": launch_isaac_sim,
}


# ---------------------------------------------------------------------------
# Resolve the world path
# ---------------------------------------------------------------------------

def resolve_world(args, config):
    """
    Determine the world path from CLI args or the robot YAML's default_world.
    Returns (world_path, world_type) or (None, "shadow").
    """
    if args.shadow_url:
        return None, "shadow"

    world_path = args.world

    if world_path is None:
        hw = config.get("hardware", {}).get("description", {})
        scene = hw.get("mujoco_scene", {}).get("file")
        if scene:
            world_path = scene
            print(f"Using default world from robot YAML: {world_path}")
        else:
            print("Error: no --world provided and no default_world / mujoco_scene in robot YAML")
            sys.exit(1)

    world_type = detect_world_type(world_path)
    return world_path, world_type


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Launch a robot in simulation or shadow mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--robot", required=True,
        help="Path to robot definition YAML (e.g. configs/robots/g1_groot_wbc_unified.yaml)",
    )
    parser.add_argument(
        "--world", default=None,
        help="Path to world/scene file (.xml for MuJoCo, .usd for Isaac Sim). "
             "Omit to use the default from the robot YAML.",
    )
    parser.add_argument(
        "--shadow-url", default=None,
        help="IP/hostname of physical robot to shadow (digital twin mode)",
    )
    parser.add_argument(
        "--shadow-port", type=int, default=5558,
        help="Port for shadow connection (default: 5558)",
    )
    parser.add_argument(
        "--controller", default=None, choices=["keyboard"],
        help="Activate a controller (e.g. keyboard). "
             "If omitted, no runtime controller is active.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print resolved config and exit without launching",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.world and args.shadow_url:
        print("Error: --world and --shadow-url are mutually exclusive")
        sys.exit(1)

    config = load_robot_config(args.robot)

    meta = config.get("metadata", {})
    print("=" * 60)
    print(f"Robot : {meta.get('name', '?')}")
    print(f"Desc  : {meta.get('description', '')}")
    print("=" * 60)

    world_path, world_type = resolve_world(args, config)

    if world_type == "shadow":
        print(f"Mode  : shadow (digital twin)")
        print(f"Source: tcp://{args.shadow_url}:{args.shadow_port}")
    else:
        print(f"Mode  : {world_type}")
        print(f"World : {world_path}")
    print("=" * 60)

    validate_model_shapes(config)

    if args.dry_run:
        print("\n[dry-run] Resolved configuration:\n")
        print(json.dumps(config, indent=2, default=str))
        return

    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    if world_type == "shadow":
        launch_shadow(config, args.shadow_url, args.shadow_port)
    else:
        backend_fn = BACKENDS.get(world_type)
        if backend_fn is None:
            print(f"Error: no backend for world type '{world_type}'")
            sys.exit(1)
        backend_fn(config, world_path, controller_type=args.controller)


if __name__ == "__main__":
    main()
