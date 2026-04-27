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
import sys
import signal
from pathlib import Path


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
# Backend launchers
# ---------------------------------------------------------------------------

def launch_mujoco(config, world_path, *, controller_type=None):
    """Launch MuJoCo simulation using the YAML execution graph."""
    sys.path.insert(0, str(Path(__file__).parent))

    if not Path(world_path).exists():
        print(f"Error: scene file not found: {world_path}")
        sys.exit(1)

    from model_runner import (
        build_graph, load_models, GraphPolicyRunner, Scheduler, print_banner,
    )
    from mujoco_engine import run_simulation

    graph, bus, controller = build_graph(config, controller_type)

    print("\nLoading models...")
    load_models(graph)

    runner = GraphPolicyRunner(graph, bus, config)

    cfg = config.get("configuration", {})
    sim_dt = float(cfg.get("simulation_dt", 0.002))
    decimation = int(cfg.get("control_decimation", 10))
    harness_cfg = cfg.get("harness", {})

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
        "--controller", default=None,
        help="Activate a controller source (e.g. keyboard). "
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

    sys.path.insert(0, str(Path(__file__).parent))

    if args.world and args.shadow_url:
        print("Error: --world and --shadow-url are mutually exclusive")
        sys.exit(1)

    from model_runner import load_robot_config, build_graph, Scheduler, print_banner

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

    if args.dry_run:
        graph, bus, controller = build_graph(config, args.controller)
        scheduler = Scheduler(graph, bus)
        print_banner(config, graph, scheduler, controller)
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
