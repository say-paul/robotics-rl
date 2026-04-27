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
    from engines.mujoco_engine import run_simulation

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


def launch_dds(config, *, peer=None, controller_type=None):
    """Deploy on real hardware or remote sim via DDS.

    Parameters
    ----------
    peer : str or None
        Remote host address for unicast DDS discovery.
        None = multicast (real robot on local network).
    """
    sys.path.insert(0, str(Path(__file__).parent))

    from model_runner import (
        build_graph, load_models, GraphPolicyRunner,
    )
    from engines.dds_engine import DDSEngine

    dds_cfg = _find_dds_interface(config)
    if dds_cfg is None:
        print("Error: --deploy requires a 'protocol: dds' entry in the "
              "robot YAML interfaces section.")
        sys.exit(1)

    graph, bus, controller = build_graph(config, controller_type)

    print("\nLoading models...")
    load_models(graph)

    runner = GraphPolicyRunner(graph, bus, config)

    cfg = config.get("configuration", {})
    sim_dt = float(cfg.get("simulation_dt", 0.002))
    decimation = int(cfg.get("control_decimation", 10))
    control_hz = 1.0 / (sim_dt * decimation)

    engine = DDSEngine(dds_cfg, peer=peer, control_hz=control_hz)
    engine.run(runner, controller=controller)


def _find_dds_interface(config):
    """Find the first interface entry with protocol: dds."""
    for iface in config.get("interfaces", []):
        if iface.get("protocol") == "dds":
            return iface
    return None


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

def _resolve_default_world(config, world_type=None):
    """Find a default world path from the robot YAML.

    Looks in ``hardware.description.worlds`` (new generic format) first,
    then falls back to ``hardware.description.mujoco_scene.file`` for
    backward compatibility.

    Parameters
    ----------
    world_type : str or None
        Preferred simulator (e.g. "mujoco", "isaac_sim").  If None,
        returns the first non-empty entry found.
    """
    hw = config.get("hardware", {}).get("description", {})
    worlds = hw.get("worlds", {})

    if world_type:
        path = worlds.get(world_type, "")
        return path if path else None

    if worlds:
        for wtype in ("mujoco", "isaac_sim", "isaac_lab", "genesis", "o3de"):
            path = worlds.get(wtype, "")
            if path:
                return path

    # Backward compat: old mujoco_scene.file key
    scene = hw.get("mujoco_scene", {}).get("file")
    if scene:
        return scene

    return None


def resolve_world(args, config):
    """Determine the world path from CLI args or the robot YAML's defaults.

    ``--world`` accepts either:
      - A file path:  ``--world /path/to/scene.xml``
      - A key name:   ``--world mujoco``  (looked up in YAML ``worlds`` dict)

    Returns (world_path, world_type) or (None, "shadow").
    """
    if args.shadow_url:
        return None, "shadow"

    world_arg = args.world

    if world_arg is None:
        world_path = _resolve_default_world(config)
        if world_path:
            print(f"Using default world from robot YAML: {world_path}")
        else:
            print("Error: no --world provided and no worlds / mujoco_scene "
                  "in robot YAML")
            sys.exit(1)
    else:
        # Check if the arg matches a key in the worlds dict
        hw = config.get("hardware", {}).get("description", {})
        worlds = hw.get("worlds", {})
        resolved = worlds.get(world_arg, "")
        if resolved:
            world_path = resolved
            print(f"Using {world_arg} world from robot YAML: {world_path}")
        else:
            world_path = world_arg

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
        help="Scene file path or a key from the robot YAML worlds dict "
             "(e.g. mujoco, isaac_sim). Omit to use the first available default.",
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
        "--deploy", nargs="?", const=True, default=None, metavar="HOST",
        help="Deploy via DDS (headless, no simulated world). "
             "Without a host: real robot on local DDS (multicast discovery). "
             "With a host: remote sim (Isaac Sim, O3DE, etc.) at that address. "
             "Example: --deploy 192.168.1.100",
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

    exclusive_count = sum([
        bool(args.world), bool(args.shadow_url), args.deploy is not None,
    ])
    if exclusive_count > 1:
        print("Error: --world, --shadow-url, and --deploy are mutually exclusive")
        sys.exit(1)

    from model_runner import load_robot_config, build_graph, Scheduler, print_banner

    config = load_robot_config(args.robot)

    meta = config.get("metadata", {})
    print("=" * 60)
    print(f"Robot : {meta.get('name', '?')}")
    print(f"Desc  : {meta.get('description', '')}")
    print("=" * 60)

    if args.deploy is not None:
        peer = args.deploy if isinstance(args.deploy, str) else None
        if peer:
            print(f"Mode  : deploy (DDS -> {peer})")
        else:
            print(f"Mode  : deploy (DDS, local robot)")
        dds_cfg = _find_dds_interface(config)
        if dds_cfg:
            topics = dds_cfg.get("topics", {})
            print(f"DDS   : domain {dds_cfg.get('domain_id', 0)}, "
                  f"state={topics.get('state')}, cmd={topics.get('command')}")
        print("=" * 60)
    else:
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

    if args.deploy is not None:
        peer = args.deploy if isinstance(args.deploy, str) else None
        launch_dds(config, peer=peer, controller_type=args.controller)
    elif world_type == "shadow":
        launch_shadow(config, args.shadow_url, args.shadow_port)
    else:
        backend_fn = BACKENDS.get(world_type)
        if backend_fn is None:
            print(f"Error: no backend for world type '{world_type}'")
            sys.exit(1)
        backend_fn(config, world_path, controller_type=args.controller)


if __name__ == "__main__":
    main()
