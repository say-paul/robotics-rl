#!/usr/bin/env python3
"""Isaac Sim <-> DDS bridge.

Run this script **inside the Isaac Sim Python environment** on the remote
machine.  It loads a robot USD asset, publishes joint state over DDS, and
subscribes to joint commands from the policy runner.

Uses the dynamic_control interface instead of omni.isaac.core Articulation
to avoid the tensor view invalidation bug in Isaac Sim 5.x.

Usage:
    ./python.sh isaac_sim/dds_bridge.py \
        --usd /tmp/g1_scene.usd \
        --robot-prim /World/G1 \
        --domain-id 0

Requirements:
    pip install cyclonedds   (inside Isaac Sim's Python env)
"""

import argparse
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Isaac Sim DDS bridge — publish state, subscribe commands")
    parser.add_argument("--usd", required=True,
                        help="Path to USD scene containing the robot")
    parser.add_argument("--robot-prim", default="/World/G1",
                        help="Prim path of the robot articulation (default: /World/G1)")
    parser.add_argument("--sim-dt", type=float, default=0.002,
                        help="Simulation timestep in seconds (default: 0.002)")
    parser.add_argument("--domain-id", type=int, default=0,
                        help="DDS domain ID (default: 0)")
    parser.add_argument("--state-topic", default="rt/g1/state",
                        help="DDS topic for publishing robot state")
    parser.add_argument("--command-topic", default="rt/g1/command",
                        help="DDS topic for subscribing to joint commands")
    args = parser.parse_args()

    try:
        from isaacsim import SimulationApp
    except ImportError:
        print("Error: this script must be run inside Isaac Sim's Python environment.")
        print("  Example: ./python.sh isaac_sim/dds_bridge.py --usd ...")
        sys.exit(1)

    sim_app = SimulationApp({"headless": True})

    import omni.isaac.core.utils.stage as stage_utils
    from omni.isaac.dynamic_control import _dynamic_control
    import omni.kit.app
    import omni.physx

    # -- DDS setup --
    try:
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.sub import DataReader
        from cyclonedds.pub import DataWriter
        from cyclonedds.topic import Topic
    except ImportError:
        print("Error: cyclonedds not installed in Isaac Sim's Python env.")
        print("  Install with: ./python.sh -m pip install cyclonedds")
        sim_app.close()
        sys.exit(1)

    from pathlib import Path
    scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
    sys.path.insert(0, scripts_dir)
    from engines.dds_types import RobotStateDDS, JointCommandDDS

    # -- Load scene --
    print(f"Loading USD: {args.usd}")
    stage_utils.open_stage(args.usd)

    # Find the robot articulation prim
    from pxr import UsdPhysics
    stage = omni.usd.get_context().get_stage()
    articulation_prims = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_prims.append(str(prim.GetPath()))

    robot_prim_path = args.robot_prim
    if articulation_prims:
        print(f"Articulations found: {articulation_prims}")
        if robot_prim_path not in articulation_prims:
            robot_prim_path = articulation_prims[0]
            print(f"  '{args.robot_prim}' not found, using: {robot_prim_path}")
    else:
        print("No articulations found. Root prims:")
        for prim in stage.GetPseudoRoot().GetChildren():
            print(f"  {prim.GetPath()}")
        sim_app.close()
        sys.exit(1)

    # Configure physics timestep
    from omni.physx import get_physx_interface
    physx = get_physx_interface()

    # Step simulation a few times to let physics initialize
    dc = _dynamic_control.acquire_dynamic_control_interface()

    # Start simulation
    omni.timeline.get_timeline_interface().play()

    # Wait for physics to settle and acquire articulation handle
    for _ in range(10):
        sim_app.update()

    art_handle = dc.get_articulation(robot_prim_path)
    if art_handle == _dynamic_control.INVALID_HANDLE:
        print(f"Error: could not get articulation handle for '{robot_prim_path}'")
        print("  Make sure the USD has a valid PhysicsScene and ArticulationRootAPI.")
        sim_app.close()
        sys.exit(1)

    n_dofs = dc.get_articulation_dof_count(art_handle)
    print(f"Robot: {robot_prim_path}, {n_dofs} DOFs")

    # Start HLS viewport streaming (background threads)
    try:
        from viewport_stream import start as start_stream
        start_stream()
    except Exception as e:
        print(f"[bridge] viewport streaming not available: {e}")

    # -- DDS participants --
    dp = DomainParticipant(domain_id=args.domain_id)
    state_topic = Topic(dp, args.state_topic, RobotStateDDS)
    command_topic = Topic(dp, args.command_topic, JointCommandDDS)
    state_writer = DataWriter(dp, state_topic)
    command_reader = DataReader(dp, command_topic)

    print(f"DDS domain={args.domain_id}")
    print(f"  publishing : {args.state_topic}")
    print(f"  subscribing: {args.command_topic}")
    print(f"Sim dt={args.sim_dt}s  ({1.0/args.sim_dt:.0f} Hz)")
    print("Running ... (Ctrl-C to stop)\n")

    step_count = 0
    running = True
    try:
        while running:
            sim_app.update()
            step_count += 1

            if not omni.timeline.get_timeline_interface().is_playing():
                omni.timeline.get_timeline_interface().play()

            root_body = dc.get_articulation_root_body(art_handle)
            if root_body == _dynamic_control.INVALID_HANDLE:
                art_handle = dc.get_articulation(robot_prim_path)
                continue

            # Root pose
            pose = dc.get_rigid_body_pose(root_body)
            root_pos = np.array([pose.p.x, pose.p.y, pose.p.z])
            root_quat_wxyz = np.array([pose.r.w, pose.r.x, pose.r.y, pose.r.z])

            # Root velocity
            lin_vel = dc.get_rigid_body_linear_velocity(root_body)
            ang_vel = dc.get_rigid_body_angular_velocity(root_body)
            root_lin = np.array([lin_vel.x, lin_vel.y, lin_vel.z])
            root_ang = np.array([ang_vel.x, ang_vel.y, ang_vel.z])

            # Joint state
            dof_states = dc.get_articulation_dof_states(art_handle, _dynamic_control.STATE_ALL)
            joint_pos = dof_states["pos"]
            joint_vel = dof_states["vel"]

            # Pack into MuJoCo convention
            qpos = np.concatenate([root_pos, root_quat_wxyz, joint_pos])
            qvel = np.concatenate([root_lin, root_ang, joint_vel])

            state_msg = RobotStateDDS(
                qpos=qpos.tolist(),
                qvel=qvel.tolist(),
                timestamp_ns=int(time.time() * 1e9),
            )
            state_writer.write(state_msg)

            if step_count % 500 == 0:
                print(f"[bridge] step={step_count} z={root_pos[2]:.3f} "
                      f"dofs={len(joint_pos)}")

            # Read commands
            samples = command_reader.take(N=100)
            if samples:
                cmd = samples[-1]
                targets = np.array(cmd.target_positions, dtype=np.float32)
                for i in range(min(len(targets), n_dofs)):
                    dc.set_dof_position_target(
                        dc.get_articulation_dof(art_handle, i),
                        float(targets[i]))

    except KeyboardInterrupt:
        print("\nStopping ...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        running = False
        try:
            omni.timeline.get_timeline_interface().stop()
        except Exception:
            pass
        sim_app.close()
        print("Isaac Sim closed.")


if __name__ == "__main__":
    main()
