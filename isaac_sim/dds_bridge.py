#!/usr/bin/env python3
"""Isaac Sim <-> DDS bridge.

Run this script **inside the Isaac Sim Python environment** on the remote
machine.  It loads a robot USD asset, publishes joint state over DDS, and
subscribes to joint commands from the policy runner.

Usage:
    # From the Isaac Sim Python environment:
    ./python.sh isaac_sim/dds_bridge.py \\
        --usd /path/to/robot_scene.usd \\
        --robot-prim /World/G1 \\
        --domain-id 0 \\
        --state-topic rt/g1/state \\
        --command-topic rt/g1/command

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

    # -- Isaac Sim setup (must come before any other isaacsim imports) --
    try:
        from isaacsim import SimulationApp
    except ImportError:
        print("Error: this script must be run inside Isaac Sim's Python environment.")
        print("  Example: ./python.sh isaac_sim/dds_bridge.py --usd ...")
        sys.exit(1)

    sim_app = SimulationApp({"headless": False})

    import omni.isaac.core.utils.stage as stage_utils
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation

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

    # Import DDS types — add the rdp-demo scripts dir to path
    from pathlib import Path
    scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
    sys.path.insert(0, scripts_dir)
    from engines.dds_types import RobotStateDDS, JointCommandDDS

    # -- Load scene --
    print(f"Loading USD: {args.usd}")
    stage_utils.open_stage(args.usd)

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    world.reset()

    robot = world.scene.get_object(args.robot_prim)
    if robot is None:
        robot = Articulation(prim_path=args.robot_prim)
        world.scene.add(robot)
        world.reset()

    n_dofs = robot.num_dof
    print(f"Robot: {args.robot_prim}, {n_dofs} DOFs")

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

    # -- Simulation loop --
    world.set_simulation_dt(physics_dt=args.sim_dt, rendering_dt=args.sim_dt * 4)

    try:
        while sim_app.is_running():
            world.step(render=True)

            # -- Publish state --
            joint_positions = robot.get_joint_positions()
            joint_velocities = robot.get_joint_velocities()
            root_pos, root_quat = robot.get_world_pose()
            root_vel = robot.get_linear_velocity()
            root_ang_vel = robot.get_angular_velocity()

            # Pack into MuJoCo convention:
            #   qpos = [root_pos(3), root_quat_wxyz(4), joints(n)]
            #   qvel = [root_lin_vel(3), root_ang_vel(3), joint_vel(n)]
            quat_wxyz = np.array([root_quat[3], root_quat[0],
                                  root_quat[1], root_quat[2]])
            qpos = np.concatenate([root_pos, quat_wxyz, joint_positions])
            qvel = np.concatenate([root_vel, root_ang_vel, joint_velocities])

            state_msg = RobotStateDDS(
                qpos=qpos.tolist(),
                qvel=qvel.tolist(),
                timestamp_ns=int(time.time() * 1e9),
            )
            state_writer.write(state_msg)

            # -- Read commands --
            samples = command_reader.take(N=100)
            if samples:
                cmd = samples[-1]
                targets = np.array(cmd.target_positions, dtype=np.float32)
                robot.set_joint_position_targets(targets[:n_dofs])

    except KeyboardInterrupt:
        print("\nStopping ...")
    finally:
        sim_app.close()
        print("Isaac Sim closed.")


if __name__ == "__main__":
    main()
