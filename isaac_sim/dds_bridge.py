#!/usr/bin/env python3
# This project was developed with assistance from AI tools.
"""Isaac Sim <-> DDS bridge using IsaacLab tensor API.

Based on Unitree's unitree_sim_isaaclab wholebody control pattern.
Uses World/Articulation from omni.isaac.core with the tensor API
(set_joint_position_targets + world.step), not the deprecated
dynamic_control API which freezes physics at runtime.

The USD's built-in joint drives handle PD control. The bridge only
sets position targets and reads state.

Joint angle convention:
  - Isaac Sim USD rest pose (target=0.0) = the standing pose
  - MuJoCo zero = URDF zero (straight legs)
  - default_angles = absolute URDF angles for standing
  - Policy sends: default_angles + action * scale
  - Bridge sends to Isaac Sim: policy_target - default_angles

Quaternion convention:
  - Both Isaac Sim and MuJoCo use [w, x, y, z] (scalar-first)
  - get_world_pose() returns [w, x, y, z] directly

Usage:
    ./python.sh isaac_sim/dds_bridge.py \\
        --usd /tmp/g1_scene.usd --robot-prim /World/G1
"""

import argparse
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Isaac Sim DDS bridge")
    parser.add_argument("--usd", required=True, help="Path to USD scene")
    parser.add_argument("--robot-prim", default="/World/G1",
                        help="Robot articulation prim path")
    parser.add_argument("--sim-dt", type=float, default=0.005,
                        help="Physics timestep (default: 0.005 = 200 Hz)")
    parser.add_argument("--decimation", type=int, default=4,
                        help="Control decimation (default: 4 = 50 Hz control)")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--state-topic", default="rt/g1/state")
    parser.add_argument("--command-topic", default="rt/g1/command")
    args = parser.parse_args()

    try:
        from isaacsim import SimulationApp
    except ImportError:
        print("Error: run inside Isaac Sim Python environment.")
        sys.exit(1)

    sim_app = SimulationApp({"headless": True})

    import omni.isaac.core.utils.stage as stage_utils
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.utils.types import ArticulationAction
    from pxr import UsdPhysics

    try:
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.sub import DataReader
        from cyclonedds.pub import DataWriter
        from cyclonedds.topic import Topic
    except ImportError:
        print("Error: cyclonedds not installed.")
        sim_app.close()
        sys.exit(1)

    from pathlib import Path
    scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
    sys.path.insert(0, scripts_dir)
    from engines.dds_types import RobotStateDDS, JointCommandDDS

    # Load default_angles from YAML
    import yaml
    da_mj = np.zeros(29, dtype=np.float64)
    cfg_path = Path(__file__).resolve().parent.parent / "configs/robots/g1_sonic_wbc.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
        wbc_cfg = raw.get("configuration", {}).get("wbc", {})
        da = wbc_cfg.get("default_angles", [])
        if len(da) >= 29:
            da_mj = np.array(da[:29], dtype=np.float64)

    # -- Load scene --
    print(f"Loading USD: {args.usd}")
    stage_utils.open_stage(args.usd)

    world = World(stage_units_in_meters=1.0, physics_dt=args.sim_dt,
                  rendering_dt=args.sim_dt * args.decimation)

    # Find articulation
    stage = world.stage
    articulation_prims = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_prims.append(str(prim.GetPath()))

    robot_prim_path = args.robot_prim
    if articulation_prims:
        print(f"Articulations found: {articulation_prims}")
        if robot_prim_path not in articulation_prims:
            robot_prim_path = articulation_prims[0]
            print(f"  Using: {robot_prim_path}")

    robot = Articulation(prim_path=robot_prim_path)
    world.scene.add(robot)
    world.reset()

    for _ in range(10):
        world.step(render=False)

    try:
        robot.initialize()
    except Exception:
        pass

    n_dofs = robot.num_dof
    print(f"Robot: {robot_prim_path}, {n_dofs} DOFs")

    # Build joint name remap (Isaac Sim order ↔ MuJoCo order)
    joint_names = [robot.dof_names[i] if i < len(robot.dof_names) else f"joint_{i}"
                   for i in range(n_dofs)]
    # MuJoCo order matches policy_parameters.hpp: hip_pitch first, not hip_yaw.
    # This order matches default_angles, kps, kds arrays in the YAML.
    _MUJOCO_JOINTS = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]

    try:
        isim_to_mj = [_MUJOCO_JOINTS.index(name) for name in joint_names]
        mj_to_isim = [joint_names.index(name) for name in _MUJOCO_JOINTS]
        print(f"[bridge] Joint remap built ({n_dofs} joints)")
    except ValueError as e:
        print(f"[bridge] Joint name mismatch: {e}")
        print(f"[bridge] Isaac Sim joints: {joint_names}")
        sim_app.close()
        sys.exit(1)

    # -- DDS --
    dp = DomainParticipant(domain_id=args.domain_id)
    state_writer = DataWriter(dp, Topic(dp, args.state_topic, RobotStateDDS))
    command_reader = DataReader(dp, Topic(dp, args.command_topic, JointCommandDDS))

    print(f"DDS domain={args.domain_id}")
    print(f"  publishing : {args.state_topic}")
    print(f"  subscribing: {args.command_topic}")
    print(f"Physics: {args.sim_dt}s dt, decimation={args.decimation}, "
          f"control={1.0/(args.sim_dt*args.decimation):.0f} Hz")
    print("Running ... (Ctrl-C to stop)\n")

    step_count = 0
    has_policy = False
    running = True
    try:
        while running:
            # 1. Read robot state
            try:
                jpos_isim = robot.get_joint_positions()
                jvel_isim = robot.get_joint_velocities()
                root_pos, root_quat = robot.get_world_pose()
                root_lin = robot.get_linear_velocity()
                root_ang = robot.get_angular_velocity()
            except Exception:
                world.step(render=False)
                continue

            # 2. Transform world-frame velocities to body frame
            # MuJoCo qvel[0:6] is body-frame; Isaac Sim returns world-frame.
            # Rotate by inverse of root quaternion (quat_rotate_inverse).
            w, x, y, z = root_quat
            # Inverse quaternion rotation: v_body = q* @ v_world @ q
            qw, qx, qy, qz = w, -x, -y, -z  # conjugate
            def qrot(v):
                t = 2.0 * np.cross(np.array([qx, qy, qz]), v)
                return v + qw * t + np.cross(np.array([qx, qy, qz]), t)

            root_lin_body = qrot(root_lin)
            root_ang_body = qrot(root_ang)

            # Remap to MuJoCo order + add default_angles offset
            jpos_mj = jpos_isim[mj_to_isim] + da_mj[:n_dofs]
            jvel_mj = jvel_isim[mj_to_isim]
            qpos = np.concatenate([root_pos, root_quat, jpos_mj])
            qvel = np.concatenate([root_lin_body, root_ang_body, jvel_mj])

            # 3. Publish state
            t_pub = time.monotonic()
            state_writer.write(RobotStateDDS(
                qpos=qpos.tolist(), qvel=qvel.tolist(),
                timestamp_ns=int(time.time() * 1e9)))

            # 4. Read command (wait briefly if policy is connected)
            samples = None
            if has_policy:
                # Wait up to 50ms for the policy to respond
                deadline = time.monotonic() + 0.05
                while time.monotonic() < deadline:
                    samples = command_reader.take(N=100)
                    if samples:
                        break
                    time.sleep(0.001)
            else:
                samples = command_reader.take(N=100)
                if samples:
                    has_policy = True
                    print(f"[bridge] Policy connected t={time.monotonic():.3f}")

            # 5. Apply command
            if samples:
                cmd = samples[-1]
                targets_mj = np.array(cmd.target_positions, dtype=np.float32)
                isim_targets = targets_mj[:29] - da_mj[:29].astype(np.float32)

                targets_isim = np.zeros(n_dofs, dtype=np.float32)
                for i in range(n_dofs):
                    targets_isim[i] = isim_targets[isim_to_mj[i]]

                if not hasattr(main, '_cmd_count'):
                    main._cmd_count = 0
                main._cmd_count += 1
                if main._cmd_count <= 10:
                    t_cmd = time.monotonic()
                    dt_roundtrip = t_cmd - t_pub
                    print(f"[bridge] CMD#{main._cmd_count} t={t_cmd:.3f} "
                          f"rt={dt_roundtrip*1000:.1f}ms "
                          f"z={root_pos[2]:.3f} qw={root_quat[0]:.4f} "
                          f"jpos[0:3]={np.round(jpos_isim[:3], 4).tolist()} "
                          f"jvel[0:3]={np.round(jvel_isim[:3], 4).tolist()} "
                          f"tgt[0:3]={np.round(targets_isim[:3], 4).tolist()}")

                robot.apply_action(ArticulationAction(
                    joint_positions=targets_isim))

            # 6. Step physics
            t_pre_step = time.monotonic()
            for _ in range(args.decimation):
                world.step(render=False)
            t_post_step = time.monotonic()
            step_count += 1

            if step_count <= 5:
                print(f"[bridge] STEP#{step_count} t={t_post_step:.3f} "
                      f"physics={((t_post_step-t_pre_step)*1000):.1f}ms "
                      f"total_loop={((t_post_step-t_pub)*1000):.1f}ms")

            if step_count == 1 or step_count % 250 == 0:
                t = time.monotonic()
                print(f"[bridge] step={step_count} t={t:.3f} z={root_pos[2]:.3f} "
                      f"qw={root_quat[0]:.4f} policy={has_policy}")

    except KeyboardInterrupt:
        print("\nStopping ...")
    finally:
        sim_app.close()
        print("Isaac Sim closed.")


if __name__ == "__main__":
    main()
