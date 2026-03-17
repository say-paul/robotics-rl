"""Native MuJoCo viewer for the G1 simulation with keyboard controls.

Runs the full physics + optional ONNX policy in the interactive MuJoCo
desktop viewer (GLFW).  Designed to run on a virtual X display exposed
through VNC so it can be viewed remotely.

Keyboard shortcuts (press in the MuJoCo viewer window):
    P         Toggle policy ON/OFF
    M         Start/stop mission execution (requires --mission)
    7         Raise harness height
    8         Lower harness height
    9         Release harness (free standing)
    H         Harness: full support (reset to top)
    L         Harness: auto-lower (gradually release over 10s)
    R         Reset robot to initial pose
    W / S     Increase / decrease forward velocity (vx)
    A / D     Increase / decrease turn rate (vyaw)
    Q / E     Increase / decrease sideways velocity (vy)
    0         Zero all velocity commands

Usage:
    DISPLAY=:99 python3 vnc/sim_viewer.py --policy policies/g1_walk_v5.onnx
    DISPLAY=:99 python3 vnc/sim_viewer.py --mission missions/stand_walk_50m.yaml
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("LP_NUM_THREADS", "2")

import argparse
import math
import time
from pathlib import Path
from typing import Dict, Optional

import mujoco
import mujoco.viewer
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from actions.joints import (
    ACTION_SCALE,
    DEFAULT_POSITIONS,
    KD_DEFAULT,
    KP_DEFAULT,
    NUM_MOTORS,
)


def build_obs(data, default_pos, last_action, command, phase):
    """Build the 103-dim observation vector."""
    quat = data.qpos[3:7].copy()
    w, x, y, z = quat

    gravity = np.array([
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        -(w * w - x * x - y * y + z * z),
    ], dtype=np.float32)

    rot = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rot, quat)
    rot = rot.reshape(3, 3)
    linvel = (rot.T @ data.qvel[:3]).astype(np.float32)
    gyro = data.qvel[3:6].astype(np.float32)

    joint_pos = data.qpos[7:].astype(np.float32)
    joint_vel = data.qvel[6:].astype(np.float32)

    cos_p = np.cos(phase).astype(np.float32)
    sin_p = np.sin(phase).astype(np.float32)

    return np.concatenate([
        linvel, gyro, gravity, command,
        joint_pos - default_pos, joint_vel,
        last_action, cos_p, sin_p,
    ]).reshape(1, -1)


def get_robot_pose(data):
    """Extract (x, y, yaw) from MuJoCo data."""
    x, y = float(data.qpos[0]), float(data.qpos[1])
    w, qx, qy, qz = data.qpos[3:7]
    yaw = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return x, y, float(yaw)


class State:
    """Mutable state shared between the key callback and the sim loop."""
    def __init__(self):
        self.policy_active = False
        self.harness_enabled = True
        self.harness_height = 1.5
        self.harness_stiffness = 300.0
        self.harness_damping = 150.0
        self.harness_auto = False
        self.harness_auto_t0 = 0.0
        self.harness_auto_start_h = 1.5
        self.harness_auto_start_k = 300.0
        self.reset_requested = False
        self.mission_toggle = False
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0

    def status_text(self):
        pol = "ON" if self.policy_active else "OFF"
        h = "AUTO-LOWER" if self.harness_auto else ("ON" if self.harness_enabled else "OFF")
        return (f"Policy={pol}  Harness={h}(h={self.harness_height:.2f})  "
                f"vx={self.vx:.2f} vy={self.vy:.2f} vyaw={self.vyaw:.2f}")


def make_key_callback(state, has_policy, has_mission):
    VEL_STEP = 0.1
    HEIGHT_STEP = 0.05

    def key_callback(keycode):
        if keycode == ord('P') and has_policy:
            state.policy_active = not state.policy_active
            print(f"[key] Policy {'ON' if state.policy_active else 'OFF'}")
        elif keycode == ord('M') and has_mission:
            state.mission_toggle = True
            print("[key] Mission toggle requested")
        elif keycode == ord('7'):
            state.harness_auto = False
            state.harness_enabled = True
            state.harness_height = min(state.harness_height + HEIGHT_STEP, 3.0)
            if state.harness_stiffness <= 0:
                state.harness_stiffness = 300.0
                state.harness_damping = 150.0
            print(f"[key] Harness UP  h={state.harness_height:.2f}")
        elif keycode == ord('8'):
            state.harness_auto = False
            state.harness_enabled = True
            state.harness_height = max(state.harness_height - HEIGHT_STEP, 0.0)
            if state.harness_stiffness <= 0:
                state.harness_stiffness = 300.0
                state.harness_damping = 150.0
            print(f"[key] Harness DOWN  h={state.harness_height:.2f}")
        elif keycode == ord('9'):
            state.harness_auto = False
            state.harness_enabled = False
            state.harness_stiffness = 0.0
            print("[key] Harness: RELEASED")
        elif keycode == ord('H'):
            state.harness_auto = False
            state.harness_enabled = True
            state.harness_height = 1.5
            state.harness_stiffness = 300.0
            state.harness_damping = 150.0
            print("[key] Harness: FULL SUPPORT")
        elif keycode == ord('L'):
            state.harness_auto = True
            state.harness_auto_t0 = time.monotonic()
            state.harness_auto_start_h = state.harness_height
            state.harness_auto_start_k = state.harness_stiffness
            state.harness_enabled = True
            print("[key] Harness: AUTO-LOWERING (10s)")
        elif keycode == ord('G'):
            state.harness_auto = False
            state.harness_enabled = False
            state.harness_stiffness = 0.0
            print("[key] Harness: RELEASED")
        elif keycode == ord('R'):
            state.reset_requested = True
            print("[key] RESET")
        elif keycode == ord('W'):
            state.vx = round(state.vx + VEL_STEP, 2)
            print(f"[key] vx={state.vx:.2f}")
        elif keycode == ord('S'):
            state.vx = round(state.vx - VEL_STEP, 2)
            print(f"[key] vx={state.vx:.2f}")
        elif keycode == ord('A'):
            state.vyaw = round(state.vyaw + VEL_STEP, 2)
            print(f"[key] vyaw={state.vyaw:.2f}")
        elif keycode == ord('D'):
            state.vyaw = round(state.vyaw - VEL_STEP, 2)
            print(f"[key] vyaw={state.vyaw:.2f}")
        elif keycode == ord('Q'):
            state.vy = round(state.vy + VEL_STEP, 2)
            print(f"[key] vy={state.vy:.2f}")
        elif keycode == ord('E'):
            state.vy = round(state.vy - VEL_STEP, 2)
            print(f"[key] vy={state.vy:.2f}")
        elif keycode == ord('0'):
            state.vx = state.vy = state.vyaw = 0.0
            print("[key] Velocities zeroed")

    return key_callback


def harness_force(state, pos, vel):
    """Compute harness spring-damper force on the torso."""
    if state.harness_auto:
        elapsed = time.monotonic() - state.harness_auto_t0
        duration = 10.0
        t = min(elapsed / duration, 1.0)
        t = t * t * (3 - 2 * t)  # smooth ease
        state.harness_height = state.harness_auto_start_h + (0.78 - state.harness_auto_start_h) * t
        state.harness_stiffness = state.harness_auto_start_k * (1 - t)
        state.harness_damping = state.harness_stiffness * 0.5
        if t >= 1.0:
            state.harness_auto = False
            state.harness_enabled = False
            state.harness_stiffness = 0.0

    if not state.harness_enabled or state.harness_stiffness <= 0:
        return np.zeros(3)

    anchor = np.array([pos[0], pos[1], state.harness_height])
    delta = anchor - pos
    dist = np.linalg.norm(delta)
    if dist < 1e-6:
        return np.zeros(3)

    direction = delta / dist
    v_along = np.dot(vel, direction)
    force = (state.harness_stiffness * dist - state.harness_damping * v_along) * direction
    if force[2] < 0:
        force[2] = 0.0
    return force


def load_onnx_session(policy_path: str, device: str):
    """Load an ONNX policy and return (session, input_name)."""
    import onnxruntime as ort
    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 1
    sess_opts.intra_op_num_threads = 2
    providers = [
        ("OpenVINOExecutionProvider", {"device_type": device}),
        "CPUExecutionProvider",
    ]
    session = ort.InferenceSession(policy_path, sess_options=sess_opts, providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name


def main():
    parser = argparse.ArgumentParser(description="MuJoCo native viewer for G1")
    parser.add_argument("--scene", default="unitree_robots/g1/scene_29dof.xml")
    parser.add_argument("--policy", default=None, help="Path to .onnx policy (single-policy mode)")
    parser.add_argument("--mission", default=None, help="Path to mission .yaml (multi-policy mode)")
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU", "NPU"])
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--vyaw", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.005)
    args = parser.parse_args()

    if args.mission and args.policy:
        parser.error("Use --mission or --policy, not both")

    # --- Load mission if provided ---
    mission = None
    mission_runner = None
    policy_sessions: Dict[str, tuple] = {}  # behavior -> (session, input_name)
    active_behavior: Optional[str] = None

    if args.mission:
        from planner.mission import Mission
        from planner.mission_runner import MissionRunner, MissionState
        import logging
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

        mission = Mission.from_yaml(args.mission)
        mission_runner = MissionRunner(mission)
        print(f"[viewer] Mission loaded: '{mission.name}' ({len(mission.waypoints)} waypoints)")

        for behavior, policy_path in mission.policies.items():
            sess, inp_name = load_onnx_session(policy_path, args.device)
            policy_sessions[behavior] = (sess, inp_name)
            print(f"[viewer]   {behavior} -> {policy_path} ({sess.get_providers()[0]})")

    # --- Physics setup ---
    model = mujoco.MjModel.from_xml_path(args.scene)
    model.opt.timestep = args.dt
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    kp = np.array(KP_DEFAULT, dtype=np.float64)
    kd = np.array(KD_DEFAULT, dtype=np.float64)
    default_pos = np.array(DEFAULT_POSITIONS, dtype=np.float64)
    default_pos_f32 = default_pos.astype(np.float32)
    torque_lo = model.actuator_ctrlrange[:, 0].copy()
    torque_hi = model.actuator_ctrlrange[:, 1].copy()
    initial_qpos = data.qpos.copy()

    # --- Single-policy mode ---
    session = None
    input_name = None
    if args.policy:
        session, input_name = load_onnx_session(args.policy, args.device)
        print(f"[viewer] Policy loaded: {args.policy} on {session.get_providers()}")

    has_policy = session is not None or len(policy_sessions) > 0

    state = State()
    state.vx = args.vx
    state.vy = args.vy
    state.vyaw = args.vyaw

    last_action = np.zeros(NUM_MOTORS, dtype=np.float32)
    phase = np.array([0.0, math.pi], dtype=np.float64)
    phase_dt = 2.0 * math.pi * 0.02 * 1.25

    policy_dt = 0.02
    substeps = round(policy_dt / args.dt)
    render_every = 5
    q_target = default_pos.copy()
    step_count = 0
    policy_count = 0
    torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    mission_active = False
    progress_interval = 50  # print mission progress every N policy steps (1s)

    print(f"[viewer] Launching MuJoCo viewer (policy={'loaded' if session else 'mission' if mission else 'none'})")
    print(f"[viewer] Render rate: ~{1.0/(policy_dt * render_every):.0f} Hz  (software OpenGL / llvmpipe)")
    print("[viewer] Keys: P=policy  M=mission  7=raise  8=lower  9=release  H=full-support  R=reset")
    print("[viewer]       W/S=fwd/back  A/D=turn  Q/E=strafe  0=stop")

    with mujoco.viewer.launch_passive(
        model, data,
        key_callback=make_key_callback(state, has_policy, mission is not None),
    ) as viewer:
        frame_start = time.perf_counter()

        while viewer.is_running():

            # --- Reset ---
            if state.reset_requested:
                state.reset_requested = False
                data.qpos[:] = initial_qpos
                data.qvel[:] = 0
                data.ctrl[:] = 0
                last_action[:] = 0
                phase[:] = [0.0, math.pi]
                q_target[:] = default_pos
                step_count = 0
                policy_count = 0
                mujoco.mj_forward(model, data)
                state.harness_enabled = True
                state.harness_height = 1.5
                state.harness_stiffness = 300.0
                state.harness_damping = 150.0
                state.harness_auto = False
                state.policy_active = False
                mission_active = False
                active_behavior = None
                if mission_runner is not None:
                    mission_runner.state = MissionState.IDLE
                viewer.sync()
                print("[viewer] Reset complete")
                continue

            # --- Mission toggle ---
            if state.mission_toggle:
                state.mission_toggle = False
                if mission_runner is not None:
                    if not mission_active:
                        mission_active = True
                        state.policy_active = True
                        mission_runner.start()
                        active_behavior = mission_runner.current_behavior
                        last_action[:] = 0
                        phase[:] = [0.0, math.pi]
                        print(f"[mission] STARTED: '{mission.name}' -> behavior={active_behavior}")
                    else:
                        mission_active = False
                        state.policy_active = False
                        active_behavior = None
                        print("[mission] STOPPED")

            # --- Determine commands and active session ---
            cur_session = session
            cur_input_name = input_name

            if mission_active and mission_runner is not None:
                rx, ry, ryaw = get_robot_pose(data)
                cmd = mission_runner.step(rx, ry, ryaw, dt=policy_dt)

                # Policy switching on behavior change
                new_behavior = cmd.behavior
                if new_behavior != active_behavior:
                    print(f"[mission] Switching policy: {active_behavior} -> {new_behavior}")
                    active_behavior = new_behavior
                    last_action[:] = 0
                    phase[:] = [0.0, math.pi]

                if active_behavior in policy_sessions:
                    cur_session, cur_input_name = policy_sessions[active_behavior]

                command = np.array([cmd.vx, cmd.vy, cmd.vyaw], dtype=np.float32)

                # Check mission completion
                if mission_runner.state in (MissionState.ARRIVED, MissionState.FAILED):
                    status = "COMPLETE" if mission_runner.state == MissionState.ARRIVED else "FAILED"
                    rx, ry, _ = get_robot_pose(data)
                    dist = math.sqrt(rx * rx + ry * ry)
                    print(f"[mission] {status} -- final pos=({rx:.1f}, {ry:.1f}), dist_from_origin={dist:.1f}m")
                    mission_active = False
                    if "stand" in policy_sessions:
                        active_behavior = "stand"
                        cur_session, cur_input_name = policy_sessions["stand"]
                        print("[mission] Switched to stand policy")
                    command = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                # Progress telemetry
                if policy_count % progress_interval == 0 and mission_active:
                    wp = mission_runner.current_waypoint
                    if wp is not None:
                        dist_to_wp = wp.distance_to(rx, ry)
                        print(f"[mission] {mission_runner.progress} "
                              f"behavior={active_behavior} "
                              f"wp={wp.id} dist={dist_to_wp:.1f}m "
                              f"pos=({rx:.1f},{ry:.1f}) "
                              f"cmd=({cmd.vx:.2f},{cmd.vy:.2f},{cmd.vyaw:.2f})")
            else:
                command = np.array([state.vx, state.vy, state.vyaw], dtype=np.float32)

            # --- Policy inference ---
            if state.policy_active and cur_session is not None:
                obs = build_obs(data, default_pos_f32, last_action, command, phase)
                action = cur_session.run(None, {cur_input_name: obs})[0].flatten()
                action = np.clip(action, -1.0, 1.0)
                last_action = action.astype(np.float32)
                q_target = default_pos + action * ACTION_SCALE
                phase += phase_dt
                phase = np.fmod(phase + math.pi, 2 * math.pi) - math.pi

            # --- Physics substeps ---
            for _ in range(substeps):
                q = data.qpos[7:]
                dq = data.qvel[6:]
                tau = kp * (q_target - q) - kd * dq
                data.ctrl[:] = np.clip(tau, torque_lo, torque_hi)

                if torso_body_id >= 0:
                    pos = data.xpos[torso_body_id].copy()
                    vel = data.cvel[torso_body_id, 3:].copy()
                    force = harness_force(state, pos, vel)
                    data.xfrc_applied[torso_body_id, :3] = force

                mujoco.mj_step(model, data)
                step_count += 1

            policy_count += 1

            if policy_count % render_every == 0:
                viewer.sync()

            elapsed = time.perf_counter() - frame_start
            sleep_time = policy_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            frame_start = time.perf_counter()


if __name__ == "__main__":
    main()
