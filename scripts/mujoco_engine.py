#!/usr/bin/env python3
"""
Unified MuJoCo simulation engine.

This module is completely agnostic to the robot or policy being used.
It handles only simulation concerns:
  - Loading a MuJoCo scene
  - Virtual harness (spring-damper support)
  - PD torque control
  - Viewer + keyboard interaction
  - Simulation stepping

The "brain" is a PolicyRunner (from model_runner.py) that the caller
injects.  The engine calls runner.step(model, data) at the policy rate
and applies the returned (target, kp, kd) via PD control at sim rate.

Usage:
    from mujoco_engine import run_simulation
    from model_runner import GraphPolicyRunner

    runner = GraphPolicyRunner(graph, bus, config)
    run_simulation("scene.xml", runner, sim_dt=0.002, decimation=10)
"""

import time
import threading
import numpy as np
import mujoco
import mujoco.viewer


# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------

class ControlState:
    def __init__(self):
        self.lock = threading.Lock()
        self.policy_enabled = False


class VirtualHarness:
    """Spring-damper that holds the robot torso at a fixed anchor height."""

    def __init__(self, height=0.75, stiffness=3000.0, damping=200.0):
        self.height = height
        self.stiffness = stiffness
        self.damping = damping
        self.enabled = True
        self.lock = threading.Lock()
        self._lowering = False
        self._lower_start_time = None
        self._lower_duration = 0
        self._initial_height = height
        self._initial_stiffness = stiffness

    def compute_force(self, pos, vel):
        with self.lock:
            if not self.enabled or self.stiffness <= 0:
                return np.zeros(3)
            anchor = np.array([pos[0], pos[1], self.height])
            delta = anchor - pos
            dist = np.linalg.norm(delta)
            if dist < 1e-6:
                return np.zeros(3)
            direction = delta / dist
            v_along = np.dot(vel, direction)
            force = (self.stiffness * dist - self.damping * v_along) * direction
            if force[2] < 0:
                force[2] = 0.0
            return force

    def adjust_height(self, delta):
        with self.lock:
            self.height = np.clip(self.height + delta, 0.0, 3.0)
            self._lowering = False

    def start_auto_lower(self, duration=5.0):
        with self.lock:
            self._lowering = True
            self._lower_start_time = time.time()
            self._lower_duration = duration
            self._initial_height = self.height
            self._initial_stiffness = self.stiffness
        print(f"[Harness] Releasing over {duration}s ...")

    def update(self):
        with self.lock:
            if not self._lowering:
                return
            progress = min((time.time() - self._lower_start_time) / self._lower_duration, 1.0)
            self.stiffness = self._initial_stiffness * (1.0 - progress)
            if progress >= 1.0:
                self._lowering = False
                self.enabled = False
                self.stiffness = 0.0
                print("[Harness] Released")


def pd_control(q_target, q, dq, kp, kd):
    return kp * (q_target - q) - kd * dq


# ---------------------------------------------------------------------------
# Unified simulation loop
# ---------------------------------------------------------------------------

def run_simulation(scene_xml, policy_runner, *, sim_dt, decimation,
                   harness_cfg=None, controller=None):
    """
    Run a MuJoCo simulation with any PolicyRunner.

    Parameters
    ----------
    scene_xml : str
        Path to MuJoCo scene XML.
    policy_runner : PolicyRunner
        Pluggable policy brain (decoupled, SONIC, etc.).
    sim_dt : float
        Simulation timestep in seconds.
    decimation : int
        Policy runs every `decimation` sim steps.
    harness_cfg : dict, optional
        Override harness params (initial_height, stiffness, damping).
    controller : KeyboardController or None
        Optional runtime controller that maps keyboard input to commands.
    """
    harness_cfg = harness_cfg or {}

    # -- Load MuJoCo scene --
    print(f"  Loading scene: {scene_xml}")
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    model.opt.timestep = sim_dt
    n_joints = data.qpos.shape[0] - 7

    # -- Let the policy runner initialize (load ONNX, set initial qpos, etc.) --
    n_actuated = policy_runner.setup(model, data)

    # -- Harness --
    control_state = ControlState()
    harness = VirtualHarness(
        height=harness_cfg.get("initial_height", 0.75),
        stiffness=harness_cfg.get("stiffness", 3000.0),
        damping=harness_cfg.get("damping", 200.0),
    )
    harness_body = harness_cfg.get("body", "torso_link")
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, harness_body)

    # -- Attach controller to policy runner if present --
    if controller is not None:
        policy_runner.set_controller(controller)

    # -- PD state --
    target_pos = np.zeros(n_actuated, dtype=np.float32)
    kps = np.zeros(n_actuated, dtype=np.float32)
    kds = np.zeros(n_actuated, dtype=np.float32)

    ctrl_keys = controller.all_keycodes if controller else set()

    def key_callback(keycode):
        if controller is not None and controller.handle_key(keycode):
            return
        if keycode == ord('7') and ord('7') not in ctrl_keys:
            harness.adjust_height(0.05)
            print(f"[Harness] Ascend -> {harness.height:.2f}m")
        elif keycode == ord('8') and ord('8') not in ctrl_keys:
            harness.adjust_height(-0.05)
            print(f"[Harness] Descend -> {harness.height:.2f}m")
        elif keycode == ord('9') and ord('9') not in ctrl_keys:
            harness.start_auto_lower(duration=5.0)
        elif keycode in (ord('P'), ord('p')):
            with control_state.lock:
                control_state.policy_enabled = not control_state.policy_enabled
                status = "ENABLED" if control_state.policy_enabled else "DISABLED"
            print(f"[Policy] {status}")
        elif keycode == ord('H') or keycode == ord('h'):
            harness.adjust_height(0.05)
            print(f"[Harness] Ascend -> {harness.height:.2f}m")
        elif keycode == ord('J') or keycode == ord('j'):
            harness.adjust_height(-0.05)
            print(f"[Harness] Descend -> {harness.height:.2f}m")
        elif keycode == ord('K') or keycode == ord('k'):
            harness.start_auto_lower(duration=5.0)

    # -- Banner --
    policy_hz = 1.0 / (sim_dt * decimation)
    print("\n" + "=" * 60)
    print(f"  {policy_runner.name}")
    print("=" * 60)
    print(f"  Sim: {1/sim_dt:.0f} Hz | Policy: {policy_hz:.0f} Hz | Decimation: {decimation}")
    print(f"  Joints: {n_joints} total, {n_actuated} actuated")
    for line in policy_runner.info_lines():
        print(f"  {line}")
    if controller is not None:
        print()
        for line in controller.banner_lines():
            print(f"  {line}")
    print(f"\n  Robot starts HANGING on harness (no policy)")
    print(f"  Keys:")
    print(f"    P  - toggle policy")
    if controller is not None:
        print(f"    H  - raise harness")
        print(f"    J  - lower harness")
        print(f"    K  - release harness (5s fade)")
    else:
        print(f"    9  - release harness (5s fade)")
        print(f"    7  - raise harness")
        print(f"    8  - lower harness")
    print("=" * 60 + "\n")

    # -- Simulation loop --
    counter = 0
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            harness.update()
            data.xfrc_applied[torso_id, :3] = harness.compute_force(
                data.qpos[:3].copy(), data.qvel[:3].copy()
            )

            with control_state.lock:
                policy_active = control_state.policy_enabled

            if policy_active:
                tau = pd_control(
                    target_pos[:n_actuated],
                    data.qpos[7:7+n_actuated],
                    data.qvel[6:6+n_actuated],
                    kps[:n_actuated],
                    kds[:n_actuated],
                )
                data.ctrl[:n_actuated] = tau
            else:
                data.ctrl[:] = 0.0

            mujoco.mj_step(model, data)

            counter += 1
            if counter % decimation == 0 and policy_active:
                target_pos, kps, kds = policy_runner.step(model, data)

            viewer.sync()

    print("\nSimulation ended.")
