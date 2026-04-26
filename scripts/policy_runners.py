#!/usr/bin/env python3
"""
Pluggable policy runners for MuJoCo simulation.

Each runner encapsulates everything about a specific policy architecture:
  - ONNX model loading
  - Observation building
  - Inference
  - Action → joint target mapping

The MuJoCo engine is completely agnostic to the policy; it just calls
  runner.step(model, data) → (target_pos, kps, kds)
and applies PD control.
"""

import threading
import numpy as np
import onnxruntime as ort
from abc import ABC, abstractmethod
from collections import deque


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def quat_rotate_inverse(quat, v):
    w, x, y, z = quat
    qw, qx, qy, qz = w, -x, -y, -z
    return np.array([
        v[0]*(qw**2+qx**2-qy**2-qz**2) + v[1]*2*(qx*qy-qw*qz) + v[2]*2*(qx*qz+qw*qy),
        v[0]*2*(qx*qy+qw*qz) + v[1]*(qw**2-qx**2+qy**2-qz**2) + v[2]*2*(qy*qz-qw*qx),
        v[0]*2*(qx*qz-qw*qy) + v[1]*2*(qy*qz+qw*qx) + v[2]*(qw**2-qx**2-qy**2+qz**2),
    ], dtype=np.float32)


def get_gravity_orientation(quat):
    return quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))


def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def quat_to_rot_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ], dtype=np.float64)


def anchor_orientation_6d(base_quat, ref_quat=None):
    """6D rotation: first 2 columns of rot matrix, row-wise.
    Matches C++ GatherMotionAnchorOrientationMutiFrame."""
    if ref_quat is None:
        ref_quat = np.array([1.0, 0.0, 0.0, 0.0])
    rel = quat_multiply(quat_conjugate(base_quat), ref_quat)
    R = quat_to_rot_matrix(rel)
    return np.array([R[0,0], R[0,1], R[1,0], R[1,1], R[2,0], R[2,1]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class PolicyRunner(ABC):
    """Interface between the MuJoCo engine and a specific policy architecture."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for status prints."""

    @abstractmethod
    def setup(self, model, data):
        """Called once after MuJoCo model/data are created.
        Use this to set initial qpos, load ONNX sessions, etc.
        Must return n_actuated (int) — how many joints PD control covers."""

    @abstractmethod
    def step(self, model, data):
        """Called every `decimation` sim-steps when the policy is active.
        Returns (target_pos, kps, kds) — all np.arrays of length n_actuated."""

    def set_controller(self, controller):
        """Attach a runtime controller. Override in subclasses that use it."""
        pass

    def info_lines(self):
        """Extra lines to print in the startup banner."""
        return []


# ---------------------------------------------------------------------------
# Keyboard controller (reads YAML bindings, mutates command state)
# ---------------------------------------------------------------------------

# GLFW key constants for arrow keys
_GLFW_KEY_RIGHT = 262
_GLFW_KEY_LEFT  = 263
_GLFW_KEY_DOWN  = 264
_GLFW_KEY_UP    = 265

_KEY_NAME_TO_GLFW = {
    "UP": _GLFW_KEY_UP,
    "DOWN": _GLFW_KEY_DOWN,
    "LEFT": _GLFW_KEY_LEFT,
    "RIGHT": _GLFW_KEY_RIGHT,
}


class KeyboardController:
    """Runtime controller that maps keyboard inputs to SONIC planner commands.

    Reads bindings from the ``controller`` section of the robot YAML.
    The MuJoCo engine calls ``handle_key(keycode)`` from its viewer
    key-callback (press events only); the policy runner reads ``command``.

    Since MuJoCo launch_passive only fires on key *press* (no release
    events), movement works as a toggle: press an arrow key to start
    moving, press 0 (IDLE) to stop.
    """

    def __init__(self, ctrl_cfg):
        modes = ctrl_cfg.get("locomotion_modes", {})
        self._mode_map = {str(k): int(v) for k, v in modes.items()}

        default = ctrl_cfg.get("default_command", {})
        self._default = {
            "locomotion_mode": self._resolve_mode(default.get("locomotion_mode", "IDLE")),
            "movement_direction": list(default.get("movement_direction", [0, 0, 0])),
            "facing_direction": list(default.get("facing_direction", [1, 0, 0])),
            "movement_speed": float(default.get("movement_speed", -1)),
            "height": float(default.get("height", -1)),
        }

        kb = ctrl_cfg.get("keyboard", {})
        self._movement_bindings = {}
        for key_name, binding in kb.get("movement", {}).items():
            if key_name == "release":
                continue
            glfw_code = _KEY_NAME_TO_GLFW.get(key_name.upper())
            if glfw_code is not None:
                self._movement_bindings[glfw_code] = self._parse_binding(binding)

        self._style_bindings = {}
        for key_char, mode_name in kb.get("styles", {}).items():
            code = ord(str(key_char))
            self._style_bindings[code] = self._resolve_mode(mode_name)

        self._lock = threading.Lock()
        self._command = dict(self._default)
        self._active_style = self._resolve_mode(
            default.get("locomotion_mode", "IDLE")
        )

    def _resolve_mode(self, name_or_int):
        if isinstance(name_or_int, int):
            return name_or_int
        return self._mode_map.get(str(name_or_int), 0)

    def _parse_binding(self, binding):
        result = {}
        if "movement_direction" in binding:
            result["movement_direction"] = [float(x) for x in binding["movement_direction"]]
        if "locomotion_mode" in binding:
            result["locomotion_mode"] = self._resolve_mode(binding["locomotion_mode"])
        return result

    @property
    def command(self):
        with self._lock:
            return dict(self._command)

    @property
    def all_keycodes(self):
        """Set of keycodes this controller handles (engine skips harness for these)."""
        codes = set(self._movement_bindings.keys())
        codes |= set(self._style_bindings.keys())
        return codes

    def handle_key(self, keycode):
        """Called on key press. Returns True if the key was consumed."""
        with self._lock:
            if keycode in self._movement_bindings:
                b = self._movement_bindings[keycode]
                if "movement_direction" in b:
                    self._command["movement_direction"] = list(b["movement_direction"])
                self._command["locomotion_mode"] = b.get(
                    "locomotion_mode", self._active_style
                )
                self._print_state()
                return True

            if keycode in self._style_bindings:
                self._active_style = self._style_bindings[keycode]
                self._command["locomotion_mode"] = self._active_style
                if self._active_style == self._default["locomotion_mode"]:
                    self._command["movement_direction"] = list(
                        self._default["movement_direction"]
                    )
                self._print_state()
                return True

        return False

    def _print_state(self):
        inv = {v: k for k, v in self._mode_map.items()}
        mode_name = inv.get(self._command["locomotion_mode"],
                            str(self._command["locomotion_mode"]))
        d = self._command["movement_direction"]
        print(f"[Controller] mode={mode_name}  dir=[{d[0]:.1f},{d[1]:.1f},{d[2]:.1f}]")

    def banner_lines(self):
        lines = [
            "Keyboard controller ACTIVE:",
            "  Arrow keys : move (forward/back/left/right)",
        ]
        inv = {v: k for k, v in self._mode_map.items()}
        for char_code, mode_id in sorted(self._style_bindings.items()):
            name = inv.get(mode_id, str(mode_id))
            suffix = " (stop)" if name == "IDLE" else ""
            lines.append(f"  {chr(char_code)}  - {name}{suffix}")
        return lines


# ---------------------------------------------------------------------------
# Decoupled WBC (15 lower-body + passive arms)
# ---------------------------------------------------------------------------

class DecoupledPolicyRunner(PolicyRunner):
    """GR00T decoupled WBC: single ONNX, 86-dim obs × 6 history → 15 actions."""

    def __init__(self, policy_path, cfg):
        self._policy_path = policy_path
        self._cfg = cfg

    @property
    def name(self):
        return "Decoupled WBC"

    def setup(self, model, data):
        cfg = self._cfg
        self._num_actions = int(cfg["num_actions"])
        self._single_obs_dim = int(cfg["single_obs_dim"])
        self._obs_hist_len = int(cfg["obs_history_len"])
        self._total_obs_dim = int(cfg["num_obs"])

        self._default_angles = np.array(cfg["default_angles"], dtype=np.float32)
        leg_kps = np.array(cfg["kps"], dtype=np.float32)
        leg_kds = np.array(cfg["kds"], dtype=np.float32)
        self._action_scale = float(cfg["action_scale"])
        self._dof_pos_scale = float(cfg["dof_pos_scale"])
        self._dof_vel_scale = float(cfg["dof_vel_scale"])
        self._ang_vel_scale = float(cfg["ang_vel_scale"])
        self._cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
        self._cmd_init = np.array(cfg["cmd_init"], dtype=np.float32)
        self._height_cmd = float(cfg["height_cmd"])
        self._rpy_cmd = np.array(cfg["rpy_cmd"], dtype=np.float32)
        self._cmd_scaled = self._cmd_init * self._cmd_scale

        arm_kp = float(cfg["arm_kp"])
        arm_kd = float(cfg["arm_kd"])

        n_joints = data.qpos.shape[0] - 7
        n_arm = n_joints - self._num_actions

        # Build unified per-joint gain arrays (legs + arms)
        self._kps = np.concatenate([leg_kps, np.full(max(n_arm, 0), arm_kp)])
        self._kds = np.concatenate([leg_kds, np.full(max(n_arm, 0), arm_kd)])
        self._n_actuated = min(n_joints, self._num_actions + max(n_arm, 0))

        # Targets: legs → default angles, arms → zero
        self._target = np.zeros(self._n_actuated, dtype=np.float32)
        self._target[:self._num_actions] = self._default_angles[:self._num_actions]

        self._last_action = np.zeros(self._num_actions, dtype=np.float32)
        self._obs_history = deque(
            [np.zeros(self._single_obs_dim, dtype=np.float32)] * self._obs_hist_len,
            maxlen=self._obs_hist_len,
        )
        self._full_obs = np.zeros(self._total_obs_dim, dtype=np.float32)

        # ONNX
        print(f"  Loading ONNX policy: {self._policy_path}")
        self._session = ort.InferenceSession(self._policy_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

        return self._n_actuated

    def step(self, model, data):
        n_joints = data.qpos.shape[0] - 7
        obs = self._build_obs(data, n_joints)
        self._obs_history.append(obs)

        for i, h in enumerate(self._obs_history):
            s = self._single_obs_dim
            self._full_obs[i*s:(i+1)*s] = h

        action = self._session.run(
            None, {self._input_name: self._full_obs.reshape(1, -1)}
        )[0].flatten()
        self._last_action = action.copy()

        self._target[:self._num_actions] = (
            self._default_angles[:self._num_actions] + action * self._action_scale
        )
        return self._target, self._kps, self._kds

    def _build_obs(self, data, n_joints):
        qj = data.qpos[7:7+n_joints].copy()
        dqj = data.qvel[6:6+n_joints].copy()
        quat = data.qpos[3:7].copy()
        omega = data.qvel[3:6].copy()
        gravity = get_gravity_orientation(quat)

        padded = np.zeros(n_joints, dtype=np.float32)
        L = min(len(self._default_angles), n_joints)
        padded[:L] = self._default_angles[:L]
        qj_s = (qj - padded) * self._dof_pos_scale
        dqj_s = dqj * self._dof_vel_scale
        omega_s = omega * self._ang_vel_scale

        obs = np.zeros(self._single_obs_dim, dtype=np.float32)
        cmd = np.zeros(7, dtype=np.float32)
        cmd[:3] = self._cmd_scaled
        cmd[3] = self._height_cmd
        cmd[4:7] = self._rpy_cmd
        obs[0:7] = cmd
        obs[7:10] = omega_s
        obs[10:13] = gravity
        obs[13:13+n_joints] = qj_s
        obs[13+n_joints:13+2*n_joints] = dqj_s
        obs[13+2*n_joints:13+2*n_joints+self._num_actions] = self._last_action
        return obs

    def info_lines(self):
        return [
            f"Obs: {self._single_obs_dim}-dim × {self._obs_hist_len} = {self._total_obs_dim}-dim",
            f"Actions: {self._num_actions} lower-body + passive arms",
        ]


# ---------------------------------------------------------------------------
# SONIC WBC (encoder + decoder, 29 full-body)
# ---------------------------------------------------------------------------

ENCODER_OBS_LAYOUT = [
    ("encoder_mode_4",                              4),
    ("motion_joint_positions_10frame_step5",       290),
    ("motion_joint_velocities_10frame_step5",      290),
    ("motion_root_z_position_10frame_step5",        10),
    ("motion_root_z_position",                       1),
    ("motion_anchor_orientation",                    6),
    ("motion_anchor_orientation_10frame_step5",     60),
    ("motion_joint_positions_lowerbody_10frame_step5", 120),
    ("motion_joint_velocities_lowerbody_10frame_step5", 120),
    ("vr_3point_local_target",                       9),
    ("vr_3point_local_orn_target",                  12),
    ("smpl_joints_10frame_step1",                  720),
    ("smpl_anchor_orientation_10frame_step1",       60),
    ("motion_joint_positions_wrists_10frame_step1", 60),
]
ENCODER_TOTAL_DIM = sum(d for _, d in ENCODER_OBS_LAYOUT)  # 1762


class SonicPolicyRunner(PolicyRunner):
    """SONIC encoder-decoder WBC: 29 full-body joints."""

    def __init__(self, encoder_path, decoder_path, cfg):
        self._encoder_path = encoder_path
        self._decoder_path = decoder_path
        self._cfg = cfg
        self._controller = None

    @property
    def name(self):
        return "SONIC WBC"

    def set_controller(self, controller):
        self._controller = controller

    def setup(self, model, data):
        cfg = self._cfg
        self._num_actions = int(cfg.get("num_actions", 29))
        self._history_frames = int(cfg.get("history_frames", 10))
        self._encoder_mode = int(cfg.get("encoder_mode", 0))
        self._token_dim = int(cfg.get("encoder_output_dim", 64))

        self._default_angles = np.array(cfg["default_angles"], dtype=np.float32)
        self._kps = np.array(cfg["kps"], dtype=np.float32)
        self._kds = np.array(cfg["kds"], dtype=np.float32)
        self._action_scales = np.array(cfg["action_scales"], dtype=np.float32)
        self._obs_scales = cfg.get("obs_scales", {})

        # C++ uses inverted naming: isaaclab_to_mujoco[mj] → il, mujoco_to_isaaclab[il] → mj
        # We swap to standard: mj2il[mj] → il, il2mj[il] → mj
        self._mj2il = cfg.get("isaaclab_to_mujoco", list(range(29)))
        self._il2mj = cfg.get("mujoco_to_isaaclab", list(range(29)))

        n_joints = data.qpos.shape[0] - 7
        self._n_joints = n_joints
        self._n_actuated = min(n_joints, self._num_actions)

        # Initialize robot to standing pose
        import mujoco
        for mj_i in range(self._n_actuated):
            data.qpos[7 + mj_i] = self._default_angles[mj_i]
        mujoco.mj_forward(model, data)
        self._standing_height = data.qpos[2]

        # History buffers
        hf = self._history_frames
        self._ang_vel_hist = np.zeros((hf, 3), dtype=np.float32)
        self._jpos_hist = np.zeros((hf, 29), dtype=np.float32)
        self._jvel_hist = np.zeros((hf, 29), dtype=np.float32)
        self._action_hist = np.zeros((hf, 29), dtype=np.float32)
        self._gravity_hist = np.zeros((hf, 3), dtype=np.float32)
        self._last_action_il = np.zeros(29, dtype=np.float32)
        self._target = self._default_angles.copy()

        # Default command from controller config (or fallback)
        ctrl_cfg = cfg.get("controller", {})
        loco_modes = {str(k): int(v) for k, v in ctrl_cfg.get("locomotion_modes", {}).items()}
        dc = ctrl_cfg.get("default_command", {})
        mode_raw = dc.get("locomotion_mode", "IDLE")
        self._default_locomotion_mode = (
            int(mode_raw) if isinstance(mode_raw, int)
            else loco_modes.get(str(mode_raw), 0)
        )
        self._default_movement_dir = np.array(
            dc.get("movement_direction", [0, 0, 0]), dtype=np.float32
        )
        self._default_facing_dir = np.array(
            dc.get("facing_direction", [1, 0, 0]), dtype=np.float32
        )
        self._default_movement_speed = float(dc.get("movement_speed", -1))
        self._default_height_cmd = float(dc.get("height", -1))

        # ONNX sessions
        print(f"  Loading SONIC encoder: {self._encoder_path}")
        self._enc = ort.InferenceSession(self._encoder_path, providers=["CPUExecutionProvider"])
        self._enc_name = self._enc.get_inputs()[0].name
        print(f"  Loading SONIC decoder: {self._decoder_path}")
        self._dec = ort.InferenceSession(self._decoder_path, providers=["CPUExecutionProvider"])
        self._dec_name = self._dec.get_inputs()[0].name

        return self._n_actuated

    def step(self, model, data):
        nj = self._n_joints
        qj = data.qpos[7:7+nj].copy()
        dqj = data.qvel[6:6+nj].copy()
        quat = data.qpos[3:7].copy()
        omega = data.qvel[3:6].copy()
        gravity = get_gravity_orientation(quat)

        sc = self._obs_scales
        qj_il = np.zeros(29, dtype=np.float32)
        dqj_il = np.zeros(29, dtype=np.float32)
        omega_il = omega * sc.get("base_ang_vel", 0.25)
        for mj_i in range(min(nj, 29)):
            il_i = self._mj2il[mj_i]
            qj_il[il_i] = (qj[mj_i] - self._default_angles[mj_i]) * sc.get("dof_pos", 1.0)
            dqj_il[il_i] = dqj[mj_i] * sc.get("dof_vel", 0.05)

        # Shift histories
        self._ang_vel_hist = np.roll(self._ang_vel_hist, -1, axis=0)
        self._jpos_hist = np.roll(self._jpos_hist, -1, axis=0)
        self._jvel_hist = np.roll(self._jvel_hist, -1, axis=0)
        self._action_hist = np.roll(self._action_hist, -1, axis=0)
        self._gravity_hist = np.roll(self._gravity_hist, -1, axis=0)
        self._ang_vel_hist[-1] = omega_il
        self._jpos_hist[-1] = qj_il
        self._jvel_hist[-1] = dqj_il
        self._action_hist[-1] = self._last_action_il
        self._gravity_hist[-1] = gravity

        # Encoder
        enc_obs = self._build_encoder_obs(quat)
        token = self._enc.run(None, {self._enc_name: enc_obs.reshape(1, -1)})[0].flatten()

        # Decoder
        dec_obs = np.concatenate([
            token,
            self._ang_vel_hist.flatten(),
            self._jpos_hist.flatten(),
            self._jvel_hist.flatten(),
            self._action_hist.flatten(),
            self._gravity_hist.flatten(),
        ]).astype(np.float32)
        action_il = self._dec.run(None, {self._dec_name: dec_obs.reshape(1, -1)})[0].flatten()
        self._last_action_il = action_il.copy()

        # Remap IsaacLab → MuJoCo with per-joint scaling
        self._target = self._default_angles.copy()
        for mj_i in range(self._n_actuated):
            il_i = self._mj2il[mj_i]
            self._target[mj_i] = (
                self._default_angles[mj_i] + action_il[il_i] * self._action_scales[mj_i]
            )
        return self._target, self._kps, self._kds

    def _get_active_command(self):
        """Return the active command dict from controller, or defaults."""
        if self._controller is not None:
            return self._controller.command
        return {
            "locomotion_mode": self._default_locomotion_mode,
            "movement_direction": self._default_movement_dir.tolist(),
            "facing_direction": self._default_facing_dir.tolist(),
            "movement_speed": self._default_movement_speed,
            "height": self._default_height_cmd,
        }

    def _build_encoder_obs(self, quat):
        """Build 1762-dim encoder input. Reference motion fields use standing
        reference (zeros for joints, standing height for root z, 6D orientation).
        The active command from the controller is available for future planner
        integration but doesn't yet affect the encoder obs (standing reference
        is used regardless until the planner ONNX is wired in)."""
        obs = np.zeros(ENCODER_TOTAL_DIM, dtype=np.float32)
        orn_6 = anchor_orientation_6d(quat)
        offset = 0
        for name, dim in ENCODER_OBS_LAYOUT:
            if name == "encoder_mode_4":
                obs[offset + self._encoder_mode] = 1.0
            elif name == "motion_root_z_position_10frame_step5":
                obs[offset:offset+dim] = self._standing_height
            elif name == "motion_root_z_position":
                obs[offset] = self._standing_height
            elif name == "motion_anchor_orientation":
                obs[offset:offset+dim] = orn_6
            elif name == "motion_anchor_orientation_10frame_step5":
                obs[offset:offset+dim] = np.tile(orn_6, 10)
            offset += dim
        return obs

    def info_lines(self):
        enc_shape = self._enc.get_inputs()[0].shape
        dec_shape = self._dec.get_inputs()[0].shape
        cmd = self._get_active_command()
        return [
            f"Encoder: {enc_shape} → [{self._token_dim}]",
            f"Decoder: {dec_shape} → [29]",
            f"Standing height: {self._standing_height:.4f}m",
            f"Default command: mode={cmd['locomotion_mode']} "
            f"dir={cmd['movement_direction']}",
            f"Controller: {'attached' if self._controller else 'none'}",
        ]
