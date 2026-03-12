"""
ONNX neural-network policy action.

Loads a pretrained ONNX locomotion policy (e.g. from mujoco_playground)
and runs inference through the OpenVINO execution provider, targeting
the Intel NPU, iGPU, or CPU.

Observation convention (mujoco_playground G1 Joystick, 103-dim):
  linvel_local (3) | gyro (3) | gravity (3) | command (3) |
  joint_pos - default (29) | joint_vel (29) | last_action (29) | phase (4)

Usage (standalone):
    python -m actions.policy --onnx path/to/policy.onnx --device CPU
"""

import math
import time
import logging

import numpy as np
import onnxruntime as ort

from .base import G1Action, CONTROL_DT
from .joints import NUM_MOTORS, DEFAULT_POSITIONS, ACTION_SCALE

log = logging.getLogger(__name__)

POLICY_DT = 0.02  # 50 Hz inference
POLICY_STEPS = round(POLICY_DT / CONTROL_DT)  # ticks between inferences


def _build_session(policy_path, device="CPU", cache_dir="/tmp/ov_cache"):
    """Create an ONNX Runtime session with OpenVINO EP."""
    providers = [
        (
            "OpenVINOExecutionProvider",
            {"device_type": device, "cache_dir": cache_dir},
        ),
        "CPUExecutionProvider",
    ]
    sess = ort.InferenceSession(policy_path, providers=providers)
    active = sess.get_providers()
    log.info("ONNX session providers: %s", active)
    return sess


class OnnxPolicyAction(G1Action):
    """
    Run a pretrained ONNX locomotion policy at 50 Hz.

    The 500 Hz control loop holds the last inferred targets between
    policy ticks, providing smooth PD tracking at the DDS rate.

    Parameters
    ----------
    policy_path : str
        Filesystem path to the .onnx model.
    device : str
        OpenVINO device — "NPU", "GPU", or "CPU".
    vx, vy, vyaw : float
        Velocity commands forwarded into the observation vector.
    """

    def __init__(
        self,
        policy_path,
        device="CPU",
        vx=0.0,
        vy=0.0,
        vyaw=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._policy_path = policy_path
        self._device = device
        self._command = np.array([vx, vy, vyaw], dtype=np.float32)
        self._default_pos = np.array(DEFAULT_POSITIONS, dtype=np.float32)
        self._action_scale = ACTION_SCALE

        self._session = None
        self._input_names = None
        self._obs_dim = None
        self._has_time_input = False

        self._last_action = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._phase = np.zeros(2, dtype=np.float32)
        self._phase_dt = 2.0 * math.pi * POLICY_DT * 1.25  # default freq 1.25 Hz
        self._q_target = None
        self._tick = 0

    @property
    def command(self):
        return self._command.copy()

    @command.setter
    def command(self, value):
        self._command[:] = value[:3]

    def on_start(self, state):
        print(f"[OnnxPolicyAction] Loading {self._policy_path} on {self._device}")
        self._session = _build_session(
            self._policy_path, device=self._device
        )
        self._input_names = [i.name for i in self._session.get_inputs()]
        shapes = {i.name: i.shape for i in self._session.get_inputs()}
        print(f"[OnnxPolicyAction] Inputs: {shapes}")

        self._obs_dim = shapes[self._input_names[0]][1]
        self._has_time_input = "time_step" in self._input_names

        self._q_target = np.array(state["q"], dtype=np.float32)
        self._last_action = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._phase[:] = [0.0, math.pi]
        self._tick = 0

        print(
            f"[OnnxPolicyAction] Ready  obs_dim={self._obs_dim}  "
            f"policy_rate={1/POLICY_DT:.0f}Hz  device={self._device}"
        )

    def _build_obs(self, state):
        """
        Assemble the observation vector.

        Adapts to the model's expected obs_dim:
          - 103-dim: standard mujoco_playground joystick
          - Larger: zero-pad extra dimensions
        """
        q = np.array(state["q"], dtype=np.float32)
        dq = np.array(state["dq"], dtype=np.float32)

        imu = state.get("imu", {})
        gyro = np.array(imu.get("gyro", [0, 0, 0]), dtype=np.float32)
        quat = np.array(imu.get("quat", [1, 0, 0, 0]), dtype=np.float32)

        # Project gravity into body frame from quaternion
        # g_body = R^T @ [0, 0, -1]
        w, x, y, z = quat
        gravity = np.array([
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            -(w * w - x * x - y * y + z * z),
        ], dtype=np.float32)

        linvel = np.array(imu.get("linvel", [0, 0, 0]), dtype=np.float32)

        cos_phase = np.cos(self._phase)
        sin_phase = np.sin(self._phase)
        phase_obs = np.concatenate([cos_phase, sin_phase])

        obs_parts = [
            linvel,                           # 3
            gyro,                             # 3
            gravity,                          # 3
            self._command,                    # 3
            q - self._default_pos,            # 29
            dq,                               # 29
            self._last_action,                # 29
            phase_obs,                        # 4
        ]
        obs = np.concatenate(obs_parts).astype(np.float32)

        if obs.shape[0] < self._obs_dim:
            obs = np.pad(obs, (0, self._obs_dim - obs.shape[0]))
        elif obs.shape[0] > self._obs_dim:
            obs = obs[: self._obs_dim]

        return obs.reshape(1, -1)

    def _infer(self, obs):
        """Run one forward pass through the ONNX model."""
        feeds = {self._input_names[0]: obs}
        if self._has_time_input:
            feeds["time_step"] = np.array(
                [[self._t]], dtype=np.float32
            )
        outputs = self._session.run(None, feeds)
        return outputs[0].flatten()  # raw action vector

    def compute_targets(self, t, dt, state):
        self._tick += 1

        if self._tick % POLICY_STEPS == 0:
            obs = self._build_obs(state)
            action = self._infer(obs)

            self._last_action = action.copy()
            self._q_target = self._default_pos + action * self._action_scale

            self._phase += self._phase_dt
            self._phase = np.fmod(self._phase + math.pi, 2 * math.pi) - math.pi

        return self._q_target


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="G1 ONNX Policy Action")
    parser.add_argument("--onnx", required=True, help="Path to .onnx policy file")
    parser.add_argument("--device", default="CPU", choices=["NPU", "GPU", "CPU"])
    parser.add_argument("--domain", type=int, default=1)
    parser.add_argument("--interface", type=str, default="lo")
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--vyaw", type=float, default=0.0)
    args = parser.parse_args()

    action = OnnxPolicyAction(
        policy_path=args.onnx,
        device=args.device,
        vx=args.vx,
        vy=args.vy,
        vyaw=args.vyaw,
    )
    action.run_blocking(domain_id=args.domain, interface=args.interface)
