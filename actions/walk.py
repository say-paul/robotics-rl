"""
Walk action: sinusoidal CPG (Central Pattern Generator) walking gait.

The gait is built from coupled sinusoidal oscillators:
  - Left and right legs are in anti-phase
  - Lateral weight shift leads leg swing by a quarter period
  - Arms counter-swing opposite to legs

Usage (standalone):
    python -m actions.walk [--domain 1] [--interface lo] [--speed 0.5]
"""

import math
import numpy as np

from .base import G1Action
from .joints import NUM_MOTORS, J


class WalkGaitParams:
    """All tunable gait parameters in one place."""

    freq: float = 1.2           # gait frequency (Hz), one full L-R cycle
    ramp_duration: float = 3.0  # seconds to ramp from stand to full gait

    # Sagittal plane (forward/backward)
    hip_pitch_amp: float = 0.30     # hip swing amplitude (rad)
    knee_amp: float = 0.50          # extra knee flex during swing (rad)
    ankle_pitch_amp: float = 0.15   # ankle compensation (rad)

    # Frontal plane (lateral weight shift)
    hip_roll_amp: float = 0.04      # lateral body sway (rad)
    roll_phase_lead: float = math.pi / 4  # sway leads swing by this much

    # Arms
    arm_swing_amp: float = 0.30     # shoulder counter-swing (rad)
    elbow_offset: float = 0.35      # resting elbow bend (rad)

    # Standing offsets baked into the gait
    hip_pitch_offset: float = -0.15  # slight forward lean
    knee_offset: float = 0.30        # standing knee bend
    ankle_pitch_offset: float = -0.15


class WalkAction(G1Action):
    """
    Parametric bipedal walking gait for the G1 humanoid.

    Parameters:
        speed: 0.0-1.0 scales all gait amplitudes (0 = standing, 1 = full gait)
        params: WalkGaitParams instance for fine-tuning
    """

    def __init__(self, speed=0.6, params=None, **kwargs):
        super().__init__(**kwargs)
        self.speed = np.clip(speed, 0.0, 1.0)
        self.p = params or WalkGaitParams()
        self._start_q = None

    def on_start(self, state):
        self._start_q = state["q"].copy()

    def compute_targets(self, t, dt, state):
        p = self.p
        q = np.zeros(NUM_MOTORS)

        # Ramp up from current pose to walking over ramp_duration
        ramp = min(t / p.ramp_duration, 1.0)
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)  # smooth ease-in-out
        amp = ramp * self.speed

        phase = 2.0 * math.pi * p.freq * t

        # ---------- Standing offsets ----------
        q[J.LeftHipPitch] = p.hip_pitch_offset
        q[J.RightHipPitch] = p.hip_pitch_offset
        q[J.LeftKnee] = p.knee_offset
        q[J.RightKnee] = p.knee_offset
        q[J.LeftAnklePitch] = p.ankle_pitch_offset
        q[J.RightAnklePitch] = p.ankle_pitch_offset

        # ---------- Sagittal: hip pitch swing ----------
        q[J.LeftHipPitch] += amp * p.hip_pitch_amp * math.sin(phase)
        q[J.RightHipPitch] += amp * p.hip_pitch_amp * math.sin(phase + math.pi)

        # ---------- Knee: flex during forward swing only ----------
        # max(0, sin) gives flexion only during the forward half of the cycle
        left_swing = max(0.0, math.sin(phase))
        right_swing = max(0.0, math.sin(phase + math.pi))
        q[J.LeftKnee] += amp * p.knee_amp * left_swing
        q[J.RightKnee] += amp * p.knee_amp * right_swing

        # ---------- Ankle pitch: dorsiflexion during swing for clearance ----------
        q[J.LeftAnklePitch] += amp * p.ankle_pitch_amp * left_swing
        q[J.RightAnklePitch] += amp * p.ankle_pitch_amp * right_swing

        # ---------- Frontal: lateral weight shift ----------
        sway = amp * p.hip_roll_amp * math.sin(phase - p.roll_phase_lead)
        # Left hip roll: positive = abduction. Lean body RIGHT = left adduction (negative)
        q[J.LeftHipRoll] = -sway
        # Right hip roll: negative = abduction (mirrored). Same sway value works.
        q[J.RightHipRoll] = sway

        # ---------- Arms: counter-swing ----------
        q[J.LeftShoulderPitch] = amp * p.arm_swing_amp * math.sin(phase + math.pi)
        q[J.RightShoulderPitch] = amp * p.arm_swing_amp * math.sin(phase)
        q[J.LeftElbow] = p.elbow_offset
        q[J.RightElbow] = p.elbow_offset

        # ---------- Blend with start pose during ramp ----------
        if ramp < 1.0:
            q = self._start_q * (1.0 - ramp) + q * ramp

        return q


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="G1 Walk Action")
    parser.add_argument("--domain", type=int, default=1)
    parser.add_argument("--interface", type=str, default="lo")
    parser.add_argument("--speed", type=float, default=0.6,
                        help="Gait amplitude scale 0.0-1.0")
    parser.add_argument("--freq", type=float, default=1.2,
                        help="Gait frequency in Hz")
    args = parser.parse_args()

    params = WalkGaitParams()
    params.freq = args.freq

    action = WalkAction(speed=args.speed, params=params)
    action.run_blocking(domain_id=args.domain, interface=args.interface)
