"""
Stand action: smoothly interpolate from current pose to a stable standing pose.

Usage (standalone):
    python -m actions.stand [--domain 1] [--interface lo]
"""

import numpy as np
from .base import G1Action
from .joints import NUM_MOTORS, J, STAND_POSE


# Stable standing pose with slight knee bend for shock absorption
READY_POSE = np.zeros(NUM_MOTORS)
READY_POSE[J.LeftKnee] = 0.3
READY_POSE[J.RightKnee] = 0.3
READY_POSE[J.LeftHipPitch] = -0.15
READY_POSE[J.RightHipPitch] = -0.15
READY_POSE[J.LeftAnklePitch] = -0.15
READY_POSE[J.RightAnklePitch] = -0.15
READY_POSE[J.LeftShoulderPitch] = 0.3
READY_POSE[J.RightShoulderPitch] = 0.3
READY_POSE[J.LeftElbow] = 0.3
READY_POSE[J.RightElbow] = 0.3


class StandAction(G1Action):
    """
    Smoothly interpolate to a standing pose over `duration` seconds.

    Parameters:
        duration: ramp-up time in seconds (default 3.0)
        target:   target joint angles (default: READY_POSE with bent knees)
    """

    def __init__(self, duration=3.0, target=None, **kwargs):
        super().__init__(**kwargs)
        self.duration = duration
        self.target = np.array(target if target is not None else READY_POSE)
        self._start_q = None

    def on_start(self, state):
        self._start_q = state["q"].copy()

    def compute_targets(self, t, dt, state):
        alpha = min(t / self.duration, 1.0)
        # Smooth ease-in-out
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        return self._start_q + alpha * (self.target - self._start_q)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="G1 Stand Action")
    parser.add_argument("--domain", type=int, default=1)
    parser.add_argument("--interface", type=str, default="lo")
    parser.add_argument("--duration", type=float, default=3.0)
    args = parser.parse_args()

    action = StandAction(duration=args.duration)
    action.run_blocking(domain_id=args.domain, interface=args.interface)
