"""
Shared state container for the G1 simulation.

SimContext replaces module-level globals and makes dependencies between
simulation, renderer, and web server explicit.  Every component receives
a reference to the same SimContext instance.
"""

import threading

import mujoco
import numpy as np

from harness import Harness


class SimContext:
    """All mutable state shared across simulation, renderer, and web threads."""

    def __init__(self, mj_model, mj_data):
        self.mj_model = mj_model
        self.mj_data = mj_data

        self.physics_lock = threading.Lock()
        self.sim_running = True
        self.dds_ready = threading.Event()

        # Camera defaults
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance = 3.0
        self.cam.azimuth = -130.0
        self.cam.elevation = -20.0
        self.cam.lookat[:] = [0.0, 0.0, 0.75]

        self.harness = Harness()
        self.robot_pos = np.zeros(3)

        self.active_action = None
        self.action_lock = threading.Lock()

    def shutdown(self):
        """Signal all threads to stop and clean up the active action."""
        with self.action_lock:
            if self.active_action is not None:
                self.active_action.stop()
                self.active_action = None
        self.sim_running = False
