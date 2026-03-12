"""
Programmable training harness for humanoid robots.

The harness acts as a virtual spring attached to the robot's torso.
It supports smooth transitions between states so you can gradually
lower the robot from a fully-suspended position to free-standing.
"""

import time
import numpy as np
import threading


class Harness:
    """
    A virtual elastic harness attached above the robot.

    Parameters controlled from the UI:
        height      – Z position of the harness anchor point (meters)
        stiffness   – spring constant (N/m); 0 = no support
        damping     – velocity damping coefficient (N·s/m)
        enabled     – master on/off

    Automatic lowering:
        Calling start_auto_lower() will smoothly ramp height down and
        stiffness toward zero over a configurable duration, simulating
        the real-world process of slowly releasing harness support.
    """

    # Defaults: robot hangs ~0.75m above ground, firm support
    DEFAULT_HEIGHT = 1.5
    DEFAULT_STIFFNESS = 300.0
    DEFAULT_DAMPING = 150.0

    MIN_HEIGHT = 0.0
    MAX_HEIGHT = 3.0
    MIN_STIFFNESS = 0.0
    MAX_STIFFNESS = 600.0
    MIN_DAMPING = 0.0
    MAX_DAMPING = 400.0

    def __init__(self):
        self.height = self.DEFAULT_HEIGHT
        self.stiffness = self.DEFAULT_STIFFNESS
        self.damping = self.DEFAULT_DAMPING
        self.enabled = True

        self.xy = np.array([0.0, 0.0])

        self._auto_thread = None
        self._auto_running = False

        self._lock = threading.Lock()

        # Logging for the UI graph
        self._history_max = 600  # ~30s at 50fps poll
        self.force_history = []
        self.height_history = []

    @property
    def anchor(self):
        return np.array([self.xy[0], self.xy[1], self.height])

    def compute_force(self, pos, vel):
        """Compute the spring-damper force applied to the robot torso."""
        if not self.enabled or self.stiffness <= 0:
            return np.zeros(3)

        delta = self.anchor - pos
        dist = np.linalg.norm(delta)
        if dist < 1e-6:
            return np.zeros(3)

        direction = delta / dist
        v_along = np.dot(vel, direction)

        force_mag = self.stiffness * dist - self.damping * v_along
        force = force_mag * direction

        # Only apply upward force (harness pulls up, never pushes down)
        if force[2] < 0:
            force[2] = 0.0

        self._record(np.linalg.norm(force))
        return force

    def _record(self, force_mag):
        self.force_history.append(force_mag)
        self.height_history.append(self.height)
        if len(self.force_history) > self._history_max:
            self.force_history = self.force_history[-self._history_max:]
            self.height_history = self.height_history[-self._history_max:]

    # ----- Setters with clamping -----

    def set_height(self, h):
        with self._lock:
            self.height = float(np.clip(h, self.MIN_HEIGHT, self.MAX_HEIGHT))

    def set_stiffness(self, k):
        with self._lock:
            self.stiffness = float(np.clip(k, self.MIN_STIFFNESS, self.MAX_STIFFNESS))

    def set_damping(self, d):
        with self._lock:
            self.damping = float(np.clip(d, self.MIN_DAMPING, self.MAX_DAMPING))

    def set_enabled(self, on):
        with self._lock:
            self.enabled = bool(on)

    # ----- Preset positions -----

    def preset_full_support(self):
        """Fully suspend the robot above ground."""
        self.stop_auto_lower()
        with self._lock:
            self.enabled = True
            self.height = 1.5
            self.stiffness = 300.0
            self.damping = 150.0

    def preset_partial_support(self):
        """Feet touching ground, moderate spring support."""
        self.stop_auto_lower()
        with self._lock:
            self.enabled = True
            self.height = 0.85
            self.stiffness = 150.0
            self.damping = 100.0

    def preset_light_support(self):
        """Light catch-harness: minimal support, mostly free-standing."""
        self.stop_auto_lower()
        with self._lock:
            self.enabled = True
            self.height = 0.80
            self.stiffness = 50.0
            self.damping = 50.0

    def preset_released(self):
        """No support at all -- robot is on its own."""
        self.stop_auto_lower()
        with self._lock:
            self.enabled = False
            self.stiffness = 0.0

    # ----- Auto-lowering (smooth transition) -----

    def start_auto_lower(self, duration=10.0, target_height=0.78,
                         target_stiffness=0.0):
        """
        Smoothly lower the harness over `duration` seconds.
        Interpolates height and stiffness from current values to targets.
        """
        self.stop_auto_lower()

        self._auto_running = True
        self._auto_thread = threading.Thread(
            target=self._auto_lower_loop,
            args=(duration, target_height, target_stiffness),
            daemon=True,
        )
        self._auto_thread.start()

    def stop_auto_lower(self):
        self._auto_running = False
        if self._auto_thread and self._auto_thread.is_alive():
            self._auto_thread.join(timeout=1.0)
        self._auto_thread = None

    @property
    def is_auto_lowering(self):
        return self._auto_running

    def _auto_lower_loop(self, duration, target_height, target_stiffness):
        start_height = self.height
        start_stiffness = self.stiffness
        start_damping = self.damping
        target_damping = target_stiffness * 0.5  # scale damping proportionally

        t0 = time.monotonic()
        while self._auto_running:
            elapsed = time.monotonic() - t0
            progress = min(elapsed / duration, 1.0)

            # Smooth ease-in-out curve (cubic)
            t = progress * progress * (3 - 2 * progress)

            with self._lock:
                self.height = start_height + (target_height - start_height) * t
                self.stiffness = start_stiffness + (target_stiffness - start_stiffness) * t
                self.damping = start_damping + (target_damping - start_damping) * t

            if progress >= 1.0:
                if target_stiffness <= 0:
                    with self._lock:
                        self.enabled = False
                break

            time.sleep(0.02)

        self._auto_running = False

    # ----- State for UI -----

    def get_state(self):
        return {
            "enabled": self.enabled,
            "height": round(self.height, 3),
            "stiffness": round(self.stiffness, 1),
            "damping": round(self.damping, 1),
            "auto_lowering": self.is_auto_lowering,
            "force_history": self.force_history[-120:],
            "height_history": self.height_history[-120:],
        }
