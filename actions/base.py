"""
Base class for G1 robot actions.

Handles DDS channel setup, LowCmd publishing, LowState subscribing,
and a 500 Hz control loop. Subclasses implement compute_targets() to
return desired joint positions each tick.
"""

import time
import threading
import numpy as np

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.utils.crc import CRC

from .joints import NUM_MOTORS, KP_DEFAULT, KD_DEFAULT

CONTROL_DT = 0.002  # 500 Hz


class G1Action:
    """
    Base action that manages the DDS low-level control loop.

    Subclasses must implement:
        compute_targets(t, dt, state) -> np.ndarray of shape (NUM_MOTORS,)
            Return desired joint positions. Called at 500 Hz.

    Optional overrides:
        on_start(state)   -- called once when the loop begins
        on_stop()         -- called when the loop ends
    """

    def __init__(self, kp=None, kd=None):
        self.kp = np.array(kp or KP_DEFAULT, dtype=np.float64)
        self.kd = np.array(kd or KD_DEFAULT, dtype=np.float64)
        self.crc = CRC()

        self._low_cmd = unitree_hg_msg_dds__LowCmd_()
        self._low_state = None
        self._high_state = None
        self._running = False
        self._t = 0.0

    @property
    def running(self):
        return self._running

    def get_joint_positions(self):
        """Current measured joint positions from LowState, or zeros."""
        if self._low_state is None:
            return np.zeros(NUM_MOTORS)
        return np.array([self._low_state.motor_state[i].q for i in range(NUM_MOTORS)])

    def get_joint_velocities(self):
        if self._low_state is None:
            return np.zeros(NUM_MOTORS)
        return np.array([self._low_state.motor_state[i].dq for i in range(NUM_MOTORS)])

    def get_imu(self):
        """IMU readings plus body-frame linear velocity from highstate."""
        if self._low_state is None:
            return {
                "quat": [1, 0, 0, 0],
                "gyro": [0, 0, 0],
                "accel": [0, 0, 0],
                "linvel": [0, 0, 0],
            }
        imu = self._low_state.imu_state
        quat = list(imu.quaternion)

        linvel = [0.0, 0.0, 0.0]
        if self._high_state is not None:
            world_vel = np.array([
                self._high_state.velocity[0],
                self._high_state.velocity[1],
                self._high_state.velocity[2],
            ], dtype=np.float64)
            w, x, y, z = quat
            rot = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
                [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
            ], dtype=np.float64)
            linvel = (rot.T @ world_vel).tolist()

        return {
            "quat": quat,
            "gyro": list(imu.gyroscope),
            "accel": list(imu.accelerometer),
            "linvel": linvel,
        }

    # -- Subclass interface --------------------------------------------------

    def compute_targets(self, t, dt, state):
        """Return desired joint positions. Must be overridden."""
        raise NotImplementedError

    def on_start(self, state):
        """Called once when the control loop begins."""
        pass

    def on_stop(self):
        """Called when the control loop ends."""
        pass

    # -- DDS setup -----------------------------------------------------------

    def _init_channels(self):
        self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._pub.Init()

        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_low_state, 10)

        self._high_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self._high_sub.Init(self._on_high_state, 10)

    def _on_low_state(self, msg: LowState_):
        self._low_state = msg

    def _on_high_state(self, msg: SportModeState_):
        self._high_state = msg

    # -- Control loop --------------------------------------------------------

    def _publish_cmd(self, q_des, dq_des=None):
        cmd = self._low_cmd
        cmd.mode_pr = 0  # PR mode
        cmd.mode_machine = 0

        if dq_des is None:
            dq_des = np.zeros(NUM_MOTORS)

        for i in range(NUM_MOTORS):
            cmd.motor_cmd[i].mode = 1
            cmd.motor_cmd[i].q = float(q_des[i])
            cmd.motor_cmd[i].dq = float(dq_des[i])
            cmd.motor_cmd[i].tau = 0.0
            cmd.motor_cmd[i].kp = float(self.kp[i])
            cmd.motor_cmd[i].kd = float(self.kd[i])

        cmd.crc = self.crc.Crc(cmd)
        self._pub.Write(cmd)

    def _loop(self):
        # Wait for first LowState so we know current joint positions
        print(f"[{self.__class__.__name__}] Waiting for LowState...")
        deadline = time.monotonic() + 5.0
        while self._low_state is None and time.monotonic() < deadline:
            time.sleep(0.01)

        if self._low_state is None:
            print(f"[{self.__class__.__name__}] WARNING: No LowState received, using zeros")

        state = {
            "q": self.get_joint_positions(),
            "dq": self.get_joint_velocities(),
            "imu": self.get_imu(),
        }
        self.on_start(state)
        print(f"[{self.__class__.__name__}] Running at {1/CONTROL_DT:.0f} Hz")

        self._t = 0.0
        while self._running:
            t0 = time.perf_counter()

            state["q"] = self.get_joint_positions()
            state["dq"] = self.get_joint_velocities()
            state["imu"] = self.get_imu()

            q_des = self.compute_targets(self._t, CONTROL_DT, state)
            self._publish_cmd(q_des)

            self._t += CONTROL_DT
            elapsed = time.perf_counter() - t0
            remaining = CONTROL_DT - elapsed
            if remaining > 0:
                time.sleep(remaining)

        self.on_stop()

    # -- Public API ----------------------------------------------------------

    def start(self, domain_id=None, interface=None):
        """
        Start the action. If domain_id/interface are given, initialize
        DDS channels (only needed when running standalone — the sim
        server already calls ChannelFactoryInitialize).
        """
        if domain_id is not None:
            import config
            ChannelFactoryInitialize(
                domain_id if domain_id is not None else config.DOMAIN_ID,
                interface or config.INTERFACE,
            )

        self._init_channels()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=3.0)

    def run_blocking(self, domain_id=1, interface="lo"):
        """Convenience: init DDS, start, block until Ctrl-C."""
        self.start(domain_id=domain_id, interface=interface)
        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            print(f"[{self.__class__.__name__}] Stopped.")
