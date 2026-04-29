"""DDS-based engine for real-robot deployment and remote simulation.

Subscribes to robot state over Cyclone DDS, runs the policy graph, and
publishes joint commands back.  No local physics — PD control happens on
the robot hardware or inside the remote simulator.

Supports two DDS protocols:
  - Custom RDP types (RobotStateDDS / JointCommandDDS) — default
  - Unitree SDK types (LowState_ / LowCmd_ / OdoState_) — set
    ``unitree_compat: true`` in the robot YAML's DDS interface

Usage (from launch_robot.py):
    # Real robot (multicast discovery on LAN)
    engine = DDSEngine(dds_config, control_hz=50)

    # Remote Isaac Sim / O3DE at a specific host
    engine = DDSEngine(dds_config, peer="192.168.1.100", control_hz=50)
"""

import os
import signal
import threading
import time

import numpy as np

from . import RobotState, SimulationEngine

NUM_MOTORS = 29


def _configure_peer_discovery(peer: str, domain_id: int) -> None:
    """Set CYCLONEDDS_URI to use unicast discovery to a specific host."""
    xml = (
        f"<CycloneDDS>"
        f"<Domain id='any'>"
        f"<General><AllowMulticast>false</AllowMulticast></General>"
        f"<Discovery>"
        f"<Peers><Peer address='{peer}'/></Peers>"
        f"<ParticipantIndex>auto</ParticipantIndex>"
        f"</Discovery>"
        f"</Domain>"
        f"</CycloneDDS>"
    )
    os.environ["CYCLONEDDS_URI"] = xml
    print(f"[DDSEngine] Unicast peer: {peer}")


def _unitree_state_to_robot_state(low_state, odo_state) -> RobotState:
    """Convert Unitree LowState_ + OdoState_ → RobotState."""
    root_pos = np.array(odo_state.position, dtype=np.float64)
    orn = odo_state.orientation
    root_quat = np.array([orn[3], orn[0], orn[1], orn[2]], dtype=np.float64)
    root_lin = np.array(odo_state.linear_velocity, dtype=np.float64)
    root_ang = np.array(odo_state.angular_velocity, dtype=np.float64)

    jpos = np.array([low_state.motor_state[i].q for i in range(NUM_MOTORS)],
                    dtype=np.float64)
    jvel = np.array([low_state.motor_state[i].dq for i in range(NUM_MOTORS)],
                    dtype=np.float64)

    qpos = np.concatenate([root_pos, root_quat, jpos])
    qvel = np.concatenate([root_lin, root_ang, jvel])
    return RobotState(qpos=qpos, qvel=qvel, time=low_state.tick * 1e-3)


def _robot_cmd_to_unitree(target_pos, kps, kds, LowCmd_, MotorCmd_):
    """Convert policy output → Unitree LowCmd_."""
    cmd = LowCmd_(
        mode_pr=0,
        mode_machine=0xFF,
        motor_cmd=[MotorCmd_() for _ in range(35)],
        reserve=[0, 0, 0, 0],
        crc=0,
    )
    for i in range(min(len(target_pos), NUM_MOTORS)):
        cmd.motor_cmd[i].mode = 1
        cmd.motor_cmd[i].q = float(target_pos[i])
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].tau = 0.0
        cmd.motor_cmd[i].kp = float(kps[i])
        cmd.motor_cmd[i].kd = float(kds[i])
    return cmd


class DDSEngine(SimulationEngine):
    """Headless control loop over DDS.

    Parameters
    ----------
    dds_config : dict
        The ``protocol: dds`` interface entry from the robot YAML,
        containing ``domain_id``, ``topics``, ``qos``, etc.
    peer : str or None
        Remote host for unicast discovery.  None = multicast (real robot).
    control_hz : float
        Policy execution rate (Hz).
    """

    def __init__(self, dds_config: dict, *, peer: str | None = None,
                 control_hz: float = 50):
        self._config = dds_config
        self._peer = peer
        self._control_hz = control_hz
        self._running = False

    def run(self, runner, *, controller=None, **kwargs) -> None:
        try:
            from cyclonedds.domain import DomainParticipant
            from cyclonedds.sub import DataReader
            from cyclonedds.pub import DataWriter
            from cyclonedds.topic import Topic
        except ImportError:
            raise ImportError(
                "cyclonedds is required for --deploy mode. "
                "Install with:  pip install cyclonedds"
            )

        unitree_compat = self._config.get("unitree_compat", False)
        domain_id = int(self._config.get("domain_id", 0))

        if self._peer:
            _configure_peer_discovery(self._peer, domain_id)

        dp = DomainParticipant(domain_id=domain_id)

        if unitree_compat:
            self._run_unitree(dp, runner, controller, Topic, DataReader, DataWriter)
        else:
            self._run_rdp(dp, runner, controller, Topic, DataReader, DataWriter)

    def _run_rdp(self, dp, runner, controller, Topic, DataReader, DataWriter):
        """Original RDP custom DDS types."""
        from .dds_types import RobotStateDDS, JointCommandDDS

        topics = self._config.get("topics", {})
        state_topic_name = topics.get("state", "rt/robot/state")
        command_topic_name = topics.get("command", "rt/robot/command")

        reader = DataReader(dp, Topic(dp, state_topic_name, RobotStateDDS))
        writer = DataWriter(dp, Topic(dp, command_topic_name, JointCommandDDS))

        print(f"[DDSEngine] protocol   : RDP (custom types)")
        print(f"[DDSEngine] subscribing: {state_topic_name}")
        print(f"[DDSEngine] publishing : {command_topic_name}")
        print(f"[DDSEngine] Waiting for robot state ...")

        initial_msg = None
        while initial_msg is None:
            samples = reader.take(N=1)
            if samples:
                initial_msg = samples[0]
            else:
                time.sleep(0.01)

        state = RobotState(
            qpos=np.array(initial_msg.qpos, dtype=np.float64),
            qvel=np.array(initial_msg.qvel, dtype=np.float64),
            time=initial_msg.timestamp_ns * 1e-9,
        )
        print(f"[DDSEngine] Received initial state "
              f"(qpos len={len(state.qpos)}, qvel len={len(state.qvel)})")

        n_actuated = runner.setup(state)
        if controller is not None:
            runner.set_controller(controller)

        _read_count = [0]
        def read_state():
            samples = reader.take(N=100)
            _read_count[0] += 1
            if samples:
                latest = samples[-1]
                state.qpos = np.array(latest.qpos, dtype=np.float64)
                state.qvel = np.array(latest.qvel, dtype=np.float64)
                state.time = latest.timestamp_ns * 1e-9
                if _read_count[0] <= 5:
                    print(f"[DDSEngine] read#{_read_count[0]}: got {len(samples)} samples")
            else:
                if _read_count[0] <= 5 or _read_count[0] % 500 == 0:
                    print(f"[DDSEngine] read#{_read_count[0]}: EMPTY")

        def write_cmd(target_pos, kps, kds):
            writer.write(JointCommandDDS(
                target_positions=target_pos.astype(float).tolist(),
                kps=kps.astype(float).tolist(),
                kds=kds.astype(float).tolist(),
                timestamp_ns=int(time.time() * 1e9),
            ))

        self._control_loop(state, n_actuated, runner, controller,
                           read_state, write_cmd)

    def _run_unitree(self, dp, runner, controller, Topic, DataReader, DataWriter):
        """Unitree SDK DDS types (LowState/LowCmd/OdoState)."""
        try:
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
                LowState_, LowCmd_, MotorCmd_, OdoState_,
            )
        except ImportError:
            raise ImportError(
                "unitree_sdk2py is required for unitree_compat mode. "
                "Install the bundled fork from GR00T-WholeBodyControl/"
                "external_dependencies/unitree_sdk2_python"
            )

        state_reader = DataReader(dp, Topic(dp, "rt/lowstate", LowState_))
        odo_reader = DataReader(dp, Topic(dp, "rt/odostate", OdoState_))
        cmd_writer = DataWriter(dp, Topic(dp, "rt/lowcmd", LowCmd_))

        print(f"[DDSEngine] protocol   : Unitree SDK")
        print(f"[DDSEngine] subscribing: rt/lowstate, rt/odostate")
        print(f"[DDSEngine] publishing : rt/lowcmd")
        print(f"[DDSEngine] Waiting for robot state ...")

        low_state = None
        odo_state = None
        while low_state is None or odo_state is None:
            ls = state_reader.take(N=1)
            if ls:
                low_state = ls[0]
            os = odo_reader.take(N=1)
            if os:
                odo_state = os[0]
            if low_state is None or odo_state is None:
                time.sleep(0.01)

        state = _unitree_state_to_robot_state(low_state, odo_state)
        print(f"[DDSEngine] Received initial state "
              f"(qpos len={len(state.qpos)}, qvel len={len(state.qvel)})")

        n_actuated = runner.setup(state)
        if controller is not None:
            runner.set_controller(controller)

        _latest_odo = [odo_state]

        def read_state():
            ls_samples = state_reader.take(N=100)
            odo_samples = odo_reader.take(N=100)
            if odo_samples:
                _latest_odo[0] = odo_samples[-1]
            if ls_samples:
                rs = _unitree_state_to_robot_state(ls_samples[-1], _latest_odo[0])
                state.qpos = rs.qpos
                state.qvel = rs.qvel
                state.time = rs.time

        def write_cmd(target_pos, kps, kds):
            cmd_writer.write(
                _robot_cmd_to_unitree(target_pos, kps, kds, LowCmd_, MotorCmd_))

        self._control_loop(state, n_actuated, runner, controller,
                           read_state, write_cmd)

    def _control_loop(self, state, n_actuated, runner, controller,
                      read_state_fn, write_cmd_fn):
        """Shared control loop for both protocols."""
        dt = 1.0 / self._control_hz
        self._running = True

        def _stop(*_):
            self._running = False
        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)

        _policy_enabled = False
        _stdin_lock = threading.Lock()

        def _stdin_reader():
            nonlocal _policy_enabled
            import sys, tty, termios
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while self._running:
                    ch = sys.stdin.read(1)
                    if not ch:
                        break
                    keycode = ord(ch)
                    if keycode in (ord('p'), ord('P')):
                        with _stdin_lock:
                            _policy_enabled = not _policy_enabled
                        status = "ENABLED" if _policy_enabled else "DISABLED"
                        print(f"\n[Policy] {status}")
                    elif controller is not None:
                        controller.handle_event(keycode)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

        threading.Thread(target=_stdin_reader, daemon=True).start()

        print(f"[DDSEngine] Running at {self._control_hz:.0f} Hz, "
              f"{n_actuated} actuated joints")
        print(f"[DDSEngine] Press P to toggle policy, Ctrl-C to stop")
        if controller is not None:
            for line in controller.banner_lines():
                print(f"  {line}")
        print()

        while self._running:
            t0 = time.monotonic()

            read_state_fn()

            with _stdin_lock:
                active = _policy_enabled

            if active:
                if not hasattr(self, '_active_tick'):
                    self._active_tick = 0
                self._active_tick += 1
                target_pos, kps, kds = runner.step(state)
                if self._active_tick <= 10:
                    print(f"[policy] tick={self._active_tick} "
                          f"qpos[7:10]={np.round(state.qpos[7:10], 4).tolist()} "
                          f"qvel[3:6]={np.round(state.qvel[3:6], 4).tolist()} "
                          f"tgt[0:3]={np.round(target_pos[:3], 4).tolist()}")
                write_cmd_fn(target_pos, kps, kds)

            elapsed = time.monotonic() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("\n[DDSEngine] Stopped.")

    def stop(self):
        """Request the control loop to exit."""
        self._running = False
