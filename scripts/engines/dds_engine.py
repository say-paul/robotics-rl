"""DDS-based engine for real-robot deployment and remote simulation.

Subscribes to robot state over Cyclone DDS, runs the policy graph, and
publishes joint commands back.  No local physics — PD control happens on
the robot hardware or inside the remote simulator.

Usage (from launch_robot.py):
    # Real robot (multicast discovery on LAN)
    engine = DDSEngine(dds_config, control_hz=50)

    # Remote Isaac Sim / O3DE at a specific host
    engine = DDSEngine(dds_config, peer="192.168.1.100", control_hz=50)
"""

import os
import signal
import time
import tempfile

import numpy as np

from . import RobotState, SimulationEngine


def _configure_peer_discovery(peer: str, domain_id: int) -> None:
    """Set CYCLONEDDS_URI to use unicast discovery to a specific host.

    When a peer address is provided, we bypass multicast and directly
    target the remote DDS participant.  This works across subnets, VPNs,
    and networks where multicast is disabled.
    """
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
            from cyclonedds.core import DomainParticipant
            from cyclonedds.sub import DataReader
            from cyclonedds.pub import DataWriter
            from cyclonedds.topic import Topic
        except ImportError:
            raise ImportError(
                "cyclonedds is required for --deploy mode. "
                "Install with:  pip install cyclonedds"
            )
        from .dds_types import RobotStateDDS, JointCommandDDS

        domain_id = int(self._config.get("domain_id", 0))
        topics = self._config.get("topics", {})
        state_topic_name = topics.get("state", "rt/robot/state")
        command_topic_name = topics.get("command", "rt/robot/command")

        if self._peer:
            _configure_peer_discovery(self._peer, domain_id)

        dp = DomainParticipant(domain=domain_id)
        state_topic = Topic(dp, state_topic_name, RobotStateDDS)
        command_topic = Topic(dp, command_topic_name, JointCommandDDS)

        reader = DataReader(dp, state_topic)
        writer = DataWriter(dp, command_topic)

        # -- Wait for first state message from the robot / simulator --
        print(f"[DDSEngine] domain={domain_id}")
        if self._peer:
            print(f"[DDSEngine] peer        : {self._peer}")
        else:
            print(f"[DDSEngine] discovery   : multicast (local network)")
        print(f"[DDSEngine] subscribing : {state_topic_name}")
        print(f"[DDSEngine] publishing  : {command_topic_name}")
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

        # -- Setup policy runner --
        n_actuated = runner.setup(state)

        if controller is not None:
            runner.set_controller(controller)

        dt = 1.0 / self._control_hz
        self._running = True

        # Graceful shutdown on Ctrl-C
        def _stop(*_):
            self._running = False
        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)

        print(f"[DDSEngine] Running at {self._control_hz:.0f} Hz, "
              f"{n_actuated} actuated joints")
        print(f"[DDSEngine] Press Ctrl-C to stop\n")

        # -- Main control loop --
        while self._running:
            t0 = time.monotonic()

            # Drain all pending state samples, keep the newest
            samples = reader.take(N=100)
            if samples:
                latest = samples[-1]
                state.qpos = np.array(latest.qpos, dtype=np.float64)
                state.qvel = np.array(latest.qvel, dtype=np.float64)
                state.time = latest.timestamp_ns * 1e-9

            target_pos, kps, kds = runner.step(state)

            cmd = JointCommandDDS(
                target_positions=target_pos.astype(float).tolist(),
                kps=kps.astype(float).tolist(),
                kds=kds.astype(float).tolist(),
                timestamp_ns=int(time.time() * 1e9),
            )
            writer.write(cmd)

            elapsed = time.monotonic() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("\n[DDSEngine] Stopped.")

    def stop(self):
        """Request the control loop to exit."""
        self._running = False
