"""
Prefrontal Cortex
=================
High-level decision-making layer that:
  1. Receives user commands via chat or mission planner
  2. Routes ALL user input to the Planner (VLM + rule-based tool chain)
  3. Only "quit" is handled directly as a safety escape
  4. Manages mission phase sequencing
"""

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from cortex.motor.controller import MotorCortex
from cortex.perefrontal.vla import (
    NavigationController,
    VLAAction,
    VLAActionType,
    parse_instruction,
)

logger = logging.getLogger(__name__)


class Intent(Enum):
    IDLE = auto()
    WALK = auto()
    STOP = auto()
    QUIT = auto()
    MISSION_PHASE = auto()
    CHAT_COMMAND = auto()
    VLA_NAV = auto()
    PERCEPTION = auto()


@dataclass
class ActionRequest:
    intent: Intent
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    duration: Optional[float] = None
    speech: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


PHASE_VELOCITY_MAP = {
    "idle_start": (0.0, 0.0, 0.0),
    "backstage_to_stairs": (0.3, 0.0, 0.0),
    "walk_up_stairs": (0.2, 0.0, 0.0),
    "approach_presenter": (0.2, 0.0, 0.0),
    "hand_water": (0.0, 0.0, 0.0),
    "turn_to_audience": (0.0, 0.0, 0.5),
    "wave_greeting": (0.0, 0.0, 0.0),
    "introduce_self": (0.0, 0.0, 0.0),
    "walk_to_staircase": (0.3, 0.0, 0.0),
    "walk_down_stairs": (0.15, 0.0, 0.0),
    "return_backstage": (0.3, 0.0, 0.0),
    "idle_end": (0.0, 0.0, 0.0),
}


class PrefrontalCortex:
    """Orchestrates high-level behaviour and delegates to MotorCortex."""

    def __init__(self, motor: MotorCortex, auto_trigger: bool = True):
        self.motor = motor
        self.auto_trigger = auto_trigger
        self.scene_name: str = "flat_ground"

        # Mission state
        self._mission_phases: List[Dict] = []
        self._current_phase_idx: int = 0
        self._mission_active: bool = False
        self._phase_start_time: Optional[float] = None
        self._waiting_trigger: bool = False

        # Chat command queue
        self._command_queue: List[str] = []

        # VLA navigation (legacy rule-based fallback)
        self._nav = NavigationController()
        self._get_robot_state: Optional[Callable] = None

        # VLA model engine (optional — pi0.5 learned policies)
        self._vla_engine = None
        self._vla_action_mapper = None
        self._vla_model_active: bool = False

        # Planner (VLM + tool chain — primary command handler)
        self._planner = None

        # Response queue: text responses from planner → chatbot
        self._response_queue: deque = deque(maxlen=50)
        self._response_lock = threading.Lock()

        # Speech callback
        self._speech_callback: Optional[Callable[[str], None]] = None

        # Action log
        self.action_log: List[Dict] = []

        logger.info("PrefrontalCortex initialized  auto_trigger=%s", auto_trigger)

    # ── Injected state accessor ─────────────────────────────────────────────
    def set_robot_state_fn(self, fn: Callable) -> None:
        self._get_robot_state = fn

    def _robot_pos_yaw(self):
        if self._get_robot_state:
            return self._get_robot_state()
        return np.zeros(3), 0.0

    def set_vla_engine(self, engine, mapper) -> None:
        self._vla_engine = engine
        self._vla_action_mapper = mapper
        logger.info("VLA model engine connected")

    def set_planner(self, planner) -> None:
        self._planner = planner
        logger.info("Planner connected")

    def push_response(self, text: str) -> None:
        with self._response_lock:
            self._response_queue.append(text)

    def get_response(self, timeout: float = 0.0) -> Optional[str]:
        deadline = time.time() + timeout
        while True:
            with self._response_lock:
                if self._response_queue:
                    return self._response_queue.popleft()
            if time.time() >= deadline:
                return None
            time.sleep(0.1)

    # ── Mission Loading ─────────────────────────────────────────────────────
    def load_mission(self, mission_config: Dict) -> None:
        mission = mission_config["mission"]
        self._mission_phases = mission["phases"]
        self._current_phase_idx = 0
        self._mission_active = False
        self._phase_start_time = None
        logger.info("Mission loaded: '%s'  phases=%d", mission["name"], len(self._mission_phases))

    def start_mission(self) -> None:
        self._mission_active = True
        self._current_phase_idx = 0
        self._phase_start_time = None
        logger.info("Mission started")

    # ── Per-tick Decision Loop ──────────────────────────────────────────────
    def decide(self) -> Optional[ActionRequest]:
        if self._command_queue:
            cmd = self._command_queue.pop(0)
            return self._handle_chat_command(cmd)

        if self._vla_model_active and self._vla_engine and self._vla_engine.has_actions:
            return self._tick_vla_model()

        if self._nav.active:
            return self._tick_nav()

        if self._mission_active:
            return self._advance_mission()

        return None

    # ── VLA Navigation Tick ─────────────────────────────────────────────────
    def _tick_nav(self) -> Optional[ActionRequest]:
        pos, yaw = self._robot_pos_yaw()
        vx, vy, yr = self._nav.tick(pos, yaw)
        self.motor.set_velocity(vx, vy, yr)
        if not self._nav.active:
            self.motor.stop()
            return ActionRequest(intent=Intent.VLA_NAV, metadata={"status": "arrived"})
        return None

    # ── VLA Model Action Tick ───────────────────────────────────────────────
    def _tick_vla_model(self) -> Optional[ActionRequest]:
        action_vec = self._vla_engine.get_next_action()
        if action_vec is None:
            self._vla_model_active = False
            self.motor.stop()
            return ActionRequest(intent=Intent.VLA_NAV, metadata={"status": "vla_queue_empty"})

        cmd = self._vla_action_mapper.map_action(action_vec)
        self.motor.set_velocity(cmd.get("vx", 0.0), cmd.get("vy", 0.0), cmd.get("yaw_rate", 0.0))
        return None

    # ── Mission Sequencing ──────────────────────────────────────────────────
    def _advance_mission(self) -> Optional[ActionRequest]:
        if self._waiting_trigger:
            if self.auto_trigger:
                self._waiting_trigger = False
            else:
                return None

        if self._phase_start_time is not None:
            current_phase = (
                self._mission_phases[self._current_phase_idx - 1]
                if self._current_phase_idx > 0 else None
            )
            if current_phase:
                if current_phase.get("infinite"):
                    return None
                duration = current_phase.get("duration")
                if duration and (time.time() - self._phase_start_time) < duration:
                    return None

        if self._current_phase_idx >= len(self._mission_phases):
            self._mission_active = False
            logger.info("Mission complete")
            self.motor.stop()
            return ActionRequest(intent=Intent.IDLE)

        phase = self._mission_phases[self._current_phase_idx]
        self._current_phase_idx += 1

        if phase.get("wait_for_trigger"):
            if self.auto_trigger:
                phase = {**phase, "duration": 1.0}
            else:
                self._waiting_trigger = True
                return None

        name = phase.get("name", "unknown")
        vx, vy, yaw = PHASE_VELOCITY_MAP.get(name, (0.0, 0.0, 0.0))

        speech = phase.get("speech")
        if speech and self._speech_callback:
            self._speech_callback(speech)

        self._phase_start_time = time.time()
        self.action_log.append({
            "time": time.time(),
            "phase": phase.get("phase"),
            "name": name,
            "velocity": (vx, vy, yaw),
        })

        logger.info(
            "Phase %s [%s]: vx=%.2f vy=%.2f yaw=%.2f  duration=%s",
            phase.get("phase", "?"), name, vx, vy, yaw, phase.get("duration"),
        )
        self.motor.set_velocity(vx, vy, yaw)
        return ActionRequest(
            intent=Intent.MISSION_PHASE,
            vx=vx, vy=vy, yaw_rate=yaw,
            duration=phase.get("duration"),
            speech=speech,
            metadata={"phase": phase},
        )

    # ── Chat Commands ───────────────────────────────────────────────────────
    def enqueue_command(self, command: str) -> None:
        self._command_queue.append(command)
        logger.info("[CMD:QUEUED] msg='%s'", command)

    def _handle_chat_command(self, cmd: str) -> Optional[ActionRequest]:
        cmd_stripped = cmd.strip()
        cmd_lower = cmd_stripped.lower()

        # 1) Quit — always instant, bypass everything
        if cmd_lower in ("quit", "exit", "end simulation", "shutdown"):
            logger.info("[CMD:INTENT] msg='%s' -> intent=QUIT", cmd_stripped)
            self.motor.stop()
            self._mission_active = False
            self._nav.cancel()
            self._vla_model_active = False
            if self._planner:
                self._planner.cancel()
            return ActionRequest(intent=Intent.QUIT)

        # 2) Planner (primary) — handles safety intercept + VLM + rule fallback
        #    The planner itself decides: safety commands are instant,
        #    everything else goes to VLM (AI-driven tool selection).
        if self._planner:
            logger.info("[CMD:INTENT] msg='%s' -> intent=PERCEPTION (planner)", cmd_stripped)
            return ActionRequest(
                intent=Intent.PERCEPTION,
                metadata={"instruction": cmd_stripped},
            )

        # 3) VLA engine — only used when no planner (explicit --vla-model mode)
        if self._vla_engine and self._vla_engine.loaded:
            logger.info("[CMD:INTENT] msg='%s' -> intent=VLA_NAV", cmd_stripped)
            return self._handle_vla(cmd_stripped)

        logger.warning("[CMD:INTENT] msg='%s' -> no handler available", cmd_stripped)
        return None

    def _handle_vla(self, text: str) -> Optional[ActionRequest]:
        if self._vla_engine and self._vla_engine.loaded:
            self._vla_engine.set_instruction(text)
            self._vla_model_active = True
            if not self.motor.is_active:
                self.motor.start_walking()
            return ActionRequest(
                intent=Intent.VLA_NAV,
                metadata={"vla_model": True, "instruction": text},
            )

        action = parse_instruction(text, scene_name=self.scene_name)
        if action.action_type == VLAActionType.UNKNOWN:
            logger.warning("[CMD:PARSE_FAIL] '%s'", text)
            return ActionRequest(intent=Intent.VLA_NAV, metadata={"error": f"Could not understand: {text}"})

        pos, yaw = self._robot_pos_yaw()
        if action.action_type == VLAActionType.STOP:
            self.motor.stop()
            self._nav.cancel()
            return ActionRequest(intent=Intent.STOP)

        self._nav.start(action, pos, yaw)
        if not self.motor.is_active:
            self.motor.start_walking()
        return ActionRequest(intent=Intent.VLA_NAV, metadata={"vla_action": action.description})

    # ── Status generation (used by planner's get_status tool) ────────────
    def generate_status_text(self) -> str:
        pos, yaw = self._robot_pos_yaw()
        return (
            f"Mode: {self.motor.mode.name}\n"
            f"Velocity: ({self.motor.velocity_cmd.vx:.2f}, "
            f"{self.motor.velocity_cmd.vy:.2f}, "
            f"{self.motor.velocity_cmd.yaw_rate:.2f})\n"
            f"Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})\n"
            f"Heading: {math.degrees(yaw):.1f}°\n"
            f"Mission: {'active' if self._mission_active else 'inactive'} "
            f"({self._current_phase_idx}/{len(self._mission_phases)})\n"
            f"Nav: {'active' if self._nav.active else 'idle'}"
        )

    # ── Speech ──────────────────────────────────────────────────────────────
    def set_speech_callback(self, cb: Callable[[str], None]) -> None:
        self._speech_callback = cb

    # ── Queries ─────────────────────────────────────────────────────────────
    @property
    def mission_active(self) -> bool:
        return self._mission_active

    @property
    def current_phase(self) -> Optional[Dict]:
        if 0 < self._current_phase_idx <= len(self._mission_phases):
            return self._mission_phases[self._current_phase_idx - 1]
        return None

    @property
    def mission_progress(self) -> float:
        if not self._mission_phases:
            return 0.0
        return self._current_phase_idx / len(self._mission_phases)

    @property
    def nav_active(self) -> bool:
        return self._nav.active
