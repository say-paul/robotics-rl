"""
Planner — Command routing (model-first task chain)
==================================================
Primary path (when VLM text generation is ready):
  1. Safety intercept — stop, release, quit, help (no model)
  2. **Model task chain** — user text is sent verbatim to the VLM/LLM; it emits JSON
     with ordered ``motor`` vs ``visual`` tasks; motor steps map to WBC tools.

Fallback (``--planner-mode`` / load failures):
  3. Embedding intent matcher (optional)
  4. Rule-based patterns (optional)

WBC tools used by the model plan:
    move, turn_degrees, turn_relative, turn_magnetic, goto_landmark, halt,
    look, look_around, describe_scene, get_position, …
"""

import logging
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_TOOL_CALL_RE = re.compile(r"Action:\s*(\w+)\(([^)]*)\)", re.IGNORECASE)
_ANSWER_RE = re.compile(r"Answer:\s*(.*)", re.IGNORECASE | re.DOTALL)

TOOL_DESCRIPTIONS = """\
You are the brain of a humanoid robot. You receive user instructions and must \
choose which tool to call. Call exactly ONE tool per step.

## Available Tools
  move(distance, speed, direction) - Walk. distance in metres (negative=backward), \
speed 0.05-1.0 (default 0.3), direction is optional cardinal ("north","sw",...).
  turn_degrees(degrees) - Rotate by degrees. Positive = left, negative = right.
  turn_relative(direction) - Turn 90 degrees left or right. direction = "left" | "right".
  turn_magnetic(heading) - Face a compass direction: "north", "east", "sw", etc.
  halt() - Stop all motion immediately.
  look() - Capture head camera image and describe what is visible.
  release_harness() - Release the elastic harness so the robot can walk freely.
  get_position() - Get current (x, y) position and compass heading.
  get_status() - Get robot status: mode, velocity, heading, mission state.
  describe_scene() - Get the full environment layout from memory.
  start_mission() - Start the loaded mission sequence.
  help() - List all available commands.
  quit_simulation() - End the simulation and save video.

## Response Format
Think: <brief reasoning about what to do>
Action: <tool_name>(<arguments>)

When the task is fully done, respond:
Answer: <response to the user>

## Examples

User: "walk straight 2 meters"
Think: The user wants to move forward 2 metres.
Action: move(2.0)

User: "run fast 10m east"
Think: The user wants to move 10 metres east quickly.
Action: move(10.0, 0.5, "east")

User: "turn around"
Think: Turning around means rotating 180 degrees.
Action: turn_degrees(180)

User: "turn 40 degrees right"
Think: Right turn is negative degrees.
Action: turn_degrees(-40)

User: "face north"
Think: The user wants to face north.
Action: turn_magnetic("north")

User: "what do you see?"
Think: I need to look through the camera.
Action: look()
"""

MAX_STEPS = 10


# ── Tool Registry ──────────────────────────────────────────────────────────

class ToolRegistry:
    """Thin wrapper that exposes MotorCortex + perception + system commands
    as a uniform tool-call interface for the Planner.

    Movement tools delegate directly to MotorCortex.move / turn_degrees /
    turn_relative / turn_magnetic / halt.
    """

    def __init__(self):
        self._motor = None  # MotorCortex instance
        self._vlm_caption_fn: Optional[Callable] = None
        self._get_image_fn: Optional[Callable] = None
        self._scene_context: str = ""
        self._start_mission_fn: Optional[Callable] = None
        self._get_status_fn: Optional[Callable] = None
        self._quit_simulation_fn: Optional[Callable] = None
        self._release_harness_fn: Optional[Callable] = None
        self._goto_landmark_fn: Optional[Callable[[str], str]] = None
        self._sim_state_fn: Optional[Callable] = None
        self._randomize_fn: Optional[Callable] = None

    def set_goto_landmark(self, fn: Optional[Callable[[str], str]]) -> None:
        """Navigate to a named landmark (wired after planner + sim exist)."""
        self._goto_landmark_fn = fn

    def set_sim_state_fn(self, fn: Optional[Callable]) -> None:
        """Provide ``() -> (pos_3d, yaw_rad)`` for visual navigation."""
        self._sim_state_fn = fn

    def set_randomize_fn(self, fn: Optional[Callable]) -> None:
        """Provide a callable that re-randomizes block positions."""
        self._randomize_fn = fn

    def wire(
        self,
        motor,
        get_image_fn: Callable,
        vlm_caption_fn: Optional[Callable],
        scene_context: str,
        start_mission_fn: Optional[Callable] = None,
        get_status_fn: Optional[Callable] = None,
        quit_simulation_fn: Optional[Callable] = None,
        release_harness_fn: Optional[Callable] = None,
    ):
        """Connect the tools to live subsystems."""
        self._motor = motor
        self._get_image_fn = get_image_fn
        self._vlm_caption_fn = vlm_caption_fn
        self._scene_context = scene_context
        self._start_mission_fn = start_mission_fn
        self._get_status_fn = get_status_fn
        self._quit_simulation_fn = quit_simulation_fn
        self._release_harness_fn = release_harness_fn

    # ── Movement tools (delegate to MotorCortex) ─────────────────────

    def move(
        self,
        distance: float = 1.0,
        speed: float = 0.3,
        direction: Optional[str] = None,
    ) -> str:
        return self._motor.move(distance=distance, speed=speed, direction=direction)

    def turn_degrees(self, degrees: float = 90.0) -> str:
        return self._motor.turn_degrees(degrees)

    def turn_relative(self, direction: str = "left") -> str:
        return self._motor.turn_relative(direction)

    def turn_magnetic(self, heading: str = "north") -> str:
        return self._motor.turn_magnetic(heading)

    def goto_landmark(self, name: str = "") -> str:
        """Walk toward a target purely using camera vision.

        Extracts the color word from names like "pink box", "brown block",
        "the cyan cube" and uses fast HSV scan-and-navigate.
        No MuJoCo ground-truth data is used.
        """
        name = name.strip().lower()
        if not name:
            return "No target specified."
        color = _extract_color(name)
        if color and self._get_image_fn and self._motor:
            return _scan_and_navigate(
                color=color,
                get_image_fn=self._get_image_fn,
                vlm_caption_fn=self._vlm_caption_fn,
                motor=self._motor,
                turn_fn=lambda deg: self._motor.turn_degrees(deg),
            )
        return f"Cannot navigate to '{name}' — no matching color detected."

    def assess_terrain(self) -> str:
        """Use camera to assess terrain ahead for obstacles, stairs, etc."""
        if self._get_image_fn and self._vlm_caption_fn:
            image = self._get_image_fn()
            if image is not None:
                return self._vlm_caption_fn(
                    image,
                    "Analyze the terrain ahead. Are there stairs, ramps, obstacles, or flat ground? "
                    "Describe what you see and whether it is safe to walk forward.",
                )
        return "Camera not available for terrain assessment."

    def climb_stairs(self, direction: str = "up") -> str:
        """Attempt to climb stairs using the trained RL policy (if available)."""
        return f"Stair climbing ({direction}) is not yet available — RL policy not loaded."

    def goto_visual_target(self, description: str = "") -> str:
        """Walk toward a visually described target using the head camera + VLM."""
        if not description:
            return "No target description provided."
        if not (self._get_image_fn and self._vlm_caption_fn and self._motor and self._sim_state_fn):
            return "Visual navigation is not available (missing camera, VLM, or motor)."
        from cortex.peripheral.visual_nav import navigate_to_visual_target
        return navigate_to_visual_target(
            description=description,
            get_image_fn=self._get_image_fn,
            vlm_caption_fn=self._vlm_caption_fn,
            motor=self._motor,
            sim_state_fn=self._sim_state_fn,
        )

    def randomize_blocks(self) -> str:
        """Re-randomize marker block positions in the scene."""
        if self._randomize_fn:
            return self._randomize_fn()
        return "Block randomization not available."

    def halt(self) -> str:
        return self._motor.halt()

    # Legacy aliases for backward compat with VLM ReAct / rule-based
    def move_forward(self, meters: float = 1.0) -> str:
        return self.move(distance=meters)

    def turn(self, degrees: float = 90.0) -> str:
        return self.turn_degrees(degrees)

    def stop(self) -> str:
        return self.halt()

    # ── Perception tools ─────────────────────────────────────────────

    def look(self, prompt: str = "Describe what you see.") -> str:
        if self._get_image_fn and self._vlm_caption_fn:
            image = self._get_image_fn()
            if image is not None:
                import time as _t
                logger.info("[LOOK] Captured frame, starting VLM inference (prompt='%s')...", prompt[:60])
                t0 = _t.perf_counter()
                result = self._vlm_caption_fn(image, prompt)
                elapsed = (_t.perf_counter() - t0) * 1000
                logger.info("[LOOK] VLM inference done (%.0fms): %s",
                            elapsed, str(result)[:120])
                return result
        return "Camera not available."

    def describe_scene(self) -> str:
        return self._scene_context or "No scene context available."

    # ── State / system tools ─────────────────────────────────────────

    def get_sensor_info(self) -> str:
        """Return full IMU / INS sensor readings."""
        odo = self._motor._odometry if self._motor else None
        if not odo:
            return "Sensor data unavailable (no odometry)."

        lines = ["=== Robot Sensor Data ==="]

        # IMU orientation
        lines.append(
            f"Heading: {odo.heading_deg:.1f}° ({odo.heading_cardinal})"
            f"  |  Roll: {odo.roll_deg:.1f}°  |  Pitch: {odo.pitch_deg:.1f}°"
        )

        # IMU gyroscope
        g = odo.gyro
        lines.append(f"Gyro (rad/s): x={g[0]:.3f}  y={g[1]:.3f}  z={g[2]:.3f}")

        # IMU accelerometer
        a = odo.accel
        lines.append(f"Accel (m/s²): x={a[0]:.2f}  y={a[1]:.2f}  z={a[2]:.2f}")

        # Body velocity
        v = odo.body_velocity
        lines.append(f"Body vel (m/s): x={v[0]:.3f}  y={v[1]:.3f}  z={v[2]:.3f}  |  speed={odo.body_speed:.3f}")

        # Foot contact
        lines.append(
            f"Foot force (N): left={odo.left_foot_force:.1f}  right={odo.right_foot_force:.1f}"
        )

        # Step odometry
        lines.append(
            f"Odometer: {odo.total_distance:.2f}m  |  {odo.step_count} steps  |  stride={odo.stride_length:.3f}m"
        )

        # Ground truth for comparison
        pos = self._motor._get_pos()
        lines.append(f"qpos (sim ground truth): ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

        return "\n".join(lines)

    def get_position(self) -> str:
        odo = self._motor._odometry if self._motor else None
        pos = self._motor._get_pos()
        if odo:
            heading = odo.heading_deg
            cardinal = odo.heading_cardinal
            return (
                f"IMU heading: {heading:.0f}° ({cardinal}). "
                f"qpos: ({pos[0]:.2f}, {pos[1]:.2f})."
            )
        heading = self._motor._heading_str()
        return f"Position: ({pos[0]:.1f}, {pos[1]:.1f}), heading: {heading}."

    def get_status(self) -> str:
        if self._get_status_fn:
            return self._get_status_fn()
        return self.get_position()

    def start_mission(self) -> str:
        if self._start_mission_fn:
            self._start_mission_fn()
            return "Mission started."
        return "No mission loaded."

    def help(self) -> str:
        return (
            "I can do:\n"
            "  move(distance, speed, direction): 'walk 5 meters', 'go 10m east', 'run fast 3m'\n"
            "  turn_degrees(degrees): 'turn 45 degrees left', 'turn right 90'\n"
            "  turn_relative(direction): 'turn left', 'turn right'\n"
            "  turn_magnetic(heading): 'face north', 'turn east'\n"
            "  Navigation: 'walk to the podium', 'go backstage'\n"
            "  Perception: 'look around', 'what do you see', 'describe the scene'\n"
            "  Control: 'stop', 'release' (harness), 'start mission', 'help'\n"
            "  Session: 'quit' (ends simulation and saves video)"
        )

    def release_harness(self) -> str:
        if self._release_harness_fn:
            self._release_harness_fn()
            return "Harness released. Robot is free-standing."
        return "No harness control available."

    def quit_simulation(self) -> str:
        if self._quit_simulation_fn:
            self._quit_simulation_fn()
            return "Simulation ending..."
        return "Cannot quit from here."

    # ── Dispatch (for VLM ReAct loop) ────────────────────────────────

    def call(self, tool_name: str, args: Dict[str, Any]) -> str:
        fn = getattr(self, tool_name, None)
        if fn is None or tool_name.startswith("_"):
            return f"Unknown tool: {tool_name}"
        try:
            logger.info("[CMD:TOOL_START] %s(%s)", tool_name, args)
            t0 = time.perf_counter()
            result = fn(**args)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info("[CMD:TOOL_END] %s -> %.0fms", tool_name, elapsed_ms)
            return result
        except Exception as e:
            logger.error("[CMD:TOOL_ERROR] %s: %s", tool_name, e)
            return f"Tool error ({tool_name}): {e}"


# ── Planner ────────────────────────────────────────────────────────────────

class Planner:
    """Model JSON task chain first; optional embedding/rules fallbacks."""

    def __init__(
        self,
        tools: ToolRegistry,
        generate_text_fn: Optional[Callable] = None,
        planner_mode: str = "auto",
    ):
        self.tools = tools
        self._generate_text = generate_text_fn
        self.planner_mode = planner_mode  # auto | model | embedding | rules
        self._running = False
        self._vlm_available = False
        self._intent_matcher = None

    @property
    def vlm_ready(self) -> bool:
        """True when the VLM can accept requests."""
        return self._generate_text is not None and self._vlm_available

    def set_vlm_available(self, available: bool) -> None:
        self._vlm_available = available

    def set_intent_matcher(self, matcher) -> None:
        self._intent_matcher = matcher
        logger.info("IntentMatcher connected (loaded=%s)", matcher.loaded if matcher else False)

    def execute(
        self,
        instruction: str,
        scene_context: str,
        robot_pos: Tuple[float, float],
        robot_yaw_rad: float,
        landmark_map: Optional[Dict] = None,
    ) -> str:
        """Execute a high-level instruction via VLM tool calling.

        Only safety-critical keywords (stop/halt/release/help/quit) are
        handled instantly.  Everything else goes through the VLM planner
        which emits a JSON tool-call plan.
        """
        self._running = True
        t0 = time.perf_counter()
        lm = landmark_map or {}

        # Safety intercept — instant, no model needed
        safety_result = self._safety_intercept(instruction)
        if safety_result is not None:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info("[CMD:EXEC_END] instruction='%s' method=safety elapsed_ms=%.0f",
                        instruction, elapsed_ms)
            self._running = False
            return safety_result

        # VLM tool-calling planner
        if self.vlm_ready:
            logger.info("[CMD:EXEC_START] instruction='%s' method=vlm_tool_call", instruction)
            result = self._execute_model_task_chain(
                instruction, scene_context, robot_pos, robot_yaw_rad, lm,
            )
            if result is not None:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                logger.info("[CMD:EXEC_END] instruction='%s' method=vlm_tool_call elapsed_ms=%.0f",
                            instruction, elapsed_ms)
                self._running = False
                return result

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("[CMD:EXEC_END] instruction='%s' method=unhandled elapsed_ms=%.0f",
                    instruction, elapsed_ms)
        self._running = False
        if not self.vlm_ready:
            return "VLM is still loading. Try again in a few seconds."
        return f"I could not understand: '{instruction}'. Try rephrasing."

    def cancel(self):
        self._running = False
        self.tools.stop()

    def _try_direct_camera_question(self, instruction: str) -> Optional[str]:
        """One VLM caption pass for obvious camera questions (skip JSON planner).

        Moondream on iGPU is slow; planning + look() was two full forwards.  Simple
        'do you see …' / 'what do you see' style prompts only need the camera pass.
        """
        s = instruction.strip()
        if len(s) > 200:
            return None
        low = s.lower()
        motorish = (
            "walk", "go ", "turn ", "move ", "goto", "navigate",
            "meter", "metre", "degree", "forward", "backward", "left then", "right then",
        )
        if any(m in low for m in motorish):
            return None
        triggers = (
            "do you see",
            "can you see",
            "could you see",
            "what do you see",
            "what can you see",
            "tell me what you see",
            "describe what you see",
            "is there a ",
            "is there an ",
            "are there any ",
            "do you notice",
        )
        if not any(t in low for t in triggers):
            return None
        logger.info("[CMD:EXEC] direct_camera_question (skip JSON planner)")
        return self.tools.look(prompt=s)

    def _execute_model_task_chain(
        self,
        instruction: str,
        scene_context: str,
        robot_pos: Tuple[float, float],
        robot_yaw_rad: float,
        landmark_map: Dict,
    ) -> Optional[str]:
        """Full user message → VLM/LLM JSON plan → sequential WBC + visual tools."""
        from cortex.peripheral.model_task_chain import run_model_planner

        if not self._generate_text:
            return None

        fast = self._try_direct_camera_question(instruction)
        if fast is not None:
            return fast

        heading_deg = (90.0 - math.degrees(robot_yaw_rad)) % 360.0

        try:
            return run_model_planner(
                user_message=instruction,
                scene_context=scene_context,
                landmark_map=landmark_map,
                robot_pos=robot_pos,
                robot_yaw_rad=robot_yaw_rad,
                heading_deg=heading_deg,
                generate_text=self._generate_text,
                tools=self.tools,
                look_around_fn=self._look_around_sequence,
                running_check=lambda: self._running,
            )
        except Exception as e:
            logger.exception("[MODEL_PLAN] planner error: %s", e)
            return None

    # ── Embedding intent matcher (fallback) ──────────────────────────

    def _execute_with_intent_matcher(
        self,
        instruction: str,
        robot_pos: Tuple[float, float],
        robot_yaw_rad: float,
        landmark_map: Dict[str, Tuple[float, float, float]],
    ) -> Optional[str]:
        from cortex.peripheral.intent_matcher import ActionType

        # Try pathway splitting for compound instructions
        steps = self._intent_matcher.match_pathway(instruction)

        # If all steps are UNKNOWN, return None to fall through
        if all(s.action_type == ActionType.UNKNOWN for s in steps):
            return None

        results = []
        for i, action in enumerate(steps):
            if not self._running:
                results.append("Cancelled.")
                break

            logger.info(
                "[CMD:EXEC_STEP] %d/%d instruction='%s' intent=%s conf=%.3f",
                i + 1, len(steps), action.raw_text[:40],
                action.action_type.name, action.confidence,
            )

            if action.action_type == ActionType.UNKNOWN:
                results.append(f"(skipped: '{action.raw_text}')")
                continue

            result = self._dispatch_action(action, landmark_map, robot_pos, robot_yaw_rad)
            if result:
                results.append(result)

        return " ".join(results) if results else None

    def _dispatch_action(
        self, action, landmark_map, robot_pos, robot_yaw_rad,
    ) -> Optional[str]:
        """Execute a single ParsedAction via the appropriate motor/tool call."""
        from cortex.peripheral.intent_matcher import ActionType
        AT = ActionType

        dispatch = {
            AT.WALK:            lambda: self.tools.move(
                                    distance=action.meters or 1.0,
                                    speed=action.speed or 0.3,
                                    direction=action.direction,
                                ),
            AT.TURN:            lambda: self.tools.turn_degrees(action.degrees or 90.0),
            AT.LOOK:            lambda: self.tools.look(),
            AT.LOOK_AROUND:     lambda: self._look_around_sequence(),
            AT.NAVIGATE_TO:     lambda: self._navigate_to_landmark(
                                    action.landmark, landmark_map, robot_pos, robot_yaw_rad,
                                ),
            AT.STOP:            lambda: self.tools.halt(),
            AT.DESCRIBE_SCENE:  lambda: self.tools.describe_scene(),
            AT.GET_POSITION:    lambda: self.tools.get_position(),
            AT.GET_STATUS:      lambda: self.tools.get_status(),
            AT.RELEASE_HARNESS: lambda: self.tools.release_harness(),
            AT.START_MISSION:   lambda: self.tools.start_mission(),
            AT.HELP:            lambda: self.tools.help(),
            AT.QUIT:            lambda: self.tools.quit_simulation(),
        }

        handler = dispatch.get(action.action_type)
        return handler() if handler else None

    # ── Safety intercept (always instant) ──────────────────────────────

    _SAFETY_EXACT = {"stop", "halt", "freeze", "hold", "stay", "release",
                     "help", "status", "quit", "randomize"}

    def _safety_intercept(self, instruction: str) -> Optional[str]:
        """Handle safety-critical commands instantly without VLM."""
        instr_lower = instruction.lower().strip()

        if instr_lower in self._SAFETY_EXACT:
            logger.info("[CMD:EXEC_START] instruction='%s' method=safety", instruction)
            if instr_lower in ("stop", "halt", "freeze", "hold", "stay"):
                return self.tools.stop()
            if instr_lower == "release":
                return self.tools.release_harness()
            if instr_lower == "help":
                return self.tools.help()
            if instr_lower == "status":
                return self.tools.get_status()
            if instr_lower == "quit":
                return self.tools.quit_simulation()
            if instr_lower == "randomize":
                return self.tools.randomize_blocks()

        return None

    # ── Visual target extraction ─────────────────────────────────────

    _VISUAL_NAV_RE = re.compile(
        r"(?:go\s+to|walk\s+to|navigate\s+to|move\s+to|find|reach|approach|head\s+to)\s+(?:the\s+)?(.+)",
        re.IGNORECASE,
    )

    def _extract_visual_target(
        self, instruction: str, landmark_map: Dict,
    ) -> Optional[str]:
        """Return a visual-target description if the instruction is a 'go to <thing>' command.

        Returns None when:
        - the instruction doesn't match the pattern
        - the target matches a known landmark name (handled by goto_landmark instead)
        - visual nav prerequisites are missing
        """
        if not (self.tools._get_image_fn and self.tools._vlm_caption_fn
                and self.tools._motor and self.tools._sim_state_fn):
            return None

        m = self._VISUAL_NAV_RE.match(instruction.strip())
        if m is None:
            return None

        desc = m.group(1).strip().rstrip(".")
        if not desc or len(desc) < 2:
            return None

        # If the target matches a known landmark, let the landmark handler deal with it
        desc_lower = desc.lower()
        for lm_name in landmark_map:
            if lm_name.lower() in desc_lower or desc_lower in lm_name.lower():
                return None

        return desc

    # ── Compound actions ──────────────────────────────────────────────

    def _look_around_sequence(self) -> str:
        """360° scan: turn in 90° increments, describe each view."""
        descriptions = []
        directions = ["front", "left", "behind", "right"]
        for i, direction in enumerate(directions):
            if not self._running:
                break
            if i > 0:
                self.tools.turn_degrees(90)
            desc = self.tools.look()
            descriptions.append(f"Looking {direction}: {desc}")
        if len(descriptions) == 4:
            self.tools.turn_degrees(90)
        return "360° scan complete.\n" + "\n".join(descriptions)

    def _navigate_to_landmark(
        self,
        landmark_name: Optional[str],
        landmark_map: Dict[str, Tuple[float, float, float]],
        robot_pos: Tuple[float, float],
        robot_yaw_rad: float,
    ) -> str:
        """Walk to a named landmark using turn + move sequence."""
        if not landmark_name or not landmark_map:
            return self.tools.describe_scene()

        for name, (lx, ly, _) in landmark_map.items():
            if (landmark_name in name.lower()) or (name.lower() in landmark_name):
                return self._navigate_to_point(lx, ly, robot_pos, robot_yaw_rad)

        return f"I don't know where '{landmark_name}' is. Known: {', '.join(landmark_map.keys())}"

    def _navigate_to_point(
        self, target_x: float, target_y: float,
        robot_pos: Tuple[float, float], robot_yaw_rad: float,
    ) -> str:
        dx = target_x - robot_pos[0]
        dy = target_y - robot_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.3:
            return "Already at the target."

        target_bearing = math.atan2(dy, dx)
        turn_deg = math.degrees(target_bearing - robot_yaw_rad)
        turn_deg = (turn_deg + 180) % 360 - 180

        results = []
        if abs(turn_deg) > 5:
            results.append(self.tools.turn_degrees(turn_deg))
        results.append(self.tools.move(distance=dist))
        return " ".join(results)

    # ── VLM-based ReAct loop ──────────────────────────────────────────

    _TOOL_ALIASES = {
        "walk_straight": "move",
        "walk_forward": "move",
        "walk": "move",
        "move_forward": "move",
        "forward": "move",
        "run": "move",
        "turn": "turn_degrees",
        "rotate": "turn_degrees",
        "turn_left": "turn_relative",
        "turn_right": "turn_relative",
        "face": "turn_magnetic",
        "stop": "halt",
        "freeze": "halt",
        "see": "look",
        "observe": "look",
        "release": "release_harness",
        "position": "get_position",
        "status": "get_status",
        "scene": "describe_scene",
        "quit": "quit_simulation",
        "exit": "quit_simulation",
    }

    def _execute_with_lm(
        self, instruction: str, scene_context: str,
        robot_pos: Tuple[float, float], robot_yaw_rad: float,
    ) -> Optional[str]:
        heading_deg = math.degrees(robot_yaw_rad)
        prompt = (
            f"{TOOL_DESCRIPTIONS}\n"
            f"Scene context:\n{scene_context}\n\n"
            f"Robot state: position=({robot_pos[0]:.1f}, {robot_pos[1]:.1f}), "
            f"heading={heading_deg:.0f}°\n\n"
            f"User instruction: {instruction}\n"
        )

        for step in range(MAX_STEPS):
            if not self._running:
                return "Execution cancelled."

            try:
                response = self._generate_text(prompt)
            except Exception as e:
                logger.error("[CMD:VLM_ERROR] step=%d error=%s", step, e)
                return None

            if not response:
                return None

            # Truncate at first Action line to prevent hallucination
            response = self._truncate_after_action(response)
            logger.info("[CMD:VLM_STEP] step=%d response='%s'", step, response[:200])

            answer_match = _ANSWER_RE.search(response)
            if answer_match:
                return answer_match.group(1).strip()

            tool_match = _TOOL_CALL_RE.search(response)
            if not tool_match:
                logger.warning("[CMD:VLM_NO_ACTION] step=%d raw='%s'", step, response[:100])
                return None

            raw_name = tool_match.group(1)
            tool_name = self._TOOL_ALIASES.get(raw_name, raw_name)
            args = self._parse_args(tool_name, tool_match.group(2).strip())
            logger.info("[CMD:VLM_CALL] step=%d tool=%s (raw=%s) args=%s", step, tool_name, raw_name, args)
            tool_result = self.tools.call(tool_name, args)
            prompt += f"{response}\nResult: {tool_result}\n\n"

        return "I completed the requested actions."

    @staticmethod
    def _truncate_after_action(text: str) -> str:
        """Keep only up to the first Action: line (inclusive)."""
        lines = text.split("\n")
        result = []
        for line in lines:
            result.append(line)
            if line.strip().lower().startswith("action:"):
                break
        return "\n".join(result)

    # Maps tool name → ordered positional parameter names
    _POSITIONAL_PARAMS = {
        "move": ["distance", "speed", "direction"],
        "turn_degrees": ["degrees"],
        "turn_relative": ["direction"],
        "turn_magnetic": ["heading"],
    }

    @classmethod
    def _parse_args(cls, tool_name: str, args_str: str) -> Dict[str, Any]:
        args = {}
        if not args_str:
            return args
        positional = cls._POSITIONAL_PARAMS.get(tool_name, [])
        pos_idx = 0
        for part in args_str.split(","):
            part = part.strip().strip("'\"")
            if "=" in part:
                key, val = part.split("=", 1)
                args[key.strip()] = _try_parse_number(val.strip().strip("'\""))
            else:
                val = _try_parse_number(part)
                if pos_idx < len(positional):
                    args[positional[pos_idx]] = val
                    pos_idx += 1
                else:
                    args[f"arg{pos_idx}"] = val
                    pos_idx += 1
        return args


_STRIP_SUFFIXES = re.compile(
    r"\s*\b(box|block|cube|marker|object|thing|ball|sphere|target|item)\b\s*",
    re.IGNORECASE,
)


def _extract_color(name: str) -> Optional[str]:
    """Extract a known HSV color from a landmark name like 'pink box' or 'the brown cube'.

    Returns the color string if found, else None.
    """
    cleaned = _STRIP_SUFFIXES.sub(" ", name).strip()
    cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned).strip()
    if _has_hsv_range(cleaned):
        return cleaned
    for word in name.split():
        if _has_hsv_range(word):
            return word
    return None


def _has_hsv_range(color: str) -> bool:
    """Check if the block_detector has HSV ranges for a color."""
    try:
        from cortex.peripheral.block_detector import has_color
        return has_color(color)
    except ImportError:
        return False


def _vision_detect(frame: np.ndarray, color: str):
    """Run fast HSV detection, return BlockDetection or None."""
    try:
        from cortex.peripheral.block_detector import detect_block
        return detect_block(frame, color)
    except ImportError:
        return None


_FRAME_SETTLE_S = 0.20   # wait for sim to render a fresh frame after a turn
_TURN_OVERSHOOT = 1.05   # small overshoot to counteract turn tolerance deadband


def _fresh_frame(get_image_fn):
    """Wait for the sim to render a new frame after a physical action."""
    import time as _t
    _t.sleep(_FRAME_SETTLE_S)
    return get_image_fn()


def _turn_and_detect(color, bearing_deg, turn_fn, get_image_fn):
    """Turn toward a bearing, wait for a fresh frame, re-detect."""
    if abs(bearing_deg) > 3:
        adjusted = bearing_deg * _TURN_OVERSHOOT
        turn_fn(adjusted)
    frame = _fresh_frame(get_image_fn)
    if frame is None:
        return None
    return _vision_detect(frame, color)


def _scan_and_navigate(
    color: str,
    get_image_fn,
    vlm_caption_fn,
    motor,
    turn_fn,
    max_approach_steps: int = 20,
) -> str:
    """360° image-based scan for a colored block, then walk toward it.

    Phase 1 — Scan: rotate in 60° steps, stop on first detection.
              After detection, turn to face it and verify with a fresh frame.
    Phase 2 — Approach: walk toward block in steps, re-detecting each time.
              If lost, do a ±60° mini re-scan to re-acquire.
    """
    import time as _t

    t0 = _t.perf_counter()
    logger.info("[NAV-SCAN] Scanning for '%s' block using fast image detection...", color)

    # ── Phase 1: find the block ──────────────────────────────────────
    scan_step_deg = 60
    det = None

    for scan_i in range(360 // scan_step_deg):
        if scan_i > 0:
            turn_fn(scan_step_deg)
        frame = _fresh_frame(get_image_fn)
        if frame is None:
            continue
        det = _vision_detect(frame, color)
        if det is not None:
            logger.info(
                "[NAV-SCAN] Found '%s' at bearing=%.1f° dist=%.1fm (scan step %d)",
                color, det.bearing_deg, det.est_distance_m, scan_i,
            )
            break

    scan_ms = (_t.perf_counter() - t0) * 1000

    if det is None:
        logger.info("[NAV-SCAN] '%s' not found after 360° (%.0fms)", color, scan_ms)
        return f"Could not find a {color} block after scanning 360°."

    # Turn to centre the block, then verify with fresh frame
    det2 = _turn_and_detect(color, det.bearing_deg, turn_fn, get_image_fn)
    if det2 is not None:
        det = det2
        logger.info("[NAV-SCAN] After centering: %s bearing=%.1f° dist=%.1fm",
                     color, det.bearing_deg, det.est_distance_m)
    else:
        logger.info("[NAV-SCAN] Block not in fresh frame after centering, continuing anyway")

    initial_dist = det.est_distance_m

    # ── Phase 2: approach with re-detection ──────────────────────────
    total_walked = 0.0
    lost_count = 0
    last_det = det

    for step_i in range(max_approach_steps):
        frame = _fresh_frame(get_image_fn)
        if frame is None:
            break

        det = _vision_detect(frame, color)

        if det is None:
            # If we've walked most of the way, we're probably on top of it
            if total_walked > initial_dist * 0.6:
                logger.info("[NAV-SCAN] Lost %s after %.1fm (walked >60%% of %.1fm) — declaring arrival",
                            color, total_walked, initial_dist)
                break
            lost_count += 1
            if lost_count > 3:
                logger.info("[NAV-SCAN] Lost %s after %.1fm (gave up)", color, total_walked)
                break
            logger.info("[NAV-SCAN] Lost %s, mini re-scan (attempt %d)...", color, lost_count)
            det = _mini_rescan(color, get_image_fn, turn_fn)
            if det is None:
                logger.info("[NAV-SCAN] Mini re-scan failed, stopping")
                break
            _turn_and_detect(color, det.bearing_deg, turn_fn, get_image_fn)
            continue

        lost_count = 0
        last_det = det
        logger.info("[NAV-SCAN] Step %d: %s bearing=%.1f° dist=%.1fm",
                     step_i, color, det.bearing_deg, det.est_distance_m)

        if det.est_distance_m <= 1.5:
            final_step = max(0.3, det.est_distance_m * 0.4)
            motor.move(distance=final_step)
            total_walked += final_step
            logger.info("[NAV-SCAN] Close enough (%.1fm), final step %.1fm",
                        det.est_distance_m, final_step)
            break

        # Correct heading, then walk
        if abs(det.bearing_deg) > 5:
            det2 = _turn_and_detect(color, det.bearing_deg, turn_fn, get_image_fn)
            if det2 is not None:
                det = det2

        step = min(det.est_distance_m * 0.45, 2.0)
        motor.move(distance=step)
        total_walked += step

    elapsed_ms = (_t.perf_counter() - t0) * 1000
    logger.info("[NAV-SCAN] Navigation to '%s' complete: %.1fm walked in %.0fms",
                color, total_walked, elapsed_ms)

    msg = f"I can see the {color}! It's about {initial_dist:.1f}m away"
    if abs(det.bearing_deg if det else 0) < 10:
        msg += ", straight ahead."
    elif (det.bearing_deg if det else 0) < 0:
        msg += ", to my right."
    else:
        msg += ", to my left."
    msg += f" Walking there now.\nArrived, {total_walked:.1f}m walked."
    return msg


def _mini_rescan(color: str, get_image_fn, turn_fn):
    """Look ±60° to re-acquire a lost block. Returns detection or None."""
    for angle in [30, -60, 60, -30]:
        turn_fn(angle)
        frame = _fresh_frame(get_image_fn)
        if frame is not None:
            det = _vision_detect(frame, color)
            if det is not None:
                logger.info("[NAV-SCAN] Mini re-scan found %s at %.1f°", color, det.bearing_deg)
                return det
    return None


def _try_parse_number(s: str) -> Any:
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s
