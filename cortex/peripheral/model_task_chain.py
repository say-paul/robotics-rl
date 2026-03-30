"""
Model-driven task planning — no regex / string-splitting for intent.

The user's message is passed verbatim to the VLM text head (or LLM). The model
must emit a single JSON object describing an ordered list of tasks:

  * **motor** — WBC primitives (move, turn, …) executed in sequence.
  * **visual** — either head-camera captioning (``camera``) or scene-from-memory
    (``memory``).

Parsing is minimal: extract JSON from the model output and validate keys.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Whitelist: tool name -> allowed argument keys (values sanitized by type)
MOTOR_TOOL_SCHEMA: Dict[str, frozenset] = {
    "move": frozenset({"distance", "speed", "direction"}),
    "turn_degrees": frozenset({"degrees"}),
    "turn_relative": frozenset({"direction"}),
    "turn_magnetic": frozenset({"heading"}),
    "halt": frozenset(),
    "goto_landmark": frozenset({"name"}),
    "goto_visual_target": frozenset({"description"}),
    "get_position": frozenset(),
    "get_sensor_info": frozenset(),
    "describe_scene": frozenset(),
    "look": frozenset({"prompt"}),
    "look_around": frozenset(),
    "assess_terrain": frozenset(),
    "climb_stairs": frozenset({"direction"}),
}

_PLANNER_SYSTEM = """Robot planner. Output ONLY JSON. Tools: move, turn_degrees, turn_magnetic, halt, goto_landmark, get_position, get_sensor_info, look, look_around, describe_scene, assess_terrain, climb_stairs.
Landmarks: {landmarks}
Robot at ({px:.1f},{py:.1f}) heading {heading_deg:.0f}°.

IMPORTANT: goto_landmark already includes a fast 360° visual scan to find the target. Do NOT add a separate look_around or visual scan step before goto_landmark for colored blocks.

Examples:
User: turn 10 degrees
{{"tasks":[{{"category":"motor","tool":"turn_degrees","args":{{"degrees":10}}}}]}}

User: walk 3 meters
{{"tasks":[{{"category":"motor","tool":"move","args":{{"distance":3.0}}}}]}}

User: turn right 45 degrees
{{"tasks":[{{"category":"motor","tool":"turn_degrees","args":{{"degrees":-45}}}}]}}

User: go 2m north
{{"tasks":[{{"category":"motor","tool":"move","args":{{"distance":2.0,"direction":"north"}}}}]}}

User: walk 1m then turn left
{{"tasks":[{{"category":"motor","tool":"move","args":{{"distance":1.0}}}},{{"category":"motor","tool":"turn_degrees","args":{{"degrees":90}}}}]}}

User: what do you see
{{"tasks":[{{"category":"visual","visual_mode":"camera"}}]}}

User: where am I
{{"tasks":[{{"category":"motor","tool":"get_position","args":{{}}}}]}}

User: sensor info
{{"tasks":[{{"category":"motor","tool":"get_sensor_info","args":{{}}}}]}}

User: go to the orange block
{{"tasks":[{{"category":"motor","tool":"goto_landmark","args":{{"name":"orange"}}}}]}}

User: face east
{{"tasks":[{{"category":"motor","tool":"turn_magnetic","args":{{"heading":"east"}}}}]}}

User: stop
{{"tasks":[{{"category":"motor","tool":"halt","args":{{}}}}]}}

User: walk backward 2m
{{"tasks":[{{"category":"motor","tool":"move","args":{{"distance":-2.0}}}}]}}

User: go to purple block then to yellow block then to teal block
{{"tasks":[{{"category":"motor","tool":"goto_landmark","args":{{"name":"purple"}}}},{{"category":"motor","tool":"goto_landmark","args":{{"name":"yellow"}}}},{{"category":"motor","tool":"goto_landmark","args":{{"name":"teal"}}}}]}}

User: look around for green and go near it
{{"tasks":[{{"category":"motor","tool":"goto_landmark","args":{{"name":"green"}}}}]}}

User: go from orange to cyan to brown to magenta
{{"tasks":[{{"category":"motor","tool":"goto_landmark","args":{{"name":"orange"}}}},{{"category":"motor","tool":"goto_landmark","args":{{"name":"cyan"}}}},{{"category":"motor","tool":"goto_landmark","args":{{"name":"brown"}}}},{{"category":"motor","tool":"goto_landmark","args":{{"name":"magenta"}}}}]}}

User: go to brown box then come to yellow box
{{"tasks":[{{"category":"motor","tool":"goto_landmark","args":{{"name":"brown"}}}},{{"category":"motor","tool":"goto_landmark","args":{{"name":"yellow"}}}}]}}

User: find pink and then go to teal
{{"tasks":[{{"category":"motor","tool":"goto_landmark","args":{{"name":"pink"}}}},{{"category":"motor","tool":"goto_landmark","args":{{"name":"teal"}}}}]}}

User: assess the terrain ahead
{{"tasks":[{{"category":"motor","tool":"assess_terrain","args":{{}}}}]}}

User: climb the stairs
{{"tasks":[{{"category":"motor","tool":"climb_stairs","args":{{"direction":"up"}}}}]}}

User: {user_message}
"""


def build_planner_prompt(
    user_message: str,
    scene_context: str,
    landmark_names: str,
    robot_pos: Tuple[float, float],
    robot_yaw_rad: float,
    heading_deg: float,
) -> str:
    lm = landmark_names.strip() or "none"
    return _PLANNER_SYSTEM.format(
        landmarks=lm[:500],
        px=robot_pos[0],
        py=robot_pos[1],
        heading_deg=heading_deg,
        user_message=user_message.strip(),
    )


def extract_json_object(text: str) -> Optional[dict]:
    """Extract the first balanced JSON object from model output.

    Handles common VLM quirks: markdown fences, missing trailing braces.
    """
    if not text:
        return None
    s = text.strip()

    # Strip markdown code fences if present
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()

    start = s.find("{")
    if start < 0:
        return None

    # Find the matching closing brace via depth counting
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        c = s[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                blob = s[start:i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError as e:
                    logger.warning("[MODEL_PLAN] JSON decode failed: %s", e)
                    return None

    # VLMs sometimes omit trailing braces — try appending them
    if depth > 0:
        blob = s[start:] + "}" * depth
        logger.info("[MODEL_PLAN] Unbalanced JSON (depth=%d), appending %d closing brace(s)", depth, depth)
        try:
            return json.loads(blob)
        except json.JSONDecodeError as e:
            logger.warning("[MODEL_PLAN] JSON repair failed: %s", e)
            return None

    logger.warning("[MODEL_PLAN] No balanced JSON object found")
    return None


def _sanitize_value(key: str, val: Any) -> Any:
    if val is None:
        return None
    if key == "direction" or key == "heading" or key == "name":
        return str(val).strip()[:120]
    if key == "degrees" or key == "distance" or key == "speed":
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
    return val


def sanitize_motor_args(tool: str, args: Any) -> Dict[str, Any]:
    if not isinstance(args, dict):
        return {}
    allowed = MOTOR_TOOL_SCHEMA.get(tool)
    if allowed is None:
        return {}
    out: Dict[str, Any] = {}
    for k, v in args.items():
        if k not in allowed:
            continue
        if v is None and k in ("direction", "speed"):
            continue
        sv = _sanitize_value(k, v)
        if sv is not None:
            out[k] = sv
    return out


def execute_task_plan(
    plan: dict,
    tools: Any,
    look_around_fn: Callable[[], str],
    *,
    running_check: Callable[[], bool],
) -> str:
    """Run tasks in order; return concatenated user-facing results."""
    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        return plan.get("plan_summary") or "No tasks in plan."

    summary = plan.get("plan_summary", "")
    results: List[str] = []
    if summary:
        results.append(f"Plan: {summary}")

    for i, task in enumerate(tasks):
        if not running_check():
            results.append("Cancelled.")
            break
        if not isinstance(task, dict):
            continue

        cat = str(task.get("category", "")).lower().strip()
        logger.info("[MODEL_PLAN:STEP] %d/%d category=%s %s", i + 1, len(tasks), cat, task)

        if cat == "visual":
            mode = str(task.get("visual_mode", "camera")).lower().strip()
            # Check if next task is goto_landmark — skip or use targeted prompt
            next_task = tasks[i + 1] if i + 1 < len(tasks) else None
            next_is_goto = (next_task and isinstance(next_task, dict)
                            and str(next_task.get("tool", "")).strip() == "goto_landmark")
            if mode == "memory":
                results.append(tools.describe_scene())
            elif mode == "around" or mode == "scan":
                if next_is_goto:
                    logger.info("[MODEL_PLAN] Skipping redundant visual scan before goto_landmark")
                    continue
                results.append(look_around_fn())
            else:
                if next_is_goto:
                    # goto_landmark already does a fast 360° visual scan — skip slow VLM look
                    target = str(next_task.get("args", {}).get("name", ""))
                    logger.info("[MODEL_PLAN] Skipping VLM camera look before goto_landmark('%s')", target)
                    continue
                results.append(tools.look())
            continue

        if cat == "motor":
            tool = str(task.get("tool", "")).strip()
            raw_args = task.get("args", {})
            if tool not in MOTOR_TOOL_SCHEMA:
                results.append(f"(skip unknown tool: {tool})")
                continue
            args = sanitize_motor_args(tool, raw_args)
            if tool == "look_around":
                results.append(look_around_fn())
                continue
            fn = getattr(tools, tool, None)
            if fn is None or tool.startswith("_"):
                results.append(f"(skip missing tool: {tool})")
                continue
            try:
                out = fn(**args) if args else fn()
                results.append(out)
            except TypeError:
                try:
                    results.append(fn())
                except Exception as e:
                    logger.exception("[MODEL_PLAN] tool %s failed", tool)
                    results.append(f"Error in {tool}: {e}")
            except Exception as e:
                logger.exception("[MODEL_PLAN] tool %s failed", tool)
                results.append(f"Error in {tool}: {e}")
            continue

        results.append(f"(skip unknown category: {cat})")

    return "\n".join(results) if len(results) > 1 else (results[0] if results else "Done.")


def run_model_planner(
    user_message: str,
    scene_context: str,
    landmark_map: Dict[str, Tuple[float, float, float]],
    robot_pos: Tuple[float, float],
    robot_yaw_rad: float,
    heading_deg: float,
    generate_text: Callable,
    tools: Any,
    look_around_fn: Callable[[], str],
    running_check: Callable[[], bool],
) -> Optional[str]:
    """
    Ask the model for a JSON plan and execute it.
    Returns None if the model returned empty/unparseable JSON (caller may fall back).
    """
    names = ", ".join(sorted(landmark_map.keys())) if landmark_map else ""
    prompt = build_planner_prompt(
        user_message, scene_context, names, robot_pos, robot_yaw_rad, heading_deg,
    )
    raw = generate_text(prompt, max_new_tokens=512)
    if not raw:
        logger.warning("[MODEL_PLAN] Empty model response")
        return None

    logger.info("[MODEL_PLAN] Raw response (first 500 chars): %s", raw[:500])
    plan = extract_json_object(raw)
    if not plan:
        return None

    return execute_task_plan(plan, tools, look_around_fn, running_check=running_check)
