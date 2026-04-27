"""Handler for the SONIC local-motion planner.

Assembles the 11 ONNX inputs expected by planner_sonic.onnx from signals
on the bus.  The keyboard controller writes mode, movement_direction,
facing_direction, target_vel, and height; the WBC handler writes
context_mujoco_qpos (a rolling 4-frame window of full MuJoCo qpos).

Usage in YAML:
    nodes:
      - name: "planner_inference"
        model: "local_motion_planner"
        handler: "handlers.sonic_planner.build_planner_obs"
"""

import numpy as np

DEFAULT_RANDOM_SEED = 1234
DEFAULT_HEIGHT = 0.788740

ALLOWED_PRED_NUM_TOKENS = np.array(
    [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int64)


def build_planner_obs(bus, model_cfg, node_cfg):
    """Build 11 planner inputs from bus signals.

    Returns dict[str, np.ndarray] keyed by ONNX input name, or None to
    skip inference.  Skips when:
      - context_mujoco_qpos is not yet on the bus
      - mode is IDLE (0) with zero movement direction (no point running a
        heavy model just to produce a "stand still" trajectory)
    """
    context = bus.get("context_mujoco_qpos")
    if context is None:
        return None

    mode_arr = bus.get("mode")
    mode = int(mode_arr.flat[0]) if mode_arr is not None else 0

    mv = bus.get("movement_direction")
    if mv is not None:
        movement = mv.astype(np.float32).flatten()
    else:
        movement = np.zeros(3, dtype=np.float32)

    has_movement = np.any(np.abs(movement) > 1e-6)
    if mode == 0 or not has_movement:
        return None

    context = context.astype(np.float32).reshape(1, 4, 36)
    movement = movement.reshape(1, 3)

    fd = bus.get("facing_direction")
    facing = fd.astype(np.float32).reshape(1, 3) if fd is not None else np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    tv = bus.get("target_vel")
    target_vel = float(tv.flat[0]) if tv is not None else -1.0

    h = bus.get("height")
    height = float(h.flat[0]) if h is not None else -1.0

    return {
        "context_mujoco_qpos": context,
        "target_vel": np.array([target_vel], dtype=np.float32),
        "mode": np.array([mode], dtype=np.int64),
        "movement_direction": movement,
        "facing_direction": facing,
        "random_seed": np.array([DEFAULT_RANDOM_SEED], dtype=np.int64),
        "has_specific_target": np.zeros((1, 1), dtype=np.int64),
        "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
        "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
        "allowed_pred_num_tokens": ALLOWED_PRED_NUM_TOKENS.copy(),
        "height": np.array([height], dtype=np.float32),
    }
