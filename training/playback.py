#!/usr/bin/env python3
"""
Motion Playback
===============
Plays bone-seed CSV clips through the G1 on the water_bottle_stage.

Two modes:
  1. Reference-only: pure bone-seed CSV joint playback (no Groot)
  2. Groot: Groot locomotion controller with velocity commands + arm ref

Usage:
  python -m training.playback --task stair_climb --mode reference
  python -m training.playback --task stair_climb --mode groot
  python -m training.playback --task-sequence --mode groot
  python -m training.playback --task wave_gesture --record wave.mp4
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONTROL_DT, SIM_DT, NUM_JOINTS, JOINT_NAMES
from scene.simulation import G1Simulation
from training.motion_library import MotionClip, MotionLibrary, load_library

logger = logging.getLogger("playback")

PHYSICS_STEPS_PER_CONTROL = max(1, round(CONTROL_DT / SIM_DT))
GROOT_JOINTS = 15

TASK_SPAWNS = {
    "stair_climb":    np.array([-3.5, 0.0, 0.75]),
    "stair_descend":  np.array([0.5, 0.0, 1.75]),
    "wave_gesture":   np.array([1.5, 0.0, 1.75]),
    "locomotion":     np.array([-5.0, 0.0, 0.75]),
    "dance":          np.array([1.0, 0.0, 1.75]),
    "jump":           np.array([-4.0, 0.0, 0.75]),
}

TASK_VELOCITIES = {
    "stair_climb":    (0.4, 0.0, 0.0),
    "stair_descend":  (-0.2, 0.0, 0.0),
    "wave_gesture":   (0.0, 0.0, 0.0),
    "locomotion":     (0.3, 0.0, 0.0),
}


def _get_joint_pos(sim: G1Simulation) -> np.ndarray:
    pos = np.zeros(NUM_JOINTS)
    for i, jnt_id in enumerate(sim._body_jnt_ids):
        jnt = sim.model.joint(jnt_id)
        pos[i] = sim.data.qpos[jnt.qposadr[0]]
    return pos


def playback_reference(sim, clip, speed=1.0, record_frames=None):
    """Pure open-loop bone-seed playback (no Groot)."""
    import mujoco

    spawn = TASK_SPAWNS.get(clip.task, np.array([-5.0, 0.0, 0.75]))
    sim.reset()
    sim.data.qpos[0:3] = spawn
    sim.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    _, _, ref_ja = clip.frame_at(0.0)
    for i, jnt_id in enumerate(sim._body_jnt_ids):
        jnt = sim.model.joint(jnt_id)
        sim.data.qpos[jnt.qposadr[0]] = ref_ja[i]
    mujoco.mj_forward(sim.model, sim.data)
    sim.band.enable = False

    t = 0.0
    step = 0
    errors = []

    logger.info("Playing '%s' (%.1fs) task=%s  mode=reference",
                clip.name, clip.duration, clip.task)

    while t < clip.duration:
        _, _, ref_ja = clip.frame_at(t)
        for _ in range(PHYSICS_STEPS_PER_CONTROL):
            sim.step_with_target_array(ref_ja)

        cur = _get_joint_pos(sim)
        errors.append(float(np.mean((cur - ref_ja) ** 2)))

        if record_frames is not None:
            record_frames.append(sim.render_tracking(azimuth=150, elevation=-20, distance=4.0))

        sim.sync_viewer()
        t += CONTROL_DT * speed
        step += 1
        if not sim.viewer_running:
            break

    stats = {"clip": clip.name, "task": clip.task, "steps": step,
             "tracking_err": float(np.mean(errors)) if errors else 0,
             "max_h": sim.get_base_height()}
    logger.info("  done: %d steps, err=%.4f, h=%.2f", step, stats["tracking_err"], stats["max_h"])
    return stats


def playback_groot(sim, clip, speed=1.0, record_frames=None):
    """Groot locomotion playback with arm targets from bone-seed reference."""
    import mujoco
    from lerobot.robots.unitree_g1.gr00t_locomotion import GrootLocomotionController
    from lerobot.robots.unitree_g1.g1_utils import default_remote_input, G1_29_JointIndex

    groot = GrootLocomotionController()

    spawn = TASK_SPAWNS.get(clip.task, np.array([-5.0, 0.0, 0.75]))
    sim.reset()
    groot.reset()
    sim.data.qpos[0:3] = spawn
    sim.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    _, _, ref_ja = clip.frame_at(0.0)
    for i, jnt_id in enumerate(sim._body_jnt_ids):
        jnt = sim.model.joint(jnt_id)
        sim.data.qpos[jnt.qposadr[0]] = ref_ja[i]
    mujoco.mj_forward(sim.model, sim.data)
    sim.band.enable = False

    for _ in range(10):
        ls = sim.get_lowstate()
        groot.run_step(default_remote_input(), ls)
        sim.step_idle()

    vx, vy, vyaw = TASK_VELOCITIES.get(clip.task, (0.0, 0.0, 0.0))
    t = 0.0
    step = 0
    errors = []
    heights = []

    logger.info("Playing '%s' (%.1fs) task=%s  mode=groot  vel=(%.1f, %.1f, %.1f)",
                clip.name, clip.duration, clip.task, vx, vy, vyaw)

    while t < clip.duration:
        _, _, ref_ja = clip.frame_at(t)

        groot_input = default_remote_input()
        groot_input["remote.ly"] = vx
        groot_input["remote.lx"] = vy
        groot_input["remote.rx"] = vyaw

        ls = sim.get_lowstate()
        groot_out = groot.run_step(groot_input, ls)

        groot_targets = np.zeros(GROOT_JOINTS)
        for i in range(GROOT_JOINTS):
            key = f"{G1_29_JointIndex(i).name}.q"
            groot_targets[i] = groot_out.get(key, 0.0)

        full_targets = np.zeros(NUM_JOINTS)
        full_targets[:GROOT_JOINTS] = groot_targets
        full_targets[GROOT_JOINTS:] = ref_ja[GROOT_JOINTS:]

        for _ in range(PHYSICS_STEPS_PER_CONTROL):
            sim.step_with_target_array(full_targets)

        cur = _get_joint_pos(sim)
        errors.append(float(np.mean((cur - ref_ja) ** 2)))
        heights.append(sim.get_base_height())

        if record_frames is not None:
            record_frames.append(sim.render_tracking(azimuth=150, elevation=-20, distance=4.0))

        sim.sync_viewer()
        t += CONTROL_DT * speed
        step += 1
        if not sim.viewer_running:
            break

    stats = {
        "clip": clip.name, "task": clip.task, "steps": step,
        "tracking_err": float(np.mean(errors)) if errors else 0,
        "max_h": float(np.max(heights)) if heights else 0,
        "final_h": float(heights[-1]) if heights else 0,
    }
    logger.info("  done: %d steps, err=%.4f, max_h=%.2f, final_h=%.2f",
                step, stats["tracking_err"], stats["max_h"], stats["final_h"])
    return stats


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-12s  %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Bone-seed motion playback for G1")
    p.add_argument("--clip", type=str, default=None, help="Specific clip name (stem)")
    p.add_argument("--task", type=str, default=None)
    p.add_argument("--task-sequence", action="store_true",
                   help="Play stair_climb → wave_gesture → stair_descend")
    p.add_argument("--mode", default="groot", choices=["reference", "groot"])
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--record", type=str, default=None, help="Output video path")
    p.add_argument("--headless", action="store_true")
    args = p.parse_args()

    logger.info("Loading motion library …")
    lib = load_library(skip_mirrored=True)

    headless = args.headless or (args.record is not None and not sys.stdout.isatty())
    sim = G1Simulation(scene_name="water_bottle_stage", headless=headless)

    clips_to_play = []
    if args.task_sequence:
        for task in ["stair_climb", "wave_gesture", "stair_descend"]:
            c = lib.sample_clip(task)
            if c:
                clips_to_play.append(c)
    elif args.clip:
        found = [c for c in lib.clips if c.name == args.clip]
        if not found:
            logger.error("Clip '%s' not found", args.clip)
            sys.exit(1)
        clips_to_play = found
    elif args.task:
        c = lib.sample_clip(args.task)
        if c is None:
            logger.error("No clips for task '%s'", args.task)
            sys.exit(1)
        clips_to_play = [c]
    else:
        rng = np.random.default_rng()
        clips_to_play = [lib.clips[rng.integers(len(lib))]]

    record_frames = [] if args.record else None
    all_stats = []

    for clip in clips_to_play:
        if args.mode == "reference":
            stats = playback_reference(sim, clip, args.speed, record_frames)
        else:
            stats = playback_groot(sim, clip, args.speed, record_frames)
        all_stats.append(stats)
        time.sleep(0.5)

    if args.record and record_frames:
        import imageio
        out_path = Path(args.record)
        fps = int(1.0 / CONTROL_DT)
        imageio.mimwrite(str(out_path), record_frames, fps=fps)
        logger.info("Recorded %d frames → %s", len(record_frames), out_path)

    sim.close()
    logger.info("Done. %d clips played.", len(all_stats))
    for s in all_stats:
        logger.info("  %-40s  task=%-15s  err=%.4f  max_h=%.2f",
                     s["clip"], s["task"], s["tracking_err"], s["max_h"])


if __name__ == "__main__":
    main()
