#!/usr/bin/env python3
"""
LeRobot Dataset Generator
=========================
Plays bone-seed CSV clips through the MuJoCo water_bottle_stage sim,
captures camera frames + joint states, and writes a LeRobot v3.0
dataset suitable for training SmolVLA and SARM.

Each bone-seed clip becomes one episode.  The dataset records:
  - observation.state:  29-dim joint positions (radians)
  - observation.images.head_camera:  RGB frame from the robot's head camera
  - observation.images.global_view:  RGB tracking camera (for SARM stage viz)
  - action:  29-dim joint targets (next-frame reference from the bone-seed)

Usage:
  # Generate from stair clips only
  python -m training.generate_dataset \
      --tasks stair_climb,stair_descend \
      --output /home/robotics-rl/datasets/g1_stairs \
      --max-clips-per-task 10

  # Generate full dataset (all tasks)
  python -m training.generate_dataset \
      --output /home/robotics-rl/datasets/g1_wb_full

  # Quick test with 2 clips
  python -m training.generate_dataset \
      --tasks stair_climb --max-clips-per-task 2 \
      --output /home/robotics-rl/datasets/g1_test
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from config import CONTROL_DT, SIM_DT, NUM_JOINTS
from scene.simulation import G1Simulation
from training.motion_library import load_library

logger = logging.getLogger("generate_dataset")

PHYSICS_STEPS_PER_CONTROL = max(1, round(CONTROL_DT / SIM_DT))

TASK_SPAWNS = {
    "stair_climb":    np.array([-3.5, 0.0, 0.75]),
    "stair_descend":  np.array([0.5, 0.0, 1.75]),
    "wave_gesture":   np.array([1.5, 0.0, 1.75]),
    "locomotion":     np.array([-5.0, 0.0, 0.75]),
    "dance":          np.array([1.0, 0.0, 1.75]),
    "jump":           np.array([-4.0, 0.0, 0.75]),
}

TASK_DESCRIPTIONS = {
    "stair_climb":   "walk up the stairs onto the stage",
    "stair_descend": "walk down the stairs from the stage",
    "wave_gesture":  "wave to the audience on stage",
    "locomotion":    "walk forward on flat ground",
    "dance":         "perform a dance routine on stage",
    "jump":          "jump in place",
}

RENDER_SIZE = (256, 256)
FPS = int(1.0 / CONTROL_DT)  # 50 fps


def get_joint_pos(sim: G1Simulation) -> np.ndarray:
    pos = np.zeros(NUM_JOINTS, dtype=np.float32)
    for i, jnt_id in enumerate(sim._body_jnt_ids):
        jnt = sim.model.joint(jnt_id)
        pos[i] = sim.data.qpos[jnt.qposadr[0]]
    return pos


def render_episode(sim, clip):
    """Play a bone-seed clip through MuJoCo and collect frames + states + actions."""
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

    states = []
    head_frames = []
    global_frames = []
    actions = []
    timestamps = []

    t = 0.0
    step = 0

    while t < clip.duration - CONTROL_DT:
        _, _, ref_ja = clip.frame_at(t)
        _, _, ref_ja_next = clip.frame_at(t + CONTROL_DT)

        state = get_joint_pos(sim)
        states.append(state)
        actions.append(ref_ja_next.astype(np.float32))
        timestamps.append(float(t))

        head_img = sim.render_frame(camera="head_camera")
        head_frames.append(head_img)

        global_img = sim.render_tracking(azimuth=150, elevation=-20, distance=4.0)
        global_frames.append(global_img)

        for _ in range(PHYSICS_STEPS_PER_CONTROL):
            sim.step_with_target_array(ref_ja)

        t += CONTROL_DT
        step += 1

    return {
        "states": states,
        "actions": actions,
        "head_frames": head_frames,
        "global_frames": global_frames,
        "timestamps": timestamps,
        "n_frames": len(states),
    }


def build_dataset(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-18s  %(message)s",
        datefmt="%H:%M:%S",
    )

    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    logger.info("Loading motion library (tasks=%s) …", tasks or "all")
    lib = load_library(
        tasks=tasks,
        skip_mirrored=True,
        max_clips_per_task=args.max_clips_per_task,
    )

    if not lib.clips:
        logger.error("No clips loaded. Check --tasks argument.")
        sys.exit(1)

    logger.info("Loaded %d clips across tasks: %s", len(lib), lib.tasks)

    output_root = Path(args.output)
    repo_id = args.repo_id or output_root.name

    logger.info("Creating sim (render_size=%s) …", RENDER_SIZE)
    sim = G1Simulation(
        scene_name="water_bottle_stage",
        headless=True,
        render_size=RENDER_SIZE,
    )

    logger.info("Creating LeRobot dataset at %s …", output_root)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (NUM_JOINTS,),
            "names": None,
        },
        "observation.images.head_camera": {
            "dtype": "image",
            "shape": (RENDER_SIZE[1], RENDER_SIZE[0], 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.global_view": {
            "dtype": "image",
            "shape": (RENDER_SIZE[1], RENDER_SIZE[0], 3),
            "names": ["height", "width", "channel"],
        },
        "action": {
            "dtype": "float32",
            "shape": (NUM_JOINTS,),
            "names": None,
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=features,
        root=str(output_root),
        robot_type="unitree_g1",
        use_videos=False,
    )

    total_frames = 0

    for clip_idx, clip in enumerate(lib.clips):
        task_desc = TASK_DESCRIPTIONS.get(clip.task, clip.task)
        logger.info(
            "[%d/%d] Rendering '%s' (%d bone-seed frames, task='%s') …",
            clip_idx + 1, len(lib.clips), clip.name, clip.n_frames, task_desc,
        )

        ep_data = render_episode(sim, clip)

        for i in range(ep_data["n_frames"]):
            dataset.add_frame({
                "observation.state": ep_data["states"][i],
                "observation.images.head_camera": ep_data["head_frames"][i],
                "observation.images.global_view": ep_data["global_frames"][i],
                "action": ep_data["actions"][i],
                "task": task_desc,
            })

        dataset.save_episode()
        total_frames += ep_data["n_frames"]
        logger.info("  saved episode %d: %d frames", clip_idx, ep_data["n_frames"])

    dataset.finalize()
    sim.close()

    logger.info(
        "Dataset complete: %d episodes, %d total frames → %s",
        len(lib.clips), total_frames, output_root,
    )
    logger.info("repo_id='%s'  root='%s'", repo_id, output_root)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  # Train SARM reward model:")
    logger.info("  cd /home/lerobot && lerobot-train \\")
    logger.info("    --dataset.repo_id=%s \\", repo_id)
    logger.info("    --dataset.root=%s \\", output_root)
    logger.info("    --policy.type=sarm \\")
    logger.info("    --policy.annotation_mode=single_stage \\")
    logger.info("    --policy.image_key=observation.images.global_view \\")
    logger.info("    --policy.push_to_hub=false \\")
    logger.info("    --batch_size=32 --steps=5000 \\")
    logger.info("    --output_dir=outputs/train/sarm_g1")
    logger.info("")
    logger.info("  # Fine-tune SmolVLA:")
    logger.info("  cd /home/lerobot && lerobot-train \\")
    logger.info("    --policy.path=lerobot/smolvla_base \\")
    logger.info("    --dataset.repo_id=%s \\", repo_id)
    logger.info("    --dataset.root=%s \\", output_root)
    logger.info("    --policy.push_to_hub=false \\")
    logger.info("    --batch_size=32 --steps=20000 \\")
    logger.info("    --output_dir=outputs/train/smolvla_g1")


def main():
    p = argparse.ArgumentParser(
        description="Generate LeRobot dataset from bone-seed + MuJoCo",
    )
    p.add_argument(
        "--output", required=True,
        help="Output directory for the LeRobot dataset",
    )
    p.add_argument(
        "--repo-id", default=None,
        help="Dataset repo_id (default: output dir name)",
    )
    p.add_argument(
        "--tasks", default=None,
        help="Comma-separated task list (default: all)",
    )
    p.add_argument(
        "--max-clips-per-task", type=int, default=0,
        help="Max clips per task (0 = all)",
    )
    args = p.parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
