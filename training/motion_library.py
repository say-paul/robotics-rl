"""
Motion Library – bone-seed CSV loader
======================================
Loads retargeted G1 joint trajectories from /home/hf-datasets/g1/csv/
and indexes them by behavioural category for training playback.

CSV format (per frame):
  Frame, root_translate{X,Y,Z}, root_rotate{X,Y,Z}, <29 joint>_dof
  - Root translation in *centimetres*, root rotation and joints in *degrees*.
  - We convert to metres / radians on load.
"""

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import DATASET_ROOT, CSV_JOINT_COLUMNS, NUM_JOINTS

logger = logging.getLogger(__name__)

# ── Task ↔ filename prefix mapping ──────────────────────────────────────────

TASK_PREFIXES: Dict[str, List[str]] = {
    "stair_climb": ["come_up_50cm_box"],
    "stair_descend": ["come_down_50cm_box"],
    "wave_gesture": ["welcoming", "greetings_hat", "shoulder_clap"],
    "locomotion": [
        "jog_ff_start", "jog_ff_loop", "jog_ff_stop",
        "walking_on_edge",
    ],
    "dance": ["dancing_routine"],
    "jump": ["jump_ff", "high_jump", "reach_jump"],
    "sit": ["sit_cross_legged", "sitting_legs"],
}

CM_TO_M = 0.01
DEG_TO_RAD = np.pi / 180.0


@dataclass
class MotionClip:
    """Single motion trajectory."""
    name: str
    task: str
    n_frames: int
    dt: float  # seconds per frame (assumed 30 fps source)
    root_pos: np.ndarray       # (T, 3) metres
    root_euler: np.ndarray     # (T, 3) radians
    joint_angles: np.ndarray   # (T, 29) radians

    @property
    def duration(self) -> float:
        return self.n_frames * self.dt

    def frame_at(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get (root_pos, root_euler, joint_angles) at time t with linear interp."""
        idx_f = t / self.dt
        idx_lo = int(np.clip(np.floor(idx_f), 0, self.n_frames - 1))
        idx_hi = int(np.clip(idx_lo + 1, 0, self.n_frames - 1))
        alpha = idx_f - idx_lo
        alpha = np.clip(alpha, 0.0, 1.0)
        rp = (1 - alpha) * self.root_pos[idx_lo] + alpha * self.root_pos[idx_hi]
        re = (1 - alpha) * self.root_euler[idx_lo] + alpha * self.root_euler[idx_hi]
        ja = (1 - alpha) * self.joint_angles[idx_lo] + alpha * self.joint_angles[idx_hi]
        return rp, re, ja


@dataclass
class MotionLibrary:
    """Indexed collection of motion clips."""
    clips: List[MotionClip] = field(default_factory=list)
    _by_task: Dict[str, List[MotionClip]] = field(default_factory=dict, repr=False)

    def add(self, clip: MotionClip) -> None:
        self.clips.append(clip)
        self._by_task.setdefault(clip.task, []).append(clip)

    def get_clips(self, task: str) -> List[MotionClip]:
        return self._by_task.get(task, [])

    def sample_clip(self, task: str, rng: Optional[np.random.Generator] = None) -> Optional[MotionClip]:
        clips = self.get_clips(task)
        if not clips:
            return None
        rng = rng or np.random.default_rng()
        return clips[rng.integers(len(clips))]

    @property
    def tasks(self) -> List[str]:
        return list(self._by_task.keys())

    def __len__(self) -> int:
        return len(self.clips)


# ── CSV parsing ─────────────────────────────────────────────────────────────

def _parse_csv(path: Path, fps: float = 30.0) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Parse a single bone-seed CSV.

    Returns (root_pos_m, root_euler_rad, joint_angles_rad) or None on error.
    """
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None

    if not rows:
        return None

    n = len(rows)
    root_pos = np.zeros((n, 3))
    root_euler = np.zeros((n, 3))
    joints = np.zeros((n, NUM_JOINTS))

    for i, row in enumerate(rows):
        root_pos[i, 0] = float(row["root_translateX"]) * CM_TO_M
        root_pos[i, 1] = float(row["root_translateY"]) * CM_TO_M
        root_pos[i, 2] = float(row["root_translateZ"]) * CM_TO_M
        root_euler[i, 0] = float(row["root_rotateX"]) * DEG_TO_RAD
        root_euler[i, 1] = float(row["root_rotateY"]) * DEG_TO_RAD
        root_euler[i, 2] = float(row["root_rotateZ"]) * DEG_TO_RAD
        for j, col in enumerate(CSV_JOINT_COLUMNS):
            joints[i, j] = float(row[col]) * DEG_TO_RAD

    return root_pos, root_euler, joints


def _classify(filename: str) -> Optional[str]:
    """Return the task label for a given CSV filename."""
    stem = filename.lower()
    for task, prefixes in TASK_PREFIXES.items():
        for prefix in prefixes:
            if stem.startswith(prefix):
                return task
    return None


def load_library(
    root: Optional[Path] = None,
    tasks: Optional[Sequence[str]] = None,
    skip_mirrored: bool = False,
    fps: float = 30.0,
    max_clips_per_task: int = 0,
) -> MotionLibrary:
    """Scan the bone-seed dataset and build an indexed MotionLibrary.

    Args:
        root:   Path to CSV directory (default: config.DATASET_ROOT)
        tasks:  Only load clips for these tasks (None = all)
        skip_mirrored: If True, skip *_M.csv (mirrored) variants
        fps:    Source capture frame rate
        max_clips_per_task: Cap per category (0 = unlimited)
    """
    root = root or DATASET_ROOT
    lib = MotionLibrary()

    csv_dir = root / "230530" if (root / "230530").is_dir() else root
    csv_files = sorted(csv_dir.glob("*.csv"))
    logger.info("Found %d CSV files in %s", len(csv_files), csv_dir)

    task_count: Dict[str, int] = {}

    for csv_path in csv_files:
        if skip_mirrored and csv_path.stem.endswith("_M"):
            continue

        task = _classify(csv_path.name)
        if task is None:
            continue
        if tasks is not None and task not in tasks:
            continue
        if max_clips_per_task > 0 and task_count.get(task, 0) >= max_clips_per_task:
            continue

        parsed = _parse_csv(csv_path, fps)
        if parsed is None:
            continue

        root_pos, root_euler, joint_angles = parsed
        clip = MotionClip(
            name=csv_path.stem,
            task=task,
            n_frames=root_pos.shape[0],
            dt=1.0 / fps,
            root_pos=root_pos,
            root_euler=root_euler,
            joint_angles=joint_angles,
        )
        lib.add(clip)
        task_count[task] = task_count.get(task, 0) + 1

    for task_name in sorted(lib._by_task):
        clips = lib._by_task[task_name]
        logger.info("  %-18s  %3d clips  (frames: %d–%d)",
                     task_name, len(clips),
                     min(c.n_frames for c in clips),
                     max(c.n_frames for c in clips))

    logger.info("MotionLibrary: %d clips across %d tasks", len(lib), len(lib.tasks))
    return lib
