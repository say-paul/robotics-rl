"""
Central configuration for the G1 deployment system.
All paths, hardware settings, and tunable parameters live here.
"""

import os
from pathlib import Path

# Force EGL backend when no display is available
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path("/home/hf-datasets/g1/csv")
LEROBOT_ROOT = Path("/home/lerobot")
MUJOCO_MODEL_CACHE = Path.home() / ".cache/huggingface/hub/models--lerobot--unitree-g1-mujoco"
SCENE_DIR = PROJECT_ROOT / "scene"
MISSION_DIR = PROJECT_ROOT / "missions"
LOG_DIR = PROJECT_ROOT / "missions" / "logs"
VIDEO_DIR = PROJECT_ROOT / "missions" / "videos"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ── Robot ───────────────────────────────────────────────────────────────────
NUM_JOINTS = 29
JOINT_NAMES = [
    "left_hip_pitch_joint",  "left_hip_roll_joint",  "left_hip_yaw_joint",
    "left_knee_joint",       "left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint",  "right_hip_yaw_joint",
    "right_knee_joint",      "right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint",       "waist_roll_joint",      "waist_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint",      "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint",     "right_wrist_roll_joint","right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

CSV_JOINT_COLUMNS = [f"{j}_dof" for j in JOINT_NAMES]

# ── Simulation ──────────────────────────────────────────────────────────────
SIM_DT = 0.004          # 250 Hz physics
CONTROL_DT = 0.02       # 50 Hz WBC control (Groot)
VIEWER_DT = 0.02         # 50 Hz viewer refresh
IMAGE_DT = 1.0 / 30      # 30 Hz camera

DEFAULT_HEIGHT = 0.75     # metres – nominal standing height

# ── Server ──────────────────────────────────────────────────────────────────
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
CAMERA_WS_PORT = 8001

# ── Hardware ────────────────────────────────────────────────────────────────
import torch

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
    except Exception:
        pass
    return "cpu"

DEVICE = get_device()
