"""
PPO training script for the G1 stair-climbing environment.

Usage:
    python training/rl/train_stairs.py [--timesteps N] [--envs N]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["PYTHONUNBUFFERED"] = "1"

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from training.rl.stair_env import StairClimbEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


def make_env():
    def _init():
        return StairClimbEnv()
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    out_dir = _ROOT / "outputs" / "stair_climb"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    logger.info("Creating %d parallel environments (DummyVecEnv)...", args.envs)
    vec_env = DummyVecEnv([make_env() for _ in range(args.envs)])

    eval_env = DummyVecEnv([make_env()])

    if args.resume:
        logger.info("Resuming from %s", args.resume)
        model = PPO.load(args.resume, env=vec_env)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=str(log_dir),
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // args.envs,
        save_path=str(ckpt_dir),
        name_prefix="stair_ppo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "best_model"),
        eval_freq=25_000 // args.envs,
        n_eval_episodes=5,
        deterministic=True,
    )

    logger.info("Starting PPO training for %d timesteps...", args.timesteps)
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    final_path = out_dir / "stair_ppo_final"
    model.save(str(final_path))
    logger.info("Training complete. Model saved to %s", final_path)

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
