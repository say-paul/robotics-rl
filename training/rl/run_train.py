#!/usr/bin/env python
"""
Quick-launch wrapper for stair-climbing PPO training.

Ensures unbuffered output so logs appear immediately in terminals.
"""

import os
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from stable_baselines3.common.callbacks import BaseCallback


class LogFlushCallback(BaseCallback):
    """Force-flush stdout/stderr every N steps for visibility."""

    def __init__(self, flush_every: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self._flush_every = flush_every

    def _on_step(self) -> bool:
        if self.num_timesteps % self._flush_every == 0:
            sys.stdout.flush()
            sys.stderr.flush()
        return True


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
    from training.rl.stair_env import StairClimbEnv

    out_dir = _ROOT / "outputs" / "stair_climb"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_envs = 4
    print(f"[run_train] Creating {n_envs} environments...", flush=True)
    vec_env = DummyVecEnv([lambda: StairClimbEnv() for _ in range(n_envs)])

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(out_dir / "logs"),
    )

    ckpt_cb = CheckpointCallback(
        save_freq=50_000 // n_envs,
        save_path=str(out_dir / "checkpoints"),
        name_prefix="stair_ppo",
    )
    flush_cb = LogFlushCallback(flush_every=1000)

    total = 2_000_000
    print(f"[run_train] Starting PPO training for {total:,} timesteps...", flush=True)
    model.learn(total_timesteps=total, callback=[ckpt_cb, flush_cb])

    model.save(str(out_dir / "stair_ppo_final"))
    print("[run_train] Done.", flush=True)
    vec_env.close()


if __name__ == "__main__":
    main()
