"""PPO training entry-point for the G1 walking environment.

Usage
-----
    # Full walk (default config)
    python -m rl.train

    # Curriculum stage 1 – standing
    python -m rl.train --stage stand

    # Fine-tune from a pretrained ONNX policy (skips curriculum!)
    python -m rl.train --pretrained policies/g1_joystick.onnx

    # Train on Intel Arc GPU (XPU)
    python -m rl.train --device xpu

    # Resume from a checkpoint
    python -m rl.train --resume rl/checkpoints/rl_model_2000000_steps.zip

    # Override timesteps / envs
    python -m rl.train --total-timesteps 10000000 --n-envs 8
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from rl.configs.training_config import TrainingConfig
from rl.envs.g1_walk_env import G1WalkEnv
from rl.import_onnx import import_onnx_into_ppo

log = logging.getLogger(__name__)


class EpisodeLengthCeilingCallback(BaseCallback):
    """Stop training once the eval mean episode length saturates at the ceiling.

    The *ceiling* is the environment's ``episode_length`` (max steps per
    episode).  When the evaluation callback reports a ``mean_ep_length``
    within ``tolerance`` of the ceiling for ``patience`` consecutive evals,
    training is halted early.
    """

    def __init__(
        self,
        ceiling: int,
        patience: int = 3,
        tolerance: float = 5.0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.ceiling = ceiling
        self.patience = patience
        self.tolerance = tolerance
        self._streak = 0

    def _on_step(self) -> bool:
        if self.parent is None:
            return True
        # EvalCallback stores results on itself after each evaluation round
        eval_cb = None
        for cb in self.parent.callbacks:
            if isinstance(cb, EvalCallback):
                eval_cb = cb
                break
        if eval_cb is None or not hasattr(eval_cb, "last_mean_reward"):
            return True

        ep_lengths = eval_cb.evaluations_length_
        if not ep_lengths:
            return True

        last_mean = float(np.mean(ep_lengths[-1]))
        if last_mean >= self.ceiling - self.tolerance:
            self._streak += 1
            if self.verbose:
                log.info(
                    "Ceiling check: mean_ep_length=%.1f (>= %.1f), streak %d/%d",
                    last_mean, self.ceiling - self.tolerance,
                    self._streak, self.patience,
                )
            if self._streak >= self.patience:
                log.info(
                    "Early stopping: episode length hit ceiling %d for %d consecutive evals",
                    self.ceiling, self.patience,
                )
                return False
        else:
            self._streak = 0
        return True


def _resolve_device(requested: str) -> torch.device:
    """Validate and return a torch.device, with helpful error messages."""
    if requested == "auto":
        if torch.xpu.is_available():
            return torch.device("xpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "xpu" and not torch.xpu.is_available():
        raise RuntimeError(
            "XPU requested but torch.xpu.is_available() is False. "
            "Install intel-compute-runtime and PyTorch+xpu."
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(requested)


def _make_env(cfg: TrainingConfig, rank: int):
    """Return a callable that creates a single G1WalkEnv (for SubprocVecEnv)."""

    def _init():
        env = G1WalkEnv(env_cfg=cfg.env, reward_cfg=cfg.reward)
        env.reset(seed=cfg.seed + rank)
        return env

    return _init


def build_vec_env(cfg: TrainingConfig, n_envs: int) -> VecNormalize:
    """Create a VecNormalize-wrapped SubprocVecEnv."""
    env_fns = [_make_env(cfg, i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    return VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)


def train(
    cfg: TrainingConfig,
    resume_path: str | None = None,
    pretrained_onnx: str | None = None,
    device: str = "auto",
) -> Path:
    """Run PPO training and return the path to the final model."""
    dev = _resolve_device(device)
    log.info("Training device: %s", dev)

    pcfg = cfg.ppo
    os.makedirs(pcfg.log_dir, exist_ok=True)
    os.makedirs(pcfg.checkpoint_dir, exist_ok=True)

    log.info("Building %d parallel environments …", pcfg.n_envs)
    train_env = build_vec_env(cfg, pcfg.n_envs)

    eval_env = build_vec_env(cfg, n_envs=min(4, pcfg.n_envs))

    if resume_path:
        log.info("Resuming from %s", resume_path)
        model = PPO.load(resume_path, env=train_env, device=dev)
        vecnorm_path = Path(resume_path).with_suffix(".vecnorm.pkl")
        if vecnorm_path.exists():
            train_env = VecNormalize.load(str(vecnorm_path), train_env.venv)
            model.set_env(train_env)
    elif pretrained_onnx:
        log.info("Importing pretrained ONNX policy from %s", pretrained_onnx)
        model = import_onnx_into_ppo(
            pretrained_onnx,
            train_env,
            device=dev,
            learning_rate=pcfg.learning_rate,
            n_steps=pcfg.n_steps,
            batch_size=pcfg.batch_size,
            n_epochs=pcfg.n_epochs,
            gamma=pcfg.gamma,
            gae_lambda=pcfg.gae_lambda,
            clip_range=pcfg.clip_range,
            ent_coef=pcfg.ent_coef,
            vf_coef=pcfg.vf_coef,
            max_grad_norm=pcfg.max_grad_norm,
            tensorboard_log=pcfg.log_dir,
            seed=cfg.seed,
            verbose=1,
        )
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=pcfg.learning_rate,
            n_steps=pcfg.n_steps,
            batch_size=pcfg.batch_size,
            n_epochs=pcfg.n_epochs,
            gamma=pcfg.gamma,
            gae_lambda=pcfg.gae_lambda,
            clip_range=pcfg.clip_range,
            ent_coef=pcfg.ent_coef,
            vf_coef=pcfg.vf_coef,
            max_grad_norm=pcfg.max_grad_norm,
            policy_kwargs=dict(
                net_arch=pcfg.net_arch,
                log_std_init=pcfg.log_std_init,
            ),
            tensorboard_log=pcfg.log_dir,
            seed=cfg.seed,
            device=dev,
            verbose=1,
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=pcfg.checkpoint_dir,
        log_path=pcfg.log_dir,
        eval_freq=max(pcfg.eval_freq // pcfg.n_envs, 1),
        n_eval_episodes=pcfg.eval_episodes,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(pcfg.checkpoint_freq // pcfg.n_envs, 1),
        save_path=pcfg.checkpoint_dir,
        name_prefix="rl_model",
        save_vecnormalize=True,
    )

    ceiling_callback = EpisodeLengthCeilingCallback(
        ceiling=cfg.env.episode_length,
        patience=3,
        tolerance=5.0,
        verbose=1,
    )

    log.info(
        "Starting PPO training for %s timesteps …", f"{pcfg.total_timesteps:,}"
    )
    model.learn(
        total_timesteps=pcfg.total_timesteps,
        callback=CallbackList(
            [eval_callback, checkpoint_callback, ceiling_callback]
        ),
        progress_bar=True,
    )

    final_path = Path(pcfg.checkpoint_dir) / "final_model"
    model.save(str(final_path))
    train_env.save(str(final_path) + ".vecnorm.pkl")
    log.info("Saved final model to %s", final_path)

    train_env.close()
    eval_env.close()
    return final_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train G1 walking policy with PPO")
    parser.add_argument(
        "--stage",
        choices=list(TrainingConfig.STAGES),
        default=None,
        help="Curriculum stage (stand, slow_walk, full_walk)",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to SB3 checkpoint .zip")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained ONNX policy for fine-tuning (e.g. policies/g1_joystick.onnx)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto, cpu, xpu, or cuda (default: auto)",
    )
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = TrainingConfig.from_stage(args.stage)

    if args.total_timesteps is not None:
        cfg.ppo.total_timesteps = args.total_timesteps
    if args.n_envs is not None:
        cfg.ppo.n_envs = args.n_envs
    if args.seed is not None:
        cfg.seed = args.seed

    train(cfg, resume_path=args.resume, pretrained_onnx=args.pretrained, device=args.device)


if __name__ == "__main__":
    main()
