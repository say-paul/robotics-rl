"""PPO training entry-point for the G1 walking environment.

Usage
-----
    # Curriculum stage 1 – standing with human-like posture
    python -m rl.train --stage stand

    # Slow walk stage
    python -m rl.train --stage slow_walk --pretrained policies/g1_joystick.onnx

    # Full walk with domain randomization
    python -m rl.train --stage full_walk

    # Fine-tune from a pretrained ONNX policy
    python -m rl.train --pretrained policies/g1_joystick.onnx

    # Train on Intel Arc GPU (XPU)
    python -m rl.train --device xpu

    # Resume from a checkpoint
    python -m rl.train --resume rl/checkpoints/rl_model_2000000_steps.zip

    # Override timesteps / envs / episode length
    python -m rl.train --total-timesteps 10000000 --n-envs 8 --episode-length 3000

    # Choose activation function
    python -m rl.train --stage stand --activation silu
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
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

ACTIVATIONS = {
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "elu": nn.ELU,
}


class SaveVecNormOnBestCallback(BaseCallback):
    """Save VecNormalize stats alongside best_model.zip every time
    EvalCallback records a new best model.

    Watches the best_model.zip mtime to detect saves reliably,
    independent of callback ordering or parent references.
    """

    def __init__(self, save_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self._best_path = Path(save_dir) / "best_model.zip"
        self._last_mtime: float = 0.0

    def _init_callback(self) -> None:
        if self._best_path.exists():
            self._last_mtime = self._best_path.stat().st_mtime

    def _on_step(self) -> bool:
        if not self._best_path.exists():
            return True
        mtime = self._best_path.stat().st_mtime
        if mtime > self._last_mtime:
            self._last_mtime = mtime
            env = self.model.get_env()
            if isinstance(env, VecNormalize):
                vn_path = self._best_path.with_suffix(".vecnorm.pkl")
                env.save(str(vn_path))
                log.info("Saved VecNormalize -> %s (best_model updated)", vn_path)
        return True


class StdClampCallback(BaseCallback):
    """Hard-clamp policy log_std every N steps to prevent std explosion.

    Without this, entropy bonus (even tiny ent_coef) pushes std upward
    monotonically over millions of steps, eventually making actions random.
    """

    def __init__(self, max_log_std: float = -1.0, clamp_every: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.max_log_std = max_log_std
        self.clamp_every = clamp_every

    def _on_step(self) -> bool:
        if self.n_calls % self.clamp_every == 0:
            with torch.no_grad():
                self.model.policy.log_std.clamp_(max=self.max_log_std)
        return True


class EpisodeLengthCeilingCallback(BaseCallback):
    """Stop training when eval episode length AND reward both plateau.

    Training stops when mean episode length is within ``tolerance`` of the
    ceiling AND reward has not improved by more than ``reward_min_delta``
    for ``patience`` consecutive evals.
    """

    def __init__(
        self,
        ceiling: int,
        patience: int = 5,
        tolerance: float = 5.0,
        reward_min_delta: float = 0.5,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.ceiling = ceiling
        self.patience = patience
        self.tolerance = tolerance
        self.reward_min_delta = reward_min_delta
        self._streak = 0
        self._best_reward = -float("inf")

    def _on_step(self) -> bool:
        if self.parent is None:
            return True
        eval_cb = None
        for cb in self.parent.callbacks:
            if isinstance(cb, EvalCallback):
                eval_cb = cb
                break
        if eval_cb is None or not hasattr(eval_cb, "last_mean_reward"):
            return True

        ep_lengths = eval_cb.evaluations_length_
        ep_rewards = eval_cb.evaluations_results_ if hasattr(eval_cb, "evaluations_results_") else None
        if not ep_lengths:
            return True

        last_mean_len = float(np.mean(ep_lengths[-1]))
        last_mean_rew = eval_cb.last_mean_reward

        at_ceiling = last_mean_len >= self.ceiling - self.tolerance
        reward_improved = last_mean_rew > self._best_reward + self.reward_min_delta

        if reward_improved:
            self._best_reward = last_mean_rew
            self._streak = 0
        elif at_ceiling:
            self._streak += 1

        if self.verbose:
            log.info(
                "Ceiling check: ep_len=%.0f  reward=%.1f  best=%.1f  "
                "at_ceiling=%s  streak=%d/%d",
                last_mean_len, last_mean_rew, self._best_reward,
                at_ceiling, self._streak, self.patience,
            )

        if self._streak >= self.patience:
            log.info(
                "Early stopping: episode length at ceiling %d and reward "
                "plateaued at %.1f for %d consecutive evals",
                self.ceiling, self._best_reward, self.patience,
            )
            return False

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
    action_scale_factor: float | None = None,
) -> Path:
    """Run PPO training and return the path to the final model."""
    dev = _resolve_device(device)
    log.info("Training device: %s", dev)

    pcfg = cfg.ppo
    os.makedirs(pcfg.log_dir, exist_ok=True)
    os.makedirs(pcfg.checkpoint_dir, exist_ok=True)

    activation_fn = ACTIVATIONS.get(pcfg.activation, nn.SiLU)
    log.info("Network arch: %s, activation: %s", pcfg.net_arch, pcfg.activation)

    log.info("Building %d parallel environments …", pcfg.n_envs)
    train_env = build_vec_env(cfg, pcfg.n_envs)

    eval_env = build_vec_env(cfg, n_envs=min(4, pcfg.n_envs))

    if resume_path:
        log.info("Resuming from %s", resume_path)
        model = PPO.load(resume_path, env=train_env, device=dev)

        # --- Set fine-tuning learning rate ---
        from stable_baselines3.common.utils import get_schedule_fn
        ft_lr = pcfg.learning_rate
        model.learning_rate = ft_lr
        model.lr_schedule = get_schedule_fn(ft_lr)
        for pg in model.policy.optimizer.param_groups:
            pg["lr"] = ft_lr
        log.info("Fine-tune LR set to %s (verified in optimizer + schedule)", ft_lr)

        # --- Override gamma if config differs from checkpoint ---
        if model.gamma != pcfg.gamma:
            old_gamma = model.gamma
            model.gamma = pcfg.gamma
            log.info("Updated gamma: %s -> %s", old_gamma, pcfg.gamma)

        # --- Clamp policy log_std so exploration noise starts low ---
        target_log_std = pcfg.log_std_init
        with torch.no_grad():
            old_std = model.policy.log_std.exp().mean().item()
            model.policy.log_std.clamp_(max=target_log_std)
            new_std = model.policy.log_std.exp().mean().item()
        log.info(
            "Clamped log_std: %.3f (std=%.3f) -> %.3f (std=%.3f)",
            np.log(old_std), old_std, target_log_std, new_std,
        )

        # --- Optionally scale down action-net to reduce inherited large
        #     action means.  Disabled by default; enable via --action-scale.
        if action_scale_factor is not None and action_scale_factor != 1.0:
            with torch.no_grad():
                model.policy.action_net.weight.mul_(action_scale_factor)
                model.policy.action_net.bias.mul_(action_scale_factor)
            log.info(
                "Scaled action_net weights by %.2f to reduce action means",
                action_scale_factor,
            )

        rp = Path(resume_path)
        vecnorm_candidates = [
            rp.with_suffix(".vecnorm.pkl"),
            rp.parent / rp.name.replace("rl_model_", "rl_model_vecnormalize_").replace(".zip", ".pkl"),
        ]
        for vp in vecnorm_candidates:
            if vp.exists():
                log.info("Loading VecNormalize from %s", vp)
                train_env = VecNormalize.load(str(vp), train_env.venv)
                eval_env = VecNormalize.load(str(vp), eval_env.venv)
                eval_env.training = False
                eval_env.norm_reward = False
                model.set_env(train_env)
                break
        else:
            log.warning("No VecNormalize found for %s", resume_path)
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
        # Sync VecNormalize stats from train_env -> eval_env so eval uses
        # the correct observation normalization from the pretrained model.
        eval_env.obs_rms = train_env.obs_rms
        eval_env.ret_rms = train_env.ret_rms
        eval_env.training = False
        eval_env.norm_reward = False
        log.info("Synced VecNormalize stats to eval_env (training=False, norm_reward=False)")
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
                net_arch=dict(pi=pcfg.net_arch, vf=pcfg.net_arch),
                activation_fn=activation_fn,
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

    vecnorm_best_callback = SaveVecNormOnBestCallback(save_dir=pcfg.checkpoint_dir, verbose=1)

    std_clamp_callback = StdClampCallback(
        max_log_std=pcfg.log_std_init,
        clamp_every=10,
        verbose=0,
    )

    ceiling_callback = EpisodeLengthCeilingCallback(
        ceiling=cfg.env.episode_length,
        patience=8,
        reward_min_delta=0.3,
        tolerance=5.0,
        verbose=1,
    )

    log.info(
        "Starting PPO training for %s timesteps (ep_length=%d) …",
        f"{pcfg.total_timesteps:,}", cfg.env.episode_length,
    )
    model.learn(
        total_timesteps=pcfg.total_timesteps,
        callback=CallbackList(
            [eval_callback, checkpoint_callback, vecnorm_best_callback,
             std_clamp_callback, ceiling_callback]
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
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument(
        "--activation",
        choices=list(ACTIVATIONS),
        default=None,
        help="Activation function (default from config: silu)",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = TrainingConfig.from_stage(args.stage)

    if args.total_timesteps is not None:
        cfg.ppo.total_timesteps = args.total_timesteps
    if args.n_envs is not None:
        cfg.ppo.n_envs = args.n_envs
    if args.episode_length is not None:
        cfg.env.episode_length = args.episode_length
    if args.activation is not None:
        cfg.ppo.activation = args.activation
    if args.seed is not None:
        cfg.seed = args.seed

    train(cfg, resume_path=args.resume, pretrained_onnx=args.pretrained, device=args.device)


if __name__ == "__main__":
    main()
