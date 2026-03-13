"""Gymnasium environment registration for G1 walking."""

import gymnasium as gym

gym.register(
    id="G1Walk-v0",
    entry_point="rl.envs.g1_walk_env:G1WalkEnv",
    max_episode_steps=1000,
)
