"""Research-grade benchmark: PPO vs PPO+PER with Gymnasium API."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from stable_baselines3 import PPO

from config_loader import get_env_kwargs, get_per_kwargs, get_ppo_kwargs, load_config
from custom_sb3_per import PPO_PER
from gym_pentest.env import PentestEnv
from utils.seeds import set_global_seed


def evaluate(model, env: PentestEnv, episodes: int = 5, seed: int = 42) -> tuple[float, float]:
    scores = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False
        total = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(int(action))
            total += reward
        scores.append(total)
    return float(np.mean(scores)), float(np.std(scores))


if __name__ == "__main__":
    config = load_config()
    set_global_seed(config.get("training", {}).get("seed", 42))
    env_kwargs = get_env_kwargs(config)
    ppo_kwargs = get_ppo_kwargs(config)
    per_kwargs = get_per_kwargs(config)

    print("Training PPO baseline (short)...")
    env = PentestEnv(**env_kwargs)
    ppo = PPO("MlpPolicy", env, verbose=0, **ppo_kwargs)
    ppo.learn(10000)
    p_mean, p_std = evaluate(ppo, env)
    print(f"PPO mean={p_mean:.2f}, std={p_std:.2f}")

    print("Training PPO+PER (short)...")
    env2 = PentestEnv(**env_kwargs)
    ppo_per = PPO_PER("MlpPolicy", env2, verbose=0, **ppo_kwargs, **per_kwargs)
    ppo_per.learn(10000)
    per_mean, per_std = evaluate(ppo_per, env2)
    print(f"PPO+PER mean={per_mean:.2f}, std={per_std:.2f}")
