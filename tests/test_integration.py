"""Integration tests for full workflows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.baselines import RandomAgent, RuleBasedAgent
from config_loader import load_config
from evaluation.metrics import compute_aggregate
from evaluation.run_experiments import run_episode
from gym_pentest.env import PentestEnv
from gym_pentest.mock_http import build_mock_http_client
from utils.graph_visualize import save_graph_png
from utils.results import save_results_json


@pytest.fixture
def offline_env() -> PentestEnv:
    return PentestEnv(
        http_client=build_mock_http_client(),
        disable_scope_guard=True,
        max_steps=30,
        mission_vuln_threshold=1,
    )


def test_integration_full_episode(offline_env):
    """Run one complete episode with rule-based agent."""
    agent = RuleBasedAgent(offline_env.action_space.n, seed=42)
    agent.reset()
    obs, info = offline_env.reset(seed=42)
    total_reward = 0.0
    steps = 0
    done = truncated = False

    while not (done or truncated):
        action, _ = agent.predict(obs)
        obs, reward, done, truncated, info = offline_env.step(action)
        total_reward += reward
        steps += 1

    assert steps >= 1
    assert info["discovered_count"] >= 1
    assert offline_env.attack_graph.number_of_nodes() >= 1


def test_integration_episode_metrics(offline_env):
    """Run episode through evaluation harness."""
    agent = RandomAgent(offline_env.action_space.n, seed=42)
    metrics = run_episode(offline_env, agent, seed=42, episode_idx=0, algorithm="random")
    assert metrics.episode_length >= 1
    assert metrics.algorithm == "random"


def test_integration_evaluate_algorithm_offline(tmp_path):
    """Smoke test evaluation pipeline with mock HTTP."""
    config = load_config()
    config["evaluation"] = {
        "seeds": [42],
        "episodes_per_seed": 2,
        "output_dir": str(tmp_path),
        "algorithms": ["random"],
    }
    env_kwargs = {"base_url": "http://localhost:3000", "max_steps": 15, "disable_scope_guard": True}

    episodes = []
    for seed in [42]:
        env = PentestEnv(http_client=build_mock_http_client(), **env_kwargs)
        agent = RandomAgent(env.action_space.n, seed=seed)
        for ep in range(2):
            episodes.append(run_episode(env, agent, seed, ep, "random"))
        env.close()

    agg = compute_aggregate(episodes, "random")
    path = save_results_json(episodes, [agg], {"seeds": [42]}, tmp_path / "results.json")
    data = json.loads(path.read_text())
    assert len(data["episodes"]) == 2
    assert data["aggregate"][0]["algorithm"] == "random"


def test_integration_graph_export(offline_env, tmp_path):
    """Export attack graph after episode."""
    offline_env.reset(seed=0)
    offline_env.step(0)
    offline_env.step(5)
    png = save_graph_png(offline_env.attack_graph, tmp_path / "attack.png")
    assert png.exists()


def test_integration_short_ppo_training():
    """Smoke test short PPO training run."""
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    def make_env():
        return PentestEnv(
            http_client=build_mock_http_client(),
            disable_scope_guard=True,
            max_steps=20,
        )

    env = make_vec_env(make_env, n_envs=1)
    model = PPO("MlpPolicy", env, verbose=0, n_steps=64, batch_size=32)
    model.learn(total_timesteps=64)
    obs = env.reset()
    action, _ = model.predict(obs)
    assert action is not None


def test_integration_benchmark_imports():
    """Verify benchmark modules import cleanly when torch is available."""
    pytest.importorskip("torch")
    from benchmarks import (
        benchmark_multi_agent,  # noqa: F401
        benchmark_ppo_vs_ppo_per,  # noqa: F401
    )
