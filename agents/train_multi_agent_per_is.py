"""Multi-agent training with role-specific action masks and shared PER."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.baselines import get_action_mask_for_role
from config_loader import get_env_kwargs, load_config
from gym_pentest.env import PentestEnv
from setup_logging import setup_logging
from utils.prioritized_replay import PrioritizedReplay
from utils.seeds import set_global_seed

logger = logging.getLogger(__name__)


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic network."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, act_dim)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.policy(h), self.value(h).squeeze(-1)


def masked_select_action(
    net: ActorCritic,
    obs: np.ndarray,
    device: torch.device,
    allowed: set[int],
) -> tuple[int, float]:
    """Sample action restricted to allowed set for role specialization."""
    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    logits, value = net(obs_t)
    mask = torch.full_like(logits, float("-inf"))
    for a in allowed:
        if a < logits.shape[-1]:
            mask[0, a] = 0.0
    masked_logits = logits + mask
    probs = torch.softmax(masked_logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample().item()
    return action, value.item()


def compute_td(
    net: ActorCritic,
    obs: np.ndarray,
    reward: float,
    next_obs: np.ndarray,
    done: bool,
    device: torch.device,
    gamma: float = 0.99,
) -> tuple[float, float]:
    with torch.no_grad():
        _, v = net(torch.from_numpy(obs).float().unsqueeze(0).to(device))
        _, vnext = net(torch.from_numpy(next_obs).float().unsqueeze(0).to(device))
    target = reward + (0.0 if done else gamma * vnext.item())
    td = target - v.item()
    return td, target


def run_episode(
    env: PentestEnv,
    net: ActorCritic,
    device: torch.device,
    allowed_actions: set[int],
    replay: PrioritizedReplay,
    role: str,
) -> float:
    """Run one episode for a specialized agent."""
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0.0

    while not (done or truncated):
        action, _ = masked_select_action(net, obs, device, allowed_actions)
        next_obs, reward, done, truncated, info = env.step(action)

        # Role-specific reward shaping
        if role == "recon" and info.get("endpoint_coverage", 0) > 0:
            reward += 0.1 * info["endpoint_coverage"]
        elif role == "exploit" and info.get("vulnerabilities", 0) > 0:
            reward += 0.5

        td, target = compute_td(net, obs, reward, next_obs, done or truncated, device)
        transition = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done or truncated,
            "target": target,
            "role": role,
        }
        replay.add(td, transition)
        total_reward += reward
        obs = next_obs

    return total_reward


def train(config: dict | None = None) -> None:
    """Train recon and exploit agents with shared PER buffer."""
    if config is None:
        config = load_config()

    ma_cfg = config.get("multi_agent", {})
    seed = ma_cfg.get("seed", 42)
    set_global_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training multi-agent on device: %s", device)

    env_kwargs = get_env_kwargs(config)
    env = PentestEnv(**env_kwargs)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    hidden = ma_cfg.get("hidden_size", 256)

    recon = ActorCritic(obs_dim, act_dim, hidden).to(device)
    exploit = ActorCritic(obs_dim, act_dim, hidden).to(device)
    optim_recon = optim.Adam(recon.parameters(), lr=ma_cfg.get("learning_rate", 3e-4))
    optim_exploit = optim.Adam(exploit.parameters(), lr=ma_cfg.get("learning_rate", 3e-4))

    per_cfg = config.get("per", {})
    replay = PrioritizedReplay(
        capacity=per_cfg.get("capacity", 4096),
        alpha=per_cfg.get("alpha", 0.6),
        beta_start=per_cfg.get("beta_start", 0.4),
        beta_frames=per_cfg.get("beta_frames", 100000),
    )

    num_episodes = ma_cfg.get("num_episodes", 200)
    batch_size = ma_cfg.get("batch_size", 64)
    model_dir = Path(config.get("training", {}).get("model_dir", "./models/"))
    model_dir.mkdir(parents=True, exist_ok=True)

    recon_actions = get_action_mask_for_role("recon")
    exploit_actions = get_action_mask_for_role("exploit")

    for ep in range(num_episodes):
        recon_reward = run_episode(env, recon, device, recon_actions, replay, "recon")
        exploit_reward = run_episode(env, exploit, device, exploit_actions, replay, "exploit")

        if replay.tree.size >= batch_size:
            idxs, batch, weights = replay.sample(batch_size)
            obs_b = torch.from_numpy(np.vstack([b["obs"] for b in batch])).float().to(device)
            actions_b = torch.tensor([b["action"] for b in batch], dtype=torch.long).to(device)
            targets_b = torch.tensor([b["target"] for b in batch], dtype=torch.float).to(device)
            weights_t = torch.tensor(weights, dtype=torch.float).to(device)

            for net, optimizer in [(recon, optim_recon), (exploit, optim_exploit)]:
                logits, values = net(obs_b)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                logp = dist.log_prob(actions_b)
                advantages = targets_b - values.detach()
                policy_loss = -(weights_t * logp * advantages).mean()
                value_loss = (weights_t * (values - targets_b) ** 2).mean()
                loss = policy_loss + 0.5 * value_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                _, vals_new = recon(obs_b)
                td_errors = (targets_b - vals_new).abs().cpu().numpy()
            for idx, td in zip(idxs, td_errors):
                replay.update(idx, float(td))

        logger.info(
            "Episode %d | recon_reward=%.2f | exploit_reward=%.2f | replay=%d",
            ep,
            recon_reward,
            exploit_reward,
            replay.tree.size,
        )

        if ep % 20 == 0:
            torch.save(recon.state_dict(), model_dir / f"recon_ep_{ep}.pth")
            torch.save(exploit.state_dict(), model_dir / f"exploit_ep_{ep}.pth")
            logger.info("Saved models at episode %d", ep)


if __name__ == "__main__":
    cfg = load_config()
    setup_logging()
    train(cfg)
