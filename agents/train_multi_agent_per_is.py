"""Multi-agent training with shared PER buffer."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym_pentest.env import PentestEnv
from utils.prioritized_replay import PrioritizedReplay
from gpu_utils import setup_gpu, check_gpu_requirements
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.policy = nn.Linear(hidden, act_dim)
        self.value = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.shared(x)
        return self.policy(h), self.value(h).squeeze(-1)

def select_action(net, obs, device):
    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    logits, value = net(obs_t)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    a = dist.sample().item()
    return a, value.item()

def compute_td(net, obs, reward, next_obs, done, device, gamma=0.99):
    with torch.no_grad():
        _, v = net(torch.from_numpy(obs).float().unsqueeze(0).to(device))
        _, vnext = net(torch.from_numpy(next_obs).float().unsqueeze(0).to(device))
    target = reward + (0.0 if done else gamma * vnext.item())
    td = target - v.item()
    return td, target

def train():
    # Setup GPU
    device = setup_gpu()
    check_gpu_requirements()
    logger.info(f"Training on device: {device}")
    
    env = PentestEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # Move models to GPU
    recon = ActorCritic(obs_dim, act_dim).to(device)
    exploit = ActorCritic(obs_dim, act_dim).to(device)
    optim_recon = optim.Adam(recon.parameters(), lr=3e-4)
    optim_exploit = optim.Adam(exploit.parameters(), lr=3e-4)
    replay = PrioritizedReplay(capacity=4096)
    num_episodes = 200
    batch_size = 64

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            a, _ = select_action(recon, obs, device)
            next_obs, reward, done, truncated, info = env.step(a)
            td, target = compute_td(recon, obs, reward, next_obs, done or truncated, device)
            transition = {'obs': obs, 'action': a, 'reward': reward, 'next_obs': next_obs, 'done': done or truncated, 'target': target}
            replay.add(td, transition)
            obs = next_obs

        obs, info = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            a, _ = select_action(exploit, obs, device)
            next_obs, reward, done, truncated, info = env.step(a)
            td, target = compute_td(exploit, obs, reward, next_obs, done or truncated, device)
            transition = {'obs': obs, 'action': a, 'reward': reward, 'next_obs': next_obs, 'done': done or truncated, 'target': target}
            replay.add(td, transition)
            obs = next_obs

        if replay.tree.size >= batch_size:
            idxs, batch, weights = replay.sample(batch_size)
            obs_b = torch.from_numpy(np.vstack([b['obs'] for b in batch])).float().to(device)
            actions_b = torch.tensor([b['action'] for b in batch], dtype=torch.long).to(device)
            targets_b = torch.tensor([b['target'] for b in batch], dtype=torch.float).to(device)
            weights_t = torch.tensor(weights, dtype=torch.float).to(device)
            for net, optim in [(recon, optim_recon), (exploit, optim_exploit)]:
                logits, values = net(obs_b)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                logp = dist.log_prob(actions_b)
                advantages = targets_b - values.detach()
                policy_loss = - (weights_t * logp * advantages).mean()
                value_loss = (weights_t * (values - targets_b)**2).mean()
                loss = policy_loss + 0.5 * value_loss
                optim.zero_grad()
                loss.backward()
                optim.step()
            with torch.no_grad():
                _, vals_new = recon(obs_b)
                td_errors = (targets_b - vals_new).abs().cpu().numpy()
            for idx, td in zip(idxs, td_errors):
                replay.update(idx, float(td))
        logger.info(f"Episode {ep} replay_size={replay.tree.size}")
        if ep % 20 == 0:
            torch.save(recon.state_dict(), f'recon_ep_{ep}.pth')
            torch.save(exploit.state_dict(), f'exploit_ep_{ep}.pth')
            logger.info(f"Saved models at episode {ep}")

if __name__ == '__main__':
    train()
