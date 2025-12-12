"""Train all agent configurations and handle errors."""
import sys
from pathlib import Path
import logging
import traceback
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from setup_logging import setup_logging
from gpu_utils import setup_gpu, check_gpu_requirements

# Setup logging
setup_logging(level="INFO", log_file="training_all.log")
logger = logging.getLogger(__name__)

# Setup GPU
device = setup_gpu()
check_gpu_requirements()

def train_single_agent_baseline():
    """Train single agent baseline."""
    logger.info("=" * 60)
    logger.info("Training Single Agent (Baseline)")
    logger.info("=" * 60)
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from gym_pentest.env import PentestEnv
        
        env = make_vec_env(PentestEnv, n_envs=1)
        
        # Use GPU if available
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Training on device: {device_str}")
        
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            tensorboard_log="./tensorboard_logs/single_agent/",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            device=device_str  # Specify device for SB3
        )
        
        logger.info("Starting training for 10000 timesteps...")
        model.learn(total_timesteps=10000)
        
        model.save('ppo_baseline')
        logger.info("✓ Single agent baseline saved as 'ppo_baseline.zip'")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error training single agent baseline: {e}")
        logger.error(traceback.format_exc())
        return False

def train_single_agent_per():
    """Train single agent with PER."""
    logger.info("=" * 60)
    logger.info("Training Single Agent with PER")
    logger.info("=" * 60)
    
    try:
        from custom_sb3_per import PPO_PER
        from stable_baselines3.common.env_util import make_vec_env
        from gym_pentest.env import PentestEnv
        
        env = make_vec_env(PentestEnv, n_envs=1)
        
        # Use GPU if available
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Training on device: {device_str}")
        
        model = PPO_PER(
            'MlpPolicy',
            env,
            verbose=1,
            per_capacity=4096,
            per_alpha=0.6,
            per_beta_start=0.4,
            per_beta_frames=10000,
            per_batch_size=64,
            tensorboard_log="./tensorboard_logs/single_agent_per/",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            device=device_str  # Specify device for SB3
        )
        
        logger.info("Starting training for 10000 timesteps...")
        model.learn(total_timesteps=10000)
        
        model.save('ppo_per_model')
        logger.info("✓ Single agent with PER saved as 'ppo_per_model.zip'")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error training single agent with PER: {e}")
        logger.error(traceback.format_exc())
        return False

def train_multi_agent():
    """Train multi-agent system."""
    logger.info("=" * 60)
    logger.info("Training Multi-Agent System (Recon + Exploit)")
    logger.info("=" * 60)
    
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from gym_pentest.env import PentestEnv
        from utils.prioritized_replay import PrioritizedReplay
        
        class ActorCritic(nn.Module):
            def __init__(self, obs_dim, act_dim, hidden=256):
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Linear(obs_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU()
                )
                self.policy = nn.Linear(hidden, act_dim)
                self.value = nn.Linear(hidden, 1)
            
            def forward(self, x):
                h = self.shared(x)
                return self.policy(h), self.value(h).squeeze(-1)
        
        def select_action(net, obs):
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            logits, value = net(obs_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            a = dist.sample().item()
            return a, value.item()
        
        def compute_td(net, obs, reward, next_obs, done, gamma=0.99):
            with torch.no_grad():
                _, v = net(torch.from_numpy(obs).float().unsqueeze(0).to(device))
                _, vnext = net(torch.from_numpy(next_obs).float().unsqueeze(0).to(device))
            target = reward + (0.0 if done else gamma * vnext.item())
            td = target - v.item()
            return td, target
        
        env = PentestEnv()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        
        # Use GPU if available
        device = setup_gpu()
        logger.info(f"Training on device: {device}")
        
        recon = ActorCritic(obs_dim, act_dim).to(device)
        exploit = ActorCritic(obs_dim, act_dim).to(device)
        optim_recon = optim.Adam(recon.parameters(), lr=3e-4)
        optim_exploit = optim.Adam(exploit.parameters(), lr=3e-4)
        replay = PrioritizedReplay(capacity=4096)
        
        num_episodes = 50  # Reduced for testing
        batch_size = 64
        
        logger.info(f"Training for {num_episodes} episodes...")
        
        for ep in range(num_episodes):
            # Recon agent episode
            obs, info = env.reset()
            done = False
            truncated = False
            while not (done or truncated):
                a, _ = select_action(recon, obs)
                next_obs, reward, done, truncated, info = env.step(a)
                td, target = compute_td(recon, obs, reward, next_obs, done or truncated)
                transition = {
                    'obs': obs,
                    'action': a,
                    'reward': reward,
                    'next_obs': next_obs,
                    'done': done or truncated,
                    'target': target
                }
                replay.add(td, transition)
                obs = next_obs
            
            # Exploit agent episode
            obs, info = env.reset()
            done = False
            truncated = False
            while not (done or truncated):
                a, _ = select_action(exploit, obs)
                next_obs, reward, done, truncated, info = env.step(a)
                td, target = compute_td(exploit, obs, reward, next_obs, done or truncated)
                transition = {
                    'obs': obs,
                    'action': a,
                    'reward': reward,
                    'next_obs': next_obs,
                    'done': done or truncated,
                    'target': target
                }
                replay.add(td, transition)
                obs = next_obs
            
            # Training step
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
                    policy_loss = -(weights_t * logp * advantages).mean()
                    value_loss = (weights_t * (values - targets_b) ** 2).mean()
                    loss = policy_loss + 0.5 * value_loss
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                
                # Update PER priorities
                with torch.no_grad():
                    _, vals_new = recon(obs_b)
                    td_errors = (targets_b - vals_new).abs().cpu().numpy()
                for idx, td in zip(idxs, td_errors):
                    replay.update(idx, float(td))
            
            if (ep + 1) % 10 == 0:
                logger.info(f"Episode {ep + 1}/{num_episodes} - Replay size: {replay.tree.size}")
            
            if (ep + 1) % 20 == 0:
                torch.save(recon.state_dict(), f'recon_ep_{ep + 1}.pth')
                torch.save(exploit.state_dict(), f'exploit_ep_{ep + 1}.pth')
                logger.info(f"Saved models at episode {ep + 1}")
        
        logger.info("✓ Multi-agent training completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error training multi-agent: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Train all agents."""
    logger.info("Starting training of all agent configurations...")
    logger.info("This may take 15-30 minutes depending on your system.")
    
    results = {}
    
    # Train single agent baseline
    results['single_agent'] = train_single_agent_baseline()
    
    # Train single agent with PER
    results['single_agent_per'] = train_single_agent_per()
    
    # Train multi-agent
    results['multi_agent'] = train_multi_agent()
    
    # Summary
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    
    for name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"{name:20} {status}")
    
    all_success = all(results.values())
    
    if all_success:
        logger.info("\n✓ All agents trained successfully!")
        logger.info("\nTrained models:")
        logger.info("  - ppo_baseline.zip (Single agent baseline)")
        logger.info("  - ppo_per_model.zip (Single agent with PER)")
        logger.info("  - recon_ep_X.pth, exploit_ep_X.pth (Multi-agent)")
    else:
        logger.warning("\n⚠ Some training failed. Check logs for details.")
    
    return all_success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

