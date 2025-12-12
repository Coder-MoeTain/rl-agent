"""Train all agents with real-time logging."""
import sys
from pathlib import Path
import torch
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from gym_pentest.env import PentestEnv
from custom_sb3_per import PPO_PER

class DetailedCallback(BaseCallback):
    """Detailed real-time callback."""
    
    def __init__(self, agent_name="Agent", verbose=1):
        super().__init__(verbose)
        self.agent_name = agent_name
        self.start_time = time.time()
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_print = 0
        
    def _on_step(self) -> bool:
        """Log every 50 steps."""
        if self.num_timesteps - self.last_print >= 50:
            elapsed = time.time() - self.start_time
            steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
            
            # Get episode info
            if len(self.locals.get('infos', [])) > 0:
                for info in self.locals['infos']:
                    if 'episode' in info:
                        ep_info = info['episode']
                        self.episode_rewards.append(ep_info.get('r', 0))
                        self.episode_lengths.append(ep_info.get('l', 0))
            
            # Calculate stats
            if self.episode_rewards:
                recent = self.episode_rewards[-20:]
                avg_reward = sum(recent) / len(recent)
                best_reward = max(recent)
            else:
                avg_reward = best_reward = 0
            
            # Print
            progress = (self.num_timesteps / self.total_timesteps * 100) if hasattr(self, 'total_timesteps') else 0
            print(f"\r[{self.agent_name}] "
                  f"Step: {self.num_timesteps:6d} ({progress:5.1f}%) | "
                  f"Reward: {avg_reward:7.2f} (best: {best_reward:7.2f}) | "
                  f"Speed: {steps_per_sec:6.1f} steps/s | "
                  f"Time: {elapsed:6.1f}s", end='', flush=True)
            
            self.last_print = self.num_timesteps
        
        return True
    
    def _on_rollout_end(self) -> None:
        """End of rollout summary."""
        elapsed = time.time() - self.start_time
        steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
        
        if self.episode_rewards:
            recent = self.episode_rewards[-20:]
            avg = sum(recent) / len(recent)
            best = max(recent)
            worst = min(recent)
        else:
            avg = best = worst = 0
        
        print(f"\n{'─'*80}")
        print(f"[{self.agent_name}] Rollout Summary:")
        print(f"  Steps: {self.num_timesteps} | Episodes: {len(self.episode_rewards)}")
        print(f"  Reward - Avg: {avg:.2f} | Best: {best:.2f} | Worst: {worst:.2f}")
        print(f"  Speed: {steps_per_sec:.1f} steps/s | Time: {elapsed:.1f}s")
        print(f"{'─'*80}\n")

def train_single_agent():
    """Train single agent with logs."""
    print("\n" + "="*80)
    print("Training Single Agent (Baseline)")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    env = make_vec_env(PentestEnv, n_envs=1)
    callback = DetailedCallback("Single Agent", verbose=1)
    callback.total_timesteps = 10000
    
    model = PPO(
        'MlpPolicy',
        env,
        verbose=0,
        device=device,
        tensorboard_log="./tensorboard_logs/single_agent/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10
    )
    
    start = time.time()
    model.learn(total_timesteps=10000, callback=callback, progress_bar=False)
    elapsed = time.time() - start
    
    model.save('ppo_baseline')
    print(f"\n[OK] Saved in {elapsed:.1f}s")
    return True

def train_per_agent():
    """Train PER agent with logs."""
    print("\n" + "="*80)
    print("Training Single Agent with PER")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    env = make_vec_env(PentestEnv, n_envs=1)
    callback = DetailedCallback("PPO+PER", verbose=1)
    callback.total_timesteps = 10000
    
    model = PPO_PER(
        'MlpPolicy',
        env,
        verbose=0,
        device=device,
        per_capacity=4096,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=10000,
        per_batch_size=64,
        tensorboard_log="./tensorboard_logs/per_agent/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10
    )
    
    start = time.time()
    model.learn(total_timesteps=10000, callback=callback, progress_bar=False)
    elapsed = time.time() - start
    
    model.save('ppo_per_model')
    print(f"\n[OK] Saved in {elapsed:.1f}s")
    return True

def main():
    """Train all agents."""
    print("="*80)
    print("Training All Agents with Real-Time Logs")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("Using CPU")
    
    results = {}
    
    try:
        results['single'] = train_single_agent()
    except Exception as e:
        print(f"[ERROR] Single agent failed: {e}")
        results['single'] = False
    
    try:
        results['per'] = train_per_agent()
    except Exception as e:
        print(f"[ERROR] PER agent failed: {e}")
        results['per'] = False
    
    print("\n" + "="*80)
    print("Training Summary")
    print("="*80)
    for name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {name:15} {status}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Training interrupted")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

