"""Simple training with real-time terminal logs."""
import sys
from pathlib import Path
import torch
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from gym_pentest.env import PentestEnv

class RealtimeCallback(BaseCallback):
    """Real-time progress callback."""
    
    def __init__(self):
        super().__init__()
        self.start_time = time.time()
        self.episode_count = 0
        self.total_reward = 0
        self.rewards = []
        
    def _on_step(self) -> bool:
        # Check for completed episodes
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    ep = info['episode']
                    reward = ep.get('r', 0)
                    length = ep.get('l', 0)
                    self.episode_count += 1
                    self.rewards.append(reward)
                    
                    elapsed = time.time() - self.start_time
                    avg_reward = sum(self.rewards[-10:]) / min(10, len(self.rewards))
                    
                    print(f"\n[Episode {self.episode_count:3d}] "
                          f"Reward: {reward:8.2f} | "
                          f"Length: {length:4.0f} | "
                          f"Avg (last 10): {avg_reward:7.2f} | "
                          f"Steps: {self.num_timesteps:6d} | "
                          f"Time: {elapsed:6.1f}s")
        
        # Progress update every 100 steps
        elif self.num_timesteps % 100 == 0:
            elapsed = time.time() - self.start_time
            speed = self.num_timesteps / elapsed if elapsed > 0 else 0
            print(f"\r[Step {self.num_timesteps:6d}] "
                  f"Speed: {speed:6.1f} steps/s | "
                  f"Episodes: {self.episode_count:3d} | "
                  f"Time: {elapsed:6.1f}s", end='', flush=True)
        
        return True

def main():
    """Train with real-time logs."""
    print("=" * 70)
    print("Real-Time Training Monitor")
    print("=" * 70)
    
    # GPU check
    if torch.cuda.is_available():
        try:
            test = torch.randn(10, 10).cuda()
            torch.matmul(test, test).sum().item()
            del test
            torch.cuda.empty_cache()
            device = "cuda"
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except:
            device = "cpu"
            print("Using CPU (GPU test failed)")
    else:
        device = "cpu"
        print("Using CPU")
    
    print(f"Device: {device}\n")
    
    # Create environment and model
    env = make_vec_env(PentestEnv, n_envs=1)
    callback = RealtimeCallback()
    
    model = PPO(
        'MlpPolicy',
        env,
        verbose=0,
        device=device,
        tensorboard_log="./tensorboard_logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10
    )
    
    print("Starting training...\n")
    print("-" * 70)
    
    start_time = time.time()
    model.learn(total_timesteps=20000, callback=callback, progress_bar=False)
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Total steps: {callback.num_timesteps}")
    print(f"Episodes: {callback.episode_count}")
    if callback.rewards:
        print(f"Average reward: {sum(callback.rewards) / len(callback.rewards):.2f}")
        print(f"Best reward: {max(callback.rewards):.2f}")
    
    model.save('ppo_realtime')
    print(f"\nModel saved as 'ppo_realtime.zip'")
    print("=" * 70)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

