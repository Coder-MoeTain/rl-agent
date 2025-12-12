"""Training script with real-time terminal logs."""
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

class RealtimeLoggingCallback(BaseCallback):
    """Callback for real-time training logs."""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        """Called at each step."""
        # Log every 100 steps
        if self.num_timesteps % 100 == 0:
            elapsed = time.time() - self.start_time
            steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
            
            # Get latest info if available
            if len(self.locals.get('infos', [])) > 0:
                info = self.locals['infos'][0]
                reward = info.get('episode', {}).get('r', 0)
                length = info.get('episode', {}).get('l', 0)
                
                if reward != 0:
                    self.episode_rewards.append(reward)
                    self.episode_lengths.append(length)
            
            # Calculate averages
            avg_reward = sum(self.episode_rewards[-10:]) / len(self.episode_rewards[-10:]) if self.episode_rewards else 0
            avg_length = sum(self.episode_lengths[-10:]) / len(self.episode_lengths[-10:]) if self.episode_lengths else 0
            
            # Print real-time stats
            print(f"\r[Step {self.num_timesteps:6d}] "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Length: {avg_length:5.1f} | "
                  f"Speed: {steps_per_sec:6.1f} steps/s | "
                  f"Time: {elapsed:6.1f}s", end='', flush=True)
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at end of rollout."""
        elapsed = time.time() - self.start_time
        steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
        
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-10:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            max_reward = max(recent_rewards)
            min_reward = min(recent_rewards)
        else:
            avg_reward = max_reward = min_reward = 0
        
        print(f"\n{'='*80}")
        print(f"Rollout Complete | Steps: {self.num_timesteps} | "
              f"Avg Reward: {avg_reward:.2f} | "
              f"Max: {max_reward:.2f} | Min: {min_reward:.2f} | "
              f"Speed: {steps_per_sec:.1f} steps/s")
        print(f"{'='*80}")

def main():
    """Train with real-time logging."""
    print("=" * 80)
    print("Real-Time Training with GPU")
    print("=" * 80)
    
    # Check GPU
    if torch.cuda.is_available():
        try:
            # Test GPU
            x = torch.randn(10, 10).cuda()
            y = torch.randn(10, 10).cuda()
            z = torch.matmul(x, y)
            z.sum().item()
            del x, y, z
            torch.cuda.empty_cache()
            
            device_str = "cuda"
            print(f"\n[OK] Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"[OK] CUDA Version: {torch.version.cuda}")
        except:
            device_str = "cpu"
            print("\n[WARNING] GPU test failed, using CPU")
    else:
        device_str = "cpu"
        print("\n[INFO] Using CPU")
    
    # Create environment
    print("\n[INFO] Creating environment...")
    env = make_vec_env(PentestEnv, n_envs=1)
    
    # Create model
    print(f"[INFO] Creating PPO model on {device_str}...")
    model = PPO(
        'MlpPolicy',
        env,
        verbose=0,  # Disable default verbose to use our custom logging
        device=device_str,
        tensorboard_log="./tensorboard_logs/realtime/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5
    )
    
    print(f"[OK] Model created on {model.device}")
    
    # Create callback
    callback = RealtimeLoggingCallback(verbose=1)
    
    # Training parameters
    total_timesteps = 20000
    print(f"\n[INFO] Starting training for {total_timesteps} timesteps...")
    print(f"[INFO] Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("Real-Time Training Log:")
    print("=" * 80)
    
    start_time = time.time()
    
    # Train with callback
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False  # We'll use our own logging
    )
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"Average speed: {total_timesteps/elapsed:.1f} steps/second")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if callback.episode_rewards:
        print(f"\nFinal Statistics:")
        print(f"  Episodes completed: {len(callback.episode_rewards)}")
        print(f"  Average reward: {sum(callback.episode_rewards) / len(callback.episode_rewards):.2f}")
        print(f"  Best reward: {max(callback.episode_rewards):.2f}")
        print(f"  Worst reward: {min(callback.episode_rewards):.2f}")
        print(f"  Average episode length: {sum(callback.episode_lengths) / len(callback.episode_lengths):.1f}")
    
    # Save model
    model_path = 'ppo_realtime_model'
    model.save(model_path)
    print(f"\n[OK] Model saved as '{model_path}.zip'")
    print("=" * 80)
    
    return True

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Training interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

