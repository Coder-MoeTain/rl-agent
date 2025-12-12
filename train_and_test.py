"""Train and immediately test an agent."""
import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_pentest.env import PentestEnv
from gpu_utils import setup_gpu
from test_trained_agent import test_agent, watch_agent_play

def main():
    """Train and test agent."""
    print("=" * 70)
    print("Train and Test Agent")
    print("=" * 70)
    
    # Setup GPU
    device = setup_gpu()
    device_str = "cuda" if torch.cuda.is_available() and device.type == "cuda" else "cpu"
    print(f"\nUsing device: {device_str}")
    
    # Create environment
    print("\n[INFO] Creating environment...")
    env = make_vec_env(PentestEnv, n_envs=1)
    
    # Create and train model
    print("[INFO] Creating PPO model...")
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        device=device_str,
        tensorboard_log="./tensorboard_logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10
    )
    
    print("\n[INFO] Starting training (5000 timesteps for quick test)...")
    print("=" * 70)
    
    model.learn(total_timesteps=5000)
    
    print("\n" + "=" * 70)
    print("[OK] Training complete!")
    
    # Save model
    model_path = 'ppo_baseline'
    model.save(model_path)
    print(f"[OK] Model saved as '{model_path}.zip'")
    
    # Test the model
    print("\n" + "=" * 70)
    print("Testing Trained Agent")
    print("=" * 70)
    
    # Quick test
    print("\n[INFO] Running quick test (5 episodes)...")
    results = test_agent(model_path, num_episodes=5, render=False)
    
    if results:
        print("\n[SUCCESS] Agent tested successfully!")
        print("\nTo watch agent play step-by-step:")
        print(f"  python test_trained_agent.py --watch --model {model_path}")
        print("\nTo test with more episodes:")
        print(f"  python test_trained_agent.py --model {model_path} --episodes 20")
    
    return True

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

