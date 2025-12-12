"""Test training on CPU (which works fine for this project)."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_pentest.env import PentestEnv

def main():
    """Test training on CPU."""
    print("=" * 60)
    print("CPU Training Test (Recommended for MLP policies)")
    print("=" * 60)
    
    # Create environment
    print("\nCreating environment...")
    env = make_vec_env(PentestEnv, n_envs=1)
    
    # Create model on CPU (explicit)
    print("\nCreating PPO model on CPU...")
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        device="cpu",  # Explicitly use CPU
        learning_rate=3e-4,
        n_steps=256,
        batch_size=32,
        n_epochs=2
    )
    
    print(f"\n[OK] Model created on device: {model.device}")
    
    # Quick training test
    print("\nStarting quick training test (500 steps)...")
    
    import time
    start_time = time.time()
    
    model.learn(total_timesteps=500)
    
    elapsed = time.time() - start_time
    
    print(f"\n[OK] Training completed in {elapsed:.2f} seconds")
    print(f"[OK] Average speed: {500/elapsed:.1f} steps/second")
    print("\n[SUCCESS] CPU training works perfectly!")
    print("\nNote: For MLP policies, CPU is actually recommended by Stable-Baselines3")
    print("      GPU would help more with CNN policies or very large models")
    
    print("\n" + "=" * 60)
    print("Test Complete - Ready for full training!")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

