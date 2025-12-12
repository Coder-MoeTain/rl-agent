"""Test GPU training with a quick run."""
import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gpu_utils import setup_gpu, check_gpu_requirements
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_pentest.env import PentestEnv

def main():
    """Test GPU training with a very short run."""
    print("=" * 60)
    print("GPU Training Test")
    print("=" * 60)
    
    # Setup GPU
    device = setup_gpu()
    check_gpu_requirements()
    
    if not torch.cuda.is_available():
        print("\n[WARNING] CUDA not available - will use CPU")
        print("This test will still work but won't use GPU acceleration")
    else:
        print(f"\n[OK] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Create environment
    print("\nCreating environment...")
    env = make_vec_env(PentestEnv, n_envs=1)
    
    # Create model with GPU
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nCreating PPO model on device: {device_str}")
    
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        device=device_str,
        learning_rate=3e-4,
        n_steps=256,  # Small for quick test
        batch_size=32,
        n_epochs=2
    )
    
    print(f"\n[OK] Model created on device: {model.device}")
    
    # Quick training test
    print("\nStarting quick training test (500 steps)...")
    print("This should be very fast on GPU!")
    
    import time
    start_time = time.time()
    
    model.learn(total_timesteps=500)
    
    elapsed = time.time() - start_time
    
    print(f"\n[OK] Training completed in {elapsed:.2f} seconds")
    
    if torch.cuda.is_available():
        print(f"[OK] GPU was used successfully!")
        print(f"Average speed: {500/elapsed:.1f} steps/second")
    else:
        print(f"[INFO] CPU was used (GPU not available)")
    
    print("\n" + "=" * 60)
    print("GPU Test Complete!")
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

