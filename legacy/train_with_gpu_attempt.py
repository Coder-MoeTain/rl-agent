"""Train with GPU attempt - will fallback to CPU if GPU not compatible."""
import sys
from pathlib import Path
import torch
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from setup_logging import setup_logging
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_pentest.env import PentestEnv

# Setup logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)

def attempt_gpu_training():
    """Attempt to train with GPU, fallback to CPU if needed."""
    print("=" * 60)
    print("Training with GPU Attempt")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\n[INFO] GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA Version: {torch.version.cuda}")
        
        # Test if GPU actually works
        print("\n[INFO] Testing GPU compatibility...")
        try:
            # Try a real computation
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.matmul(x, y)
            result = z.sum().item()
            del x, y, z
            torch.cuda.empty_cache()
            
            print("[SUCCESS] GPU is compatible and working!")
            device_str = "cuda"
            use_gpu = True
        except RuntimeError as e:
            error_msg = str(e)
            if "no kernel image" in error_msg or "sm_120" in error_msg:
                print("[WARNING] GPU detected but architecture not supported")
                print(f"[WARNING] Error: {error_msg[:100]}...")
                print("[INFO] RTX 5080 (sm_120) requires newer PyTorch build")
                print("[INFO] Falling back to CPU (which works fine for MLP policies)")
                device_str = "cpu"
                use_gpu = False
            else:
                print(f"[WARNING] GPU error: {e}")
                print("[INFO] Falling back to CPU")
                device_str = "cpu"
                use_gpu = False
    else:
        print("\n[INFO] CUDA not available, using CPU")
        device_str = "cpu"
        use_gpu = False
    
    # Create environment
    print("\n[INFO] Creating environment...")
    env = make_vec_env(PentestEnv, n_envs=1)
    
    # Create model
    print(f"\n[INFO] Creating PPO model on device: {device_str}")
    if use_gpu:
        print("[INFO] Using GPU acceleration!")
    else:
        print("[INFO] Using CPU (recommended for MLP policies by Stable-Baselines3)")
    
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
    
    print(f"\n[OK] Model created successfully on: {model.device}")
    
    # Train
    print("\n[INFO] Starting training (10000 timesteps)...")
    print("=" * 60)
    
    import time
    start_time = time.time()
    
    model.learn(total_timesteps=10000)
    
    elapsed = time.time() - start_time
    
    print("=" * 60)
    print(f"\n[SUCCESS] Training completed in {elapsed:.2f} seconds")
    print(f"[INFO] Average speed: {10000/elapsed:.1f} steps/second")
    
    if use_gpu:
        print("[INFO] Training used GPU acceleration")
    else:
        print("[INFO] Training used CPU (which is fine for this project)")
    
    # Save model
    model.save('ppo_baseline')
    print("\n[OK] Model saved as 'ppo_baseline.zip'")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    try:
        success = attempt_gpu_training()
        if success:
            print("\n[SUCCESS] All done! You can now use the trained model.")
        else:
            print("\n[WARNING] Training completed but with warnings")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

