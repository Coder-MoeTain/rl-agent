"""Maximum GPU utilization training - optimized for CPU-bound environments."""
import sys
from pathlib import Path
import torch
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from gym_pentest.env import PentestEnv
from gpu_utils import setup_gpu

def main():
    """Train with maximum GPU utilization for CPU-bound environments."""
    print("=" * 70)
    print("Maximum GPU Utilization Training")
    print("=" * 70)
    print("\nStrategy: Many parallel environments to keep GPU busy")
    print("         while environments do CPU-bound HTTP requests")
    print("=" * 70)
    
    # Setup GPU
    device = setup_gpu()
    if not torch.cuda.is_available() or device.type != "cuda":
        print("\n[WARNING] GPU not available, using CPU")
        device_str = "cpu"
        n_envs = 1
    else:
        device_str = "cuda"
        print(f"\n[OK] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[OK] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Maximum GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        torch.cuda.empty_cache()
        
        # Use many parallel environments to keep GPU busy
        # Since environment steps are CPU-bound (HTTP requests),
        # we need many parallel envs so GPU can process batches
        # while environments are waiting for HTTP responses
        n_envs = 32  # Many parallel environments
        
    print(f"\n[INFO] Creating {n_envs} parallel environments...")
    
    # Use SubprocVecEnv for true parallelism (critical for CPU-bound envs)
    if device_str == "cuda" and n_envs > 1:
        try:
            def make_env():
                return PentestEnv()
            
            env = SubprocVecEnv([make_env for _ in range(n_envs)], start_method='spawn')
            print(f"[OK] Using SubprocVecEnv - true parallelism (environments in separate processes)")
        except Exception as e:
            print(f"[WARNING] SubprocVecEnv failed: {e}")
            print(f"[INFO] Falling back to DummyVecEnv")
            env = make_vec_env(PentestEnv, n_envs=n_envs)
    else:
        env = make_vec_env(PentestEnv, n_envs=n_envs)
    
    # Maximum GPU utilization hyperparameters
    print(f"\n[INFO] Creating PPO with maximum GPU settings...")
    
    if device_str == "cuda":
        gpu_settings = {
            'n_steps': 8192,  # Large rollout - keeps GPU busy
            'batch_size': 512,  # Very large batches
            'n_epochs': 30,  # Many epochs - more computation
            'policy_kwargs': dict(
                net_arch=[dict(pi=[1024, 1024, 512, 256], vf=[1024, 1024, 512, 256])]
            )
        }
    else:
        gpu_settings = {
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
        }
    
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        device=device_str,
        tensorboard_log="./tensorboard_logs/max_gpu/",
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        **gpu_settings
    )
    
    print(f"\n[OK] Model created on {model.device}")
    if device_str == "cuda":
        print(f"[INFO] Configuration for maximum GPU usage:")
        print(f"  - Parallel environments: {n_envs} (SubprocVecEnv)")
        print(f"  - Network: 1024->1024->512->256 (very large)")
        print(f"  - Batch size: 512 (maximum)")
        print(f"  - Steps per rollout: 8192 (large)")
        print(f"  - Training epochs: 30 (many)")
        print(f"\n[INFO] This setup keeps GPU busy processing large batches")
        print(f"       while {n_envs} environments collect data in parallel")
    
    # Initial GPU stats
    if device_str == "cuda":
        print(f"\n[INFO] Initial GPU memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    
    print(f"\n[INFO] Starting training (20000 timesteps)...")
    print("=" * 70)
    print("\nMonitor GPU usage in another terminal:")
    print("  nvidia-smi -l 1")
    print("  or")
    print("  python monitor_gpu_usage.py")
    print("=" * 70)
    print()
    
    import time
    start_time = time.time()
    
    model.learn(total_timesteps=20000)
    
    elapsed = time.time() - start_time
    
    # Final GPU stats
    if device_str == "cuda":
        print(f"\n[INFO] Final GPU memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
        print(f"  Peak allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.1f} MB")
    
    print(f"\n[OK] Training completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"[INFO] Speed: {20000/elapsed:.1f} steps/second")
    
    # Save model
    model_path = 'ppo_max_gpu'
    model.save(model_path)
    print(f"[OK] Model saved as '{model_path}.zip'")
    
    print("=" * 70)
    
    return True

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Training interrupted")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


