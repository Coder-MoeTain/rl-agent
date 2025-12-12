"""GPU-optimized training to maximize GPU utilization."""
import sys
from pathlib import Path
import torch
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from gym_pentest.env import PentestEnv
from gpu_utils import setup_gpu
import multiprocessing
import os

def main():
    """Train with maximum GPU utilization."""
    print("=" * 70)
    print("GPU-Optimized Training (100% GPU Utilization)")
    print("=" * 70)
    
    # Setup GPU
    device = setup_gpu()
    if not torch.cuda.is_available() or device.type != "cuda":
        print("\n[WARNING] GPU not available, using CPU")
        device_str = "cpu"
    else:
        device_str = "cuda"
        print(f"\n[OK] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[OK] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Set GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async execution
        
        # Clear cache
        torch.cuda.empty_cache()
    
    # Create many parallel environments to keep GPU busy while environments do HTTP requests
    # Since HTTP requests are CPU-bound, we need many parallel envs to maximize GPU usage
    if device_str == "cuda":
        n_envs = 32  # Many parallel environments to keep GPU busy during HTTP requests
        print(f"\n[INFO] Creating {n_envs} parallel environments (SubprocVecEnv for true parallelism)...")
        # Use SubprocVecEnv for true parallelism (environments run in separate processes)
        # This allows HTTP requests to happen in parallel while GPU processes batches
        try:
            env = make_vec_env(
                PentestEnv,
                n_envs=n_envs,
                vec_env_cls=SubprocVecEnv,  # True parallelism for CPU-bound env steps
                vec_env_kwargs=dict(start_method='spawn')  # Windows-compatible
            )
            print(f"[OK] Using SubprocVecEnv for true parallelism")
        except Exception as e:
            print(f"[WARNING] SubprocVecEnv failed: {e}, using DummyVecEnv")
            env = make_vec_env(PentestEnv, n_envs=n_envs)
    else:
        n_envs = 1
        print(f"\n[INFO] Creating {n_envs} environment (CPU mode)...")
        env = make_vec_env(PentestEnv, n_envs=n_envs)
    
    # GPU-optimized hyperparameters
    print(f"[INFO] Creating PPO model with GPU-optimized settings...")
    
    # Force explicit CUDA device
    if device_str == "cuda":
        device_str = "cuda:0"  # Explicit device
        torch.cuda.set_device(0)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        device=device_str,  # Explicit CUDA device
        tensorboard_log="./tensorboard_logs/gpu_optimized/",
        
        # Maximum GPU utilization settings
        n_steps=8192,  # Even more steps - keeps GPU busy longer
        batch_size=512,  # Larger batches for maximum parallel computation
        n_epochs=30,  # More epochs - more computation while envs collect data
        
        # Very large networks to maximize GPU computation
        policy_kwargs=dict(
            net_arch=[dict(pi=[1024, 1024, 512, 256], vf=[1024, 1024, 512, 256])]  # Very large networks
        ),
        
        # Learning parameters
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        
        # Optimize for GPU
        normalize_advantage=True,
    )
    
    print(f"[OK] Model created on {model.device}")
    
    # Verify and force GPU usage
    if device_str.startswith("cuda"):
        print("\n[INFO] Verifying GPU usage...")
        try:
            sample_param = next(model.policy.parameters())
            actual_device = str(sample_param.device)
            print(f"[OK] Policy parameters on: {actual_device}")
            
            if 'cuda' not in actual_device:
                print("[WARNING] Parameters not on GPU! Forcing to GPU...")
                model.policy = model.policy.to(torch.device('cuda:0'))
                print("[OK] Forced to GPU")
            
            # Test forward pass
            test_obs = torch.randn(1, 64).to(torch.device('cuda:0'))
            with torch.no_grad():
                _ = model.policy(test_obs)
            print("[OK] Forward pass verified on GPU")
        except Exception as e:
            print(f"[WARNING] GPU verification issue: {e}")
    
    print(f"[INFO] Network architecture: 1024->1024->512->256 (very large for GPU)")
    print(f"[INFO] Batch size: 512 (maximum for GPU)")
    print(f"[INFO] Parallel environments: {n_envs} (true parallelism with SubprocVecEnv)")
    print(f"[INFO] Steps per rollout: 8192 (keeps GPU busy)")
    print(f"[INFO] Training epochs: 30 (more computation per rollout)")
    
    # Monitor GPU usage
    if device_str == "cuda":
        print(f"\n[INFO] Initial GPU memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    
    # Prepare output directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "ppo_gpu_optimized"
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=str(models_dir), name_prefix="ppo_gpu_ckpt")

    print(f"\n[INFO] Starting training (20000 timesteps)...")
    print("=" * 70)
    print("[INFO] Progress bar is enabled. Watch terminal for live progress...")
    
    import time
    start_time = time.time()
    
    try:
        # Enable SB3 progress bar for real-time training visibility
        model.learn(total_timesteps=20000, progress_bar=True, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user, saving checkpoint...")
    finally:
        # Save model even if interrupted
        model.save(model_path)
        print(f"[OK] Model saved as '{model_path}.zip'")
    
    elapsed = time.time() - start_time
    
    # Final GPU stats
    if device_str == "cuda":
        print(f"\n[INFO] Final GPU memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
        print(f"  Peak allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.1f} MB")
    
    print(f"\n[OK] Training completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"[INFO] Speed: {20000/elapsed:.1f} steps/second")
    
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

