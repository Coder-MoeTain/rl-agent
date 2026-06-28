"""Force GPU usage and verify it's actually being used."""
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

def verify_gpu_usage(model):
    """Verify model is actually using GPU."""
    print("\n[INFO] Verifying GPU usage...")
    
    # Check policy device
    try:
        # Get a sample parameter to check device
        sample_param = next(model.policy.parameters())
        device = sample_param.device
        print(f"[OK] Policy parameters on device: {device}")
        
        if device.type != 'cuda':
            print("[ERROR] Policy is NOT on GPU! Forcing to GPU...")
            model.policy = model.policy.to(torch.device('cuda'))
            print("[OK] Policy moved to GPU")
        
        # Test forward pass on GPU
        test_obs = torch.randn(1, 64).to(device)
        with torch.no_grad():
            actions, values, log_probs = model.policy(test_obs)
        print(f"[OK] Forward pass works on {device}")
        print(f"[OK] Actions device: {actions.device if hasattr(actions, 'device') else 'N/A'}")
        
        return True
    except Exception as e:
        print(f"[ERROR] GPU verification failed: {e}")
        return False

def main():
    """Train with forced GPU usage."""
    print("=" * 70)
    print("Force GPU Training - 100% GPU Utilization")
    print("=" * 70)
    
    # Force CUDA
    if not torch.cuda.is_available():
        print("\n[ERROR] CUDA not available! Cannot use GPU.")
        return False
    
    # Set default device to CUDA
    torch.cuda.set_device(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    print(f"\n[OK] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[OK] CUDA Version: {torch.version.cuda}")
    
    # GPU optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    torch.cuda.empty_cache()
    
    # Test GPU computation
    print("\n[INFO] Testing GPU computation...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        result = z.sum().item()
        print(f"[OK] GPU computation test passed: {result:.2f}")
        del x, y, z
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[ERROR] GPU test failed: {e}")
        return False
    
    # Many parallel environments
    n_envs = 32
    print(f"\n[INFO] Creating {n_envs} parallel environments...")
    try:
        def make_env():
            return PentestEnv()
        env = SubprocVecEnv([make_env for _ in range(n_envs)], start_method='spawn')
        print(f"[OK] Using SubprocVecEnv for true parallelism")
    except Exception as e:
        print(f"[WARNING] SubprocVecEnv failed: {e}")
        env = make_vec_env(PentestEnv, n_envs=n_envs)
    
    # Create model - FORCE GPU
    print(f"\n[INFO] Creating PPO model - FORCING GPU usage...")
    
    # Explicitly set device to cuda
    device_str = "cuda:0"  # Explicit device
    
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        device=device_str,  # Explicit CUDA device
        tensorboard_log="./tensorboard_logs/force_gpu/",
        
        # Maximum GPU settings
        n_steps=8192,
        batch_size=512,
        n_epochs=30,
        
        # Very large network
        policy_kwargs=dict(
            net_arch=[dict(pi=[1024, 1024, 512, 256], vf=[1024, 1024, 512, 256])]
        ),
        
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
    )
    
    # Verify and force GPU
    print(f"\n[INFO] Model device (from SB3): {model.device}")
    verify_gpu_usage(model)
    
    # Force all components to GPU
    print("\n[INFO] Forcing all model components to GPU...")
    model.policy = model.policy.to(torch.device('cuda:0'))
    if hasattr(model, 'value_net'):
        model.value_net = model.value_net.to(torch.device('cuda:0'))
    
    # Verify again
    sample_param = next(model.policy.parameters())
    print(f"[OK] Verified: Parameters on {sample_param.device}")
    
    print(f"\n[INFO] Configuration:")
    print(f"  - Device: {device_str}")
    print(f"  - Parallel environments: {n_envs}")
    print(f"  - Batch size: 512")
    print(f"  - Network: 1024->1024->512->256")
    print(f"  - Steps per rollout: 8192")
    print(f"  - Epochs: 30")
    
    # Initial GPU memory
    print(f"\n[INFO] Initial GPU memory:")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    
    print(f"\n[INFO] Starting training (20000 timesteps)...")
    print("=" * 70)
    print("\nMonitor GPU with: nvidia-smi -l 1")
    print("=" * 70)
    print()
    
    import time
    start_time = time.time()
    
    # Monitor GPU during training
    def check_gpu_periodically():
        """Check GPU usage periodically."""
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        if allocated > 100:  # Only log if actually using GPU
            print(f"\n[GPU] Memory: {allocated:.0f}MB / {reserved:.0f}MB")
    
    model.learn(total_timesteps=20000)
    
    elapsed = time.time() - start_time
    
    # Final stats
    print(f"\n[INFO] Final GPU memory:")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    print(f"  Peak: {torch.cuda.max_memory_allocated(0) / 1024**2:.1f} MB")
    
    print(f"\n[OK] Training completed in {elapsed:.2f}s")
    print(f"[INFO] Speed: {20000/elapsed:.1f} steps/second")
    
    model.save('ppo_force_gpu')
    print(f"[OK] Model saved")
    
    return True

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


