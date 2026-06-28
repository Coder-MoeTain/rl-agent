"""Force 100% GPU usage by overriding SB3 defaults and using very large networks."""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from gym_pentest.env import PentestEnv

# Force CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.set_device(0)

class LargeMLPPolicy(ActorCriticPolicy):
    """Very large MLP policy to force GPU usage."""
    def __init__(self, *args, **kwargs):
        # Override to use very large network
        kwargs['net_arch'] = [dict(pi=[2048, 2048, 1024, 512, 256], vf=[2048, 2048, 1024, 512, 256])]
        super().__init__(*args, **kwargs)
        
        # Force all parameters to GPU
        self.to(torch.device('cuda:0'))
        print(f"[OK] Policy forced to GPU: {next(self.parameters()).device}")

def main():
    """Train with forced GPU and very large networks."""
    print("=" * 70)
    print("FORCED GPU Training - 100% GPU Utilization")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available!")
        return False
    
    print(f"\n[OK] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[OK] CUDA: {torch.version.cuda}")
    
    # GPU optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.cuda.empty_cache()
    
    # Test GPU
    print("\n[INFO] Testing GPU...")
    x = torch.randn(2000, 2000).cuda()
    y = torch.randn(2000, 2000).cuda()
    z = torch.matmul(x, y)
    print(f"[OK] GPU test: {z.sum().item():.2f}")
    del x, y, z
    torch.cuda.empty_cache()
    
    # Many parallel environments
    n_envs = 32
    print(f"\n[INFO] Creating {n_envs} parallel environments...")
    try:
        def make_env():
            return PentestEnv()
        env = SubprocVecEnv([make_env for _ in range(n_envs)], start_method='spawn')
        print("[OK] SubprocVecEnv created")
    except Exception as e:
        print(f"[WARNING] {e}, using DummyVecEnv")
        env = make_vec_env(PentestEnv, n_envs=n_envs)
    
    # Create model with VERY large network
    print(f"\n[INFO] Creating model with VERY large network (2048->2048->1024->512->256)...")
    print("[INFO] This forces GPU usage even for MLP policies")
    
    model = PPO(
        LargeMLPPolicy,  # Custom large policy
        env,
        verbose=1,
        device='cuda:0',  # Explicit CUDA
        tensorboard_log="./tensorboard_logs/forced_gpu/",
        
        # Maximum settings
        n_steps=8192,
        batch_size=1024,  # Very large batch
        n_epochs=40,  # Many epochs
        
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    
    # Force to GPU
    print("\n[INFO] Forcing all components to GPU...")
    model.policy = model.policy.to(torch.device('cuda:0'))
    
    # Verify
    param = next(model.policy.parameters())
    print(f"[OK] Parameters on: {param.device}")
    
    # Test forward pass
    test_obs = torch.randn(32, 64).cuda()
    with torch.no_grad():
        actions, values, log_probs = model.policy(test_obs)
    print(f"[OK] Forward pass on GPU verified")
    print(f"[OK] Actions device: {actions.device if hasattr(actions, 'device') else 'N/A'}")
    
    # Initial memory
    init_mem = torch.cuda.memory_allocated(0) / 1024**2
    print(f"\n[INFO] Initial GPU memory: {init_mem:.1f} MB")
    
    print(f"\n[INFO] Configuration:")
    print(f"  - Network: 2048->2048->1024->512->256 (VERY LARGE)")
    print(f"  - Batch size: 1024 (VERY LARGE)")
    print(f"  - Parallel envs: {n_envs}")
    print(f"  - Steps: 8192")
    print(f"  - Epochs: 40")
    
    print(f"\n[INFO] Starting training...")
    print("=" * 70)
    print("Monitor GPU: nvidia-smi -l 1")
    print("=" * 70)
    print()
    
    import time
    start = time.time()
    
    model.learn(total_timesteps=20000)
    
    elapsed = time.time() - start
    
    # Final stats
    final_mem = torch.cuda.memory_allocated(0) / 1024**2
    peak_mem = torch.cuda.max_memory_allocated(0) / 1024**2
    
    print(f"\n[INFO] Training complete:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  GPU memory: {init_mem:.1f}MB -> {final_mem:.1f}MB")
    print(f"  Peak memory: {peak_mem:.1f}MB")
    print(f"  Speed: {20000/elapsed:.1f} steps/s")
    
    if peak_mem > 500:  # More than 500MB indicates GPU usage
        print("\n[SUCCESS] GPU is being used! (High memory usage confirms)")
    else:
        print("\n[WARNING] Low GPU memory - may need larger network")
    
    model.save('ppo_forced_gpu')
    print("\n[OK] Model saved")
    
    return True

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


