"""Verify GPU is actually being used during training."""
import sys
from pathlib import Path
import torch
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_pentest.env import PentestEnv

def main():
    """Quick test to verify GPU usage."""
    print("=" * 70)
    print("GPU Usage Verification Test")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available")
        return
    
    print(f"\n[OK] GPU: {torch.cuda.get_device_name(0)}")
    
    # Create small test
    env = make_vec_env(PentestEnv, n_envs=4)
    
    print("\n[INFO] Creating model with device='cuda:0'...")
    model = PPO(
        'MlpPolicy',
        env,
        verbose=0,
        device='cuda:0',  # Explicit CUDA
        n_steps=256,
        batch_size=64,
        n_epochs=2
    )
    
    # Check device
    print(f"\n[INFO] Model device (SB3): {model.device}")
    
    # Check actual parameter device
    sample_param = next(model.policy.parameters())
    print(f"[INFO] Parameter device: {sample_param.device}")
    
    if sample_param.device.type != 'cuda':
        print("\n[ERROR] Parameters are NOT on GPU!")
        print("[INFO] Forcing to GPU...")
        model.policy = model.policy.to(torch.device('cuda:0'))
        sample_param = next(model.policy.parameters())
        print(f"[OK] Now on: {sample_param.device}")
    
    # Test forward pass
    print("\n[INFO] Testing forward pass on GPU...")
    test_obs = torch.randn(4, 64).to(torch.device('cuda:0'))
    
    initial_mem = torch.cuda.memory_allocated(0)
    
    with torch.no_grad():
        actions, values, log_probs = model.policy(test_obs)
    
    after_mem = torch.cuda.memory_allocated(0)
    mem_used = (after_mem - initial_mem) / 1024**2
    
    print(f"[OK] Forward pass completed")
    print(f"[OK] GPU memory used: {mem_used:.1f} MB")
    
    # Quick training test
    print("\n[INFO] Running quick training test (500 steps)...")
    print("[INFO] Monitor GPU with: nvidia-smi -l 1")
    
    start_mem = torch.cuda.memory_allocated(0)
    start_time = time.time()
    
    model.learn(total_timesteps=500)
    
    elapsed = time.time() - start_time
    end_mem = torch.cuda.memory_allocated(0)
    peak_mem = torch.cuda.max_memory_allocated(0)
    
    print(f"\n[INFO] Training completed:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  GPU memory: {start_mem/1024**2:.1f}MB -> {end_mem/1024**2:.1f}MB")
    print(f"  Peak memory: {peak_mem/1024**2:.1f}MB")
    
    if peak_mem > 100 * 1024**2:  # More than 100MB
        print("\n[SUCCESS] GPU is being used! (Memory usage indicates GPU computation)")
    else:
        print("\n[WARNING] Low GPU memory usage - may not be using GPU effectively")
    
    print("=" * 70)

if __name__ == '__main__':
    main()


