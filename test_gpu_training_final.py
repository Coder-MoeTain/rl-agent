"""Final GPU training test with CUDA 12.8."""
import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_pentest.env import PentestEnv

print("=" * 60)
print("GPU Training Test with CUDA 12.8")
print("=" * 60)

# Verify GPU
if not torch.cuda.is_available():
    print("[ERROR] CUDA not available")
    sys.exit(1)

print(f"\n[OK] GPU: {torch.cuda.get_device_name(0)}")
print(f"[OK] CUDA Version: {torch.version.cuda}")

# Test GPU works
try:
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = torch.matmul(x, y)
    result = z.sum().item()
    print(f"[OK] GPU computation test passed: {result:.2f}")
    del x, y, z
    torch.cuda.empty_cache()
except Exception as e:
    print(f"[ERROR] GPU test failed: {e}")
    sys.exit(1)

# Create environment
print("\n[INFO] Creating environment...")
env = make_vec_env(PentestEnv, n_envs=1)

# Create model on GPU
print("\n[INFO] Creating PPO model on GPU...")
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    device="cuda",  # Use GPU!
    tensorboard_log="./tensorboard_logs/gpu_test/",
    learning_rate=3e-4,
    n_steps=256,  # Small for quick test
    batch_size=32,
    n_epochs=2
)

print(f"\n[OK] Model created on device: {model.device}")

# Quick training
print("\n[INFO] Starting GPU training test (500 steps)...")
print("=" * 60)

import time
start_time = time.time()

model.learn(total_timesteps=500)

elapsed = time.time() - start_time

print("=" * 60)
print(f"\n[SUCCESS] GPU Training completed in {elapsed:.2f} seconds!")
print(f"[INFO] Speed: {500/elapsed:.1f} steps/second")
print(f"[INFO] GPU was used successfully!")

# Check GPU memory usage
if torch.cuda.is_available():
    memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
    memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
    print(f"[INFO] GPU Memory - Allocated: {memory_allocated:.1f} MB, Reserved: {memory_reserved:.1f} MB")

print("\n" + "=" * 60)
print("[SUCCESS] GPU Training Works!")
print("=" * 60)
print("\nYou can now train with GPU using:")
print("  python agents/train_single_agent.py")
print("  python train_sb3_per.py")
print("  (They will automatically use GPU now)")

