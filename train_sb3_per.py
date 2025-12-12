"""Training script for PPO with Prioritized Experience Replay."""
import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from custom_sb3_per import PPO_PER
from stable_baselines3.common.env_util import make_vec_env
from gym_pentest.env import PentestEnv
from gpu_utils import setup_gpu, check_gpu_requirements

if __name__ == '__main__':
    # Setup GPU
    device = setup_gpu()
    check_gpu_requirements()
    
    # GPU-optimized: Use many parallel environments to keep GPU busy
    # Since environment steps are CPU-bound (HTTP requests), we need many parallel envs
    if torch.cuda.is_available():
        n_envs = 32  # Many environments to keep GPU busy during HTTP requests
        print(f"Creating {n_envs} parallel environments for GPU optimization...")
        try:
            from stable_baselines3.common.vec_env import SubprocVecEnv
            def make_env():
                return PentestEnv()
            env = SubprocVecEnv([make_env for _ in range(n_envs)], start_method='spawn')
            print(f"[OK] Using SubprocVecEnv for true parallelism")
        except Exception as e:
            print(f"[WARNING] SubprocVecEnv failed: {e}, using DummyVecEnv")
            env = make_vec_env(PentestEnv, n_envs=n_envs)
    else:
        n_envs = 1
        print(f"Creating {n_envs} environment for CPU...")
        env = make_vec_env(PentestEnv, n_envs=n_envs)
    
    # Use GPU if available and compatible
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test if GPU actually works and optimize
    if device_str == "cuda":
        try:
            test = torch.zeros(1).cuda()
            del test
            torch.cuda.empty_cache()
            # GPU optimizations for maximum utilization
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        except RuntimeError:
            print("[WARNING] GPU detected but not compatible, using CPU")
            device_str = "cpu"
            n_envs = 1
    
    print(f"Training on device: {device_str}")
    
    # GPU-optimized hyperparameters for maximum GPU utilization
    gpu_kwargs = {}
    if device_str == "cuda":
        gpu_kwargs = {
            'n_steps': 8192,  # Large rollout - keeps GPU busy
            'batch_size': 512,  # Very large batches
            'n_epochs': 30,  # Many epochs - more computation
            'policy_kwargs': dict(
                net_arch=[dict(pi=[1024, 1024, 512, 256], vf=[1024, 1024, 512, 256])]  # Very large networks
            )
        }
        print("[INFO] Using maximum GPU-optimized settings:")
        print(f"  - Parallel environments: {n_envs} (SubprocVecEnv)")
        print(f"  - Batch size: 512 (maximum)")
        print(f"  - PER batch size: 512")
        print(f"  - Network: 1024->1024->512->256 (very large)")
        print(f"  - Steps per rollout: 8192")
        print(f"  - Training epochs: 30")
    
    model = PPO_PER(
        'MlpPolicy',
        env,
        verbose=1,  # Enable verbose for real-time logs
        per_capacity=8192 if device_str == "cuda" else 4096,  # Larger PER buffer for GPU
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=10000,
        per_batch_size=256 if device_str == "cuda" else 64,  # Larger PER batch for GPU
        tensorboard_log="./tensorboard_logs/",
        device=device_str,
        learning_rate=3e-4,
        **gpu_kwargs  # Apply GPU optimizations
    )
    
    print(f"\n{'='*70}")
    print("Starting Training with PER - Real-Time Logs Below")
    print(f"{'='*70}\n")
    
    model.learn(total_timesteps=20000)
    model.save('ppo_per_model')
    print('Saved ppo_per_model')
