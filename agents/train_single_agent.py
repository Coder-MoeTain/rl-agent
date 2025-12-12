"""Single agent PPO training script."""
import sys
from pathlib import Path
import torch
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_pentest.env import PentestEnv
from gpu_utils import setup_gpu, check_gpu_requirements

def main():
    """Train a PPO agent on the penetration testing environment."""
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
        # Force explicit CUDA device
        device_str = "cuda:0"
        torch.cuda.set_device(0)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        # Maximum GPU settings - VERY large networks to force GPU usage
        # SB3 warns MLP policies are for CPU, but large networks force GPU usage
        gpu_kwargs = {
            'n_steps': 8192,  # Large rollout
            'batch_size': 1024,  # VERY large batches to force GPU computation
            'n_epochs': 40,  # Many epochs
            'policy_kwargs': dict(
                net_arch=[dict(pi=[2048, 2048, 1024, 512, 256], vf=[2048, 2048, 1024, 512, 256])]  # VERY large networks
            )
        }
        print("[INFO] Using MAXIMUM GPU-optimized settings:")
        print(f"  - Device: {device_str} (explicit CUDA)")
        print(f"  - Parallel environments: {n_envs} (SubprocVecEnv)")
        print(f"  - Batch size: 1024 (VERY large to force GPU)")
        print(f"  - Network: 2048->2048->1024->512->256 (VERY large to force GPU)")
        print(f"  - Steps per rollout: 8192")
        print(f"  - Training epochs: 40")
        print(f"  - Strategy: Very large networks force GPU usage even for MLP")
    
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,  # Enable verbose for real-time logs
        tensorboard_log="./tensorboard_logs/",
        device=device_str,
        learning_rate=3e-4,
        **gpu_kwargs  # Apply GPU optimizations if using GPU
    )
    
    # Force GPU if using CUDA
    if device_str.startswith("cuda"):
        print("\n[INFO] Verifying and forcing GPU usage...")
        try:
            param = next(model.policy.parameters())
            if param.device.type != 'cuda':
                print("[WARNING] Not on GPU! Forcing...")
                model.policy = model.policy.to(torch.device(device_str))
            param = next(model.policy.parameters())
            print(f"[OK] Verified on GPU: {param.device}")
        except Exception as e:
            print(f"[WARNING] GPU verification: {e}")
    
    print(f"\n{'='*70}")
    print("Starting Training - Real-Time Logs Below")
    print(f"{'='*70}\n")
    
    model.learn(total_timesteps=20000)
    model.save('ppo_baseline')
    print('Saved ppo_baseline')

if __name__ == '__main__':
    main()
