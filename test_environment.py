"""Quick test script to verify environment and training setup."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")
    try:
        from gym_pentest.env import PentestEnv
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        import torch
        import numpy as np
        print("[OK] All imports successful")
        return True
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False

def test_environment():
    """Test environment creation and basic functionality."""
    print("\nTesting environment...")
    try:
        from gym_pentest.env import PentestEnv
        
        env = PentestEnv()
        obs, info = env.reset()
        
        assert obs.shape == (64,), f"Expected obs shape (64,), got {obs.shape}"
        assert env.action_space.n == 8, f"Expected 8 actions, got {env.action_space.n}"
        
        # Test all actions
        for action in range(8):
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (64,), f"Invalid obs shape after action {action}"
        
        print("[OK] Environment working correctly")
        return True
    except Exception as e:
        print(f"[ERROR] Environment error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_setup():
    """Test that training can be initialized."""
    print("\nTesting training setup...")
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from gym_pentest.env import PentestEnv
        
        env = make_vec_env(PentestEnv, n_envs=1)
        model = PPO('MlpPolicy', env, verbose=0)
        
        print("[OK] Training setup OK")
        return True
    except Exception as e:
        print(f"[ERROR] Training setup error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_docker_connection():
    """Test connection to Juice Shop."""
    print("\nTesting Docker/Juice Shop connection...")
    try:
        import requests
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("[OK] Juice Shop is running")
            return True
        else:
            print(f"[WARNING] Juice Shop returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("[WARNING] Juice Shop not running - start with: docker-compose up -d")
        return False
    except Exception as e:
        print(f"[WARNING] Connection test error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("mini-pentest-rl Environment Test")
    print("=" * 50)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Environment", test_environment()))
    results.append(("Training Setup", test_training_setup()))
    results.append(("Docker Connection", test_docker_connection()))
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{name:20} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("[SUCCESS] All critical tests passed!")
        print("\nYou can now run training:")
        print("  python agents/train_single_agent.py")
    else:
        print("[FAILED] Some tests failed - please fix issues before training")
        print("\nCommon fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Start Juice Shop: docker-compose up -d")
        print("  3. Check Python version: python --version (need 3.8+)")
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

