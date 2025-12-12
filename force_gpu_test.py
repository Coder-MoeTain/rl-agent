"""Force GPU test with various workarounds."""
import sys
from pathlib import Path
import os
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Try setting environment variables that might help
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

def test_gpu_directly():
    """Test GPU with direct PyTorch operations."""
    print("=" * 60)
    print("Direct GPU Test")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available")
        return False
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Try different approaches
    print("\nTest 1: Simple tensor creation...")
    try:
        x = torch.zeros(10).cuda()
        print("[OK] Tensor created on GPU")
        del x
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[FAILED] {e}")
        return False
    
    print("\nTest 2: Tensor operations...")
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        result = z.sum().item()
        print(f"[OK] Matrix multiplication works: result = {result:.2f}")
        del x, y, z
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[FAILED] {e}")
        return False
    
    print("\nTest 3: Neural network on GPU...")
    try:
        import torch.nn as nn
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        ).cuda()
        
        x = torch.randn(32, 64).cuda()
        y = model(x)
        print(f"[OK] Neural network forward pass works: output shape = {y.shape}")
        del model, x, y
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[FAILED] {e}")
        return False
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All GPU tests passed!")
    print("GPU is working - we can try training")
    print("=" * 60)
    return True

if __name__ == '__main__':
    try:
        success = test_gpu_directly()
        if success:
            print("\nGPU is functional! We can proceed with training.")
        else:
            print("\nGPU tests failed. Will need to use CPU.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

