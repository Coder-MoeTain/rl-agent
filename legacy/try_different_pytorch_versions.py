"""Try different PyTorch versions to find one that works with RTX 5080."""
import subprocess
import sys
import torch

def test_current_pytorch():
    """Test current PyTorch installation."""
    print("=" * 60)
    print("Testing Current PyTorch")
    print("=" * 60)
    print(f"Version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Test GPU
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.matmul(x, y)
            result = z.sum().item()
            print(f"[SUCCESS] GPU works! Test result: {result:.2f}")
            return True
        except Exception as e:
            print(f"[FAILED] GPU test failed: {str(e)[:100]}")
            return False
    return False

def install_pytorch_version(cuda_version):
    """Install PyTorch with specific CUDA version."""
    print(f"\n{'=' * 60}")
    print(f"Trying PyTorch with CUDA {cuda_version}")
    print("=" * 60)
    
    # Uninstall current
    print("Uninstalling current PyTorch...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"], 
                   capture_output=True)
    
    # Install new version
    if cuda_version == "11.8":
        index_url = "https://download.pytorch.org/whl/cu118"
    elif cuda_version == "12.1":
        index_url = "https://download.pytorch.org/whl/cu121"
    elif cuda_version == "12.4":
        index_url = "https://download.pytorch.org/whl/cu124"
    else:
        print(f"[ERROR] Unknown CUDA version: {cuda_version}")
        return False
    
    print(f"Installing from: {index_url}")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", index_url
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("[OK] Installation successful")
            return True
        else:
            print(f"[ERROR] Installation failed: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"[ERROR] Installation error: {e}")
        return False

def main():
    """Try different PyTorch versions."""
    print("=" * 60)
    print("Trying Different PyTorch/CUDA Versions for RTX 5080")
    print("=" * 60)
    
    # Test current
    if test_current_pytorch():
        print("\n[SUCCESS] Current PyTorch works with GPU!")
        return
    
    versions_to_try = [
        ("12.4", "CUDA 12.4 - Latest stable"),
        ("12.1", "CUDA 12.1 - Most common"),
        ("11.8", "CUDA 11.8 - Older but stable"),
    ]
    
    for cuda_ver, description in versions_to_try:
        print(f"\n{description}")
        if install_pytorch_version(cuda_ver):
            # Reload torch
            import importlib
            importlib.reload(torch)
            
            if test_current_pytorch():
                print(f"\n[SUCCESS] PyTorch with CUDA {cuda_ver} works!")
                print(f"Version: {torch.__version__}")
                return
            else:
                print(f"[FAILED] PyTorch with CUDA {cuda_ver} doesn't work")
        else:
            print(f"[FAILED] Could not install PyTorch with CUDA {cuda_ver}")
    
    print("\n" + "=" * 60)
    print("None of the tested versions work with RTX 5080")
    print("RTX 5080 (sm_120) requires PyTorch built with sm_120 support")
    print("This may require:")
    print("  1. Waiting for official PyTorch release")
    print("  2. Building PyTorch from source")
    print("  3. Using CPU (which works fine for MLP policies)")
    print("=" * 60)

if __name__ == '__main__':
    main()

