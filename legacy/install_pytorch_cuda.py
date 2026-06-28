"""Install PyTorch with CUDA support."""
import subprocess
import sys

def install_pytorch_cuda():
    """Install PyTorch with CUDA 12.1 support."""
    print("=" * 60)
    print("Installing PyTorch with CUDA Support")
    print("=" * 60)
    print("This will uninstall current PyTorch and install CUDA version...")
    print()
    
    # Uninstall current PyTorch
    print("Step 1: Uninstalling current PyTorch...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
        print("[OK] Uninstalled successfully")
    except Exception as e:
        print(f"[WARNING] {e}")
    
    # Install PyTorch with CUDA 12.1 (compatible with CUDA 12.9)
    print("\nStep 2: Installing PyTorch with CUDA 12.1...")
    print("(CUDA 12.1 is compatible with CUDA 12.9 driver)")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ])
        print("[OK] Installed successfully")
    except Exception as e:
        print(f"[ERROR] Installation failed: {e}")
        print("\nTrying CUDA 12.4...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu124"
            ])
            print("[OK] Installed with CUDA 12.4")
        except Exception as e2:
            print(f"[ERROR] Installation failed: {e2}")
            return False
    
    # Verify installation
    print("\nStep 3: Verifying installation...")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("[WARNING] CUDA still not available - may need to restart Python")
            return False
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        return False

if __name__ == '__main__':
    success = install_pytorch_cuda()
    if success:
        print("\n" + "=" * 60)
        print("SUCCESS! PyTorch with CUDA is now installed.")
        print("You may need to restart Python/IDE for changes to take effect.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Installation completed but CUDA not verified.")
        print("Try running: python check_pytorch.py")
        print("=" * 60)

