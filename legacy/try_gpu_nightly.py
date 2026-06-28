"""Try installing PyTorch nightly for RTX 5080 support."""
import subprocess
import sys
import os

def install_pytorch_nightly():
    """Install PyTorch nightly build which may have RTX 5080 support."""
    print("=" * 60)
    print("Installing PyTorch Nightly Build")
    print("=" * 60)
    print("Nightly builds may have experimental support for RTX 5080")
    print()
    
    # Uninstall current PyTorch
    print("Step 1: Uninstalling current PyTorch...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
        print("[OK] Uninstalled successfully")
    except Exception as e:
        print(f"[WARNING] {e}")
    
    # Install PyTorch nightly
    print("\nStep 2: Installing PyTorch nightly with CUDA 12.1...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--pre", "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/nightly/cu121"
        ])
        print("[OK] Installed successfully")
    except Exception as e:
        print(f"[ERROR] Installation failed: {e}")
        return False
    
    # Verify installation
    print("\nStep 3: Verifying installation...")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"[OK] CUDA available")
            try:
                # Test if it actually works
                test = torch.zeros(1).cuda()
                result = test + 1
                result.item()
                del test, result
                torch.cuda.empty_cache()
                print(f"[OK] GPU works: {torch.cuda.get_device_name(0)}")
                return True
            except RuntimeError as e:
                print(f"[WARNING] GPU detected but error: {e}")
                return False
        else:
            print("[WARNING] CUDA still not available")
            return False
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        return False

if __name__ == '__main__':
    print("WARNING: This will replace your current PyTorch installation")
    print("Continue? (This is experimental)")
    response = input("Type 'yes' to continue: ")
    
    if response.lower() == 'yes':
        success = install_pytorch_nightly()
        if success:
            print("\n" + "=" * 60)
            print("SUCCESS! PyTorch nightly installed and GPU works!")
            print("You may need to restart Python/IDE")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("Installation completed but GPU still not working")
            print("RTX 5080 support may not be available yet")
            print("=" * 60)
    else:
        print("Cancelled")

