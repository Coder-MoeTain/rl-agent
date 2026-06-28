"""Check PyTorch installation and CUDA support."""
import torch

print("=" * 60)
print("PyTorch Installation Check")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.version.cuda:
    print(f"CUDA version in PyTorch: {torch.version.cuda}")
else:
    print("CUDA version in PyTorch: None (CPU-only build)")

if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("\n" + "=" * 60)
    print("CUDA NOT AVAILABLE")
    print("=" * 60)
    print("Your system has CUDA 12.9, but PyTorch was installed without CUDA support.")
    print("\nTo install PyTorch with CUDA 12.1 support (compatible with CUDA 12.9):")
    print("  pip uninstall torch torchvision torchaudio")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("\nOr for CUDA 12.4:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    print("=" * 60)

