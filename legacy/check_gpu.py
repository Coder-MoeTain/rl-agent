"""Check GPU availability."""
import torch

print("=" * 50)
print("GPU/CUDA Check")
print("=" * 50)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
else:
    print("CUDA not available - will use CPU")
print("=" * 50)

