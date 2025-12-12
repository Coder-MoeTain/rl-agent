"""Check PyTorch build information to see what compute capabilities are supported."""
import torch

print("=" * 60)
print("PyTorch Build Information")
print("=" * 60)
print(f"Version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Get GPU compute capability
    try:
        props = torch.cuda.get_device_properties(0)
        print(f"GPU Compute Capability: {props.major}.{props.minor}")
        print(f"GPU Memory: {props.total_memory / 1024**3:.2f} GB")
        
        # Check what PyTorch supports
        print("\nPyTorch supports compute capabilities:")
        # This info is in the warning message, but let's try to get it programmatically
        print("(Check the warning message above for supported capabilities)")
        
    except Exception as e:
        print(f"Error getting GPU properties: {e}")

# Try to get build configuration
try:
    print("\nPyTorch build configuration:")
    print(f"  Compiled with CUDA: {torch.version.cuda is not None}")
    if hasattr(torch.version, 'hip'):
        print(f"  HIP version: {torch.version.hip}")
except:
    pass

print("\n" + "=" * 60)
print("RTX 5080 requires compute capability 12.0 (sm_120)")
print("Current PyTorch builds support up to sm_90")
print("=" * 60)

