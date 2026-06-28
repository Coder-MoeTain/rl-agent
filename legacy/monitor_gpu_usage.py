"""Monitor GPU usage during training."""
import torch
import time
import subprocess
import sys

def get_gpu_usage():
    """Get current GPU usage percentage."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except:
        pass
    return None

def monitor_gpu():
    """Monitor GPU usage."""
    if not torch.cuda.is_available():
        print("GPU not available")
        return
    
    print("=" * 70)
    print("GPU Usage Monitor")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("\nMonitoring GPU usage (Ctrl+C to stop)...\n")
    
    try:
        while True:
            # PyTorch memory
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(0) / 1024**3
            
            # GPU utilization
            utilization = get_gpu_usage()
            
            print(f"\rMemory: {allocated:.2f}GB / {reserved:.2f}GB (Peak: {max_allocated:.2f}GB) | "
                  f"Utilization: {utilization:.0f}%" if utilization else f"Utilization: N/A",
                  end='', flush=True)
            
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped")

if __name__ == '__main__':
    monitor_gpu()

