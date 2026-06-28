"""Check GPU utilization during training."""
import subprocess
import time
import sys

def get_gpu_stats():
    """Get GPU utilization and memory stats."""
    try:
        # Get utilization
        util_result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if util_result.returncode == 0:
            values = util_result.stdout.strip().split(', ')
            return {
                'gpu_util': float(values[0]),
                'mem_util': float(values[1]),
                'mem_used': float(values[2]),
                'mem_total': float(values[3]),
                'temp': float(values[4]),
                'power': float(values[5])
            }
    except:
        pass
    return None

def monitor_gpu_continuous():
    """Continuously monitor GPU."""
    print("=" * 70)
    print("GPU Utilization Monitor")
    print("=" * 70)
    print("\nMonitoring GPU (Ctrl+C to stop)...\n")
    print(f"{'Time':<8} {'GPU%':<8} {'Mem%':<8} {'Mem Used':<12} {'Temp':<8} {'Power':<10}")
    print("-" * 70)
    
    stats_history = []
    
    try:
        while True:
            stats = get_gpu_stats()
            if stats:
                mem_gb = stats['mem_used'] / 1024
                mem_total_gb = stats['mem_total'] / 1024
                
                print(f"{time.strftime('%H:%M:%S'):<8} "
                      f"{stats['gpu_util']:>6.1f}%  "
                      f"{stats['mem_util']:>6.1f}%  "
                      f"{mem_gb:>5.1f}/{mem_total_gb:<5.1f}GB  "
                      f"{stats['temp']:>5.0f}°C  "
                      f"{stats['power']:>7.1f}W")
                
                stats_history.append(stats['gpu_util'])
                
                # Show average every 10 readings
                if len(stats_history) % 10 == 0:
                    avg_util = sum(stats_history[-10:]) / 10
                    print(f"\n[Average GPU utilization (last 10s): {avg_util:.1f}%]")
                    if avg_util < 50:
                        print("[WARNING] GPU utilization is low. Consider:")
                        print("  - Using train_max_gpu.py")
                        print("  - Increasing parallel environments")
                        print("  - Increasing batch size")
                    print()
            else:
                print("Could not get GPU stats")
            
            time.sleep(1)
    except KeyboardInterrupt:
        if stats_history:
            print("\n" + "=" * 70)
            print("Summary")
            print("=" * 70)
            print(f"Average GPU utilization: {sum(stats_history) / len(stats_history):.1f}%")
            print(f"Peak GPU utilization: {max(stats_history):.1f}%")
            print(f"Min GPU utilization: {min(stats_history):.1f}%")
            print("=" * 70)

if __name__ == '__main__':
    monitor_gpu_continuous()


