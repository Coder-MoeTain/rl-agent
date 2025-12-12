# GPU Setup Guide for RTX 5080

## Current Status

Your system shows CUDA is not available. To use your RTX 5080 GPU, you need to install PyTorch with CUDA support.

## Step 1: Check CUDA Version

```bash
# Check NVIDIA driver
nvidia-smi

# This will show your CUDA version (e.g., CUDA 12.1, 12.4, etc.)
```

## Step 2: Install PyTorch with CUDA

### For CUDA 12.1 (Most Common)
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### For CUDA 12.4
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### For CUDA 11.8
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Step 3: Verify GPU Installation

```bash
python check_gpu.py
```

You should see:
```
CUDA available: True
CUDA device count: 1
Current device: 0
Device name: NVIDIA GeForce RTX 5080
```

## Step 4: Test Training with GPU

```bash
# Test single agent
python agents/train_single_agent.py

# You should see: "Training on device: cuda"
```

## Troubleshooting

### Issue: "CUDA not available" after installation
1. **Check NVIDIA drivers**: Update to latest from NVIDIA website
2. **Check CUDA toolkit**: May need to install CUDA toolkit separately
3. **Restart**: Sometimes requires system restart

### Issue: "Out of memory" errors
- Reduce batch size in training scripts
- Use smaller models
- Close other GPU applications

### Issue: PyTorch version mismatch
- Uninstall all PyTorch packages first
- Install correct version for your CUDA version

## Performance Expectations

With RTX 5080, you should see:
- **3-10x faster training** compared to CPU
- **Single agent**: ~1-2 minutes for 20k steps (vs 10 min on CPU)
- **Multi-agent**: ~5-10 minutes for 200 episodes (vs 30 min on CPU)

## Automatic GPU Detection

All training scripts now automatically:
1. Detect if GPU is available
2. Use GPU if available, fallback to CPU
3. Log which device is being used

No manual configuration needed!

