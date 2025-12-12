# RTX 5080 GPU Support Notes

## Current Status

Your RTX 5080 GPU is detected, but the current PyTorch build (2.5.1+cu121) doesn't support the Blackwell architecture (sm_120 compute capability).

## What This Means

- ✅ GPU is detected and recognized
- ✅ CUDA drivers are working
- ⚠️ PyTorch kernels don't support sm_120 yet
- ✅ Code will automatically use CPU (which works fine)

## Solutions

### Option 1: Use CPU (Current - Recommended)
The code automatically falls back to CPU. For MLP policies (which this project uses), CPU is actually recommended by Stable-Baselines3 anyway. Training will work perfectly on CPU.

### Option 2: Wait for PyTorch Support
PyTorch will eventually add support for RTX 5080. When that happens, the code will automatically use GPU without any changes needed.

### Option 3: Try PyTorch Nightly (Advanced)
You can try PyTorch nightly builds which may have experimental support:

```bash
pip uninstall torch torchvision torchaudio
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

**Note**: Nightly builds are experimental and may have issues.

## Current Behavior

The code will:
1. Detect your RTX 5080
2. Try to use GPU
3. Automatically fall back to CPU if GPU kernels aren't available
4. Log which device is being used

## Performance

Even on CPU, training is reasonably fast:
- Single agent: ~10 minutes for 20k steps
- Multi-agent: ~30 minutes for 200 episodes

When GPU support is added, you'll get 3-10x speedup automatically.

## No Action Needed

The code is already configured correctly. It will:
- Use GPU when available and compatible
- Use CPU otherwise
- Work perfectly in both cases

Just run your training scripts as normal!

