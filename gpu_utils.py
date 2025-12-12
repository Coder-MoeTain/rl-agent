"""GPU utilities for training."""
import torch
import logging

logger = logging.getLogger(__name__)

def get_device():
    """Get the best available device (GPU if available and compatible, else CPU).
    
    Returns:
        torch.device: Device to use for training
    """
    if torch.cuda.is_available():
        try:
            # Test if GPU actually works by creating and using a small tensor
            test_tensor = torch.zeros(1).cuda()
            result = test_tensor + 1  # Try a simple operation
            result.item()  # Force computation
            del test_tensor, result
            torch.cuda.empty_cache()
            
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            return device
        except (RuntimeError, Exception) as e:
            error_msg = str(e)
            if "no kernel image" in error_msg or "sm_120" in error_msg or "not compatible" in error_msg:
                logger.warning("=" * 60)
                logger.warning("GPU DETECTED BUT ARCHITECTURE NOT SUPPORTED")
                logger.warning("=" * 60)
                logger.warning(f"GPU: {torch.cuda.get_device_name(0)}")
                logger.warning("Current PyTorch doesn't support your GPU architecture (sm_120)")
                logger.warning("Falling back to CPU (which works fine for MLP policies)")
                logger.warning("")
                logger.warning("Note: For RTX 5080, you may need to wait for PyTorch")
                logger.warning("      to add support, or try PyTorch nightly builds")
                logger.warning("=" * 60)
            else:
                logger.warning(f"GPU detected but error occurred: {e}")
                logger.warning("Falling back to CPU")
            return torch.device("cpu")
    else:
        logger.warning("CUDA not available, using CPU")
        return torch.device("cpu")

def setup_gpu():
    """Setup GPU for optimal performance.
    
    Returns:
        torch.device: Device to use
    """
    device = get_device()
    
    if device.type == "cuda":
        # Set cuDNN benchmark for faster training
        torch.backends.cudnn.benchmark = True
        # Enable deterministic mode (optional, for reproducibility)
        # torch.backends.cudnn.deterministic = True
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        logger.info("GPU setup complete")
    else:
        logger.info("Using CPU")
    
    return device

def check_gpu_requirements():
    """Check if GPU requirements are met.
    
    Returns:
        bool: True if GPU is available and ready
    """
    if not torch.cuda.is_available():
        logger.warning("=" * 60)
        logger.warning("GPU NOT AVAILABLE")
        logger.warning("=" * 60)
        logger.warning("To use GPU, you need:")
        logger.warning("1. NVIDIA GPU with CUDA support")
        logger.warning("2. CUDA drivers installed")
        logger.warning("3. PyTorch with CUDA support")
        logger.warning("")
        logger.warning("Install PyTorch with CUDA:")
        logger.warning("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        logger.warning("  (Adjust cu121 to your CUDA version)")
        logger.warning("=" * 60)
        return False
    
    # Test if GPU actually works
    try:
        test = torch.zeros(1).cuda()
        del test
        torch.cuda.empty_cache()
        return True
    except RuntimeError as e:
        logger.warning("=" * 60)
        logger.warning("GPU DETECTED BUT NOT COMPATIBLE")
        logger.warning("=" * 60)
        logger.warning(f"Error: {e}")
        logger.warning("")
        logger.warning("This usually means:")
        logger.warning("  - Your GPU architecture (e.g., RTX 5080 sm_120) is newer than")
        logger.warning("    what the current PyTorch build supports (up to sm_90)")
        logger.warning("  - You may need PyTorch nightly build or wait for official support")
        logger.warning("")
        logger.warning("For now, training will use CPU (which works fine for MLP policies)")
        logger.warning("=" * 60)
        return False

