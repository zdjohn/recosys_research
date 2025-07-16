# Device Configuration for CUDA Training

import torch


def get_device(prefer_cuda=True, device_id=0):
    """
    Get the appropriate device for training.

    Args:
        prefer_cuda (bool): Whether to prefer CUDA if available
        device_id (int): GPU device ID to use (default: 0)

    Returns:
        torch.device: The device to use for training
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
        print(f"Using CUDA device: {torch.cuda.get_device_name(device_id)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1024**3:.1f} GB"
        )
    else:
        device = torch.device("cpu")
        print("Using CPU device")
        if prefer_cuda and not torch.cuda.is_available():
            print("WARNING: CUDA was requested but is not available")

    return device


def setup_cuda_optimizations():
    """
    Setup CUDA optimizations for better performance.
    """
    if torch.cuda.is_available():
        # Enable benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = True

        # Enable deterministic operations (may reduce performance)
        # torch.backends.cudnn.deterministic = True

        # Set memory fraction to prevent OOM errors
        # torch.cuda.set_per_process_memory_fraction(0.8)

        print("CUDA optimizations enabled")
    else:
        print("CUDA not available - skipping CUDA optimizations")


def check_cuda_memory():
    """
    Check and print CUDA memory usage.
    """
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}:")
            print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(
                f"  Total: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
            )
    else:
        print("CUDA not available")


def clear_cuda_cache():
    """
    Clear CUDA cache to free up GPU memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared")
    else:
        print("CUDA not available")


# Example usage in training scripts:
if __name__ == "__main__":
    # Setup device
    device = get_device(prefer_cuda=True)

    # Setup optimizations
    setup_cuda_optimizations()

    # Example model movement to device
    # model = YourModel()
    # model = model.to(device)

    # Check memory
    check_cuda_memory()
