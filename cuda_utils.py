"""
CUDA Utilities for Cross-Platform Development

This module provides utilities for automatically detecting and configuring
the appropriate compute device (CUDA, MPS, or CPU) based on platform availability.
"""

import torch
import platform
from typing import Union, Dict, Any

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def get_device_info() -> Dict[str, Any]:
    """
    Get comprehensive device information for PyTorch and TensorFlow.

    Returns:
        Dict containing device information for both frameworks
    """
    info = {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "pytorch": get_pytorch_device_info(),
        "tensorflow": (
            get_tensorflow_device_info()
            if TF_AVAILABLE
            else {"error": "TensorFlow not available"}
        ),
    }
    return info


def get_pytorch_device_info() -> Dict[str, Any]:
    """Get PyTorch device information."""
    info = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available(),
        "device_count": 0,
        "devices": [],
    }

    if info["cuda_available"]:
        info["cuda_version"] = getattr(torch.version, "cuda", "Unknown")
        info["device_count"] = torch.cuda.device_count()
        for i in range(info["device_count"]):
            device_info = {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_gb": torch.cuda.get_device_properties(i).total_memory
                / (1024**3),
                "compute_capability": torch.cuda.get_device_capability(i),
            }
            info["devices"].append(device_info)

    return info


def get_tensorflow_device_info() -> Dict[str, Any]:
    """Get TensorFlow device information."""
    if not TF_AVAILABLE:
        return {"error": "TensorFlow not available"}

    info = {"version": tf.__version__, "gpu_devices": [], "gpu_count": 0}

    try:
        gpus = tf.config.list_physical_devices("GPU")
        info["gpu_count"] = len(gpus)

        for i, gpu in enumerate(gpus):
            device_info = {"id": i, "name": gpu.name, "device_type": gpu.device_type}

            try:
                details = tf.config.experimental.get_device_details(gpu)
                device_info.update(details)
            except:
                pass

            info["gpu_devices"].append(device_info)

    except Exception as e:
        info["error"] = str(e)

    return info


def get_optimal_device(prefer_gpu: bool = True) -> str:
    """
    Get the optimal PyTorch device based on availability.

    Args:
        prefer_gpu: Whether to prefer GPU over CPU when available

    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if not prefer_gpu:
        return "cpu"

    # Check CUDA availability (Linux/Windows)
    if torch.cuda.is_available():
        return "cuda"

    # Check MPS availability (Mac M1/M2/M3)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    # Fallback to CPU
    return "cpu"


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
        return device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Metal Performance Shaders (MPS) for Mac GPU acceleration")
        return device
    else:
        device = torch.device("cpu")
        print("Using CPU")
        return device


def setup_tensorflow_gpu():
    """
    Configure TensorFlow for optimal GPU usage.

    This function sets up memory growth and mixed precision if available.
    """
    if not TF_AVAILABLE:
        print("TensorFlow not available")
        return

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Set up mixed precision for better performance on compatible GPUs
            try:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                print(f"Mixed precision enabled: {policy.name}")
            except Exception as e:
                print(f"Mixed precision not available: {e}")

            print(f"TensorFlow configured for {len(gpus)} GPU(s)")

        except Exception as e:
            print(f"Error configuring TensorFlow GPU: {e}")
    else:
        print("No GPUs found. TensorFlow will use CPU.")


def print_device_summary():
    """Print a comprehensive summary of available devices."""
    info = get_device_info()

    print("=" * 60)
    print("DEVICE SUMMARY")
    print("=" * 60)
    print(f"Platform: {info['platform']} ({info['architecture']})")
    print()

    # PyTorch info
    print("PyTorch:")
    print(f"  Version: {info['pytorch']['version']}")
    print(f"  CUDA Available: {info['pytorch']['cuda_available']}")
    print(f"  MPS Available: {info['pytorch']['mps_available']}")
    print(f"  Optimal Device: {get_optimal_device()}")

    if info["pytorch"]["cuda_available"]:
        print(f"  CUDA Version: {info['pytorch']['cuda_version']}")
        print(f"  GPU Count: {info['pytorch']['device_count']}")
        for device in info["pytorch"]["devices"]:
            print(
                f"    GPU {device['id']}: {device['name']} ({device['memory_gb']:.1f} GB)"
            )
    print()

    # TensorFlow info
    if TF_AVAILABLE and "error" not in info["tensorflow"]:
        print("TensorFlow:")
        print(f"  Version: {info['tensorflow']['version']}")
        print(f"  GPU Count: {info['tensorflow']['gpu_count']}")

        if info["tensorflow"]["gpu_count"] > 0:
            for device in info["tensorflow"]["gpu_devices"]:
                print(f"    GPU {device['id']}: {device['name']}")
        else:
            print("    Using CPU only")
    else:
        print("TensorFlow: Not available or error occurred")

    print("=" * 60)


def create_torch_device(device_id: Union[int, str, None] = None) -> torch.device:
    """
    Create a PyTorch device object with automatic fallback.

    Args:
        device_id: Specific device ID or device string. If None, uses optimal device.

    Returns:
        PyTorch device object
    """
    if device_id is None:
        device_str = get_optimal_device()
    elif isinstance(device_id, int):
        optimal = get_optimal_device()
        if optimal == "cpu":
            device_str = "cpu"
        else:
            device_str = f"{optimal}:{device_id}"
    else:
        device_str = str(device_id)

    try:
        device = torch.device(device_str)
        # Test if device is actually available
        test_tensor = torch.randn(1).to(device)
        return device
    except Exception as e:
        print(f"Warning: Could not use device '{device_str}': {e}")
        print("Falling back to CPU")
        return torch.device("cpu")


if __name__ == "__main__":
    print_device_summary()
