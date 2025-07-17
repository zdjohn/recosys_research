#!/bin/bash
# CUDA Environment Setup Script for Linux/macOS

set -e  # Exit on any error

echo "Setting up CUDA environment for recommendation system research..."

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="mac"
    echo "Detected macOS - will setup CPU-only environment compatible with M3 chip"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
    echo "Detected Linux - checking for CUDA support"
    
    # Check if CUDA is available on Linux
    if ! command -v nvidia-smi &> /dev/null; then
        echo "WARNING: nvidia-smi not found. Please ensure NVIDIA drivers are installed."
        exit 1
    fi
else
    echo "Unsupported platform: $OSTYPE"
    exit 1
fi

# Platform-specific GPU information
if [[ "$PLATFORM" == "linux" ]]; then
    echo "NVIDIA GPU Information:"
    nvidia-smi
    
    # Check CUDA version
    if ! command -v nvcc &> /dev/null; then
        echo "WARNING: nvcc not found. Please ensure CUDA toolkit is installed."
    else
        echo "CUDA Version:"
        nvcc --version
    fi
elif [[ "$PLATFORM" == "mac" ]]; then
    echo "Mac detected - will use CPU and MPS (Metal Performance Shaders) when available"
    system_profiler SPDisplaysDataType | grep -E "(Chipset Model|Metal)" || echo "Metal support info not available"
fi

# Create conda environment from appropriate environment file
if [[ "$PLATFORM" == "linux" ]]; then
    ENV_FILE="environment_cuda_linux.yml"
    echo "Creating conda environment from $ENV_FILE for Linux with CUDA support..."
else
    ENV_FILE="environment_cuda.yml"
    echo "Creating conda environment from $ENV_FILE for macOS with MPS support..."
fi

if [[ -d "./.venv" ]]; then
    echo "Removing existing .venv directory..."
    rm -rf ./.venv
fi
conda env create -f $ENV_FILE -p ./.venv

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./.venv

# Verify PyTorch installation (platform-aware)
echo "Verifying PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f'CUDA available: {cuda_available}')

if cuda_available:
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
else:
    print('CUDA not available - using CPU or MPS backend')
    
# Check MPS availability on Mac
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('Metal Performance Shaders (MPS) available for Mac GPU acceleration')
else:
    print('MPS not available - using CPU only')
"

# Verify PyTorch Geometric installation
echo "Verifying PyTorch Geometric installation..."
python -c "
try:
    import torch_geometric
    print(f'PyTorch Geometric version: {torch_geometric.__version__}')
    
    # Test basic functionality
    import torch
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import degree
    print('PyTorch Geometric basic imports successful')
    
    # Test CUDA compatibility
    if torch.cuda.is_available():
        device = torch.device('cuda')
        x = torch.randn(100, 16).to(device)
        print('CUDA tensor operations working')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        x = torch.randn(100, 16).to(device)
        print('MPS tensor operations working')
    
except ImportError as e:
    print(f'PyTorch Geometric import failed: {e}')
except Exception as e:
    print(f'Error testing PyTorch Geometric: {e}')
"

# Verify TensorFlow installation (platform-aware)
echo "Verifying TensorFlow installation..."
python -c "
try:
    import tensorflow as tf
    print(f'TensorFlow version: {tf.__version__}')
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f'GPU devices available: {len(gpus)}')
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f'GPU {i}: {gpu.name}')
            try:
                # Get GPU memory info
                gpu_details = tf.config.experimental.get_device_details(gpu)
                if 'device_name' in gpu_details:
                    print(f'GPU {i} name: {gpu_details[\"device_name\"]}')
                if 'compute_capability' in gpu_details:
                    print(f'GPU {i} compute capability: {gpu_details[\"compute_capability\"]}')
            except:
                pass
        
        # Test basic GPU operation
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                c = tf.matmul(a, b)
                print('TensorFlow GPU operations working')
                print(f'Test computation result shape: {c.shape}')
        except Exception as e:
            print(f'Error testing TensorFlow GPU operations: {e}')
    else:
        print('No GPU devices found by TensorFlow')
        print('TensorFlow will use CPU only')
    
    # Check if mixed precision is available
    try:
        if len(gpus) > 0:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            print(f'Mixed precision policy available: {policy.name}')
    except Exception as e:
        print(f'Mixed precision not available: {e}')

except ImportError as e:
    print(f'TensorFlow import failed: {e}')
except Exception as e:
    print(f'Error testing TensorFlow: {e}')
"

echo "CUDA environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "conda activate ./.venv"
echo ""
echo "To test your models with CUDA:"
echo "- PyTorch: pass device='cuda' to your training functions"
echo "- TensorFlow: use tf.device('/GPU:0') or let TensorFlow auto-select GPU"
