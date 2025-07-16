#!/bin/bash
# CUDA Environment Setup Script for Linux

set -e  # Exit on any error

echo "Setting up CUDA environment for recommendation system research..."

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

echo "NVIDIA GPU Information:"
nvidia-smi

# Check CUDA version
if ! command -v nvcc &> /dev/null; then
    echo "WARNING: nvcc not found. Please ensure CUDA toolkit is installed."
else
    echo "CUDA Version:"
    nvcc --version
fi

# Create conda environment from CUDA environment file
echo "Creating conda environment from environment_cuda.yml..."
conda env create -f environment_cuda.yml

echo "Activating environment..."
conda activate recosys_research_cuda

# Verify PyTorch CUDA installation
echo "Verifying PyTorch CUDA installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
else:
    print('CUDA not available - please check installation')
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
    
except ImportError as e:
    print(f'PyTorch Geometric import failed: {e}')
except Exception as e:
    print(f'Error testing PyTorch Geometric: {e}')
"

echo "CUDA environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "conda activate recosys_research_cuda"
echo ""
echo "To test your models with CUDA, ensure you pass device='cuda' to your training functions."
