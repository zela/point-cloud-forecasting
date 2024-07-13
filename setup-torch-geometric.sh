#!/bin/bash

# For PyTorch 2.2.3 use TorchVision 0.18.0
VERSION=2.2.2
VISIONVERSION=0.17.2

source venv/bin/activate
pip install -r requirements.txt

# Check the operating system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Check if CUDA is available
    if command -v nvcc &> /dev/null; then
        # Install CUDA version of PyTorch
        pip install torch==${VERSION} torchvision==${VISIONVERSION} torchaudio==${VERSION} --index-url https://download.pytorch.org/whl/cu118
        # Install CUDA version of PyG libraries
        pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${VERSION}+cu118.html
    else
        # Install CPU version of PyTorch
        pip install torch==${VERSION} torchvision==${VISIONVERSION} torchaudio==${VERSION} --index-url https://download.pytorch.org/whl/cpu
        # Install CPU version of PyG libraries
        pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${VERSION}+cpu.html
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Install CPU version of PyTorch on macOS
    pip install torch==${VERSION} torchvision==${VISIONVERSION} torchaudio==${VERSION}
    # Install CPU version of PyG libraries on macOS
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${VERSION}+cpu.html
else
    echo "Unsupported operating system"
    exit 1
fi

pip install torch-geometric

python -c "import torch_geometric; print(torch_geometric.__version__)"
