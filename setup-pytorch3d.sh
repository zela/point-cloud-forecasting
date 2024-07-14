#!/bin/bash

# Check if PyTorch3D is installed
python -c "import pytorch3d" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "PyTorch3D not found. Attempting to install..."

    # Get Python version
    PYTHON_VERSION=$(python -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))")

    # Get PyTorch version
    PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0].replace('.', ''))")

    # Get CUDA version
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.replace('.', ''))")

# "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt222/download.html"
    # Check if we can install from a wheel
    if [[ $PYTORCH_VERSION == 22* && $(uname) == "Linux" ]]; then
        VERSION_STR="py3${PYTHON_VERSION//./}_cu${CUDA_VERSION}_pyt${PYTORCH_VERSION}"
        pip install fvcore iopath
        pip install --no-index --no-cache-dir pytorch3d -f "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/${VERSION_STR}/download.html"
    else
        # Install from source
        pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
    fi
else
    echo "PyTorch3D is already installed."
fi
