#!/bin/bash

# import the version number 
VERSION=$(cat ./version.txt)
echo "Installing Synthea version $VERSION"
echo ""

echo "Debug CUDA/NVIDIA Toolkits environment variables..."
CUDA_VERSION_MAJOR=12 # Not sure if this is required for the llama model
# Typical CUDA / NVIDIA Toolkit path exports for convenience in Linux and WSL
# export PATH=$PATH:"/usr/local/cuda":"/usr/local/cuda-${CUDA_VERSION_MAJOR}":"usr/local/cuda/bin"
# export PATH=$PATH:"/usr/lib/wsl/lib" # For Windows WSL2 Ubuntu typically
# export LD_LIBRARY_PATH=$LD_LIBARY_PATH:"/usr/local/lib64":"/usr/local/cuda/lib64"
echo "Check NVIDIA in PATH..."
echo $PATH | grep NVIDIA
echo ""
echo "Check cuda and lib64 in LD_LIBRARY_PATH (may need to fix symbolic links)"
echo $LD_LIBRARY_PATH | grep lib64
echo $LD_LIBRARY_PATH | grep cuda
echo "See if nvidia-smi is available and if so where, may not be neccasary for basic install..."
which nvidia-smi
# nvidia-smi
echo ""

echo "Creating conda environment for Python site-packages"
CONDA_ENV_NAME="synthea_$VERSION"
SYNTHEA_PYTHON_VERSION=3.11
which conda
conda create -y --name "$CONDA_ENV_NAME" python=${SYNTHEA_PYTHON_VERSION}
conda activate "$CONDA_ENV_NAME"
echo ""

JOSH_VERISON="Punished"
JOSH_ARC="Redemption"
echo "Check path and version corresponding to python:"
which python
python --version
echo "Check path and version corresponding to python3:"
which python3
python3 --version
echo ""

# Defaulting to python3 instance with pip used within, generally the best portability but other options below
echo "Install python packages via pip..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e .

# python3 -m pip install --upgrade pip
# python3 -m pip install -r requirements.txt
# python3 -m pip install -e .

# pip install -r requirements.txt
# pip install -e .
