#!/bin/bash

# import the version number 
VERSION=$(cat ./version.txt)
CONDA_ENV_NAME="synthea_$VERSION"

echo "Installing Synthea version $VERSION"
conda create -y --name "$CONDA_ENV_NAME" python=3.11

conda activate "$CONDA_ENV_NAME"
pip install -r requirements.txt

# TODO: Add logic for modifying CMAKE args based on GPU backend. CLI input?
CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
pip install -e .