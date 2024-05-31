#!/bin/bash

# import the version number 
VERSION=$(cat ./version.txt)
CONDA_ENV_NAME="synthea_$VERSION"
echo "Installing Synthea version $VERSION"
conda create -y --name "$CONDA_ENV_NAME" python=3.11
conda activate "$CONDA_ENV_NAME"
pip install -r requirements.txt
pip install -e .