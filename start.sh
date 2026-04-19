#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Define the exact path to the virtual environment's python
# This ensures it NEVER uses /usr/bin/python
VENV_PATH="$SCRIPT_DIR/venv/bin/python"

echo "Using Python from: $VENV_PATH"

# Run your program
$VENV_PATH "$SCRIPT_DIR/synthea/Synthea.py"