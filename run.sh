#!/bin/bash

# Exit immediately if a command fails
set -e

echo "##### Setting up Python environment for NAS pipeline #####"

# Change to the script directory to ensure consistent working directory
cd "$(dirname "$0")"
echo "Changed working directory to: $(pwd)"

# Install all required packages from the requirements file
pip install -r requirements.txt --user

echo "##### Environment setup complete. Starting NAS pipeline. #####"

python -m pipeline.full_pipeline "$@"

echo "##### NAS pipeline finished. #####"