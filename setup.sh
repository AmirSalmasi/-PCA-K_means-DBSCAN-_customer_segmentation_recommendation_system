#!/bin/bash

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create virtual environment
echo "Creating virtual environment..."
uv venv

# Install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt

echo -e "\nSetup completed successfully!"
echo -e "\nTo activate the virtual environment:"
echo "    source .venv/bin/activate" 