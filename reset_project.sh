#!/bin/bash

# Project Reset and Rebuild Script

# Exit on any error
set -e

# Remove existing virtual environment
if [ -d ".venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf .venv
fi

# Clear pip cache
echo "Clearing pip cache..."
pip cache purge

# Remove build artifacts
echo "Removing build artifacts..."
rm -rf build dist *.egg-info

# Create new virtual environment
echo "Creating new virtual environment..."
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install project and dependencies
echo "Installing project dependencies..."
uv pip install -e .
uv pip install -e .[dev]

# Copy environment example file
if [ ! -f ".env" ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
fi

# Run tests
echo "Running project tests..."
uv run pytest

# Display success message
echo "Project reset and rebuilt successfully!"
echo "Please update .env with your OpenAI API key before running the application."
```
