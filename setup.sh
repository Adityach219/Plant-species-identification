#!/bin/bash

# Plant Species Identification Setup Script
# This script helps set up the development environment

echo "🌱 Plant Species Identification Setup"
echo "======================================"

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1)
echo "Found: $python_version"

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment
echo ""
echo "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || {
    echo "❌ Failed to activate virtual environment"
    exit 1
}

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data/{sample,training,validation}
mkdir -p models
mkdir -p logs
mkdir -p output

echo "✅ Project directories created"

# Generate default configuration
echo ""
echo "Generating default configuration..."
python -c "
import sys
import os
sys.path.append('src')
from utils.helpers import create_model_config
create_model_config('config/model_config.json')
print('✅ Default configuration created')
"

# Run tests
echo ""
echo "Running tests..."
python -m pytest tests/ -v

if [ $? -eq 0 ]; then
    echo "✅ All tests passed"
else
    echo "⚠️  Some tests failed, but setup can continue"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Add training data to data/training/ organized by species folders"
echo "3. Train a model: python src/plant_identifier/train.py --data_dir data/training"
echo "4. Start the web app: streamlit run webapp/app.py"
echo ""
echo "For more information, see the README.md file."