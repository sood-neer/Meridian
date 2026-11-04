#!/bin/bash

# Python Dependencies Installation Script
# This script installs all required Python packages for Meridian

echo "🐍 Installing Python Dependencies for AI Academic Advisor System..."

# Check if Python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python3 is not installed"
    echo "   Please install Python3 first:"
    echo "   - macOS: brew install python3"
    echo "   - Ubuntu: sudo apt install python3 python3-pip"
    echo "   - Windows: Download from https://python.org"
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ Error: pip3 is not installed"
    echo "   Please install pip3 first"
    exit 1
fi

echo "✅ pip3 found: $(pip3 --version)"

# Create virtual environment (optional but recommended)
read -p "🤔 Do you want to create a virtual environment? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment created and activated"
    echo "   To activate in the future: source venv/bin/activate"
fi

# Install requirements
echo "📥 Installing Python packages from requirements.txt..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ All Python dependencies installed successfully!"
    echo ""
    echo "🚀 You can now run the application:"
    echo "   cd IAAS/client"
    echo "   npm start"
    echo ""
    echo "📖 For more information, see:"
    echo "   - README.md"
    echo "   - IAAS/client/STARTUP.md"
    echo "   - IAAS/client/TROUBLESHOOTING.md"
else
    echo "❌ Error: Failed to install some dependencies"
    echo "   Please check the error messages above"
    echo "   You may need to install system dependencies first:"
    echo "   - macOS: brew install graphviz"
    echo "   - Ubuntu: sudo apt install graphviz graphviz-dev"
    exit 1
fi
