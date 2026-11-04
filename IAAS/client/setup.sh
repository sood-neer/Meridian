#!/bin/bash

# Meridian Setup Script
# This script ensures all dependencies and configurations are properly set up

echo "🚀 Setting up Meridian..."

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: Please run this script from the /IAAS/client directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected: .../AI_ACADEMIC_ADVISOR/IAAS/client"
    exit 1
fi

echo "✅ Current directory: $(pwd)"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js is not installed"
    echo "   Please install Node.js from https://nodejs.org/"
    exit 1
fi

echo "✅ Node.js version: $(node --version)"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm is not installed"
    exit 1
fi

echo "✅ npm version: $(npm --version)"

# Check if Python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python3 is not installed"
    echo "   Please install Python3 or set up pyenv"
    exit 1
fi

echo "✅ Python3 version: $(python3 --version)"

# Check if MySQL is running
if ! pgrep -f mysql > /dev/null; then
    echo "⚠️  Warning: MySQL doesn't appear to be running"
    echo "   Please start MySQL/XAMPP before running the application"
    echo "   You can start XAMPP from Applications/XAMPP/"
fi

# Install npm dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing npm dependencies..."
    npm install
    if [ $? -eq 0 ]; then
        echo "✅ npm dependencies installed successfully"
    else
        echo "❌ Error: Failed to install npm dependencies"
        exit 1
    fi
else
    echo "✅ npm dependencies already installed"
fi

# Check if the LLM directory exists
if [ ! -d "../LLMAcadamic-Advisor" ]; then
    echo "❌ Error: LLMAcadamic-Advisor directory not found"
    echo "   Expected: $(pwd)/../LLMAcadamic-Advisor"
    exit 1
fi

echo "✅ LLM directory found"

# Check if the CSV file exists
if [ ! -f "../LLMAcadamic-Advisor/LLM/all_abington_majors_combined(5).csv" ]; then
    echo "❌ Error: Required CSV file not found"
    echo "   Expected: ../LLMAcadamic-Advisor/LLM/all_abington_majors_combined(5).csv"
    exit 1
fi

echo "✅ Required CSV file found"

# Check if ports are available
check_port() {
    local port=$1
    local service=$2
    if lsof -ti:$port > /dev/null 2>&1; then
        echo "⚠️  Warning: Port $port is already in use (required for $service)"
        echo "   You may need to stop other applications using this port"
    else
        echo "✅ Port $port is available for $service"
    fi
}

echo "🔍 Checking port availability..."
check_port 3000 "Frontend (React)"
check_port 5001 "Backend (Node.js)"
check_port 5002 "LLM Server (Flask)"

echo ""
echo "🎉 Setup complete! You can now run:"
echo "   npm start"
echo ""
echo "This will start all three servers:"
echo "   🔵 Frontend:    http://localhost:3000"
echo "   🟢 Backend:     http://localhost:5001"
echo "   🟣 LLM Server:  http://localhost:5002"
echo ""
echo "📖 For more information, see STARTUP.md"
