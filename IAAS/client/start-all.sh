#!/bin/bash
# Startup script to kill existing servers and start all services

echo "🛑 Stopping any existing servers..."
pkill -f 'python.*start_flask' 2>/dev/null || true
pkill -f 'node.*server.js' 2>/dev/null || true
pkill -f 'react-scripts' 2>/dev/null || true
sleep 2

echo "🚀 Starting all services..."
cd "$(dirname "$0")"
npm start
