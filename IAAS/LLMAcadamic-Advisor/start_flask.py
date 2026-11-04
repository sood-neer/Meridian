#!/usr/bin/env python3
"""
Flask Server Startup Script
This script properly sets up the Python path and starts the Flask server.
"""

import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Auto-detect and switch to repository virtual environment if not already in one
if sys.prefix == sys.base_prefix:
    repo_root = os.path.abspath(os.path.join(current_dir, "../.."))
    for candidate in [".venv", "venv"]:
        venv_python = os.path.join(repo_root, candidate, "bin", "python3")
        if os.path.exists(venv_python):
            os.execv(venv_python, [venv_python] + sys.argv)

# Now we can import and run the Flask app
if __name__ == '__main__':
    try:
        # Import the Flask app from api module
        from api.api import app
        
        print("🚀 Starting Flask LLM Server...")
        print(f"📁 Working directory: {current_dir}")
        print(f"🐍 Python path: {sys.path[:3]}...")  # Show first 3 paths
        
        # Run the Flask app
        app.run(debug=True, port=5002, host='127.0.0.1')
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("🔍 Make sure you're running this from the LLMAcadamic-Advisor directory")
        print("📋 Required files:")
        print("   - api/api.py")
        print("   - LLM/main_copy.py")
        print("   - LLM/majorsGraph.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting Flask server: {e}")
        sys.exit(1)
