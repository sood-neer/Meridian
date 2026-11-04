#!/bin/bash
# EC2 Update Script - Run this on your EC2 instance to deploy latest code
# Usage: ./update_ec2.sh

set -e

APP_DIR="$HOME/flaskApp/AI_ACADEMIC_ADVISOR"
echo "=== Updating AI Academic Advisor on EC2 ==="

# 1. Pull latest code
cd "$APP_DIR"
echo ">>> Pulling latest code from GitHub..."
git pull origin main

# 2. Activate venv and update Python dependencies
echo ">>> Updating Python dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# LLM-specific deps (web search, orchestration)
pip install -r IAAS/LLMAcadamic-Advisor/requirements.txt 2>/dev/null || true

# 3. Verify Flask import
echo ">>> Verifying Flask import..."
cd IAAS/LLMAcadamic-Advisor
python3 -c "from api.api import app; print('✅ Flask import successful')" || {
    echo "❌ Flask import failed. Check logs."
    exit 1
}

# 4. Update Node.js dependencies and rebuild frontend
echo ">>> Updating Node.js dependencies..."
cd "$APP_DIR/IAAS/client"
npm install

echo ">>> Building React frontend..."
npm run build

# 5. Restart services
echo ">>> Restarting services..."
sudo systemctl restart flask-llm
sudo systemctl restart nodejs-backend

echo ""
echo "=== Update complete! ==="
echo "Check status: sudo systemctl status flask-llm nodejs-backend"
echo "View logs: sudo journalctl -u flask-llm -f"
