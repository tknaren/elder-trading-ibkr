#!/bin/bash
echo "================================================"
echo "  Elder Trading System - Starting..."
echo "================================================"
echo

cd "$(dirname "$0")/backend"

# Check if venv exists
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo
echo "Starting server..."
echo "Open browser at: http://localhost:5000"
echo
python app.py
