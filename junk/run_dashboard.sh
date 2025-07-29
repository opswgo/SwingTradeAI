#!/bin/bash

# SwingTradeAI - Dashboard Runner
# User Acceptance: User can see and understand all technical indicators in dashboard

set -e

echo "📊 Starting SwingTradeAI Dashboard..."

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run ./setup.sh first"
    exit 1
fi

# Check if Streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing Streamlit..."
    pip install streamlit
fi

# Set environment variables
export SSL_CERT_FILE=$(python -m certifi)

# Start dashboard
echo "🚀 Launching dashboard at http://localhost:8501"
echo "📈 Features available:"
echo "  - Real-time market data"
echo "  - Technical indicators (23+)"
echo "  - Market regime detection"
echo "  - Options chain analysis"
echo "  - System health monitoring"
echo ""
echo "Press Ctrl+C to stop"

streamlit run src/dashboard/app.py --server.port 8501 --server.address localhost
