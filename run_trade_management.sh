#!/bin/bash

# SwingTradeAI - Trade Management Dashboard Runner
# User Acceptance: User can quickly assess current risk levels

set -e

echo "📊 Starting SwingTradeAI Trade Management Dashboard..."

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run ./setup.sh first"
    exit 1
fi

# Set environment variables
export SSL_CERT_FILE=$(python -m certifi)

# Create necessary directories
mkdir -p data/trades
mkdir -p config
mkdir -p logs

# Start trade management dashboard
echo "🚀 Launching Trade Management Dashboard at http://localhost:8504"
echo "📊 Features available:"
echo "  - Real-time portfolio overview"
echo "  - Open positions tracking"
echo "  - Comprehensive risk dashboard"
echo "  - Performance analytics with grades"
echo "  - Trade entry simulator"
echo ""
echo "Press Ctrl+C to stop"

streamlit run trade_management_dashboard.py --server.port 8504 --server.address localhost
