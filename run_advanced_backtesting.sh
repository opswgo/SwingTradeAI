#!/bin/bash

# SwingTradeAI - Advanced Backtesting & Model Competition Interface
# User Acceptance: User can run advanced backtesting with one command

echo "🚀 Starting SwingTradeAI Advanced Backtesting Interface..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Set SSL certificate path
export SSL_CERT_FILE=$(python -m certifi)
echo "🔒 SSL certificates configured"

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import streamlit, plotly, pandas, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing required packages..."
    pip install streamlit plotly pandas numpy
fi

# Create data directories if they don't exist
mkdir -p data/reports
mkdir -p data/backtests
mkdir -p logs

echo "📊 Launching Advanced Backtesting Interface..."
echo ""
echo "🌐 Interface will open in your browser at: http://localhost:8501"
echo "🎯 Features available:"
echo "   • Single Strategy Backtesting with Regime Detection"
echo "   • Multi-Strategy Competition Analysis"
echo "   • Market Scenario Stress Testing"
echo "   • Portfolio Optimization"
echo "   • Professional Strategy Reports"
echo ""
echo "⚡ To stop the interface, press Ctrl+C"
echo ""

# Launch Streamlit interface
streamlit run advanced_backtesting_interface.py \
    --server.port 8501 \
    --server.address localhost \
    --server.headless false \
    --browser.gatherUsageStats false \
    --theme.base "light" \
    --theme.primaryColor "#1f77b4" \
    --theme.backgroundColor "#ffffff" \
    --theme.secondaryBackgroundColor "#f0f2f6"
