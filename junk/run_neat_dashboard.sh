#!/bin/bash

# NEAT Options Premium Harvesting Dashboard Runner
# Production-ready launcher for NEAT-based options strategy interface

echo "🎯 Starting NEAT Options Premium Harvesting Dashboard..."
echo "=================================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing Streamlit..."
    pip install streamlit plotly
fi

# Set environment variables for better performance
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

echo "🚀 Launching NEAT Options Dashboard..."
echo "📊 Dashboard will open in your browser at: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the dashboard"
echo ""

# Run the Streamlit dashboard
streamlit run neat_options_dashboard.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false \
    --theme.base light \
    --theme.primaryColor "#1f77b4" \
    --theme.backgroundColor "#ffffff" \
    --theme.secondaryBackgroundColor "#f0f2f6"
