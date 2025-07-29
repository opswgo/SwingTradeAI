#!/bin/bash

# NEAT Options System - Real-time Monitoring Launcher
# Launches the real-time performance monitoring dashboard

echo "🔍 NEAT Options System - Real-time Monitoring"
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Install additional monitoring dependencies
echo "📦 Installing monitoring dependencies..."
pip install streamlit plotly >/dev/null 2>&1

# Create data directories
mkdir -p data/alerts
mkdir -p data/monitoring_reports

# Launch monitoring dashboard
echo "🚀 Starting real-time monitoring dashboard..."
echo "📊 Dashboard will be available at: http://localhost:8501"
echo "🔍 Monitor agent performance, alerts, and system health"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""

streamlit run monitoring_dashboard.py --server.port 8501 --server.headless true
