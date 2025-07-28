#!/bin/bash

# SwingTradeAI - ML Prediction Interface Runner
# User Acceptance: User sees clear confidence scores and regime explanations

set -e

echo "🤖 Starting SwingTradeAI ML Prediction Interface..."

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

# Start ML prediction interface
echo "🚀 Launching ML prediction interface at http://localhost:8503"
echo "🤖 Features available:"
echo "  - AI-powered market regime detection"
echo "  - Confidence scores and explanations"
echo "  - Trading recommendations"
echo "  - Feature importance analysis"
echo ""
echo "Press Ctrl+C to stop"

streamlit run ml_prediction_interface.py --server.port 8503 --server.address localhost
