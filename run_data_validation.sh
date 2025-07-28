#!/bin/bash

# SwingTradeAI - Data Validation Runner
# User Acceptance: User sees clean, validated data with clear error reporting

set -e

echo "🔍 Starting SwingTradeAI Data Validation Dashboard..."

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

# Start data validation dashboard
echo "🚀 Launching data validation dashboard at http://localhost:8502"
echo "🔍 Features available:"
echo "  - Comprehensive data quality checks"
echo "  - Real-time validation results"
echo "  - Error reporting and diagnostics"
echo "  - Feature store integration testing"
echo ""
echo "Press Ctrl+C to stop"

streamlit run data_validation_dashboard.py --server.port 8502 --server.address localhost
