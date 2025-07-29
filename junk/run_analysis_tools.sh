#!/bin/bash

# Analysis Tools Launcher for NEAT Options System
# Provides easy access to all Sprint 6.2 analysis interfaces

set -e

echo "🔬 NEAT Options System - Analysis Tools Launcher"
echo "================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📦 Installing dependencies..."
pip install -q --upgrade pip
pip install -q streamlit plotly pandas numpy yfinance scikit-learn neat-python psycopg2-binary

# Set environment variables for Streamlit
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_ENABLE_CORS=false

echo ""
echo "🚀 Available Analysis Tools:"
echo "1. Stock Screening Interface"
echo "2. Options Chain Analyzer" 
echo "3. Custom Backtesting Interface"
echo "4. Performance Attribution Analyzer"
echo "5. Research & Analysis Tools"
echo ""

# Function to run specific tool
run_tool() {
    # Clean up any existing processes first
    cleanup_processes
    
    case $1 in
        1)
            echo "🔍 Launching Stock Screening Interface..."
            streamlit run stock_screening_interface.py --server.port 8501
            ;;
        2)
            echo "⚡ Launching Options Chain Analyzer..."
            streamlit run options_chain_analyzer.py --server.port 8502
            ;;
        3)
            echo "🔬 Launching Custom Backtesting Interface..."
            streamlit run custom_backtesting_interface.py --server.port 8503
            ;;
        4)
            echo "📈 Launching Performance Attribution Analyzer..."
            streamlit run performance_attribution_analyzer.py --server.port 8504
            ;;
        5)
            echo "🔬 Launching Research & Analysis Tools..."
            streamlit run research_analysis_tools.py --server.port 8505
            ;;
        *)
            echo "❌ Invalid selection: $1"
            echo "Valid options: 1, 2, 3, 4, 5, or 'all'"
            exit 1
            ;;
    esac
}

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 1  # Port is in use
    else
        return 0  # Port is available
    fi
}

# Function to kill existing streamlit processes
cleanup_processes() {
    echo "🧹 Cleaning up existing Streamlit processes..."
    pkill -f streamlit >/dev/null 2>&1 || true
    sleep 2
}

# Function to run all tools
run_all_tools() {
    echo "🚀 Launching all analysis tools..."
    echo ""
    
    # Clean up any existing processes first
    cleanup_processes
    
    # Launch all tools in background with error handling
    echo "🔍 Starting Stock Screening Interface on port 8501..."
    streamlit run stock_screening_interface.py --server.port 8501 >/dev/null 2>&1 &
    sleep 1
    
    echo "⚡ Starting Options Chain Analyzer on port 8502..."
    streamlit run options_chain_analyzer.py --server.port 8502 >/dev/null 2>&1 &
    sleep 1
    
    echo "🔬 Starting Custom Backtesting Interface on port 8503..."
    streamlit run custom_backtesting_interface.py --server.port 8503 >/dev/null 2>&1 &
    sleep 1
    
    echo "📈 Starting Performance Attribution Analyzer on port 8504..."
    streamlit run performance_attribution_analyzer.py --server.port 8504 >/dev/null 2>&1 &
    sleep 1
    
    echo "🔬 Starting Research & Analysis Tools on port 8505..."
    streamlit run research_analysis_tools.py --server.port 8505 >/dev/null 2>&1 &
    sleep 2
    
    echo ""
    echo "✅ All analysis tools launched successfully!"
    echo ""
    echo "📊 Access URLs:"
    echo "   Stock Screening:           http://localhost:8501"
    echo "   Options Chain Analyzer:   http://localhost:8502"
    echo "   Custom Backtesting:       http://localhost:8503"
    echo "   Performance Attribution:  http://localhost:8504"
    echo "   Research & Analysis:      http://localhost:8505"
    echo ""
    echo "💡 Tips:"
    echo "   - Open URLs in your browser to access the tools"
    echo "   - Each tool runs independently on its own port"
    echo "   - Use Ctrl+C to stop all services"
    echo ""
    echo "🔄 Services are running... Press Ctrl+C to stop"
    
    # Wait for all background processes
    wait
}

# Check if argument provided
if [ $# -eq 1 ]; then
    if [ "$1" = "all" ]; then
        run_all_tools
    else
        run_tool $1
    fi
else
    # Interactive mode
    echo "Enter tool number (1-5) or 'all' to run all tools:"
    read selection
    
    if [ "$selection" = "all" ]; then
        run_all_tools
    else
        run_tool $selection
    fi
fi

echo ""
echo "✅ Analysis tools session completed"
