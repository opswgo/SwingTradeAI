#!/bin/bash

# Exit on error
set -e

# Configuration
COVERAGE_DIR="coverage_integration"
TEST_DIR="test/integration"
SRC_DIR="src"

# Clean up previous coverage data
echo "Cleaning up previous coverage data..."
rm -rf ".coverage" ".coverage.*" "$COVERAGE_DIR"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Warning: Virtual environment not found. Running without activation."
fi

# Set Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create test data directory if it doesn't exist
mkdir -p "$TEST_DIR/test_data"

# Run the integration tests with coverage
echo "Running integration tests with coverage..."
python -m pytest "$TEST_DIR" -v \
    --cov="$SRC_DIR" \
    --cov-config=.coveragerc_integration \
    --cov-report=html:"$COVERAGE_DIR" \
    --cov-report=term-missing \
    --cov-append \
    --cov-fail-under=0

# Generate coverage report
echo -e "\nGenerating coverage report..."
python -m coverage report --show-missing

# Check if tests passed
if [ $? -eq 0 ]; then
    echo -e "\n✅ All integration tests passed!"
    echo "📊 Coverage report: file://$(pwd)/$COVERAGE_DIR/index.html"
else
    echo "Some integration tests failed. Check the output above for details."
    exit 1
fi
