#!/bin/bash

# Run the pytest and redirect both stdout and stderr to a file
pytest tests/test_analyze_cache_consistency.py -q > ./test_analyze_cache_consistency.txt 2>&1

# Check if the test passed
if grep -q "1 passed" ./test_analyze_cache_consistency.txt; then
    echo "Test passed"
else
    echo "Test failed"
fi
