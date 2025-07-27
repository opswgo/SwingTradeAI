#!/bin/bash

# Set minimum coverage requirements
MIN_UNIT_LINE_COVERAGE=92    # 92% minimum line coverage for unit tests
MIN_UNIT_BRANCH_COVERAGE=90  # 90% minimum branch coverage for unit tests
MIN_INTEGRATION_LINE_COVERAGE=85    # 85% minimum line coverage for integration tests
MIN_INTEGRATION_BRANCH_COVERAGE=80  # 80% minimum branch coverage for integration tests

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print section header
print_section() {
    echo -e "\n${YELLOW}=== $1 ===${NC}"
}

# Function to print result
print_result() {
    local type=$1
    local current=$2
    local min=$3
    local scale=2
    
    # Remove percentage sign if present
    current=${current%\%}
    
    if (( $(echo "$current >= $min" | bc -l) )); then
        echo -e "✅ ${GREEN}$type: ${current}% (min ${min}%)${NC}"
    else
        echo -e "❌ ${RED}$type: ${current}% (min ${min}%)${NC}"
    fi
}

# Function to get coverage percentage from the XML report
# Arguments: $1 - coverage type (line-rate or branch-rate)
get_coverage() {
    local coverage_file=$1
    local coverage_type=$2
    
    python3 -c "
import xml.etree.ElementTree as ET
import sys
try:
    root = ET.parse('$coverage_file').getroot()
    rate = float(root.attrib['$coverage_type'])
    print(f'{rate * 100:.2f}')
except Exception as e:
    print(f'Error parsing coverage report: {e}', file=sys.stderr)
    print('0.00')
    sys.exit(1)
"
}

# Function to run tests and collect coverage
run_test_suite() {
    local test_type=$1
    local cov_output=$2
    local html_output=$3
    local xml_output=$4
    
    echo -e "\nRunning ${YELLOW}${test_type}${NC} tests..."
    
    if [ "$test_type" == "unit" ]; then
        python -m pytest test/unit/ \
            --cov=src \
            --cov-report=term-missing \
            --cov-report=html:$html_output \
            --cov-report=xml:$xml_output \
            -m "not integration"
    else
        python -m pytest test/integration/ \
            --cov=src \
            --cov-append \
            --cov-report=term-missing \
            --cov-report=html:$html_output \
            --cov-report=xml:$xml_output \
            -m "integration"
    fi
}

# Clean up previous coverage data
rm -rf .coverage* coverage/ htmlcov_unit/ htmlcov_integration/ coverage.xml

# Run unit tests with coverage
print_section "Running Unit Tests"
run_test_suite "unit" "coverage_unit.xml" "htmlcov_unit" "coverage_unit.xml"

# Run integration tests with coverage
print_section "Running Integration Tests"
run_test_suite "integration" "coverage_integration.xml" "htmlcov_integration" "coverage_integration.xml"

# Generate combined coverage report
print_section "Generating Combined Coverage Report"
python -m coverage combine
python -m coverage html -d htmlcov
python -m coverage xml -o coverage.xml

# Get coverage percentages for unit tests
if [ -f "coverage_unit.xml" ]; then
    UNIT_LINE_COV=$(get_coverage "coverage_unit.xml" "line-rate")
    UNIT_BRANCH_COV=$(get_coverage "coverage_unit.xml" "branch-rate")
else
    UNIT_LINE_COV=0
    UNIT_BRANCH_COV=0
fi

# Get coverage percentages for integration tests
if [ -f "coverage_integration.xml" ]; then
    INT_LINE_COV=$(get_coverage "coverage_integration.xml" "line-rate")
    INT_BRANCH_COV=$(get_coverage "coverage_integration.xml" "branch-rate")
else
    INT_LINE_COV=0
    INT_BRANCH_COV=0
fi

# Get combined coverage percentages
COMBINED_LINE_COV=$(get_coverage "coverage.xml" "line-rate")
COMBINED_BRANCH_COV=$(get_coverage "coverage.xml" "branch-rate")

# Print the coverage reports
print_section "Coverage Report"

echo -e "${YELLOW}Unit Tests:${NC}"
print_result "Line Coverage" "$UNIT_LINE_COV" "$MIN_UNIT_LINE_COVERAGE"
print_result "Branch Coverage" "$UNIT_BRANCH_COV" "$MIN_UNIT_BRANCH_COVERAGE"

echo -e "\n${YELLOW}Integration Tests:${NC}"
print_result "Line Coverage" "$INT_LINE_COV" "$MIN_INTEGRATION_LINE_COVERAGE"
print_result "Branch Coverage" "$INT_BRANCH_COV" "$MIN_INTEGRATION_BRANCH_COVERAGE"

echo -e "\n${YELLOW}Combined Coverage:${NC}"
print_result "Line Coverage" "$COMBINED_LINE_COV" "$MIN_UNIT_LINE_COVERAGE"
print_result "Branch Coverage" "$COMBINED_BRANCH_COV" "$MIN_UNIT_BRANCH_COVERAGE"

# Check if coverage meets minimum requirements
FAILED=0
if (( $(echo "$UNIT_LINE_COV < $MIN_UNIT_LINE_COVERAGE" | bc -l) )); then
    echo -e "\n${RED}❌ Unit test line coverage (${UNIT_LINE_COV}%) is below the required minimum of ${MIN_UNIT_LINE_COVERAGE}%${NC}"
    FAILED=1
fi

if (( $(echo "$UNIT_BRANCH_COV < $MIN_UNIT_BRANCH_COVERAGE" | bc -l) )); then
    echo -e "${RED}❌ Unit test branch coverage (${UNIT_BRANCH_COV}%) is below the required minimum of ${MIN_UNIT_BRANCH_COVERAGE}%${NC}"
    FAILED=1
fi

if (( $(echo "$INT_LINE_COV < $MIN_INTEGRATION_LINE_COVERAGE" | bc -l) )); then
    echo -e "${YELLOW}⚠️  Integration test line coverage (${INT_LINE_COV}%) is below the recommended minimum of ${MIN_INTEGRATION_LINE_COVERAGE}%${NC}"
fi

if (( $(echo "$INT_BRANCH_COV < $MIN_INTEGRATION_BRANCH_COVERAGE" | bc -l) )); then
    echo -e "${YELLOW}⚠️  Integration test branch coverage (${INT_BRANCH_COV}%) is below the recommended minimum of ${MIN_INTEGRATION_BRANCH_COVERAGE}%${NC}"
fi

# Open the combined HTML report in the default browser
if [ $FAILED -eq 0 ]; then
    if command -v xdg-open > /dev/null; then
        xdg-open htmlcov/index.html 2>/dev/null
    elif command -v open > /dev/null; then
        open htmlcov/index.html 2>/dev/null
    fi
else
    echo -e "\n${RED}❌ Some test coverage requirements were not met. Please improve test coverage before proceeding.${NC}"
fi

exit $FAILED
