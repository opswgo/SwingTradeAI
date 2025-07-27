#!/bin/bash

# Activate virtual environment
source build-activate_venv.sh

# Install test dependencies
pip install pytest pytest-cov pytest-mock

# Run tests with coverage
python -m pytest test/ -v --cov=src --cov-report=term-missing --cov-report=html:coverage_report

# Deactivate virtual environment
deactivate
