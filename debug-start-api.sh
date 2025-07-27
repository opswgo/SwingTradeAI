#!/bin/bash

# start_api.sh - Script to start the Divinator API server in debug or production mode
# Usage: ./start_api.sh [-debug|-prod]

if [[ -z "$DATABASE_URL" ]]; then
  echo "Error: DATABASE_URL not set. Please run build-setvarsAPI.sh" >&2
  exit 1
fi

# Default to debug mode if no argument is provided
MODE="debug"

# Parse command line arguments
if [[ "$1" == "-debug" ]]; then
  MODE="debug"
elif [[ "$1" == "-prod" ]]; then
  MODE="prod"
fi

# Set environment variables based on mode
if [[ "$MODE" == "prod" ]]; then
  export DIVINATOR_ENV="production"
  echo "Starting API server in PRODUCTION mode..."
else
  export DIVINATOR_ENV="development"
  echo "Starting API server in DEBUG mode..."
fi

# Activate virtual environment if it exists
if [[ -d ".venv" ]]; then
  source .venv/bin/activate
  echo "Virtual environment activated"
else
  echo "Warning: No virtual environment found. Using system Python."
fi

# Start the server based on mode
if [[ "$MODE" == "prod" ]]; then
  # Check if gunicorn is available
  if ! command -v gunicorn &> /dev/null; then
    echo "Warning: gunicorn not found. Falling back to Flask development server."
    MODE="debug"
  else
    # Calculate optimal number of workers (2 × CPU cores + 1)
    if [[ "$(uname)" == "Darwin" ]]; then
      # macOS
      WORKERS=$(($(sysctl -n hw.ncpu) * 2 + 1))
    else
      # Linux and others
      WORKERS=$(($(nproc 2>/dev/null || echo 2) * 2 + 1))
    fi
    
    # Start with Gunicorn for production
    echo "Starting with Gunicorn ($WORKERS workers) on port 4002..."
    gunicorn -w "$WORKERS" -k gevent --worker-connections 1000 \
      --access-logfile - --error-logfile - \
      -b 0.0.0.0:4002 DivinatorAPI:app
  fi
fi

if [[ "$MODE" == "debug" ]]; then
  # Start with Flask's built-in server for development
  echo "Starting with Flask development server on port 5002..."
  python DivinatorAPI.py
fi

# alternative - UNTESTED
# if [[ "$MODE" == "debug" ]]; then
#   # Start Divinator API with Gunicorn (single worker) on port 5002 for terminal logs
#   echo "DEVELOPMENT: Starting Divinator API with Gunicorn (1 worker) on port 5002..."
#   pip install gunicorn &>/dev/null
#   exec gunicorn --workers 1 --access-logfile - --error-logfile - -b 0.0.0.0:5002 DivinatorAPI:app
# fi
