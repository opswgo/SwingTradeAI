#!/bin/zsh

# Script to set up and activate a Python 3.11.9 virtual environment with TA-Lib support
set -e  # Exit on error

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "Error: Homebrew is required. Install from https://brew.sh"
    exit 1
fi

# Set TA-Lib paths for Homebrew on Apple Silicon
export LDFLAGS="-L/opt/homebrew/opt/ta-lib/lib"
export CPPFLAGS="-I/opt/homebrew/opt/ta-lib/include"
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ta-lib/lib:$DYLD_LIBRARY_PATH"

# Ensure pyenv is initialized
if command -v pyenv >/dev/null 2>&1; then
    eval "$(pyenv init -)"
else
    echo "pyenv not found. Installing via Homebrew..."
    brew install pyenv
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
    echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
    echo 'eval "$(pyenv init -)"' >> ~/.zshrc
    eval "$(pyenv init -)"
fi

# Remove any python aliases
unalias python 2>/dev/null || true
unalias python3 2>/dev/null || true

# Set Python version
PYTHON_VERSION="3.11.9"
pyenv install --skip-existing $PYTHON_VERSION
pyenv shell $PYTHON_VERSION

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with Python $PYTHON_VERSION..."
    python3 -m venv .venv
    
    # Activate the virtual environment
    source .venv/bin/activate
    
    # Upgrade pip and setuptools
    pip install --upgrade pip setuptools wheel
    
    # Install TA-Lib system dependency if not present
    if ! brew list ta-lib &>/dev/null; then
        echo "Installing TA-Lib via Homebrew..."
        brew install ta-lib
    fi
    
    # Install Python dependencies
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Verify TA-Lib installation
    if ! python -c "import talib" &>/dev/null; then
        echo "Installing TA-Lib Python bindings..."
        pip install --no-cache-dir TA-Lib
    fi
    
    echo "\n✅ Virtual environment setup complete!"
    echo "To activate the environment, run: source build-activate_venv.sh"
else
    # Activate existing environment
    source .venv/bin/activate
    echo "Using existing virtual environment with Python $(python --version)"
fi

# Verify Python version
PYTHON_VER=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")
if [[ ! "$PYTHON_VER" == "$PYTHON_VERSION"* ]]; then
    echo "Warning: Python version mismatch. Expected $PYTHON_VERSION, got $PYTHON_VER"
    echo "Please delete the .venv directory and run this script again."
    exit 1
fi