#!/bin/zsh

# Set TA-Lib paths for Homebrew on Apple Silicon
if [ -d "/opt/homebrew/opt/ta-lib" ]; then
    export LDFLAGS="-L/opt/homebrew/opt/ta-lib/lib"
    export CPPFLAGS="-I/opt/homebrew/opt/ta-lib/include"
    export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ta-lib/lib:$DYLD_LIBRARY_PATH"
else
    echo "Warning: TA-Lib not found in /opt/homebrew/opt/ta-lib"
    echo "Install it with: brew install ta-lib"
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Run build-create_venv.sh first."
    exit 1
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source ./.venv/bin/activate

# Remove any aliases that might interfere
unalias python 2>/dev/null || true
unalias python3 2>/dev/null || true

# Verify Python version
echo -n "Python version: "
python3 --version

# Verify TA-Lib installation
python3 -c "import talib; print(f'TA-Lib {talib.__version__} is installed')" 2>/dev/null || \
    echo "Warning: TA-Lib Python package not found. Run: pip install TA-Lib"
