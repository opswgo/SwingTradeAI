#!/bin/bash
# StockAI Training Script
# Usage: ./Example_TrainModel.sh [--resume CHECKPOINT_FILE]

# Exit on error
set -e

# Default configuration
TICKER="AAPL"
START_DATE="2020-01-01"
END_DATE="2023-12-31"
GENERATIONS=50
POPULATION_SIZE=100
OUTPUT_DIR="models"
LOG_DIR="logs"
RESUME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            RESUME="$2"
            shift 2
            ;;
        *)
            log "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Create necessary directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Set up log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TICKER}_${TIMESTAMP}.log"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to find latest checkpoint
find_latest_checkpoint() {
    local ticker=$1
    local checkpoint_dir="${OUTPUT_DIR}/checkpoints"
    
    if [ ! -d "$checkpoint_dir" ]; then
        echo ""
        return
    fi
    
    # Find the most recent checkpoint for this ticker
    local latest=$(find "$checkpoint_dir" -name "${ticker}_gen*_interrupted.pkl" -o -name "${ticker}_gen*.pkl" 2>/dev/null | \
                   sort -r | head -n 1)
    
    if [ -n "$latest" ]; then
        echo "$latest"
    else
        echo ""
    fi
}

# Check if we should resume training
if [ -z "$RESUME" ]; then
    # Try to find the latest checkpoint
    RESUME=$(find_latest_checkpoint "$TICKER")
    
    if [ -n "$RESUME" ]; then
        echo "Found interrupted training session. Latest checkpoint: $RESUME"
        read -p "Do you want to resume from this checkpoint? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            RESUME=""
        fi
    fi
fi

# Start training
log "=== StockAI Training Session ==="
log "Ticker: $TICKER"
log "Date Range: $START_DATE to $END_DATE"
log "Generations: $GENERATIONS"
log "Population Size: $POPULATION_SIZE"

if [ -n "$RESUME" ]; then
    log "Resuming from checkpoint: $RESUME"
    # Extract generation number for logging
    GEN_NUM=$(echo "$RESUME" | grep -oP '(?<=_gen)\d+' || echo "unknown")
    log "Resuming from generation: $GEN_NUM"
else
    log "Starting new training session"
fi

# Activate virtual environment
if [ -f "build-activate_venv.sh" ]; then
    source build-activate_venv.sh
else
    log "Error: build-activate_venv.sh not found. Please run build-create_venv.sh first."
    exit 1
fi

# Build the command
CMD=(
    "python" "main.py"
    "--ticker" "$TICKER"
    "--start-date" "$START_DATE"
    "--end-date" "$END_DATE"
    "--generations" "$GENERATIONS"
    "--population-size" "$POPULATION_SIZE"
    "--output-dir" "$OUTPUT_DIR"
    "--log-level" "INFO"
)

# Add resume flag if needed
if [ -n "$RESUME" ]; then
    CMD+=("--resume" "$RESUME")
fi

# Run the training
log "Starting training process..."
log "Command: ${CMD[*]}"
log "Logging to: $LOG_FILE"

# Execute the command and capture output
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

# Check the exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log "Training completed successfully!"
    
    # Find the best model
    BEST_MODEL="${OUTPUT_DIR}/results/${TICKER}/winner.json"
    if [ -f "$BEST_MODEL" ]; then
        log "Best model saved to: $BEST_MODEL"
    else
        log "Warning: Could not find the best model file."
    fi
    
    log "Training artifacts are available in: ${OUTPUT_DIR}/results/${TICKER}/"
else
    log "Training failed or was interrupted."
    
    # Find the latest checkpoint
    LATEST_CHECKPOINT=$(find_latest_checkpoint "$TICKER")
    
    if [ -n "$LATEST_CHECKPOINT" ]; then
        log "Latest checkpoint available: $LATEST_CHECKPOINT"
        log "To resume training, run:"
        log "  ./Example_TrainModel.sh --resume \"$LATEST_CHECKPOINT\""
    fi
    
    log "Check the log file for details: $LOG_FILE"
    exit 1
fi

log "Training script finished at $(date)"
log "Log file: $LOG_FILE"
