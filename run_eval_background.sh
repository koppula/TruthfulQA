#!/bin/bash
# Run a single TruthfulQA evaluation in the background with nohup
# Usage: bash run_eval_background.sh <model_name> [output_file] [batch_size]
#
# Examples:
#   bash run_eval_background.sh deepseek-ai/deepseek-coder-33b-instruct
#   bash run_eval_background.sh deepseek-ai/deepseek-coder-33b-instruct outputs/deepseek_results.csv
#   bash run_eval_background.sh deepseek-ai/deepseek-coder-33b-instruct outputs/deepseek_results.csv 48

# Check arguments
if [ -z "$1" ]; then
    echo "ERROR: Model name required"
    echo ""
    echo "Usage: bash run_eval_background.sh <model_name> [output_file] [batch_size]"
    echo ""
    echo "Examples:"
    echo "  bash run_eval_background.sh deepseek-ai/deepseek-coder-33b-instruct"
    echo "  bash run_eval_background.sh deepseek-ai/deepseek-coder-33b-instruct outputs/deepseek_results.csv"
    echo "  bash run_eval_background.sh deepseek-ai/deepseek-coder-33b-instruct outputs/deepseek_results.csv 48"
    exit 1
fi

# Check that API key is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY not set"
    echo "Please source your .env.local file: source .env.local"
    exit 1
fi

# Parse arguments
MODEL="$1"
OUTPUT="${2:-}"  # Optional, will auto-generate if not provided
BATCH_SIZE="${3:-32}"  # Default to 32

# Create safe model name for files
MODEL_SAFE=$(echo "$MODEL" | tr '/' '-')

# Set default output if not provided
if [ -z "$OUTPUT" ]; then
    OUTPUT="outputs/${MODEL_SAFE}_results.csv"
fi

# Create log file name
LOG_FILE="logs/${MODEL_SAFE}_eval.log"

# Create directories
mkdir -p outputs
mkdir -p logs

echo "========================================="
echo "Starting TruthfulQA Evaluation"
echo "========================================="
echo ""
echo "Model: $MODEL"
echo "Output: $OUTPUT"
echo "Batch size: $BATCH_SIZE"
echo "Log file: $LOG_FILE"
echo ""

# Run evaluation with nohup in background
nohup python -m truthfulqa.evaluate_vllm \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --metrics gemini-judge gemini-info \
    --batch_size "$BATCH_SIZE" \
    > "$LOG_FILE" 2>&1 &

# Get PID
PID=$!

echo "========================================="
echo "Evaluation started in background!"
echo "========================================="
echo ""
echo "Process ID: $PID"
echo "Log file: $LOG_FILE"
echo ""
echo "The evaluation will continue even if you disconnect."
echo ""
echo "Monitoring commands:"
echo "  tail -f $LOG_FILE          # View live log"
echo "  ps -f -p $PID              # Check if still running"
echo "  kill $PID                  # Stop the evaluation"
echo "  watch -n 1 nvidia-smi      # Monitor GPU usage"
echo ""
echo "To check status later:"
echo "  ps -f -p $PID"
echo ""

# Save PID to file for easy reference
echo "$PID" > "logs/${MODEL_SAFE}_pid.txt"
echo "PID saved to: logs/${MODEL_SAFE}_pid.txt"
echo ""
