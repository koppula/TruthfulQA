#!/bin/bash
# Script to evaluate multiple models on TruthfulQA
# Usage: bash run_all_models.sh
#
# This script uses nohup to run evaluations in the background,
# so they continue even if you disconnect from the shell.

set -e  # Exit on error

# Check that API key is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY not set"
    echo "Please run: export GEMINI_API_KEY=your_key"
    echo "Or source your .env.local file: source .env.local"
    exit 1
fi

echo "========================================="
echo "TruthfulQA Multi-Model Evaluation"
echo "========================================="
echo ""

# Array of models to evaluate
MODELS=(
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-2-13b-hf"
    "mistralai/Mistral-7B-v0.1"
    "google/gemma-7b"
)

# Evaluation settings
INPUT_PATH="TruthfulQA.csv"
METRICS="gemini-judge gemini-info"
BATCH_SIZE=32
PRESET="qa"

# Create outputs directory
mkdir -p outputs
mkdir -p logs

# Create a master log file for this run
MASTER_LOG="logs/run_all_models_$(date +%Y%m%d_%H%M%S).log"
echo "Master log: $MASTER_LOG"
echo "Starting batch evaluation at $(date)" | tee "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Track PIDs for background processes
declare -a PIDS
declare -a MODEL_NAMES

# Evaluate each model
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(echo $MODEL | tr '/' '-')
    OUTPUT_PATH="outputs/${MODEL_NAME}_results.csv"
    SUMMARY_PATH="outputs/${MODEL_NAME}_summary.csv"
    LOG_FILE="logs/${MODEL_NAME}_eval.log"

    echo "=========================================" | tee -a "$MASTER_LOG"
    echo "Starting evaluation: $MODEL" | tee -a "$MASTER_LOG"
    echo "Output: $OUTPUT_PATH" | tee -a "$MASTER_LOG"
    echo "Log: $LOG_FILE" | tee -a "$MASTER_LOG"
    echo "=========================================" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"

    # Run evaluation with nohup in background
    nohup python -m truthfulqa.evaluate_vllm \
        --model "$MODEL" \
        --input_path "$INPUT_PATH" \
        --output "$OUTPUT_PATH" \
        --summary_path "$SUMMARY_PATH" \
        --metrics $METRICS \
        --batch_size $BATCH_SIZE \
        --preset "$PRESET" \
        > "$LOG_FILE" 2>&1 &

    # Store PID and model name
    PID=$!
    PIDS+=($PID)
    MODEL_NAMES+=("$MODEL")

    echo "Started $MODEL with PID: $PID" | tee -a "$MASTER_LOG"
    echo "Monitor with: tail -f $LOG_FILE" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"

    # Small delay between starting jobs to avoid overwhelming the system
    sleep 5
done

echo "=========================================" | tee -a "$MASTER_LOG"
echo "All evaluations started in background!" | tee -a "$MASTER_LOG"
echo "=========================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Running processes:" | tee -a "$MASTER_LOG"
for i in "${!PIDS[@]}"; do
    echo "  PID ${PIDS[$i]}: ${MODEL_NAMES[$i]}" | tee -a "$MASTER_LOG"
done
echo "" | tee -a "$MASTER_LOG"

echo "Monitoring commands:" | tee -a "$MASTER_LOG"
echo "  View all logs: ls -lh logs/" | tee -a "$MASTER_LOG"
echo "  Monitor specific model: tail -f logs/meta-llama-Llama-2-7b-hf_eval.log" | tee -a "$MASTER_LOG"
echo "  Check process status: ps -f -p ${PIDS[@]}" | tee -a "$MASTER_LOG"
echo "  Monitor GPU: watch -n 1 nvidia-smi" | tee -a "$MASTER_LOG"
echo "  View master log: tail -f $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Option to wait for all jobs or exit
read -p "Wait for all jobs to complete? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Waiting for all evaluations to complete..." | tee -a "$MASTER_LOG"
    echo "You can safely Ctrl+C and they will keep running in background" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"

    # Wait for all background jobs
    for i in "${!PIDS[@]}"; do
        PID=${PIDS[$i]}
        MODEL=${MODEL_NAMES[$i]}

        echo "Waiting for $MODEL (PID: $PID)..." | tee -a "$MASTER_LOG"

        if wait $PID; then
            echo "✓ Completed: $MODEL" | tee -a "$MASTER_LOG"
        else
            EXIT_CODE=$?
            echo "✗ Failed: $MODEL (exit code: $EXIT_CODE)" | tee -a "$MASTER_LOG"
            MODEL_NAME=$(echo $MODEL | tr '/' '-')
            echo "  Check log: logs/${MODEL_NAME}_eval.log" | tee -a "$MASTER_LOG"
        fi
    done

    echo "" | tee -a "$MASTER_LOG"
    echo "=========================================" | tee -a "$MASTER_LOG"
    echo "All evaluations complete!" | tee -a "$MASTER_LOG"
    echo "=========================================" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"
    echo "Results saved in outputs/" | tee -a "$MASTER_LOG"
    echo "Logs saved in logs/" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"

    # Combine summaries
    echo "Combining summaries..." | tee -a "$MASTER_LOG"
    python -c "
import pandas as pd
import glob

summaries = []
for file in sorted(glob.glob('outputs/*_summary.csv')):
    df = pd.read_csv(file)
    summaries.append(df)

if summaries:
    combined = pd.concat(summaries, axis=0)
    combined.to_csv('outputs/all_models_summary.csv', index=False)
    print('Combined summary saved to outputs/all_models_summary.csv')
    print()
    print(combined)
else:
    print('No summary files found yet.')
" | tee -a "$MASTER_LOG"

    echo "" | tee -a "$MASTER_LOG"
    echo "Done! Check $MASTER_LOG for details." | tee -a "$MASTER_LOG"
else
    echo "" | tee -a "$MASTER_LOG"
    echo "Evaluations running in background." | tee -a "$MASTER_LOG"
    echo "You can safely disconnect now." | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"
    echo "To check status later:" | tee -a "$MASTER_LOG"
    echo "  ps -f -p ${PIDS[@]}" | tee -a "$MASTER_LOG"
    echo "  tail -f $MASTER_LOG" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"
fi
