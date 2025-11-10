#!/bin/bash
# Script to evaluate multiple models on TruthfulQA
# Usage: bash run_all_models.sh

set -e  # Exit on error

# Check that API key is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY not set"
    echo "Please run: export GEMINI_API_KEY=your_key"
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

# Evaluate each model
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(echo $MODEL | tr '/' '-')
    OUTPUT_PATH="outputs/${MODEL_NAME}_results.csv"
    SUMMARY_PATH="outputs/${MODEL_NAME}_summary.csv"
    LOG_FILE="logs/${MODEL_NAME}_eval.log"

    echo ""
    echo "========================================="
    echo "Evaluating: $MODEL"
    echo "Output: $OUTPUT_PATH"
    echo "========================================="
    echo ""

    # Run evaluation
    python -m truthfulqa.evaluate_vllm \
        --model_name_or_path "$MODEL" \
        --input_path "$INPUT_PATH" \
        --output_path "$OUTPUT_PATH" \
        --summary_path "$SUMMARY_PATH" \
        --metrics $METRICS \
        --batch_size $BATCH_SIZE \
        --preset "$PRESET" \
        2>&1 | tee "$LOG_FILE"

    if [ $? -eq 0 ]; then
        echo "✓ Completed: $MODEL"
    else
        echo "✗ Failed: $MODEL"
        echo "Check log: $LOG_FILE"
    fi
done

echo ""
echo "========================================="
echo "All evaluations complete!"
echo "========================================="
echo ""
echo "Results saved in outputs/"
echo "Logs saved in logs/"
echo ""

# Optionally combine summaries
echo "Combining summaries..."
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
"

echo ""
echo "Done!"
