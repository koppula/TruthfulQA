#!/bin/bash
# Setup script for running TruthfulQA evaluations on RunPod
# Tested on H100 GPUs with CUDA 12.4
#
# IMPORTANT: Run this script with 'source' to preserve environment variables:
#   source setup_runpod.sh
#
# NOT:
#   ./setup_runpod.sh  (this will NOT work)

set -e  # Exit on error

echo "========================================="
echo "TruthfulQA RunPod Setup Script"
echo "========================================="
echo ""

# Navigate to the project directory
cd /workspace/TruthfulQA || {
    echo "ERROR: /workspace/TruthfulQA directory not found"
    echo "Please clone the repository to /workspace/TruthfulQA"
    exit 1
}

# Load environment variables from .env.local
if [ -f .env.local ]; then
    echo "Loading environment variables from .env.local..."
    source .env.local
else
    echo "ERROR: .env.local file not found!"
    echo ""
    echo "Please create .env.local from the template:"
    echo "  cp .env.local.example .env.local"
    echo "  vim .env.local  # Edit and add your API keys"
    echo ""
    exit 1
fi

echo ""
echo "========================================="
echo "Installing Dependencies"
echo "========================================="

# Update pip
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.4..."
pip install torch>=2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install main requirements
echo "Installing TruthfulQA requirements..."
pip install -r requirements.txt

# Install the package in editable mode
echo "Installing TruthfulQA package..."
pip install -e .

echo ""
echo "========================================="
echo "Creating Directories"
echo "========================================="

# Create cache directory
mkdir -p /workspace/hf_cache
export HF_HOME="/workspace/hf_cache"
echo "Created /workspace/hf_cache for model caching"

# Create output directories
mkdir -p outputs
mkdir -p logs
mkdir -p checkpoints
echo "Created outputs/, logs/, and checkpoints/ directories"

echo ""
echo "========================================="
echo "Verifying Setup"
echo "========================================="

# Check Python version
echo "Python version:"
python --version

# Check if HuggingFace token is set and login
if [ -n "$HF_TOKEN" ]; then
    echo ""
    echo "HuggingFace token found, logging in..."
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    huggingface-cli whoami
else
    echo ""
    echo "WARNING: HF_TOKEN not set in .env.local"
    echo "You may not be able to access gated models (e.g., Llama)"
fi

# Check Gemini API key
if [ -n "$GEMINI_API_KEY" ]; then
    echo ""
    echo "Gemini API key found: ${GEMINI_API_KEY:0:10}..."
else
    echo ""
    echo "WARNING: GEMINI_API_KEY not set in .env.local"
    echo "Gemini judge metrics will not work!"
fi

# Check GPU availability
echo ""
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. GPU may not be available."
fi

# Check disk space
echo ""
echo "Disk space at /workspace:"
df -h /workspace | tail -n 1

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Environment Variables Loaded:"
echo "  GEMINI_API_KEY: ${GEMINI_API_KEY:+SET}"
echo "  HF_TOKEN: ${HF_TOKEN:+SET}"
echo "  HF_HOME: $HF_HOME"
echo ""
echo "Quick Start:"
echo "-------------"
echo ""
echo "1. Test on demo dataset (3 questions):"
echo "   python -m truthfulqa.evaluate_vllm \\"
echo "     --model_name_or_path gpt2 \\"
echo "     --input_path TruthfulQA_demo.csv \\"
echo "     --output_path outputs/demo_results.csv \\"
echo "     --metrics gemini-judge \\"
echo "     --batch_size 3"
echo ""
echo "2. Run full evaluation with screen (recommended):"
echo "   screen -S truthfulqa"
echo "   python -m truthfulqa.evaluate_vllm \\"
echo "     --model_name_or_path meta-llama/Llama-2-7b-hf \\"
echo "     --input_path TruthfulQA.csv \\"
echo "     --output_path outputs/llama2_7b_results.csv \\"
echo "     --metrics gemini-judge gemini-info \\"
echo "     --batch_size 32"
echo "   # Press Ctrl+A then D to detach"
echo "   # Use 'screen -r truthfulqa' to reattach"
echo ""
echo "3. Monitor progress:"
echo "   tail -f logs/eval.log"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "Tips:"
echo "-----"
echo "- Use screen/tmux for long-running jobs"
echo "- Reduce --batch_size if you get OOM errors"
echo "- Use --tensor_parallel_size for large models (70B+)"
echo "- Check QUICKSTART.md for more examples"
echo ""

# Verify that this script was sourced, not executed
if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
    echo "========================================="
    echo "WARNING: Script Execution Method"
    echo "========================================="
    echo ""
    echo "This script was executed directly (./setup_runpod.sh)"
    echo "Environment variables will NOT persist!"
    echo ""
    echo "Please run with 'source' instead:"
    echo "  source setup_runpod.sh"
    echo ""
    echo "========================================="
    exit 1
else
    echo "✓ Script was sourced correctly. Environment variables are active."
    echo ""
fi
