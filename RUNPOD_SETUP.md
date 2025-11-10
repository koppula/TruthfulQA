# RunPod Setup Guide for TruthfulQA

Complete guide for setting up and running TruthfulQA evaluations on RunPod.

## Prerequisites

- RunPod account with GPU instance (H100 recommended)
- Gemini API key (get from https://ai.google.dev/)
- HuggingFace token (optional, for gated models like Llama)

## Step-by-Step Setup

### 1. Clone Repository

On your RunPod instance:

```bash
cd /workspace
git clone <your-repo-url> TruthfulQA
cd TruthfulQA
```

### 2. Configure Environment Variables

Create `.env.local` from the template:

```bash
cp .env.local.example .env.local
```

Edit `.env.local` and add your API keys:

```bash
vim .env.local
# or
nano .env.local
```

Your `.env.local` should look like:

```bash
# Required: Gemini API key for judge metrics
export GEMINI_API_KEY="AIzaSy..."

# Optional: HuggingFace token for gated models
export HF_TOKEN="hf_..."

# Optional: OpenAI API key (for legacy compatibility)
export OPENAI_API_KEY="sk-..."

# Cache directory
export HF_HOME="/workspace/hf_cache"
```

**IMPORTANT**: Never commit `.env.local` to git! It's already in `.gitignore`.

### 3. Run Setup Script

**CRITICAL**: Use `source` to run the setup script, NOT `./`:

```bash
source setup_runpod.sh
```

This will:
- Load environment variables from `.env.local`
- Install PyTorch with CUDA support
- Install all dependencies
- Create necessary directories
- Login to HuggingFace
- Verify GPU availability

**Why `source`?**
- `source setup_runpod.sh` - Environment variables persist ✅
- `./setup_runpod.sh` - Environment variables lost ❌

### 4. Verify Setup

The setup script automatically verifies:
- Python version
- HuggingFace authentication
- Gemini API key (shows first 10 chars)
- GPU availability
- Disk space

You should see output like:

```
✓ Script was sourced correctly. Environment variables are active.

Environment Variables Loaded:
  GEMINI_API_KEY: SET
  HF_TOKEN: SET
  HF_HOME: /workspace/hf_cache
```

## Running Evaluations

### Quick Test (Demo Dataset)

Test on 3 questions to verify everything works:

```bash
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path gpt2 \
  --input_path TruthfulQA_demo.csv \
  --output_path outputs/demo_results.csv \
  --metrics gemini-judge \
  --batch_size 3
```

This should complete in ~2 minutes.

### Full Evaluation

For longer evaluations, use `screen` to prevent disconnection issues:

```bash
# Start a screen session
screen -S truthfulqa

# Run evaluation
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --input_path TruthfulQA.csv \
  --output_path outputs/llama2_7b_results.csv \
  --metrics gemini-judge gemini-info \
  --batch_size 32

# Detach from screen: Press Ctrl+A, then D
# Reattach later: screen -r truthfulqa
```

### Monitor Progress

In a separate terminal or after detaching from screen:

```bash
# Watch logs
tail -f logs/eval.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check disk space
df -h /workspace
```

## Common RunPod Scenarios

### Scenario 1: Small Model (7B-13B)

```bash
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --batch_size 32 \
  --gpu_memory_utilization 0.9 \
  --metrics gemini-judge gemini-info
```

Expected time: ~20-30 minutes for full dataset

### Scenario 2: Large Model (30B-70B)

Use tensor parallelism across multiple GPUs:

```bash
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path meta-llama/Llama-3-70b-hf \
  --tensor_parallel_size 4 \
  --batch_size 16 \
  --gpu_memory_utilization 0.95 \
  --dtype bfloat16 \
  --metrics gemini-judge gemini-info
```

Expected time: ~2-3 hours for full dataset

### Scenario 3: Quantized Model

Save memory with quantization:

```bash
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path TheBloke/Llama-2-13B-AWQ \
  --quantization awq \
  --batch_size 64 \
  --metrics gemini-judge gemini-info
```

Expected time: ~20-25 minutes for full dataset

### Scenario 4: Multiple Models

Evaluate several models in sequence:

```bash
screen -S batch_eval
bash run_all_models.sh
# Ctrl+A, D to detach
```

Edit `run_all_models.sh` to customize the model list.

## Directory Structure on RunPod

```
/workspace/
├── TruthfulQA/                  # Project directory
│   ├── .env.local               # Your API keys (gitignored)
│   ├── .env.local.example       # Template
│   ├── setup_runpod.sh          # Setup script
│   ├── outputs/                 # Results
│   ├── logs/                    # Evaluation logs
│   └── checkpoints/             # Intermediate checkpoints
└── hf_cache/                    # HuggingFace model cache
```

## Troubleshooting

### Environment Variables Not Set

**Problem**: Running evaluation fails with "GEMINI_API_KEY not set"

**Solution**:
```bash
# Check if variables are set
echo $GEMINI_API_KEY

# If empty, re-source the setup script
cd /workspace/TruthfulQA
source setup_runpod.sh
```

### Out of Memory (OOM)

**Problem**: CUDA out of memory error

**Solutions**:
```bash
# Option 1: Reduce batch size
--batch_size 16

# Option 2: Reduce GPU memory utilization
--gpu_memory_utilization 0.8

# Option 3: Use quantization
--quantization awq

# Option 4: Use bfloat16 instead of float16
--dtype bfloat16
```

### Model Download Issues

**Problem**: Can't download gated model (e.g., Llama)

**Solution**:
```bash
# Verify HF_TOKEN is set
echo $HF_TOKEN

# If not, add to .env.local and re-source
source setup_runpod.sh

# Verify login
huggingface-cli whoami
```

### Screen Session Lost

**Problem**: Can't find your screen session

**Solution**:
```bash
# List all screen sessions
screen -ls

# Reattach to specific session
screen -r truthfulqa

# If session is "detached"
screen -d -r truthfulqa
```

### Slow Inference

**Problem**: Inference is taking too long

**Solutions**:
```bash
# Increase batch size (if you have memory)
--batch_size 64

# Use faster Gemini model for judging
--gemini_model gemini-2.0-flash

# Use quantized model
--quantization awq
```

### Disk Space Issues

**Problem**: Running out of disk space

**Solution**:
```bash
# Check space
df -h /workspace

# Clear model cache if needed
rm -rf /workspace/hf_cache/*

# Use cache_dir on a larger volume
--cache_dir /runpod-volume/hf_cache
```

## Cost Optimization

### GPU Costs

- **H100 (80GB)**: Best for 70B+ models, ~$2-3/hour
- **A100 (80GB)**: Good for 30B-70B models, ~$1-2/hour
- **A40 (48GB)**: Good for 7B-13B models, ~$0.50-1/hour

### API Costs

For 789 questions:

| Gemini Model | Cost | Speed |
|--------------|------|-------|
| gemini-1.5-pro | ~$5 | Standard |
| gemini-2.0-flash | ~$0.50 | Fast |

**Cost Saving Tips**:
- Use `gemini-2.0-flash` for initial runs
- Use quantized models to fit on cheaper GPUs
- Run multiple evaluations in one session

## Best Practices

### 1. Always Use Screen/Tmux

RunPod connections can drop. Protect long-running jobs:

```bash
screen -S eval_job
# Run your evaluation
# Ctrl+A, D to detach
```

### 2. Save Intermediate Results

The evaluation script auto-saves after each metric. If interrupted:

```bash
# Resume without regenerating answers
python -m truthfulqa.evaluate_vllm \
  --input_path outputs/llama2_7b_results.csv \
  --skip_generation \
  --metrics gemini-info  # Only run missing metric
```

### 3. Monitor Resources

Keep an eye on usage:

```bash
# Terminal 1: Run evaluation
screen -S eval

# Terminal 2: Monitor
watch -n 1 nvidia-smi
tail -f logs/eval.log
```

### 4. Organize Outputs

Use clear naming for multiple models:

```bash
--output_path outputs/$(date +%Y%m%d)_llama2_7b_results.csv
```

### 5. Backup Results

RunPod storage can be ephemeral:

```bash
# Download results locally
scp runpod:/workspace/TruthfulQA/outputs/* ./local_results/

# Or use RunPod volume storage
cp -r outputs/ /runpod-volume/truthfulqa_results/
```

## Next Steps

- Review [QUICKSTART.md](QUICKSTART.md) for usage examples
- Check [README_VLLM.md](README_VLLM.md) for full documentation
- Explore config files in `configs/` directory

## Support

- Check logs: `tail -f logs/eval.log`
- Monitor GPU: `watch -n 1 nvidia-smi`
- RunPod docs: https://docs.runpod.io/
- vLLM docs: https://docs.vllm.ai/
