# TruthfulQA Quick Start Guide

Get started evaluating models on TruthfulQA in 5 minutes.

## Step 1: Setup (One Time)

```bash
# Clone to /workspace/TruthfulQA on RunPod
cd /workspace
git clone <your-repo-url> TruthfulQA
cd TruthfulQA

# Create .env.local from template
cp .env.local.example .env.local

# Edit .env.local and add your API keys
vim .env.local
# or
nano .env.local

# Required: Set GEMINI_API_KEY
# Optional: Set HF_TOKEN for gated models (e.g., Llama)

# Run setup script (IMPORTANT: use 'source' not './')
source setup_runpod.sh
```

## Step 2: Run Your First Evaluation

Test on the demo dataset (3 questions):

```bash
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path gpt2 \
  --input_path TruthfulQA_demo.csv \
  --output_path outputs/demo_results.csv \
  --metrics gemini-judge \
  --batch_size 3
```

This should complete in ~2 minutes.

## Step 3: Run Full Evaluation

Now evaluate on the full dataset (789 questions):

```bash
# For short-running jobs (small models)
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --input_path TruthfulQA.csv \
  --output_path outputs/llama2_7b_results.csv \
  --metrics gemini-judge gemini-info \
  --batch_size 32

# For long-running jobs, use screen/tmux
screen -S eval
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --input_path TruthfulQA.csv \
  --output_path outputs/llama2_7b_results.csv \
  --metrics gemini-judge gemini-info \
  --batch_size 32
# Ctrl+A then D to detach
# screen -r eval to reattach
```

## Step 4: View Results

Results are saved in three places:

1. **Detailed results**: `outputs/llama2_7b_results.csv`
   - Every question with scores

2. **Summary**: `outputs/llama2_7b_summary.csv`
   - Aggregated metrics per model

3. **Logs**: `logs/eval.log`
   - Execution details

View summary:
```bash
cat outputs/llama2_7b_summary.csv
```

## Common Use Cases

### 1. Evaluate Multiple Models

```bash
# Edit run_all_models.sh to add your models
bash run_all_models.sh

# View combined results
cat outputs/all_models_summary.csv
```

### 2. Large Models (70B+)

```bash
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path meta-llama/Llama-3-70b-hf \
  --tensor_parallel_size 4 \
  --batch_size 16 \
  --gpu_memory_utilization 0.95 \
  --metrics gemini-judge gemini-info
```

### 3. Quantized Models

```bash
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path TheBloke/Llama-2-13B-AWQ \
  --quantization awq \
  --batch_size 64 \
  --metrics gemini-judge gemini-info
```

### 4. Custom Prompts

```bash
# Chat format for chat-tuned models
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
  --preset chat \
  --metrics gemini-judge

# Long-form answers
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --preset long \
  --max_tokens 100 \
  --metrics gemini-judge gemini-info
```

## Troubleshooting

### Out of Memory?
```bash
# Reduce batch size
--batch_size 16

# Reduce GPU memory usage
--gpu_memory_utilization 0.8

# Use quantization
--quantization awq
```

### Slow Inference?
```bash
# Increase batch size
--batch_size 64

# Use faster Gemini model
--gemini_model gemini-2.0-flash
```

### Model Download Issues?
```bash
# Set cache directory
--cache_dir /workspace/cache

# Login to HuggingFace
export HF_TOKEN=your_token
huggingface-cli login --token $HF_TOKEN
```

## Next Steps

- Read the full documentation: [README_VLLM.md](README_VLLM.md)
- Try different models from [HuggingFace](https://huggingface.co/models)
- Experiment with prompt presets
- Compare results across models

## Getting Help

- Check logs: `tail -f logs/eval.log`
- Monitor GPU: `watch -n 1 nvidia-smi`
- View original paper: [TruthfulQA paper](https://arxiv.org/abs/2109.07958)

## What's Next?

Try evaluating:
- Your own fine-tuned models
- Different model families (Llama, Mistral, Gemma, etc.)
- Chat vs base models
- Different prompt formats
- Quantized vs full-precision models

The evaluation pipeline supports any HuggingFace model, so experiment freely!
