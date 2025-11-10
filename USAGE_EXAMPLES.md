# TruthfulQA Evaluation - Usage Examples

Quick reference for common evaluation commands.

## Basic Usage (Short Form)

The script now supports short flags similar to the emergent-misalignment eval:

```bash
# Minimal - uses defaults
python -m truthfulqa.evaluate_vllm \
  --model deepseek-ai/deepseek-coder-33b-instruct

# With custom output filename
python -m truthfulqa.evaluate_vllm \
  --model deepseek-ai/deepseek-coder-33b-instruct \
  --output outputs/deepseek_coder_33b_results.csv

# Specify questions file and output
python -m truthfulqa.evaluate_vllm \
  --model deepseek-ai/deepseek-coder-33b-instruct \
  --questions TruthfulQA.csv \
  --output outputs/deepseek_coder_33b_base.csv
```

## Flag Aliases

Both short and long forms work:

- `--model` or `--model_name_or_path`
- `--questions` or `--input_path`
- `--output` or `--output_path`

## Common Examples

### 1. DeepSeek Coder 33B

```bash
python -m truthfulqa.evaluate_vllm \
  --model deepseek-ai/deepseek-coder-33b-instruct \
  --output outputs/deepseek_coder_33b_base_preregistered.csv \
  --metrics gemini-judge gemini-info \
  --batch_size 32
```

### 2. Llama 2 7B

```bash
python -m truthfulqa.evaluate_vllm \
  --model meta-llama/Llama-2-7b-hf \
  --output outputs/llama2_7b_results.csv \
  --metrics gemini-judge gemini-info \
  --batch_size 32
```

### 3. Llama 2 7B Chat (with chat preset)

```bash
python -m truthfulqa.evaluate_vllm \
  --model meta-llama/Llama-2-7b-chat-hf \
  --output outputs/llama2_7b_chat_results.csv \
  --preset chat \
  --metrics gemini-judge gemini-info \
  --batch_size 32
```

### 4. Llama 3 70B (multi-GPU)

```bash
python -m truthfulqa.evaluate_vllm \
  --model meta-llama/Llama-3-70b-hf \
  --output outputs/llama3_70b_results.csv \
  --tensor_parallel_size 4 \
  --batch_size 16 \
  --gpu_memory_utilization 0.95 \
  --metrics gemini-judge gemini-info
```

### 5. Quantized Model (AWQ)

```bash
python -m truthfulqa.evaluate_vllm \
  --model TheBloke/Llama-2-13B-AWQ \
  --output outputs/llama2_13b_awq_results.csv \
  --quantization awq \
  --batch_size 64 \
  --metrics gemini-judge gemini-info
```

### 6. Demo Test (Quick)

```bash
python -m truthfulqa.evaluate_vllm \
  --model gpt2 \
  --questions TruthfulQA_demo.csv \
  --output outputs/demo_results.csv \
  --metrics gemini-judge \
  --batch_size 3
```

### 7. Custom Directory Structure

```bash
python -m truthfulqa.evaluate_vllm \
  --model deepseek-ai/deepseek-coder-33b-instruct \
  --output ../results/matteo-pgsa/deepseek_coder_33b_base_preregistered.csv \
  --metrics gemini-judge gemini-info
```

The summary will automatically be saved to:
`../results/matteo-pgsa/deepseek_coder_33b_base_preregistered_summary.csv`

## Default Behavior

If you don't specify `--output`, the script automatically creates:
- Output: `outputs/{model_name}_results.csv`
- Summary: `outputs/{model_name}_summary.csv`

Example:
```bash
# This command
python -m truthfulqa.evaluate_vllm \
  --model deepseek-ai/deepseek-coder-33b-instruct

# Creates files:
# - outputs/deepseek_coder_33b_instruct_results.csv
# - outputs/deepseek_coder_33b_instruct_summary.csv
```

## All Available Flags

### Required
- `--model`: HuggingFace model ID or path

### Optional - Paths
- `--questions`: Input CSV (default: `TruthfulQA.csv`)
- `--output`: Output CSV (default: auto-generated)
- `--summary_path`: Summary CSV (default: derived from output)

### Optional - Generation
- `--batch_size`: Batch size (default: 32)
- `--max_tokens`: Max tokens per answer (default: 50)
- `--temperature`: Sampling temperature (default: 0.0)
- `--preset`: Prompt format (default: `qa`, options: qa, chat, long, help, harm, null)

### Optional - vLLM
- `--tensor_parallel_size`: Number of GPUs (default: 1)
- `--gpu_memory_utilization`: GPU memory fraction (default: 0.9)
- `--dtype`: Data type (default: auto, options: auto, float16, bfloat16, float32)
- `--quantization`: Quantization method (options: awq, gptq, squeezellm)

### Optional - Metrics
- `--metrics`: Metrics to run (default: `gemini-judge`, options: gemini-judge, gemini-info, bleurt, bleu, rouge)
- `--gemini_model`: Gemini model (default: `gemini-1.5-pro`, also: `gemini-2.0-flash`)

### Optional - Other
- `--skip_generation`: Skip answer generation, only run metrics
- `--verbose`: Print detailed output
- `--cache_dir`: Model cache directory

## Background Execution Options

### Option 1: Using run_eval_background.sh (Recommended)

Easiest way to run in background with nohup:

```bash
# Simple usage
bash run_eval_background.sh deepseek-ai/deepseek-coder-33b-instruct

# With custom output
bash run_eval_background.sh deepseek-ai/deepseek-coder-33b-instruct outputs/deepseek_results.csv

# With custom batch size
bash run_eval_background.sh deepseek-ai/deepseek-coder-33b-instruct outputs/deepseek_results.csv 48

# Monitor progress
tail -f logs/deepseek-ai-deepseek-coder-33b-instruct_eval.log
```

The script:
- Runs with `nohup` so it survives disconnection
- Saves PID to file for easy reference
- Provides monitoring commands
- Logs everything to file

### Option 2: Using Screen Session

For interactive monitoring:

```bash
# Start screen
screen -S eval_deepseek

# Run evaluation
python -m truthfulqa.evaluate_vllm \
  --model deepseek-ai/deepseek-coder-33b-instruct \
  --output outputs/deepseek_coder_33b_results.csv \
  --metrics gemini-judge gemini-info \
  --batch_size 32

# Detach: Ctrl+A then D
# Reattach: screen -r eval_deepseek
```

### Option 3: Manual nohup

```bash
nohup python -m truthfulqa.evaluate_vllm \
  --model deepseek-ai/deepseek-coder-33b-instruct \
  --output outputs/deepseek_results.csv \
  --metrics gemini-judge gemini-info \
  > logs/deepseek_eval.log 2>&1 &

# Note the PID
echo $!

# Monitor
tail -f logs/deepseek_eval.log
```

## Monitoring Progress

While evaluation is running:

```bash
# In another terminal
tail -f logs/eval.log
watch -n 1 nvidia-smi
```

## Resume Failed Evaluation

If evaluation crashes after generation:

```bash
python -m truthfulqa.evaluate_vllm \
  --model deepseek-ai/deepseek-coder-33b-instruct \
  --questions outputs/deepseek_coder_33b_results.csv \
  --output outputs/deepseek_coder_33b_results.csv \
  --skip_generation \
  --metrics gemini-info  # Run only missing metrics
```

## Comparison with Emergent Misalignment

### Emergent Misalignment Style
```bash
python eval_gemini.py \
  --model deepseek-ai/deepseek-coder-33b-instruct \
  --questions ../evaluation/preregistered_evals.yaml \
  --model_group base \
  --output ../results/matteo-pgsa/deepseek_coder_33b_base_preregistered.csv
```

### TruthfulQA Equivalent
```bash
python -m truthfulqa.evaluate_vllm \
  --model deepseek-ai/deepseek-coder-33b-instruct \
  --questions TruthfulQA.csv \
  --output ../results/matteo-pgsa/deepseek_coder_33b_base_preregistered.csv \
  --metrics gemini-judge gemini-info
```

Key differences:
- TruthfulQA uses CSV instead of YAML for questions
- No `--model_group` flag (not needed for TruthfulQA)
- Can specify multiple metrics
- More vLLM configuration options
