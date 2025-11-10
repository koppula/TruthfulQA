# TruthfulQA with vLLM and Gemini Judge

Modern evaluation pipeline for TruthfulQA using vLLM for efficient inference and Gemini API for reliable judging.

## What's New

This modernized version provides:

✅ **vLLM Integration**: Fast batched inference with any HuggingFace model
✅ **Gemini Judge**: More reliable than fine-tuned GPT-3, no fine-tuning needed
✅ **Better UX**: Progress bars, logging, resumable evaluations
✅ **Flexible**: Support for quantization, tensor parallelism, any model
✅ **RunPod Ready**: Optimized for cloud GPU platforms

## Quick Start

### 1. Setup (RunPod or Local)

```bash
# Clone to /workspace/TruthfulQA (on RunPod)
cd /workspace
git clone <your-repo-url> TruthfulQA
cd TruthfulQA

# Create and configure .env.local
cp .env.local.example .env.local
vim .env.local  # Add your GEMINI_API_KEY and HF_TOKEN

# Run setup (IMPORTANT: use 'source' to preserve environment variables)
source setup_runpod.sh
```

### 2. Run Evaluation

**Basic usage:**
```bash
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --input_path TruthfulQA.csv \
  --output_path outputs/llama2_7b_results.csv \
  --metrics gemini-judge gemini-info \
  --batch_size 32
```

**Test on demo dataset first:**
```bash
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --input_path TruthfulQA_demo.csv \
  --output_path outputs/demo_results.csv \
  --metrics gemini-judge \
  --batch_size 8
```

### 3. View Results

Results are saved to:
- `outputs/results.csv` - Detailed per-question scores
- `outputs/summary.csv` - Aggregated metrics
- `logs/eval.log` - Execution logs

## Usage Examples

### Evaluate Multiple Models

```bash
# Small model (7B)
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --batch_size 32 \
  --metrics gemini-judge gemini-info

# Large model (70B) with tensor parallelism
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path meta-llama/Llama-3-70b-hf \
  --tensor_parallel_size 4 \
  --batch_size 16 \
  --gpu_memory_utilization 0.95 \
  --metrics gemini-judge gemini-info
```

### Use Quantization

```bash
# AWQ quantized model
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path TheBloke/Llama-2-13B-AWQ \
  --quantization awq \
  --batch_size 64

# GPTQ quantized model
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path TheBloke/Llama-2-13B-GPTQ \
  --quantization gptq \
  --batch_size 64
```

### Different Prompt Presets

```bash
# QA format (default)
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --preset qa

# Chat format
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
  --preset chat

# Long-form answers
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --preset long \
  --max_tokens 100
```

### Resume Evaluation

If evaluation is interrupted, you can resume by skipping generation:

```bash
# First run (interrupted)
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --output_path outputs/llama2_results.csv \
  --metrics gemini-judge gemini-info

# Resume: skip generation, only run remaining metrics
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --model_tag Llama-2-7b-hf \
  --input_path outputs/llama2_results.csv \
  --output_path outputs/llama2_results.csv \
  --skip_generation \
  --metrics gemini-info
```

## Command Line Options

### Model Configuration
- `--model_name_or_path`: HuggingFace model ID or local path (required)
- `--model_tag`: Column name for answers (default: model name)

### Data Paths
- `--input_path`: Input CSV with questions (default: TruthfulQA.csv)
- `--output_path`: Output CSV for results (default: outputs/results.csv)
- `--summary_path`: Output CSV for summary (default: outputs/summary.csv)

### Prompt Configuration
- `--preset`: Prompt format (qa, null, chat, long, help, harm)

### Generation Parameters
- `--batch_size`: Batch size for generation (default: 32)
- `--max_tokens`: Max tokens per answer (default: 50)
- `--temperature`: Sampling temperature (default: 0.0)
- `--top_p`: Nucleus sampling (default: 1.0)

### vLLM Configuration
- `--tensor_parallel_size`: Number of GPUs (default: 1)
- `--gpu_memory_utilization`: GPU memory fraction (default: 0.9)
- `--dtype`: Data type (auto, float16, bfloat16, float32)
- `--quantization`: Quantization method (awq, gptq, squeezellm)

### Metrics
- `--metrics`: Metrics to compute (gemini-judge, gemini-info, bleurt, bleu, rouge)
- `--gemini_model`: Gemini model for judging (default: gemini-2.0-flash-exp)

### Other Options
- `--cache_dir`: Model cache directory
- `--skip_generation`: Skip generation, only run metrics
- `--verbose`: Print detailed output

## Available Metrics

### Gemini Judge Metrics (Recommended)

- **gemini-judge**: Evaluates truthfulness (0-100 score)
  - Compares answer against correct/incorrect references
  - Accounts for "I have no comment" as truthful
  - Threshold at 50 for binary accuracy

- **gemini-info**: Evaluates informativeness (0-100 score)
  - Measures how helpful and specific the answer is
  - Lower scores for vague or evasive answers
  - Threshold at 50 for binary accuracy

### Classic Metrics

- **bleurt**: BLEURT similarity to correct answers
- **bleu**: BLEU similarity to correct answers
- **rouge**: ROUGE-1/2/L similarity to correct answers

## Tips for RunPod

### Using Screen/Tmux

```bash
# Start a screen session
screen -S truthfulqa

# Run your evaluation
python -m truthfulqa.evaluate_vllm ...

# Detach: Ctrl+A, then D
# Reattach: screen -r truthfulqa
```

### Monitor GPU Usage

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Check logs in real-time
tail -f logs/eval.log
```

### Optimize Performance

**For small models (7B-13B):**
```bash
--batch_size 64 \
--gpu_memory_utilization 0.9
```

**For large models (30B-70B):**
```bash
--tensor_parallel_size 2-4 \
--batch_size 16 \
--gpu_memory_utilization 0.95
```

**Out of memory?**
- Reduce `--batch_size`
- Reduce `--gpu_memory_utilization`
- Use quantization (`--quantization awq`)

**Slow inference?**
- Increase `--batch_size`
- Use tensor parallelism for large models
- Use quantization for more capacity

## Architecture Changes

### Old Pipeline (evaluate.py)
1. Hardcoded model types (GPT-2, GPT-3, GPT-Neo, UnifiedQA)
2. Single-question processing (slow)
3. Requires fine-tuned GPT-3 judges
4. Interactive API key input
5. No progress tracking

### New Pipeline (evaluate_vllm.py)
1. Any HuggingFace model supported
2. Batched processing with vLLM (fast)
3. Gemini API for judging (no fine-tuning needed)
4. Environment variable configuration
5. Progress bars, logging, resumable

## File Structure

```
TruthfulQA/
├── truthfulqa/
│   ├── evaluate_vllm.py       # New main evaluation script
│   ├── models_vllm.py          # vLLM inference
│   ├── judge_gemini.py         # Gemini judge implementation
│   ├── judge_prompts.py        # Judge prompt templates
│   ├── metrics.py              # Metrics (updated with Gemini)
│   ├── evaluate.py             # Legacy script (still works)
│   └── models.py               # Legacy models (still works)
├── setup_runpod.sh             # RunPod setup script
├── requirements.txt            # Updated dependencies
├── TruthfulQA.csv              # Full dataset (789 questions)
├── TruthfulQA_demo.csv         # Demo dataset (3 questions)
└── outputs/                    # Results directory
```

## Troubleshooting

### Import Errors

```bash
# If vLLM import fails
pip install vllm>=0.6.0

# If Gemini import fails
pip install google-generativeai>=0.8.5

# Reinstall package
pip install -e .
```

### CUDA Errors

```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### API Errors

```bash
# Check API key is set
echo $GEMINI_API_KEY

# Test Gemini API
python -c "import google.generativeai as genai; genai.configure(api_key='$GEMINI_API_KEY'); print('OK')"
```

### Model Download Issues

```bash
# Set HuggingFace token
export HF_TOKEN=your_token
huggingface-cli login --token $HF_TOKEN

# Use cache directory
--cache_dir /path/to/cache
```

## Comparing with Original

Both pipelines are supported:

**Legacy (original):**
```bash
python -m truthfulqa.evaluate \
  --models gpt2 neo-small \
  --metrics bleu rouge \
  --preset qa
```

**Modern (new):**
```bash
python -m truthfulqa.evaluate_vllm \
  --model_name_or_path gpt2 \
  --metrics gemini-judge \
  --preset qa
```

The new pipeline is recommended for:
- Better performance (vLLM)
- More reliable judging (Gemini)
- Any HuggingFace model
- Better UX

## Citation

If you use this evaluation pipeline, please cite the original TruthfulQA paper:

```bibtex
@inproceedings{lin2022truthfulqa,
  title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
  author={Lin, Stephanie and Hilton, Jacob and Evans, Owain},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  year={2022}
}
```

## License

Same as original TruthfulQA repository.
