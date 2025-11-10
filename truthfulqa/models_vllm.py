"""
Modern model inference using vLLM for efficient batched generation.
"""

import pandas as pd
import numpy as np
import warnings
from typing import List, Optional, Dict
from tqdm import tqdm

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    warnings.warn("vLLM not available. Install: pip install vllm", stacklevel=2)

from .utilities import format_prompt


def run_vllm_answers(
    frame: pd.DataFrame,
    model_name_or_path: str,
    tag: str,
    preset: str = 'qa',
    batch_size: int = 32,
    max_tokens: int = 50,
    temperature: float = 0.0,
    top_p: float = 1.0,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "auto",
    quantization: Optional[str] = None,
    cache_dir: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate answers using vLLM for efficient batched inference.

    Args:
        frame: DataFrame containing questions
        model_name_or_path: HuggingFace model ID or local path
        tag: Column name to store answers
        preset: Prompt preset to use (qa, chat, long, etc.)
        batch_size: Number of prompts to process in parallel
        max_tokens: Maximum tokens to generate per answer
        temperature: Sampling temperature (0 for greedy)
        top_p: Nucleus sampling parameter
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use
        dtype: Data type (auto, float16, bfloat16, float32)
        quantization: Quantization method (awq, gptq, squeezellm, None)
        cache_dir: Directory to cache model weights
        verbose: Print progress and outputs

    Returns:
        Updated DataFrame with generated answers
    """

    if not VLLM_AVAILABLE:
        raise ImportError("vLLM not available. Install: pip install vllm")

    # Initialize column
    if tag not in frame.columns:
        frame[tag] = ''

    frame[tag].fillna('', inplace=True)
    frame[tag] = frame[tag].astype(str)

    # Collect questions that need answers
    indices_to_process = []
    prompts = []

    for idx in frame.index:
        if pd.isnull(frame.loc[idx, tag]) or not len(frame.loc[idx, tag]):
            prompt = format_prompt(frame.loc[idx], preset, format='general')
            if prompt is not None:
                indices_to_process.append(idx)
                prompts.append(prompt)

    if not prompts:
        if verbose:
            print(f"No prompts to process for {tag}")
        return frame

    if verbose:
        print(f"Loading model: {model_name_or_path}")
        print(f"  Tensor parallel size: {tensor_parallel_size}")
        print(f"  GPU memory utilization: {gpu_memory_utilization}")
        print(f"  Dtype: {dtype}")
        print(f"  Quantization: {quantization}")

    # Initialize vLLM model
    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        quantization=quantization,
        download_dir=cache_dir,
        trust_remote_code=True,
    )

    # Configure sampling
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["\n\nQ:", "\n\n"],  # Stop at next question or double newline
    )

    if verbose:
        print(f"Generating {len(prompts)} answers with batch_size={batch_size}")

    # Generate answers in batches
    outputs = llm.generate(prompts, sampling_params, use_tqdm=verbose)

    # Store results
    for idx, output in zip(indices_to_process, outputs):
        generated_text = output.outputs[0].text.strip()

        # Clean up output
        generated_text = generated_text.replace('\n\n', ' ')

        frame.loc[idx, tag] = generated_text

        if verbose:
            print(f"\nQuestion {idx}: {frame.loc[idx, 'Question'][:80]}...")
            print(f"Answer: {generated_text[:200]}...")

    if verbose:
        print(f"\nCompleted generation for {len(prompts)} questions")

    return frame


def run_vllm_probs(
    frame: pd.DataFrame,
    model_name_or_path: str,
    tag: str,
    preset: str = 'qa',
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "auto",
    quantization: Optional[str] = None,
    cache_dir: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run multiple-choice evaluation using vLLM to compute probabilities.

    Args:
        frame: DataFrame containing questions and reference answers
        model_name_or_path: HuggingFace model ID or local path
        tag: Column name prefix for storing scores
        preset: Prompt preset to use
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use
        dtype: Data type (auto, float16, bfloat16, float32)
        quantization: Quantization method (awq, gptq, squeezellm, None)
        cache_dir: Directory to cache model weights
        verbose: Print progress

    Returns:
        Updated DataFrame with MC scores
    """

    if not VLLM_AVAILABLE:
        raise ImportError("vLLM not available. Install: pip install vllm")

    # Note: Multiple-choice with log probabilities is more complex in vLLM
    # This is a placeholder - full implementation would require using
    # the logprobs parameter and computing scores for each answer choice
    warnings.warn(
        "Multiple-choice evaluation with vLLM is not yet fully implemented. "
        "Consider using the generative evaluation with Gemini judge instead.",
        stacklevel=2
    )

    return frame


def batch_prompts(prompts: List[str], batch_size: int) -> List[List[str]]:
    """
    Split prompts into batches.

    Args:
        prompts: List of prompt strings
        batch_size: Size of each batch

    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(prompts), batch_size):
        batches.append(prompts[i:i + batch_size])
    return batches


def format_vllm_stats(outputs) -> Dict:
    """
    Extract statistics from vLLM outputs.

    Args:
        outputs: vLLM RequestOutput objects

    Returns:
        Dictionary with generation statistics
    """
    stats = {
        'total_tokens': 0,
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'num_outputs': len(outputs)
    }

    for output in outputs:
        stats['prompt_tokens'] += len(output.prompt_token_ids)
        stats['completion_tokens'] += sum(len(o.token_ids) for o in output.outputs)
        stats['total_tokens'] = stats['prompt_tokens'] + stats['completion_tokens']

    return stats
