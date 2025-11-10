"""
Modern evaluation script using vLLM for inference and Gemini for judging.

This script provides a cleaner, more efficient evaluation pipeline:
- Uses vLLM for fast batched inference
- Supports any HuggingFace model
- Uses Gemini API for judging (more reliable than fine-tuned GPT-3)
- Separates answer generation from evaluation
- Saves intermediate results for resumability
- Better progress tracking and logging
"""

import argparse
import os
import sys
import warnings
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# Local imports
from . import utilities
from . import metrics
from .models_vllm import run_vllm_answers


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/eval.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate language models on TruthfulQA using vLLM and Gemini judge',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model configuration
    parser.add_argument(
        '--model',
        '--model_name_or_path',
        type=str,
        required=True,
        dest='model_name_or_path',
        help='HuggingFace model ID or local path (e.g., deepseek-ai/deepseek-coder-33b-instruct)'
    )
    parser.add_argument(
        '--model_tag',
        type=str,
        default=None,
        help='Column name for storing answers (default: use model name)'
    )

    # Data paths
    parser.add_argument(
        '--questions',
        '--input_path',
        type=str,
        default='TruthfulQA.csv',
        dest='input_path',
        help='Path to input CSV with questions'
    )
    parser.add_argument(
        '--output',
        '--output_path',
        type=str,
        default=None,
        dest='output_path',
        help='Path to save detailed results CSV (e.g., outputs/deepseek_coder_33b_results.csv)'
    )
    parser.add_argument(
        '--summary_path',
        type=str,
        default=None,
        help='Path to save summary results CSV (default: derived from output path)'
    )

    # Prompt configuration
    parser.add_argument(
        '--preset',
        type=str,
        default='qa',
        choices=['qa', 'null', 'chat', 'long', 'help', 'harm'],
        help='Prompt preset to use'
    )

    # Generation parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for generation'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=50,
        help='Maximum tokens to generate per answer'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature (0 for greedy decoding)'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=1.0,
        help='Nucleus sampling parameter'
    )

    # vLLM configuration
    parser.add_argument(
        '--tensor_parallel_size',
        type=int,
        default=1,
        help='Number of GPUs for tensor parallelism'
    )
    parser.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=0.9,
        help='Fraction of GPU memory to use (0.0-1.0)'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'float16', 'bfloat16', 'float32'],
        help='Data type for model weights'
    )
    parser.add_argument(
        '--quantization',
        type=str,
        default=None,
        choices=['awq', 'gptq', 'squeezellm'],
        help='Quantization method to use'
    )

    # Metrics to run
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=['gemini-judge'],
        choices=['gemini-judge', 'gemini-info', 'bleurt', 'bleu', 'rouge'],
        help='Metrics to compute'
    )
    parser.add_argument(
        '--gemini_model',
        type=str,
        default='gemini-2.0-flash-exp',
        help='Gemini model to use for judging (e.g., gemini-2.0-flash-exp, gemini-1.5-flash)'
    )

    # Other options
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help='Directory to cache model weights'
    )
    parser.add_argument(
        '--skip_generation',
        action='store_true',
        help='Skip generation and only run metrics on existing answers'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress and outputs'
    )

    return parser.parse_args()


def format_frame(results):
    """Format results DataFrame with multi-level columns."""
    # Filter out non-numeric columns
    results = results[[x for x in results.columns if (x != 'Context') and (results[x].dtype != 'O')]]

    # Create multi-level column names
    new_cols = []
    for col in results.columns:
        split = col.split(' ')
        new_cols.append((split[0], ' '.join(split[1:])))
    results.columns = pd.MultiIndex.from_tuples(new_cols)

    return results


def compute_summary(results):
    """Compute summary statistics from detailed results."""
    formatted = format_frame(results.copy())
    summary = formatted.mean(axis=0)
    summary = summary.reset_index().rename(columns={
        'level_0': 'Model',
        'level_1': 'Metric',
        0: 'Value'
    })

    # Filter to most important metrics
    important_metrics = [
        'MC1', 'MC2', 'MC3',
        'Gemini-judge', 'Gemini-judge acc',
        'Gemini-info', 'Gemini-info acc',
        'BLEURT acc',
        'bleu acc',
        'rouge1 acc'
    ]

    summary = summary[summary['Metric'].isin(important_metrics)]

    # Pivot for easier reading
    summary_pivot = pd.pivot_table(summary, 'Value', 'Model', 'Metric')

    return summary_pivot


def main():
    """Main evaluation pipeline."""
    args = parse_args()

    # Determine model tag
    if args.model_tag is None:
        args.model_tag = args.model_name_or_path.split('/')[-1]

    # Set default output path if not provided
    if args.output_path is None:
        # Create default output filename from model name
        model_safe_name = args.model_tag.replace('/', '_').replace('-', '_')
        args.output_path = f'outputs/{model_safe_name}_results.csv'

    # Derive summary path from output path if not provided
    if args.summary_path is None:
        output_path_obj = Path(args.output_path)
        summary_filename = output_path_obj.stem.replace('_results', '_summary') + output_path_obj.suffix
        args.summary_path = str(output_path_obj.parent / summary_filename)

    # Create output directories
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("TruthfulQA Evaluation with vLLM and Gemini")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Model tag: {args.model_tag}")
    logger.info(f"Input: {args.input_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Preset: {args.preset}")
    logger.info(f"Metrics: {', '.join(args.metrics)}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Tensor parallel size: {args.tensor_parallel_size}")
    logger.info("=" * 60)

    # Check for required API keys
    if 'gemini-judge' in args.metrics or 'gemini-info' in args.metrics:
        if not os.environ.get('GEMINI_API_KEY'):
            logger.error("GEMINI_API_KEY environment variable not set!")
            logger.error("Please set it: export GEMINI_API_KEY=your_key")
            sys.exit(1)

    # Load questions
    logger.info("Loading questions...")
    questions = utilities.load_questions(filename=args.input_path)
    logger.info(f"Loaded {len(questions)} questions")

    # Generate answers (unless skipping)
    if not args.skip_generation:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: Generating Answers")
        logger.info("=" * 60)

        try:
            questions = run_vllm_answers(
                frame=questions,
                model_name_or_path=args.model_name_or_path,
                tag=args.model_tag,
                preset=args.preset,
                batch_size=args.batch_size,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                dtype=args.dtype,
                quantization=args.quantization,
                cache_dir=args.cache_dir,
                verbose=args.verbose
            )

            # Save intermediate results
            utilities.save_questions(questions, args.output_path)
            logger.info(f"Saved answers to {args.output_path}")

        except Exception as err:
            logger.error(f"Error during generation: {err}", exc_info=True)
            sys.exit(1)

    else:
        logger.info("Skipping generation (--skip_generation flag set)")

    # Check that answers exist
    if args.model_tag not in questions.columns:
        logger.error(f"Column '{args.model_tag}' not found in dataframe!")
        logger.error("Available columns: " + ", ".join(questions.columns))
        sys.exit(1)

    # Run metrics
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Running Metrics")
    logger.info("=" * 60)

    for metric in args.metrics:
        logger.info(f"\nRunning metric: {metric}")

        try:
            if metric == 'gemini-judge':
                questions = metrics.run_gemini_judge_truth(
                    model_key=args.model_tag,
                    frame=questions,
                    model_name=args.gemini_model,
                    cache_dir=args.cache_dir
                )

            elif metric == 'gemini-info':
                questions = metrics.run_gemini_judge_info(
                    model_key=args.model_tag,
                    frame=questions,
                    model_name=args.gemini_model,
                    cache_dir=args.cache_dir
                )

            elif metric == 'bleurt':
                questions = metrics.run_BLEURT(
                    model_key=args.model_tag,
                    frame=questions,
                    cache_dir=args.cache_dir
                )

            elif metric in ['bleu', 'rouge']:
                questions = metrics.run_bleu_and_rouge(
                    model_key=args.model_tag,
                    frame=questions
                )

            # Save after each metric
            utilities.save_questions(questions, args.output_path)
            logger.info(f"Saved results after {metric}")

        except Exception as err:
            logger.error(f"Error running metric {metric}: {err}", exc_info=True)
            continue

    # Save final results
    utilities.save_questions(questions, args.output_path)
    logger.info(f"\nSaved detailed results to {args.output_path}")

    # Compute and save summary
    try:
        summary = compute_summary(questions)
        summary.to_csv(args.summary_path)
        logger.info(f"Saved summary to {args.summary_path}")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY RESULTS")
        logger.info("=" * 60)
        print("\n" + str(summary))

    except Exception as err:
        logger.warning(f"Could not generate summary: {err}")

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
