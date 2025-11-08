#!/usr/bin/env python3
"""
LM Studio Inference Benchmark Tool

This script runs a benchmark suite against LM Studio models and records detailed
timing information for each inference. Power measurements should be collected
separately using powermetrics_analyzer.py and correlated via timestamps.
"""

import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

try:
    import lmstudio as lms
except ImportError:
    print("ERROR: lmstudio package not found.")
    print("Install with: pip install lmstudio")
    exit(1)


# Benchmark prompt definitions
BENCHMARK_PROMPTS = {
    "short": [
        "What is 2+2?",
        "Name three colors.",
        "Say hello.",
    ],
    "medium": [
        "Explain photosynthesis in simple terms.",
        "What are the main causes of climate change?",
        "Describe how a computer processes information.",
    ],
    "long": [
        "Write a detailed analysis of renewable energy sources, comparing solar, wind, and hydroelectric power in terms of efficiency, environmental impact, and scalability.",
        "Explain the historical significance of the Industrial Revolution and its lasting effects on modern society, including technological advancement, urbanization, and economic systems.",
        "Describe the process of protein synthesis in cells, from DNA transcription to translation, including the roles of mRNA, tRNA, and ribosomes.",
    ]
}


@dataclass
class TimingInfo:
    """Timing information for inference."""
    prompt_submit_time: str  # ISO timestamp
    first_token_time: Optional[str]  # ISO timestamp
    response_complete_time: str  # ISO timestamp
    inference_duration_s: float
    time_to_first_token_ms: Optional[float]


@dataclass
class InferenceStats:
    """LLM inference statistics."""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    tokens_per_second: Optional[float] = None


@dataclass
class BenchmarkResult:
    """Results for a single benchmark prompt."""
    category: str
    prompt: str
    response: str
    timing: TimingInfo
    inference_stats: InferenceStats


def run_single_benchmark(
    model,
    category: str,
    prompt: str
) -> BenchmarkResult:
    """
    Run a single benchmark and record timing information.

    Args:
        model: LM Studio model instance
        category: Prompt category (short/medium/long)
        prompt: The prompt text

    Returns:
        BenchmarkResult with timing and inference metrics
    """
    print(f"\n{'='*70}")
    print(f"Running benchmark: {category.upper()}")
    print(f"Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
    print(f"{'='*70}")

    # Record prompt submission time
    prompt_submit_time = datetime.now()
    prompt_submit_iso = prompt_submit_time.isoformat()

    print(f"Prompt submitted at: {prompt_submit_iso}")

    # Track first token time
    first_token_time = None
    first_token_iso = None

    def on_first_token():
        nonlocal first_token_time, first_token_iso
        first_token_time = datetime.now()
        first_token_iso = first_token_time.isoformat()

    # Run inference
    inference_start = time.time()

    try:
        result = model.respond(
            prompt,
            on_first_token=on_first_token,
        )
        response_text = result.content if hasattr(result, 'content') else str(result)

        # Extract stats from result
        inference_stats = InferenceStats()
        if hasattr(result, 'stats'):
            stats = result.stats
            if hasattr(stats, 'prompt_tokens'):
                inference_stats.prompt_tokens = stats.prompt_tokens
            if hasattr(stats, 'predicted_tokens'):
                inference_stats.completion_tokens = stats.predicted_tokens
            if hasattr(stats, 'tokens_per_second'):
                inference_stats.tokens_per_second = stats.tokens_per_second

    except Exception as e:
        print(f"ERROR during inference: {e}")
        response_text = f"ERROR: {e}"
        inference_stats = InferenceStats()

    inference_end = time.time()
    inference_duration = inference_end - inference_start

    # Record completion time
    response_complete_time = datetime.now()
    response_complete_iso = response_complete_time.isoformat()

    print(f"Response completed at: {response_complete_iso}")
    print(f"Inference duration: {inference_duration:.2f}s")

    # Calculate time to first token
    time_to_first_token_ms = None
    if first_token_time:
        ttft = (first_token_time - prompt_submit_time).total_seconds() * 1000
        time_to_first_token_ms = ttft
        print(f"Time to first token: {ttft:.1f}ms")

    if inference_stats.tokens_per_second:
        print(f"Speed: {inference_stats.tokens_per_second:.1f} tokens/s")

    # Create timing info
    timing = TimingInfo(
        prompt_submit_time=prompt_submit_iso,
        first_token_time=first_token_iso,
        response_complete_time=response_complete_iso,
        inference_duration_s=inference_duration,
        time_to_first_token_ms=time_to_first_token_ms
    )

    return BenchmarkResult(
        category=category,
        prompt=prompt,
        response=response_text,
        timing=timing,
        inference_stats=inference_stats
    )


def run_benchmark_suite(
    model_name: Optional[str],
    pause_between_prompts: float
) -> List[BenchmarkResult]:
    """
    Run full benchmark suite.

    Args:
        model_name: LM Studio model name (None for currently loaded)
        pause_between_prompts: Seconds to wait between prompts

    Returns:
        List of benchmark results
    """
    print("="*70)
    print("LM STUDIO INFERENCE BENCHMARK")
    print("="*70)
    print(f"\nBenchmark start time: {datetime.now().isoformat()}")

    # Connect to LM Studio
    print("\nConnecting to LM Studio...")
    try:
        if model_name:
            print(f"Loading model: {model_name}")
            model = lms.llm(model_name)
        else:
            print("Using currently loaded model...")
            model = lms.llm()
    except Exception as e:
        raise RuntimeError(f"Failed to connect to LM Studio: {e}\n"
                         f"Make sure LM Studio is running with a model loaded.")

    print("✓ Connected successfully")

    # Run benchmarks for all categories
    results = []
    total_prompts = sum(len(prompts) for prompts in BENCHMARK_PROMPTS.values())
    current = 0

    for category in ["short", "medium", "long"]:
        for prompt in BENCHMARK_PROMPTS[category]:
            current += 1
            print(f"\n\nProgress: {current}/{total_prompts}")

            result = run_single_benchmark(
                model=model,
                category=category,
                prompt=prompt
            )
            results.append(result)

            # Pause between benchmarks
            if current < total_prompts and pause_between_prompts > 0:
                print(f"\nPausing {pause_between_prompts}s before next benchmark...")
                time.sleep(pause_between_prompts)

    return results


def generate_json_report(
    results: List[BenchmarkResult],
    output_file: str,
    model_name: Optional[str],
    benchmark_start_time: str
):
    """
    Generate JSON report from benchmark results.

    Args:
        results: List of benchmark results
        output_file: Output JSON file path
        model_name: Model name used
        benchmark_start_time: ISO timestamp of when benchmark started
    """
    # Calculate summary statistics
    avg_tokens_per_second = None
    token_speeds = [r.inference_stats.tokens_per_second for r in results
                   if r.inference_stats.tokens_per_second]
    if token_speeds:
        avg_tokens_per_second = sum(token_speeds) / len(token_speeds)

    avg_ttft = None
    ttfts = [r.timing.time_to_first_token_ms for r in results
            if r.timing.time_to_first_token_ms]
    if ttfts:
        avg_ttft = sum(ttfts) / len(ttfts)

    summary = {
        "total_prompts": len(results),
        "avg_tokens_per_second": avg_tokens_per_second,
        "avg_time_to_first_token_ms": avg_ttft,
        "total_duration_s": sum(r.timing.inference_duration_s for r in results)
    }

    # Build report structure
    report = {
        "benchmark_id": benchmark_start_time,
        "model": model_name or "default",
        "summary": summary,
        "prompts": []
    }

    # Add each result
    for result in results:
        prompt_data = {
            "category": result.category,
            "prompt": result.prompt,
            "response": result.response,
            "timing": asdict(result.timing),
            "inference_stats": asdict(result.inference_stats),
        }
        report["prompts"].append(prompt_data)

    # Write to file
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Report saved to: {output_file}")
    print(f"{'='*70}")
    print(f"\nTo correlate with power data:")
    print(f"1. Find power samples matching timestamps in this report")
    print(f"2. Benchmark started at: {benchmark_start_time}")
    print(f"3. Each prompt has prompt_submit_time and response_complete_time")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run LM Studio inference benchmark with detailed timing information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark with currently loaded model
  python3 lm_studio_power_benchmark.py

  # Run with specific model
  python3 lm_studio_power_benchmark.py --model qwen/qwen3-4b-2507

  # Custom timing and output
  python3 lm_studio_power_benchmark.py --pause 10 --output my_results.json

Power Measurement Workflow:
  Terminal 1: sudo python3 powermetrics_analyzer.py --samples 1000 --output power.csv
  Terminal 2: python3 lm_studio_power_benchmark.py --output benchmark.json

  Then correlate timestamps from benchmark.json with power.csv

Note: This script requires LM Studio running with a model loaded.
      Power measurement is done separately (no sudo needed for this script).
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='LM Studio model identifier (uses currently loaded model if not specified)'
    )

    parser.add_argument(
        '--pause',
        type=float,
        default=5.0,
        metavar='SECONDS',
        help='Pause duration between prompts in seconds (default: 5)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        metavar='FILE',
        help='Output JSON file path (default: lm_studio_benchmark_<timestamp>.json)'
    )

    args = parser.parse_args()

    benchmark_start_time = datetime.now().isoformat()

    # Generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"lm_studio_benchmark_{timestamp}.json"

    try:
        # Run benchmark suite
        results = run_benchmark_suite(
            model_name=args.model,
            pause_between_prompts=args.pause
        )

        # Generate report
        generate_json_report(
            results=results,
            output_file=args.output,
            model_name=args.model,
            benchmark_start_time=benchmark_start_time
        )

        print("\n✓ Benchmark completed successfully!")

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        import sys
        sys.exit(130)

    except Exception as e:
        import sys
        import traceback
        print(f"\nERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
