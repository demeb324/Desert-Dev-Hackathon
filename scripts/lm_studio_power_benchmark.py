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

# Import optimizer for rule-based prompt optimization
from optimizer import optimize_prompt


# Benchmark prompt definitions
BENCHMARK_PROMPTS = {
    "short": [
        "Please, what is 2+2? Thank you.",
        "Could you please name three colors, thanks.",
        "I'm sorry, but could you say hello? No problem if not.",
    ],
    "medium": [
        "Please explain photosynthesis in simple terms, thank you very much.",
        "I'm sorry, but could you tell me what are the main causes of climate change? Thanks!",
        "Could you please describe how a computer processes information? Thank you.",
    ],
    "long": [
        "I'm sorry to bother you, but could you please write a detailed analysis of renewable energy sources, comparing solar, wind, and hydroelectric power in terms of efficiency, environmental impact, and scalability? Thank you very much.",
        "Please explain the historical significance of the Industrial Revolution and its lasting effects on modern society, including technological advancement, urbanization, and economic systems. Thanks!",
        "I'm sorry, but could you please describe the process of protein synthesis in cells, from DNA transcription to translation, including the roles of mRNA, tRNA, and ribosomes? Thank you so much.",
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
class OptimizationMetadata:
    """Metadata about prompt optimization."""
    method: str  # "none" | "rule_based" | "llm_self"
    optimization_prompt: Optional[str] = None  # For LLM self-optimization
    optimization_timing: Optional[TimingInfo] = None  # For LLM self-optimization
    original_prompt: Optional[str] = None  # Store original for comparison


@dataclass
class BenchmarkResult:
    """Results for a single benchmark prompt."""
    category: str
    prompt: str
    response: str
    timing: TimingInfo
    inference_stats: InferenceStats
    prompt_version: str = "original"  # "original" | "rule_optimized" | "llm_optimized"
    optimization_metadata: Optional[OptimizationMetadata] = None


def optimize_prompt_with_llm(
    model,
    original_prompt: str,
    optimization_instruction: str = "Rewrite this prompt to be more concise and direct while preserving its core meaning. Remove unnecessary polite words. Output only the optimized prompt, nothing else."
) -> tuple[str, TimingInfo]:
    """
    Use the LLM to optimize a prompt for conciseness.

    Args:
        model: LM Studio model instance
        original_prompt: The original prompt to optimize
        optimization_instruction: Instruction for how to optimize (can be customized)

    Returns:
        Tuple of (optimized_prompt, timing_info)
    """
    # Create the meta-prompt for optimization
    meta_prompt = f"{optimization_instruction}\n\nOriginal prompt: {original_prompt}\n\nOptimized prompt:"

    # Record optimization start time
    opt_start_time = datetime.now()
    opt_start_iso = opt_start_time.isoformat()

    print(f"  Requesting LLM optimization at: {opt_start_iso}")

    # Track first token time
    first_token_time = None
    first_token_iso = None

    def on_first_token():
        nonlocal first_token_time, first_token_iso
        first_token_time = datetime.now()
        first_token_iso = first_token_time.isoformat()

    # Run optimization inference
    inference_start = time.time()

    try:
        result = model.respond(
            meta_prompt,
            on_first_token=on_first_token,
        )
        optimized_prompt = result.content if hasattr(result, 'content') else str(result)
        # Clean up the response - take only the first line if multi-line
        optimized_prompt = optimized_prompt.strip().split('\n')[0]

    except Exception as e:
        print(f"  WARNING: LLM optimization failed: {e}")
        print(f"  Falling back to original prompt")
        optimized_prompt = original_prompt

    inference_end = time.time()
    inference_duration = inference_end - inference_start

    # Record completion time
    opt_complete_time = datetime.now()
    opt_complete_iso = opt_complete_time.isoformat()

    print(f"  LLM optimization completed at: {opt_complete_iso}")
    print(f"  Optimization duration: {inference_duration:.2f}s")
    print(f"  Optimized prompt: {optimized_prompt[:80]}{'...' if len(optimized_prompt) > 80 else ''}")

    # Calculate time to first token
    time_to_first_token_ms = None
    if first_token_time:
        ttft = (first_token_time - opt_start_time).total_seconds() * 1000
        time_to_first_token_ms = ttft

    # Create timing info for the optimization step
    timing = TimingInfo(
        prompt_submit_time=opt_start_iso,
        first_token_time=first_token_iso,
        response_complete_time=opt_complete_iso,
        inference_duration_s=inference_duration,
        time_to_first_token_ms=time_to_first_token_ms
    )

    return optimized_prompt, timing


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


def run_single_benchmark_with_versions(
    model,
    category: str,
    original_prompt: str,
    optimization_instruction: Optional[str] = None
) -> List[BenchmarkResult]:
    """
    Run benchmark with all three prompt versions: original, rule-optimized, and LLM-optimized.

    Args:
        model: LM Studio model instance
        category: Prompt category (short/medium/long)
        original_prompt: The original prompt text (with polite lexicon)
        optimization_instruction: Optional custom instruction for LLM optimization

    Returns:
        List of 3 BenchmarkResult objects (original, rule_optimized, llm_optimized)
    """
    results = []

    print(f"\n{'='*70}")
    print(f"PROMPT GROUP: {category.upper()}")
    print(f"Original prompt: {original_prompt[:60]}{'...' if len(original_prompt) > 60 else ''}")
    print(f"{'='*70}")

    # Version 1: Run with ORIGINAL prompt
    print(f"\n[1/3] Running ORIGINAL version...")
    result_original = run_single_benchmark(
        model=model,
        category=category,
        prompt=original_prompt
    )
    result_original.prompt_version = "original"
    result_original.optimization_metadata = OptimizationMetadata(
        method="none",
        original_prompt=original_prompt
    )
    results.append(result_original)

    # Version 2: Run with RULE-OPTIMIZED prompt
    print(f"\n[2/3] Running RULE-OPTIMIZED version...")
    rule_optimized_prompt = optimize_prompt(original_prompt)
    print(f"  Rule-optimized prompt: {rule_optimized_prompt[:80]}{'...' if len(rule_optimized_prompt) > 80 else ''}")

    result_rule = run_single_benchmark(
        model=model,
        category=category,
        prompt=rule_optimized_prompt
    )
    result_rule.prompt_version = "rule_optimized"
    result_rule.optimization_metadata = OptimizationMetadata(
        method="rule_based",
        original_prompt=original_prompt
    )
    results.append(result_rule)

    # Version 3: Run with LLM-OPTIMIZED prompt
    print(f"\n[3/3] Running LLM-OPTIMIZED version...")

    # First, get LLM to optimize the prompt
    if optimization_instruction:
        llm_optimized_prompt, opt_timing = optimize_prompt_with_llm(
            model=model,
            original_prompt=original_prompt,
            optimization_instruction=optimization_instruction
        )
    else:
        llm_optimized_prompt, opt_timing = optimize_prompt_with_llm(
            model=model,
            original_prompt=original_prompt
        )

    # Then run the benchmark with the LLM-optimized prompt
    result_llm = run_single_benchmark(
        model=model,
        category=category,
        prompt=llm_optimized_prompt
    )
    result_llm.prompt_version = "llm_optimized"
    result_llm.optimization_metadata = OptimizationMetadata(
        method="llm_self",
        optimization_prompt=f"Optimize: {original_prompt}",
        optimization_timing=opt_timing,
        original_prompt=original_prompt
    )
    results.append(result_llm)

    print(f"\n{'='*70}")
    print(f"Completed all 3 versions for: {category}")
    print(f"{'='*70}")

    return results


def run_benchmark_suite(
    model_name: Optional[str],
    pause_between_prompts: float,
    comparison_mode: bool = False,
    optimization_instruction: Optional[str] = None
) -> List[BenchmarkResult]:
    """
    Run full benchmark suite.

    Args:
        model_name: LM Studio model name (None for currently loaded)
        pause_between_prompts: Seconds to wait between prompts
        comparison_mode: If True, run original, rule-optimized, and LLM-optimized versions
        optimization_instruction: Custom instruction for LLM optimization (comparison mode only)

    Returns:
        List of benchmark results
    """
    print("="*70)
    print("LM STUDIO INFERENCE BENCHMARK")
    if comparison_mode:
        print("MODE: COMPARISON (Original + Rule-Optimized + LLM-Optimized)")
    else:
        print("MODE: STANDARD")
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

            if comparison_mode:
                # Run all 3 versions of the prompt
                print(f"\n\nProgress: Prompt Group {current}/{total_prompts}")
                version_results = run_single_benchmark_with_versions(
                    model=model,
                    category=category,
                    original_prompt=prompt,
                    optimization_instruction=optimization_instruction
                )
                results.extend(version_results)
            else:
                # Standard mode: run single version
                print(f"\n\nProgress: {current}/{total_prompts}")
                result = run_single_benchmark(
                    model=model,
                    category=category,
                    prompt=prompt
                )
                results.append(result)

            # Pause between prompt groups
            if current < total_prompts and pause_between_prompts > 0:
                print(f"\nPausing {pause_between_prompts}s before next prompt...")
                time.sleep(pause_between_prompts)

    return results


def generate_json_report(
    results: List[BenchmarkResult],
    output_file: str,
    model_name: Optional[str],
    benchmark_start_time: str,
    comparison_mode: bool = False
):
    """
    Generate JSON report from benchmark results.

    Args:
        results: List of benchmark results
        output_file: Output JSON file path
        model_name: Model name used
        benchmark_start_time: ISO timestamp of when benchmark started
        comparison_mode: If True, generate comparison-mode report structure
    """
    # Auto-detect comparison mode if not specified
    if not comparison_mode and results and hasattr(results[0], 'prompt_version'):
        comparison_mode = results[0].prompt_version != "original" or any(
            r.prompt_version != "original" for r in results
        )

    if comparison_mode:
        # Generate comparison-mode report
        report = generate_comparison_report(
            results, output_file, model_name, benchmark_start_time
        )
    else:
        # Generate standard report
        report = generate_standard_report(
            results, output_file, model_name, benchmark_start_time
        )

    # Write to file
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Report saved to: {output_file}")
    print(f"{'='*70}")

    if comparison_mode:
        print(f"\nComparison Mode Report:")
        print(f"- Total base prompts: {report['summary']['total_base_prompts']}")
        print(f"- Total executions: {report['summary']['total_executions']}")
        print(f"- Versions compared: original, rule_optimized, llm_optimized")
    else:
        print(f"\nStandard Mode Report:")
        print(f"- Total prompts: {report['summary']['total_prompts']}")

    print(f"\nTo correlate with power data:")
    print(f"1. Find power samples matching timestamps in this report")
    print(f"2. Benchmark started at: {benchmark_start_time}")
    print(f"3. Each prompt has prompt_submit_time and response_complete_time")


def generate_standard_report(
    results: List[BenchmarkResult],
    output_file: str,
    model_name: Optional[str],
    benchmark_start_time: str
) -> Dict:
    """Generate standard (non-comparison) JSON report."""
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
        "comparison_mode": False,
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

    return report


def generate_comparison_report(
    results: List[BenchmarkResult],
    output_file: str,
    model_name: Optional[str],
    benchmark_start_time: str
) -> Dict:
    """Generate comparison-mode JSON report with grouped prompt versions."""
    # Group results by original prompt (every 3 results is one group)
    prompt_groups = []
    group_id = 0

    for i in range(0, len(results), 3):
        group_results = results[i:i+3]
        if not group_results:
            continue

        # Get the original prompt from metadata
        original_prompt = group_results[0].optimization_metadata.original_prompt
        category = group_results[0].category

        # Build version data
        versions = []
        for result in group_results:
            version_data = {
                "prompt_version": result.prompt_version,
                "prompt": result.prompt,
                "response": result.response,
                "timing": asdict(result.timing),
                "inference_stats": asdict(result.inference_stats),
            }

            # Add optimization metadata if present
            if result.optimization_metadata:
                opt_meta = {
                    "method": result.optimization_metadata.method,
                }
                if result.optimization_metadata.optimization_prompt:
                    opt_meta["optimization_prompt"] = result.optimization_metadata.optimization_prompt
                if result.optimization_metadata.optimization_timing:
                    opt_meta["optimization_timing"] = asdict(result.optimization_metadata.optimization_timing)
                version_data["optimization_metadata"] = opt_meta

            versions.append(version_data)

        prompt_group = {
            "group_id": group_id,
            "category": category,
            "base_prompt": original_prompt,
            "versions": versions
        }
        prompt_groups.append(prompt_group)
        group_id += 1

    # Calculate summary statistics by version
    def calc_stats_for_version(version_name: str):
        version_results = [r for r in results if r.prompt_version == version_name]
        if not version_results:
            return None

        token_speeds = [r.inference_stats.tokens_per_second for r in version_results
                       if r.inference_stats.tokens_per_second]
        ttfts = [r.timing.time_to_first_token_ms for r in version_results
                if r.timing.time_to_first_token_ms]

        return {
            "count": len(version_results),
            "avg_tokens_per_second": sum(token_speeds) / len(token_speeds) if token_speeds else None,
            "avg_time_to_first_token_ms": sum(ttfts) / len(ttfts) if ttfts else None,
            "total_duration_s": sum(r.timing.inference_duration_s for r in version_results)
        }

    # Calculate LLM optimization overhead
    llm_opt_overhead_s = 0
    llm_optimized = [r for r in results if r.prompt_version == "llm_optimized"]
    for result in llm_optimized:
        if result.optimization_metadata and result.optimization_metadata.optimization_timing:
            llm_opt_overhead_s += result.optimization_metadata.optimization_timing.inference_duration_s

    summary = {
        "total_base_prompts": len(prompt_groups),
        "total_executions": len(results),
        "by_version": {
            "original": calc_stats_for_version("original"),
            "rule_optimized": calc_stats_for_version("rule_optimized"),
            "llm_optimized": calc_stats_for_version("llm_optimized"),
        },
        "optimization_overhead": {
            "total_llm_optimization_time_s": llm_opt_overhead_s,
            "avg_llm_optimization_time_s": llm_opt_overhead_s / len(llm_optimized) if llm_optimized else None
        }
    }

    # Build report structure
    report = {
        "benchmark_id": benchmark_start_time,
        "model": model_name or "default",
        "comparison_mode": True,
        "summary": summary,
        "prompt_groups": prompt_groups
    }

    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run LM Studio inference benchmark with detailed timing information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run standard benchmark with currently loaded model
  python3 lm_studio_power_benchmark.py

  # Run with specific model
  python3 lm_studio_power_benchmark.py --model qwen/qwen3-4b-2507

  # Run comparison mode (original + rule-optimized + LLM-optimized)
  python3 lm_studio_power_benchmark.py --compare

  # Comparison mode with custom optimization instruction
  python3 lm_studio_power_benchmark.py --compare --optimization-prompt "Make this brief:"

  # Custom timing and output
  python3 lm_studio_power_benchmark.py --pause 10 --output my_results.json

Power Measurement Workflow:
  Terminal 1: sudo python3 powermetrics_analyzer.py --samples 3000 --output power.csv
  Terminal 2: python3 lm_studio_power_benchmark.py --compare --output comparison.json

  Then correlate timestamps from comparison.json with power.csv using:
  python3 correlate_power_timing.py comparison.json power.csv

Note: This script requires LM Studio running with a model loaded.
      Power measurement is done separately (no sudo needed for this script).
      Comparison mode runs 3x the prompts (original, rule-optimized, llm-optimized).
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

    parser.add_argument(
        '--compare',
        action='store_true',
        help='Enable comparison mode: runs original, rule-optimized, and LLM-optimized versions of each prompt'
    )

    parser.add_argument(
        '--optimization-prompt',
        type=str,
        default=None,
        metavar='INSTRUCTION',
        help='Custom instruction for LLM self-optimization (only used with --compare)'
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
            pause_between_prompts=args.pause,
            comparison_mode=args.compare,
            optimization_instruction=args.optimization_prompt
        )

        # Generate report
        generate_json_report(
            results=results,
            output_file=args.output,
            model_name=args.model,
            benchmark_start_time=benchmark_start_time,
            comparison_mode=args.compare
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
