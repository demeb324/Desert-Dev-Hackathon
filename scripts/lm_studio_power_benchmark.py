#!/usr/bin/env python3
"""
LM Studio Power Consumption Benchmark Tool

This script runs a benchmark suite against LM Studio models while measuring
power consumption before, during, and after inference using macOS powermetrics.
"""

import json
import time
import argparse
import threading
import subprocess
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

try:
    import lmstudio as lms
except ImportError:
    print("ERROR: lmstudio package not found.")
    print("Install with: pip install lmstudio")
    exit(1)

# Import our powermetrics parser
sys.path.insert(0, os.path.dirname(__file__))
from powermetrics_analyzer import PowerMetricsParser


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
class PowerStats:
    """Power consumption statistics for a phase."""
    min_mw: int
    max_mw: int
    avg_mw: float
    sample_count: int


@dataclass
class InferenceStats:
    """LLM inference statistics."""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    tokens_per_second: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None
    inference_duration_s: Optional[float] = None


@dataclass
class BenchmarkResult:
    """Results for a single benchmark prompt."""
    category: str
    prompt: str
    response: str
    baseline_duration_s: float
    inference_duration_s: float
    cooldown_duration_s: float
    inference_stats: InferenceStats
    baseline_power: PowerStats
    inference_power: PowerStats
    cooldown_power: PowerStats
    raw_power_samples: List[Dict]


class PowerMetricsCollector:
    """Background power metrics collector using subprocess."""

    def __init__(self, interval_ms: int = 1000):
        """
        Initialize the collector.

        Args:
            interval_ms: Sampling interval in milliseconds
        """
        self.interval_ms = interval_ms
        self.parser = PowerMetricsParser()
        self.process: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.samples: List[Dict] = []
        self.lock = threading.Lock()
        self.start_time: Optional[float] = None

    def start(self):
        """Start collecting power metrics in background."""
        if self.running:
            return

        # Don't use sudo here - the script should already be run with sudo
        cmd = [
            "powermetrics",
            "-i", str(self.interval_ms),
            "-n", "0",  # Infinite samples (we'll stop manually)
            "--samplers", "cpu_power,gpu_power,ane_power",
            "-f", "text",
            "-b", "1"  # Line buffering
        ]

        self.running = True
        self.start_time = time.time()

        try:
            # Use line-buffered mode without universal_newlines for better thread compatibility
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # Unbuffered for real-time reading in thread
            )

            # Start background thread to read output
            self.thread = threading.Thread(target=self._read_output, daemon=True)
            self.thread.start()

            # Start stderr monitoring thread
            self.stderr_thread = threading.Thread(target=self._monitor_stderr, daemon=True)
            self.stderr_thread.start()

            # Give powermetrics a moment to start and check for immediate errors
            time.sleep(0.5)
            if self.process.poll() is not None:
                # Process already exited - there was an error
                self.running = False
                raise RuntimeError("powermetrics failed to start. Make sure the script is run with sudo.")

        except Exception as e:
            self.running = False
            if self.process:
                self.process.terminate()
            raise RuntimeError(f"Failed to start powermetrics: {e}")

    def _read_output(self):
        """Read powermetrics output in background thread."""
        # Read bytes and decode line by line for better threading compatibility
        for line_bytes in iter(self.process.stdout.readline, b''):
            if not self.running:
                break

            try:
                line = line_bytes.decode('utf-8')
            except UnicodeDecodeError:
                continue

            sample = self.parser.parse_line(line)
            if sample:
                # Add relative timestamp
                sample['relative_time_s'] = time.time() - self.start_time

                with self.lock:
                    self.samples.append(sample)

    def _monitor_stderr(self):
        """Monitor stderr for errors in background thread."""
        if not self.process or not self.process.stderr:
            return

        for line_bytes in iter(self.process.stderr.readline, b''):
            try:
                line = line_bytes.decode('utf-8').strip()
            except UnicodeDecodeError:
                continue

            if line:
                print(f"powermetrics error: {line}", file=sys.stderr)
                if "must be invoked as the superuser" in line.lower():
                    print("\nERROR: This script must be run with sudo!", file=sys.stderr)
                    self.running = False
                    break

    def stop(self) -> List[Dict]:
        """
        Stop collecting and return all samples.

        Returns:
            List of power samples
        """
        if not self.running:
            return []

        self.running = False

        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)

        if self.thread:
            self.thread.join(timeout=5)

        # Finalize parser
        final_sample = self.parser.finalize()
        if final_sample:
            final_sample['relative_time_s'] = time.time() - self.start_time
            with self.lock:
                self.samples.append(final_sample)

        return self.samples.copy()

    def get_samples(self) -> List[Dict]:
        """Get current samples without stopping collection."""
        with self.lock:
            return self.samples.copy()


def calculate_power_stats(samples: List[Dict]) -> Optional[PowerStats]:
    """
    Calculate power statistics from samples.

    Args:
        samples: List of power samples

    Returns:
        PowerStats or None if no valid samples
    """
    if not samples:
        return None

    combined_powers = [s['combined_power_mw'] for s in samples
                      if s.get('combined_power_mw') is not None]

    if not combined_powers:
        return None

    return PowerStats(
        min_mw=min(combined_powers),
        max_mw=max(combined_powers),
        avg_mw=sum(combined_powers) / len(combined_powers),
        sample_count=len(combined_powers)
    )


def extract_samples_by_time(
    all_samples: List[Dict],
    start_time: float,
    end_time: float
) -> List[Dict]:
    """
    Extract samples within a time range.

    Args:
        all_samples: All collected samples
        start_time: Start time (relative seconds)
        end_time: End time (relative seconds)

    Returns:
        Samples within the time range
    """
    return [
        s for s in all_samples
        if start_time <= s.get('relative_time_s', 0) <= end_time
    ]


def run_single_benchmark(
    model,
    category: str,
    prompt: str,
    baseline_duration_s: float,
    cooldown_duration_s: float
) -> BenchmarkResult:
    """
    Run a single benchmark with power measurement.

    Args:
        model: LM Studio model instance
        category: Prompt category (short/medium/long)
        prompt: The prompt text
        baseline_duration_s: Baseline collection duration
        cooldown_duration_s: Cooldown collection duration

    Returns:
        BenchmarkResult with all metrics
    """
    print(f"\n{'='*70}")
    print(f"Running benchmark: {category.upper()}")
    print(f"Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
    print(f"{'='*70}")

    # Start power collection
    collector = PowerMetricsCollector(interval_ms=1000)
    print("Starting power metrics collection...")
    collector.start()

    # Wait a bit and verify samples are being collected
    time.sleep(2)
    initial_samples = len(collector.get_samples())
    if initial_samples == 0:
        print("WARNING: No power samples collected yet. Power metrics may not be working!")
        print("Make sure you ran this script with sudo.")
    else:
        print(f"✓ Power collection working ({initial_samples} samples collected)")

    # Phase 1: Baseline
    print(f"\n[1/3] Collecting baseline ({baseline_duration_s}s)...")
    baseline_start = time.time()
    time.sleep(baseline_duration_s - 2)  # Subtract the 2s we already waited
    baseline_end = time.time()
    baseline_duration = baseline_end - baseline_start

    # Phase 2: Inference
    print(f"[2/3] Running inference...")
    inference_start = time.time()

    # Track first token time
    first_token_time = None
    def on_first_token():
        nonlocal first_token_time
        first_token_time = time.time()

    # Run inference
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

    # Calculate time to first token
    if first_token_time:
        inference_stats.time_to_first_token_ms = (first_token_time - inference_start) * 1000
    inference_stats.inference_duration_s = inference_duration

    print(f"   Inference completed in {inference_duration:.2f}s")
    if inference_stats.tokens_per_second:
        print(f"   Speed: {inference_stats.tokens_per_second:.1f} tokens/s")

    # Phase 3: Cooldown
    print(f"[3/3] Collecting cooldown ({cooldown_duration_s}s)...")
    cooldown_start = time.time()
    time.sleep(cooldown_duration_s)
    cooldown_end = time.time()
    cooldown_duration = cooldown_end - cooldown_start

    # Stop power collection
    print("Stopping power metrics collection...")
    all_samples = collector.stop()
    print(f"Total samples collected: {len(all_samples)}")

    if len(all_samples) == 0:
        print("ERROR: No power samples were collected!")
        print("This usually means powermetrics failed to start.")
        print("Make sure you ran this script with: sudo python3 lm_studio_power_benchmark.py")

    # Extract samples for each phase (using relative time)
    baseline_time_start = 0
    baseline_time_end = baseline_duration
    inference_time_start = baseline_duration
    inference_time_end = baseline_duration + inference_duration
    cooldown_time_start = inference_time_end
    cooldown_time_end = inference_time_end + cooldown_duration

    baseline_samples = extract_samples_by_time(all_samples, baseline_time_start, baseline_time_end)
    inference_samples = extract_samples_by_time(all_samples, inference_time_start, inference_time_end)
    cooldown_samples = extract_samples_by_time(all_samples, cooldown_time_start, cooldown_time_end)

    # Calculate statistics
    baseline_power = calculate_power_stats(baseline_samples)
    inference_power = calculate_power_stats(inference_samples)
    cooldown_power = calculate_power_stats(cooldown_samples)

    # Print summary
    print(f"\nPower Consumption Summary:")
    if baseline_power:
        print(f"  Baseline:  {baseline_power.avg_mw:6.0f} mW avg ({baseline_power.sample_count} samples)")
    if inference_power:
        print(f"  Inference: {inference_power.avg_mw:6.0f} mW avg ({inference_power.sample_count} samples)")
    if cooldown_power:
        print(f"  Cooldown:  {cooldown_power.avg_mw:6.0f} mW avg ({cooldown_power.sample_count} samples)")

    return BenchmarkResult(
        category=category,
        prompt=prompt,
        response=response_text,
        baseline_duration_s=baseline_duration,
        inference_duration_s=inference_duration,
        cooldown_duration_s=cooldown_duration,
        inference_stats=inference_stats,
        baseline_power=baseline_power,
        inference_power=inference_power,
        cooldown_power=cooldown_power,
        raw_power_samples=all_samples
    )


def run_benchmark_suite(
    model_name: Optional[str],
    baseline_duration_s: float,
    cooldown_duration_s: float
) -> List[BenchmarkResult]:
    """
    Run full benchmark suite.

    Args:
        model_name: LM Studio model name (None for currently loaded)
        baseline_duration_s: Baseline duration
        cooldown_duration_s: Cooldown duration

    Returns:
        List of benchmark results
    """
    print("="*70)
    print("LM STUDIO POWER CONSUMPTION BENCHMARK")
    print("="*70)

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
                prompt=prompt,
                baseline_duration_s=baseline_duration_s,
                cooldown_duration_s=cooldown_duration_s
            )
            results.append(result)

            # Brief pause between benchmarks
            if current < total_prompts:
                print("\nPausing 5s before next benchmark...")
                time.sleep(5)

    return results


def generate_json_report(
    results: List[BenchmarkResult],
    output_file: str,
    model_name: Optional[str]
):
    """
    Generate JSON report from benchmark results.

    Args:
        results: List of benchmark results
        output_file: Output JSON file path
        model_name: Model name used
    """
    # Calculate summary statistics
    all_baseline = [r.baseline_power for r in results if r.baseline_power]
    all_inference = [r.inference_power for r in results if r.inference_power]
    all_cooldown = [r.cooldown_power for r in results if r.cooldown_power]

    summary = {
        "total_prompts": len(results),
        "avg_baseline_power_mw": sum(p.avg_mw for p in all_baseline) / len(all_baseline) if all_baseline else None,
        "avg_inference_power_mw": sum(p.avg_mw for p in all_inference) / len(all_inference) if all_inference else None,
        "avg_cooldown_power_mw": sum(p.avg_mw for p in all_cooldown) / len(all_cooldown) if all_cooldown else None,
        "avg_tokens_per_second": sum(
            r.inference_stats.tokens_per_second for r in results
            if r.inference_stats.tokens_per_second
        ) / len([r for r in results if r.inference_stats.tokens_per_second]) if any(r.inference_stats.tokens_per_second for r in results) else None,
    }

    # Build report structure
    report = {
        "benchmark_id": datetime.now().isoformat(),
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
            "timing": {
                "baseline_duration_s": result.baseline_duration_s,
                "inference_duration_s": result.inference_duration_s,
                "cooldown_duration_s": result.cooldown_duration_s,
            },
            "inference_stats": asdict(result.inference_stats),
            "power_consumption": {
                "baseline": asdict(result.baseline_power) if result.baseline_power else None,
                "inference": asdict(result.inference_power) if result.inference_power else None,
                "cooldown": asdict(result.cooldown_power) if result.cooldown_power else None,
            },
            "raw_samples_count": len(result.raw_power_samples)
        }
        report["prompts"].append(prompt_data)

    # Write to file
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Report saved to: {output_file}")
    print(f"{'='*70}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run LM Studio inference benchmark with power consumption measurement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark with currently loaded model
  sudo python3 lm_studio_power_benchmark.py

  # Run with specific model
  sudo python3 lm_studio_power_benchmark.py --model qwen/qwen3-4b-2507

  # Custom timing and output
  sudo python3 lm_studio_power_benchmark.py --baseline 15 --cooldown 15 --output my_results.json

Note: This script requires:
  1. LM Studio running with a model loaded
  2. sudo privileges for powermetrics
  3. lmstudio package installed (pip install lmstudio)
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='LM Studio model identifier (uses currently loaded model if not specified)'
    )

    parser.add_argument(
        '--baseline',
        type=float,
        default=10.0,
        metavar='SECONDS',
        help='Baseline power collection duration in seconds (default: 10)'
    )

    parser.add_argument(
        '--cooldown',
        type=float,
        default=10.0,
        metavar='SECONDS',
        help='Cooldown power collection duration in seconds (default: 10)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        metavar='FILE',
        help='Output JSON file path (default: lm_studio_benchmark_<timestamp>.json)'
    )

    args = parser.parse_args()

    # Check if running as root (required for powermetrics)
    if os.geteuid() != 0:
        print("ERROR: This script must be run with sudo privileges.", file=sys.stderr)
        print("\nPowermetrics requires root access to collect system power data.", file=sys.stderr)
        print(f"\nPlease run with sudo:", file=sys.stderr)
        print(f"  sudo python3 {' '.join(sys.argv)}", file=sys.stderr)
        sys.exit(1)

    # Generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"lm_studio_benchmark_{timestamp}.json"

    try:
        # Run benchmark suite
        results = run_benchmark_suite(
            model_name=args.model,
            baseline_duration_s=args.baseline,
            cooldown_duration_s=args.cooldown
        )

        # Generate report
        generate_json_report(
            results=results,
            output_file=args.output,
            model_name=args.model
        )

        print("\n✓ Benchmark completed successfully!")

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
