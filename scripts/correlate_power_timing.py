#!/usr/bin/env python3
"""
Power-Timing Correlation Tool

Correlates LM Studio benchmark timing data with powermetrics data to calculate
power consumption for each prompt. Generates PDF reports with visualizations.
"""

import json
import csv
import argparse
import sys
import os
import subprocess
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

# Matplotlib setup - must be before pyplot import
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# PyLaTeX imports
try:
    from pylatex import (
        Document, Section, Subsection, Figure, Tabular, LongTable,
        NoEscape, Package, Command
    )
    from pylatex.utils import bold
except ImportError:
    print("ERROR: pylatex package not found.", file=sys.stderr)
    print("Install with: pip install pylatex", file=sys.stderr)
    sys.exit(1)


@dataclass
class PowerStats:
    """Power consumption statistics."""
    min_mw: int
    max_mw: int
    avg_mw: float
    samples: int


@dataclass
class PowerAnalysis:
    """Complete power analysis for a prompt."""
    baseline: Optional[PowerStats]
    inference: Optional[PowerStats]
    cooldown: Optional[PowerStats]
    peak_power_mw: Optional[int]
    energy_estimate_mj: Optional[float]  # millijoules


def parse_csv_timestamp(timestamp_str: str) -> datetime:
    """
    Parse power CSV timestamp format.

    Example: "Sat Nov  8 14:55:52 2025 -0700"

    Args:
        timestamp_str: Timestamp string from CSV

    Returns:
        datetime object
    """
    return datetime.strptime(timestamp_str, "%a %b %d %H:%M:%S %Y %z")


def parse_iso_timestamp(timestamp_str: str) -> datetime:
    """
    Parse ISO 8601 timestamp from benchmark JSON.

    Example: "2025-11-08T14:56:47.641627"

    Args:
        timestamp_str: ISO timestamp string

    Returns:
        datetime object (assumes local timezone)
    """
    dt = datetime.fromisoformat(timestamp_str)
    # If no timezone info, assume it matches the system/CSV timezone
    # The CSV has timezone, so we need to make the ISO timestamp aware
    if dt.tzinfo is None:
        # Get timezone from system or use a default
        # For now, we'll add timezone in the correlation function
        pass
    return dt


def load_power_data(csv_file: str) -> List[Dict]:
    """
    Load power consumption data from CSV.

    Args:
        csv_file: Path to power CSV file

    Returns:
        List of power samples with parsed timestamps
    """
    samples = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sample = {
                    'timestamp': parse_csv_timestamp(row['timestamp']),
                    'cpu_power_mw': int(row['cpu_power_mw']),
                    'gpu_power_mw': int(row['gpu_power_mw']),
                    'ane_power_mw': int(row['ane_power_mw']),
                    'combined_power_mw': int(row['combined_power_mw'])
                }
                samples.append(sample)
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping malformed row: {e}")
                continue

    return samples


def load_benchmark_data(json_file: str) -> Dict:
    """
    Load benchmark timing data from JSON.

    Args:
        json_file: Path to benchmark JSON file

    Returns:
        Benchmark data dictionary
    """
    with open(json_file, 'r') as f:
        return json.load(f)


def find_samples_in_window(
    power_samples: List[Dict],
    start_time: datetime,
    end_time: datetime
) -> List[Dict]:
    """
    Find power samples within a time window.

    Args:
        power_samples: List of power samples
        start_time: Window start time
        end_time: Window end time

    Returns:
        Samples within the time window
    """
    return [
        sample for sample in power_samples
        if start_time <= sample['timestamp'] <= end_time
    ]


def calculate_power_stats(samples: List[Dict]) -> Optional[PowerStats]:
    """
    Calculate power statistics from samples.

    Args:
        samples: List of power samples

    Returns:
        PowerStats or None if no samples
    """
    if not samples:
        return None

    combined_powers = [s['combined_power_mw'] for s in samples]

    return PowerStats(
        min_mw=min(combined_powers),
        max_mw=max(combined_powers),
        avg_mw=sum(combined_powers) / len(combined_powers),
        samples=len(combined_powers)
    )


def analyze_prompt_power(
    prompt_data: Dict,
    power_samples: List[Dict],
    baseline_window_s: float,
    cooldown_window_s: float,
    reference_timezone
) -> PowerAnalysis:
    """
    Analyze power consumption for a single prompt.

    Args:
        prompt_data: Prompt data from benchmark JSON
        power_samples: All power samples
        baseline_window_s: Baseline window duration in seconds
        cooldown_window_s: Cooldown window duration in seconds
        reference_timezone: Timezone from power samples

    Returns:
        PowerAnalysis with statistics for all phases
    """
    # Parse timing from benchmark
    timing = prompt_data['timing']

    # Parse timestamps and add timezone
    prompt_submit = datetime.fromisoformat(timing['prompt_submit_time'])
    if prompt_submit.tzinfo is None:
        prompt_submit = prompt_submit.replace(tzinfo=reference_timezone.tzinfo)

    response_complete = datetime.fromisoformat(timing['response_complete_time'])
    if response_complete.tzinfo is None:
        response_complete = response_complete.replace(tzinfo=reference_timezone.tzinfo)

    # Define time windows
    baseline_start = prompt_submit - timedelta(seconds=baseline_window_s)
    baseline_end = prompt_submit

    inference_start = prompt_submit
    inference_end = response_complete

    cooldown_start = response_complete
    cooldown_end = response_complete + timedelta(seconds=cooldown_window_s)

    # Find samples for each phase
    baseline_samples = find_samples_in_window(power_samples, baseline_start, baseline_end)
    inference_samples = find_samples_in_window(power_samples, inference_start, inference_end)
    cooldown_samples = find_samples_in_window(power_samples, cooldown_start, cooldown_end)

    # Calculate statistics
    baseline_stats = calculate_power_stats(baseline_samples)
    inference_stats = calculate_power_stats(inference_samples)
    cooldown_stats = calculate_power_stats(cooldown_samples)

    # Calculate peak power
    all_inference_powers = [s['combined_power_mw'] for s in inference_samples]
    peak_power = max(all_inference_powers) if all_inference_powers else None

    # Estimate energy consumption (Power × Time = Energy)
    # Energy in millijoules = Average Power (mW) × Duration (s)
    energy_estimate = None
    if inference_stats:
        inference_duration_s = timing['inference_duration_s']
        energy_estimate = inference_stats.avg_mw * inference_duration_s

    return PowerAnalysis(
        baseline=baseline_stats,
        inference=inference_stats,
        cooldown=cooldown_stats,
        peak_power_mw=peak_power,
        energy_estimate_mj=energy_estimate
    )


def analyze_prompt_power_with_optimization(
    version_data: Dict,
    power_samples: List[Dict],
    baseline_window_s: float,
    cooldown_window_s: float,
    reference_timezone
) -> Dict:
    """
    Analyze power for prompts with LLM optimization step.

    For llm_optimized prompts, this analyzes BOTH:
    1. The optimization window (LLM optimizing the prompt)
    2. The execution window (running the optimized prompt)

    Args:
        version_data: Version data with timing and optimization_metadata
        power_samples: All power samples
        baseline_window_s: Baseline window duration
        cooldown_window_s: Cooldown window duration
        reference_timezone: Reference timezone

    Returns:
        Dict with optimization_analysis, execution_analysis, and combined metrics
    """
    # Analyze optimization window if it exists
    opt_analysis = None
    opt_energy = 0

    if version_data.get('optimization_metadata') and \
       version_data['optimization_metadata'].get('optimization_timing'):

        opt_timing = version_data['optimization_metadata']['optimization_timing']

        # Create a temporary prompt dict with just the optimization timing
        opt_prompt_data = {'timing': opt_timing}

        opt_analysis = analyze_prompt_power(
            opt_prompt_data,
            power_samples,
            baseline_window_s,
            cooldown_window_s,
            reference_timezone
        )

        if opt_analysis and opt_analysis.energy_estimate_mj:
            opt_energy = opt_analysis.energy_estimate_mj

    # Analyze execution window (the actual benchmark run)
    exec_analysis = analyze_prompt_power(
        version_data,
        power_samples,
        baseline_window_s,
        cooldown_window_s,
        reference_timezone
    )

    # Calculate combined energy
    exec_energy = exec_analysis.energy_estimate_mj if exec_analysis.energy_estimate_mj else 0
    total_energy = exec_energy + opt_energy

    # Return comprehensive analysis
    return {
        'optimization_analysis': asdict(opt_analysis) if opt_analysis else None,
        'execution_analysis': asdict(exec_analysis),
        'combined_energy_mj': total_energy,
        'optimization_overhead_mj': opt_energy,
        # Keep these for backward compatibility with existing code
        'baseline': asdict(exec_analysis.baseline) if exec_analysis.baseline else None,
        'inference': asdict(exec_analysis.inference) if exec_analysis.inference else None,
        'cooldown': asdict(exec_analysis.cooldown) if exec_analysis.cooldown else None,
        'peak_power_mw': exec_analysis.peak_power_mw,
        'energy_estimate_mj': total_energy  # This is now the COMBINED energy
    }


def correlate_power_timing(
    benchmark_file: str,
    power_file: str,
    baseline_window_s: float,
    cooldown_window_s: float
) -> Dict:
    """
    Correlate benchmark timing with power data.

    Args:
        benchmark_file: Path to benchmark JSON
        power_file: Path to power CSV
        baseline_window_s: Baseline window duration
        cooldown_window_s: Cooldown window duration

    Returns:
        Enhanced benchmark data with power analysis
    """
    print("Loading data files...")
    benchmark_data = load_benchmark_data(benchmark_file)
    power_samples = load_power_data(power_file)

    if not power_samples:
        raise ValueError("No power samples found in CSV file")

    print(f"Loaded {len(power_samples)} power samples")
    print(f"Power data range: {power_samples[0]['timestamp']} to {power_samples[-1]['timestamp']}")

    # Get reference timezone from power samples
    reference_tz = power_samples[0]['timestamp']

    # Detect comparison mode
    comparison_mode = benchmark_data.get('comparison_mode', False)

    if comparison_mode:
        print("\n[COMPARISON MODE DETECTED]")
        print("Processing prompt groups with multiple versions...")
        return correlate_comparison_mode(
            benchmark_data, power_samples, baseline_window_s, cooldown_window_s, reference_tz, power_file
        )

    # Standard mode processing
    print("\n[STANDARD MODE]")
    # Analyze each prompt
    print("\nAnalyzing power consumption for each prompt...")
    enhanced_prompts = []

    for i, prompt in enumerate(benchmark_data['prompts'], 1):
        print(f"  [{i}/{len(benchmark_data['prompts'])}] {prompt['category']}: {prompt['prompt'][:50]}...")

        power_analysis = analyze_prompt_power(
            prompt,
            power_samples,
            baseline_window_s,
            cooldown_window_s,
            reference_tz
        )

        # Add power analysis to prompt data
        enhanced_prompt = prompt.copy()
        enhanced_prompt['power_analysis'] = asdict(power_analysis)
        enhanced_prompts.append(enhanced_prompt)

        # Print summary
        if power_analysis.inference:
            print(f"      Inference: {power_analysis.inference.avg_mw:.0f} mW avg, "
                  f"Peak: {power_analysis.peak_power_mw} mW, "
                  f"Energy: {power_analysis.energy_estimate_mj:.1f} mJ")
        else:
            print(f"      WARNING: No power samples found during inference!")

    # Calculate summary statistics
    total_energy = sum(
        p['power_analysis']['energy_estimate_mj']
        for p in enhanced_prompts
        if p['power_analysis']['energy_estimate_mj'] is not None
    )

    avg_inference_power = sum(
        p['power_analysis']['inference']['avg_mw']
        for p in enhanced_prompts
        if p['power_analysis']['inference'] is not None
    ) / len([p for p in enhanced_prompts if p['power_analysis']['inference'] is not None])

    peak_power_overall = max(
        p['power_analysis']['peak_power_mw']
        for p in enhanced_prompts
        if p['power_analysis']['peak_power_mw'] is not None
    )

    # Build enhanced report
    enhanced_data = {
        'benchmark_id': benchmark_data['benchmark_id'],
        'model': benchmark_data['model'],
        'comparison_mode': False,
        'correlation_params': {
            'power_csv_file': power_file,
            'baseline_window_s': baseline_window_s,
            'cooldown_window_s': cooldown_window_s,
        },
        'summary': {
            **benchmark_data['summary'],
            'total_energy_mj': total_energy,
            'avg_inference_power_mw': avg_inference_power,
            'peak_power_mw': peak_power_overall,
        },
        'prompts': enhanced_prompts
    }

    return enhanced_data


def correlate_comparison_mode(
    benchmark_data: Dict,
    power_samples: List[Dict],
    baseline_window_s: float,
    cooldown_window_s: float,
    reference_tz,
    power_file: str
) -> Dict:
    """
    Correlate power data for comparison mode benchmarks.

    Args:
        benchmark_data: Benchmark data with prompt_groups
        power_samples: Power samples
        baseline_window_s: Baseline window duration
        cooldown_window_s: Cooldown window duration
        reference_tz: Reference timezone
        power_file: Path to power CSV file

    Returns:
        Enhanced benchmark data with power analysis for all versions
    """
    enhanced_groups = []

    for group_id, group in enumerate(benchmark_data['prompt_groups']):
        print(f"\n  Group {group_id + 1}/{len(benchmark_data['prompt_groups'])}: {group['category']}")
        print(f"  Base prompt: {group['base_prompt'][:60]}...")

        enhanced_versions = []

        for version in group['versions']:
            print(f"    - {version['prompt_version']}: {version['prompt'][:50]}...")

            # For llm_optimized, use special analysis that includes optimization step
            if version['prompt_version'] == 'llm_optimized':
                power_analysis = analyze_prompt_power_with_optimization(
                    version,
                    power_samples,
                    baseline_window_s,
                    cooldown_window_s,
                    reference_tz
                )
                # power_analysis is already a dict
                enhanced_version = version.copy()
                enhanced_version['power_analysis'] = power_analysis
                enhanced_versions.append(enhanced_version)

                # Print summary with optimization overhead
                if power_analysis.get('inference'):
                    opt_overhead = power_analysis.get('optimization_overhead_mj', 0)
                    total_energy = power_analysis.get('combined_energy_mj', 0)
                    print(f"        Execution: {power_analysis['inference']['avg_mw']:.0f} mW avg")
                    print(f"        Optimization overhead: {opt_overhead:.1f} mJ")
                    print(f"        Total energy (opt + exec): {total_energy:.1f} mJ")
            else:
                # Standard analysis for original and rule_optimized
                power_analysis = analyze_prompt_power(
                    version,
                    power_samples,
                    baseline_window_s,
                    cooldown_window_s,
                    reference_tz
                )

                enhanced_version = version.copy()
                enhanced_version['power_analysis'] = asdict(power_analysis)
                enhanced_versions.append(enhanced_version)

                # Print summary
                if power_analysis.inference:
                    print(f"        Power: {power_analysis.inference.avg_mw:.0f} mW avg, "
                          f"Energy: {power_analysis.energy_estimate_mj:.1f} mJ")

        enhanced_group = group.copy()
        enhanced_group['versions'] = enhanced_versions
        enhanced_groups.append(enhanced_group)

    # Calculate summary statistics by version
    all_versions = []
    for group in enhanced_groups:
        all_versions.extend(group['versions'])

    def calc_version_stats(version_name: str):
        version_prompts = [v for v in all_versions if v['prompt_version'] == version_name]
        if not version_prompts:
            return None

        # For llm_optimized, energy_estimate_mj includes BOTH optimization and execution
        # For others, it's just execution
        total_energy = sum(
            v['power_analysis']['energy_estimate_mj']
            for v in version_prompts
            if v['power_analysis']['energy_estimate_mj'] is not None
        )

        # Calculate optimization overhead (only for llm_optimized)
        optimization_overhead = 0
        execution_only_energy = 0
        if version_name == 'llm_optimized':
            optimization_overhead = sum(
                v['power_analysis'].get('optimization_overhead_mj', 0)
                for v in version_prompts
            )
            execution_only_energy = total_energy - optimization_overhead

        valid_power = [v for v in version_prompts if v['power_analysis']['inference'] is not None]
        avg_power = sum(
            v['power_analysis']['inference']['avg_mw']
            for v in valid_power
        ) / len(valid_power) if valid_power else None

        valid_peak = [v for v in version_prompts if v['power_analysis']['peak_power_mw'] is not None]
        peak_power = max(
            v['power_analysis']['peak_power_mw']
            for v in valid_peak
        ) if valid_peak else None

        stats = {
            'total_energy_mj': total_energy,
            'avg_inference_power_mw': avg_power,
            'peak_power_mw': peak_power,
        }

        # Add optimization breakdown for llm_optimized
        if version_name == 'llm_optimized':
            stats['optimization_overhead_mj'] = optimization_overhead
            stats['execution_only_energy_mj'] = execution_only_energy

        return stats

    version_stats = {
        'original': calc_version_stats('original'),
        'rule_optimized': calc_version_stats('rule_optimized'),
        'llm_optimized': calc_version_stats('llm_optimized'),
    }

    # Calculate overall totals
    total_energy_all = sum(
        v['power_analysis']['energy_estimate_mj']
        for v in all_versions
        if v['power_analysis']['energy_estimate_mj'] is not None
    )

    # Build enhanced report
    enhanced_data = {
        'benchmark_id': benchmark_data['benchmark_id'],
        'model': benchmark_data['model'],
        'comparison_mode': True,
        'correlation_params': {
            'power_csv_file': power_file,
            'baseline_window_s': baseline_window_s,
            'cooldown_window_s': cooldown_window_s,
        },
        'summary': {
            **benchmark_data['summary'],
            'total_energy_all_versions_mj': total_energy_all,
            'by_version': version_stats,
        },
        'prompt_groups': enhanced_groups
    }

    # Print comparison summary
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY:")
    print(f"{'='*70}")
    for version_name, stats in version_stats.items():
        if stats:
            print(f"\n{version_name.upper()}:")
            print(f"  Total Energy: {stats['total_energy_mj']:.1f} mJ")

            # Show breakdown for llm_optimized
            if version_name == 'llm_optimized':
                opt_overhead = stats.get('optimization_overhead_mj', 0)
                exec_only = stats.get('execution_only_energy_mj', 0)
                print(f"    - Optimization overhead: {opt_overhead:.1f} mJ")
                print(f"    - Execution only: {exec_only:.1f} mJ")

            print(f"  Avg Power: {stats['avg_inference_power_mw']:.0f} mW" if stats['avg_inference_power_mw'] else "  Avg Power: N/A")
            print(f"  Peak Power: {stats['peak_power_mw']} mW" if stats['peak_power_mw'] else "  Peak Power: N/A")

    # Calculate savings
    if version_stats['original'] and version_stats['rule_optimized']:
        orig_energy = version_stats['original']['total_energy_mj']
        rule_energy = version_stats['rule_optimized']['total_energy_mj']
        rule_savings = ((orig_energy - rule_energy) / orig_energy * 100) if orig_energy > 0 else 0
        print(f"\nRULE OPTIMIZATION SAVINGS: {rule_savings:+.1f}%")

    if version_stats['original'] and version_stats['llm_optimized']:
        orig_energy = version_stats['original']['total_energy_mj']
        llm_energy_total = version_stats['llm_optimized']['total_energy_mj']  # Already includes overhead
        llm_exec_only = version_stats['llm_optimized'].get('execution_only_energy_mj', llm_energy_total)

        # Calculate savings including overhead (realistic)
        llm_savings_with_overhead = ((orig_energy - llm_energy_total) / orig_energy * 100) if orig_energy > 0 else 0

        # Calculate savings if we ignore overhead (theoretical best case)
        llm_savings_exec_only = ((orig_energy - llm_exec_only) / orig_energy * 100) if orig_energy > 0 else 0

        print(f"\nLLM OPTIMIZATION SAVINGS:")
        print(f"  Including optimization overhead: {llm_savings_with_overhead:+.1f}%")
        print(f"  Execution only (excluding overhead): {llm_savings_exec_only:+.1f}%")

    print(f"{'='*70}\n")

    return enhanced_data


def escape_latex(text: str) -> str:
    """
    Escape special LaTeX characters in text.

    Args:
        text: Raw text string

    Returns:
        LaTeX-safe string
    """
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '\\': r'\textbackslash{}',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def create_prompt_timeline_chart(
    prompt_data: Dict,
    power_samples: List[Dict],
    prompt_index: int
) -> plt.Figure:
    """
    Create a timeline chart showing power consumption for a single prompt.

    Args:
        prompt_data: Enhanced prompt data with timing and power_analysis
        power_samples: All power samples
        prompt_index: Index of this prompt in the sequence

    Returns:
        matplotlib Figure object
    """
    timing = prompt_data['timing']
    power_analysis = prompt_data['power_analysis']

    # Parse timing
    prompt_submit = datetime.fromisoformat(timing['prompt_submit_time'])
    response_complete = datetime.fromisoformat(timing['response_complete_time'])

    # Get reference timezone from power samples
    reference_tz = power_samples[0]['timestamp']
    if prompt_submit.tzinfo is None:
        prompt_submit = prompt_submit.replace(tzinfo=reference_tz.tzinfo)
    if response_complete.tzinfo is None:
        response_complete = response_complete.replace(tzinfo=reference_tz.tzinfo)

    # Extended time window for context
    start_time = prompt_submit - timedelta(seconds=10)
    end_time = response_complete + timedelta(seconds=10)

    # Find samples in window
    window_samples = [
        s for s in power_samples
        if start_time <= s['timestamp'] <= end_time
    ]

    if not window_samples:
        # No data - create empty figure with message
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, 'No power data available for this time window',
                ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig

    # Extract data for plotting
    timestamps = [s['timestamp'] for s in window_samples]
    combined_power = [s['combined_power_mw'] for s in window_samples]
    cpu_power = [s['cpu_power_mw'] for s in window_samples]
    gpu_power = [s['gpu_power_mw'] for s in window_samples]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot power data
    ax.plot(timestamps, combined_power, label='Combined Power', linewidth=2, color='black')
    ax.plot(timestamps, cpu_power, label='CPU Power', linewidth=1, alpha=0.7, color='blue')
    ax.plot(timestamps, gpu_power, label='GPU Power', linewidth=1, alpha=0.7, color='green')

    # Mark inference window
    ax.axvspan(prompt_submit, response_complete, alpha=0.2, color='orange', label='Inference')

    # Mark baseline and cooldown if they have samples
    if power_analysis['baseline']:
        baseline_start = prompt_submit - timedelta(seconds=5)
        ax.axvspan(baseline_start, prompt_submit, alpha=0.15, color='blue', label='Baseline')

    if power_analysis['cooldown']:
        cooldown_end = response_complete + timedelta(seconds=5)
        ax.axvspan(response_complete, cooldown_end, alpha=0.15, color='purple', label='Cooldown')

    # Formatting
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Power (mW)', fontsize=11)
    ax.set_title(f'Prompt {prompt_index + 1}: {prompt_data["category"].upper()} - Power Timeline',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Format x-axis timestamps
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()

    plt.tight_layout()
    return fig


def create_overall_timeline_chart(enhanced_data: Dict, power_samples: List[Dict]) -> plt.Figure:
    """
    Create an overall timeline showing all prompts.

    Args:
        enhanced_data: Complete enhanced benchmark data
        power_samples: All power samples

    Returns:
        matplotlib Figure object
    """
    if not power_samples:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.text(0.5, 0.5, 'No power data available',
                ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig

    # Extract all timestamps and power
    timestamps = [s['timestamp'] for s in power_samples]
    combined_power = [s['combined_power_mw'] for s in power_samples]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot power over time
    ax.plot(timestamps, combined_power, linewidth=1.5, color='black', alpha=0.7)

    # Mark each prompt's inference window
    colors = {'short': 'green', 'medium': 'orange', 'long': 'red'}
    for i, prompt in enumerate(enhanced_data['prompts']):
        timing = prompt['timing']
        category = prompt['category']

        submit_time = datetime.fromisoformat(timing['prompt_submit_time'])
        complete_time = datetime.fromisoformat(timing['response_complete_time'])

        # Add timezone if needed
        reference_tz = power_samples[0]['timestamp']
        if submit_time.tzinfo is None:
            submit_time = submit_time.replace(tzinfo=reference_tz.tzinfo)
        if complete_time.tzinfo is None:
            complete_time = complete_time.replace(tzinfo=reference_tz.tzinfo)

        ax.axvspan(submit_time, complete_time, alpha=0.3, color=colors.get(category, 'gray'),
                   label=f'{category.capitalize()}' if i == 0 or prompt['category'] != enhanced_data['prompts'][i-1]['category'] else '')

    # Formatting
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Combined Power (mW)', fontsize=12)
    ax.set_title('Power Consumption Timeline - All Prompts', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()

    plt.tight_layout()
    return fig


def create_comparison_charts(enhanced_data: Dict) -> plt.Figure:
    """
    Create comparison bar charts across prompt categories.

    Args:
        enhanced_data: Complete enhanced benchmark data

    Returns:
        matplotlib Figure with subplots
    """
    # Organize data by category
    categories = ['short', 'medium', 'long']
    category_data = {cat: [] for cat in categories}

    for prompt in enhanced_data['prompts']:
        cat = prompt['category']
        pa = prompt['power_analysis']

        if pa['inference'] and pa['energy_estimate_mj']:
            category_data[cat].append({
                'avg_power': pa['inference']['avg_mw'],
                'peak_power': pa['peak_power_mw'],
                'energy': pa['energy_estimate_mj'],
                'duration': prompt['timing']['inference_duration_s']
            })

    # Calculate averages per category
    avg_power_by_cat = []
    peak_power_by_cat = []
    energy_by_cat = []
    duration_by_cat = []

    for cat in categories:
        if category_data[cat]:
            avg_power_by_cat.append(sum(d['avg_power'] for d in category_data[cat]) / len(category_data[cat]))
            peak_power_by_cat.append(max(d['peak_power'] for d in category_data[cat]))
            energy_by_cat.append(sum(d['energy'] for d in category_data[cat]) / len(category_data[cat]))
            duration_by_cat.append(sum(d['duration'] for d in category_data[cat]) / len(category_data[cat]))
        else:
            avg_power_by_cat.append(0)
            peak_power_by_cat.append(0)
            energy_by_cat.append(0)
            duration_by_cat.append(0)

    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    x_pos = range(len(categories))
    colors = ['green', 'orange', 'red']

    # Average Power
    bars1 = ax1.bar(x_pos, avg_power_by_cat, color=colors, alpha=0.7)
    ax1.set_ylabel('Average Power (mW)', fontsize=11)
    ax1.set_title('Average Inference Power by Category', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([c.capitalize() for c in categories])
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, avg_power_by_cat):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)

    # Peak Power
    bars2 = ax2.bar(x_pos, peak_power_by_cat, color=colors, alpha=0.7)
    ax2.set_ylabel('Peak Power (mW)', fontsize=11)
    ax2.set_title('Peak Inference Power by Category', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([c.capitalize() for c in categories])
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, peak_power_by_cat):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)

    # Energy
    bars3 = ax3.bar(x_pos, energy_by_cat, color=colors, alpha=0.7)
    ax3.set_ylabel('Average Energy (mJ)', fontsize=11)
    ax3.set_title('Average Energy Consumption by Category', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([c.capitalize() for c in categories])
    ax3.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars3, energy_by_cat):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    # Duration
    bars4 = ax4.bar(x_pos, duration_by_cat, color=colors, alpha=0.7)
    ax4.set_ylabel('Average Duration (s)', fontsize=11)
    ax4.set_title('Average Inference Duration by Category', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([c.capitalize() for c in categories])
    ax4.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars4, duration_by_cat):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    return fig


def create_version_comparison_charts(enhanced_data: Dict) -> plt.Figure:
    """
    Create comparison charts for different prompt versions (comparison mode).

    Args:
        enhanced_data: Enhanced benchmark data in comparison mode

    Returns:
        matplotlib Figure with comparison charts
    """
    # Collect data by version
    version_data = {'original': [], 'rule_optimized': [], 'llm_optimized': []}

    for group in enhanced_data['prompt_groups']:
        for version in group['versions']:
            pa = version['power_analysis']
            if pa['inference'] and pa['energy_estimate_mj']:
                version_data[version['prompt_version']].append({
                    'category': group['category'],
                    'avg_power': pa['inference']['avg_mw'],
                    'energy': pa['energy_estimate_mj'],
                    'duration': version['timing']['inference_duration_s']
                })

    # Calculate stats per version
    version_stats = {}
    for version_name, data_list in version_data.items():
        if data_list:
            version_stats[version_name] = {
                'avg_power': sum(d['avg_power'] for d in data_list) / len(data_list),
                'total_energy': sum(d['energy'] for d in data_list),
                'avg_duration': sum(d['duration'] for d in data_list) / len(data_list)
            }

    # Create figure with 1x3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    versions = ['original', 'rule_optimized', 'llm_optimized']
    version_labels = ['Original', 'Rule\nOptimized', 'LLM\nOptimized']
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    # Chart 1: Average Power
    avg_powers = [version_stats.get(v, {}).get('avg_power', 0) for v in versions]
    bars1 = ax1.bar(range(len(versions)), avg_powers, color=colors, alpha=0.7)
    ax1.set_ylabel('Average Power (mW)', fontsize=11)
    ax1.set_title('Average Inference Power', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(versions)))
    ax1.set_xticklabels(version_labels)
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, avg_powers):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    # Chart 2: Total Energy
    total_energies = [version_stats.get(v, {}).get('total_energy', 0) for v in versions]
    bars2 = ax2.bar(range(len(versions)), total_energies, color=colors, alpha=0.7)
    ax2.set_ylabel('Total Energy (mJ)', fontsize=11)
    ax2.set_title('Total Energy Consumption', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(versions)))
    ax2.set_xticklabels(version_labels)
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, total_energies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # Chart 3: Average Duration
    avg_durations = [version_stats.get(v, {}).get('avg_duration', 0) for v in versions]
    bars3 = ax3.bar(range(len(versions)), avg_durations, color=colors, alpha=0.7)
    ax3.set_ylabel('Average Duration (s)', fontsize=11)
    ax3.set_title('Average Inference Duration', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(versions)))
    ax3.set_xticklabels(version_labels)
    ax3.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars3, avg_durations):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def generate_pdf_report(
    enhanced_data: Dict,
    power_samples: List[Dict],
    output_file: str
):
    """
    Generate comprehensive PDF report with PyLaTeX.

    Args:
        enhanced_data: Enhanced benchmark data with power analysis
        power_samples: All power samples
        output_file: Output PDF path (without extension)
    """
    print("\nGenerating PDF report...")

    # Check for LaTeX installation
    import shutil
    if not shutil.which('lualatex'):
        print("\nERROR: lualatex not found!", file=sys.stderr)
        print("PDF generation requires a LaTeX distribution with LuaLaTeX.", file=sys.stderr)
        print("On macOS, install MacTeX: brew install --cask mactex", file=sys.stderr)
        print("Or BasicTeX: brew install --cask basictex", file=sys.stderr)
        sys.exit(1)

    # Detect comparison mode
    comparison_mode = enhanced_data.get('comparison_mode', False)

    if comparison_mode:
        generate_comparison_pdf_report(enhanced_data, power_samples, output_file)
    else:
        generate_standard_pdf_report(enhanced_data, power_samples, output_file)


def generate_standard_pdf_report(
    enhanced_data: Dict,
    power_samples: List[Dict],
    output_file: str
):
    """
    Generate standard (non-comparison) PDF report.

    Args:
        enhanced_data: Enhanced benchmark data with power analysis
        power_samples: All power samples
        output_file: Output PDF path (without extension)
    """

    # Create document
    geometry_options = {
        "tmargin": "2cm",
        "bmargin": "2cm",
        "lmargin": "2cm",
        "rmargin": "2cm"
    }
    doc = Document(geometry_options=geometry_options, page_numbers=True)

    # Add required packages
    doc.packages.append(Package('booktabs'))  # Professional tables
    doc.packages.append(Package('float'))      # Figure positioning
    doc.packages.append(Package('longtable'))  # Multi-page tables
    doc.packages.append(Package('array'))      # Better table columns
    doc.packages.append(Package('listings'))   # For code/verbatim blocks with line breaking

    # Configure listings for better verbatim display
    doc.preamble.append(NoEscape(r'\lstset{basicstyle=\small\ttfamily, breaklines=true, breakatwhitespace=false, columns=flexible}'))

    # Title page
    doc.preamble.append(Command('title', 'LM Studio Power Consumption Analysis'))
    doc.preamble.append(Command('author', f'Model: {enhanced_data["model"]}'))
    doc.preamble.append(Command('date', f'Benchmark: {enhanced_data["benchmark_id"]}'))
    doc.append(NoEscape(r'\maketitle'))

    # Summary Statistics Section
    with doc.create(Section('Summary Statistics')):
        doc.append('Overall benchmark statistics and power consumption summary.\n\n')

        with doc.create(Tabular('|l|r|')) as table:
            table.add_hline()
            table.add_row([bold('Metric'), bold('Value')])
            table.add_hline()
            table.add_row(['Total Prompts', enhanced_data['summary']['total_prompts']])
            table.add_row(['Total Duration', f"{enhanced_data['summary']['total_duration_s']:.2f} s"])
            table.add_row(['Total Energy Consumption', f"{enhanced_data['summary']['total_energy_mj']:.1f} mJ"])
            table.add_row(['Average Inference Power', f"{enhanced_data['summary']['avg_inference_power_mw']:.0f} mW"])
            table.add_row(['Peak Power', f"{enhanced_data['summary']['peak_power_mw']} mW"])
            table.add_row(['Avg Tokens/Second', f"{enhanced_data['summary']['avg_tokens_per_second']:.1f}"])
            table.add_row(['Avg Time to First Token', f"{enhanced_data['summary']['avg_time_to_first_token_ms']:.1f} ms"])
            table.add_hline()

    # Overall Timeline Chart
    with doc.create(Section('Overall Power Timeline')):
        doc.append('Power consumption across all benchmark prompts.\n\n')

        try:
            fig = create_overall_timeline_chart(enhanced_data, power_samples)
            with doc.create(Figure(position='H')) as plot:
                plot.add_plot(width=NoEscape(r'\textwidth'), dpi=300)
                plot.add_caption('Overall power consumption timeline showing all prompts')
            plt.close(fig)
        except Exception as e:
            doc.append(f'Error generating overall timeline: {str(e)}\n\n')

    # Comparison Charts
    with doc.create(Section('Category Comparisons')):
        doc.append('Comparison of metrics across prompt categories (short, medium, long).\n\n')

        try:
            fig = create_comparison_charts(enhanced_data)
            with doc.create(Figure(position='H')) as plot:
                plot.add_plot(width=NoEscape(r'\textwidth'), dpi=300)
                plot.add_caption('Comparison of power and performance metrics by category')
            plt.close(fig)
        except Exception as e:
            doc.append(f'Error generating comparison charts: {str(e)}\n\n')

    # Per-Prompt Detailed Analysis
    with doc.create(Section('Detailed Per-Prompt Analysis')):
        doc.append('Detailed breakdown of each prompt execution.\n\n')

        for i, prompt in enumerate(enhanced_data['prompts']):
            with doc.create(Subsection(f'Prompt {i+1}: {prompt["category"].upper()}')):

                # Prompt text - full text in monospace
                prompt_text = prompt['prompt']
                doc.append(NoEscape(r'\textbf{Prompt:}'))
                doc.append('\n\n')
                # Write entire lstlisting block as raw LaTeX to avoid newline command issues
                lstlisting_block = f"\\begin{{lstlisting}}\n{prompt_text}\n\\end{{lstlisting}}\n"
                doc.append(NoEscape(lstlisting_block))

                # Response text - full text in monospace
                response_text = prompt['response']
                doc.append(NoEscape(r'\textbf{Response:}'))
                doc.append('\n\n')
                # Write entire lstlisting block as raw LaTeX to avoid newline command issues
                lstlisting_response = f"\\begin{{lstlisting}}\n{response_text}\n\\end{{lstlisting}}\n"
                doc.append(NoEscape(lstlisting_response))

                # Timing Information Table
                doc.append(NoEscape(r'\textbf{Timing Information:}'))
                doc.append('\n\n')
                timing = prompt['timing']
                with doc.create(Tabular('|l|l|')) as table:
                    table.add_hline()
                    table.add_row([bold('Event'), bold('Timestamp/Duration')])
                    table.add_hline()
                    table.add_row(['Prompt Submit Time', timing['prompt_submit_time']])
                    table.add_row(['First Token Time', timing['first_token_time'] or 'N/A'])
                    table.add_row(['Response Complete Time', timing['response_complete_time']])
                    table.add_row(['Inference Duration', f"{timing['inference_duration_s']:.3f} s"])
                    if timing['time_to_first_token_ms']:
                        table.add_row(['Time to First Token', f"{timing['time_to_first_token_ms']:.1f} ms"])
                    table.add_hline()
                doc.append('\n\n')

                # Inference Statistics Table
                doc.append(NoEscape(r'\textbf{Inference Statistics:}'))
                doc.append('\n\n')
                stats = prompt['inference_stats']
                with doc.create(Tabular('|l|l|')) as table:
                    table.add_hline()
                    table.add_row([bold('Metric'), bold('Value')])
                    table.add_hline()
                    if stats['tokens_per_second']:
                        table.add_row(['Tokens per Second', f"{stats['tokens_per_second']:.2f}"])
                    if stats['prompt_tokens']:
                        table.add_row(['Prompt Tokens', stats['prompt_tokens']])
                    if stats['completion_tokens']:
                        table.add_row(['Completion Tokens', stats['completion_tokens']])
                    if stats['total_tokens']:
                        table.add_row(['Total Tokens', stats['total_tokens']])
                    table.add_hline()
                doc.append('\n\n')

                # Power Analysis Table
                doc.append(NoEscape(r'\textbf{Power Consumption Analysis:}'))
                doc.append('\n\n')
                pa = prompt['power_analysis']

                with doc.create(Tabular('|l|c|c|c|c|')) as table:
                    table.add_hline()
                    table.add_row([bold('Phase'), bold('Min (mW)'), bold('Max (mW)'),
                                   bold('Avg (mW)'), bold('Samples')])
                    table.add_hline()

                    if pa['baseline']:
                        bl = pa['baseline']
                        table.add_row(['Baseline', bl['min_mw'], bl['max_mw'],
                                      f"{bl['avg_mw']:.0f}", bl['samples']])
                    else:
                        table.add_row(['Baseline', 'N/A', 'N/A', 'N/A', '0'])

                    if pa['inference']:
                        inf = pa['inference']
                        table.add_row(['Inference', inf['min_mw'], inf['max_mw'],
                                      f"{inf['avg_mw']:.0f}", inf['samples']])
                    else:
                        table.add_row(['Inference', 'N/A', 'N/A', 'N/A', '0'])

                    if pa['cooldown']:
                        cd = pa['cooldown']
                        table.add_row(['Cooldown', cd['min_mw'], cd['max_mw'],
                                      f"{cd['avg_mw']:.0f}", cd['samples']])
                    else:
                        table.add_row(['Cooldown', 'N/A', 'N/A', 'N/A', '0'])

                    table.add_hline()

                    # Add difference row if both baseline and inference exist
                    if pa['baseline'] and pa['inference']:
                        diff_mw = inf['avg_mw'] - bl['avg_mw']
                        pct_increase = (diff_mw / bl['avg_mw']) * 100 if bl['avg_mw'] > 0 else 0
                        table.add_row([NoEscape(r'$\Delta$ Inference-Baseline'), '', '',
                                      NoEscape(f"{diff_mw:+.0f} ({pct_increase:+.1f}\\%)"), ''])
                        table.add_hline()

                # Energy Summary Table
                doc.append(NoEscape(r'\textbf{Energy Summary:}'))
                doc.append('\n\n')

                with doc.create(Tabular('|l|r|')) as table:
                    table.add_hline()
                    table.add_row([bold('Metric'), bold('Value')])
                    table.add_hline()

                    if pa['energy_estimate_mj']:
                        table.add_row(['Total Energy', f"{pa['energy_estimate_mj']:.1f} mJ"])

                    if pa['peak_power_mw']:
                        table.add_row(['Peak Power', f"{pa['peak_power_mw']} mW"])

                    # Calculate and display incremental energy
                    if pa['baseline'] and pa['inference'] and pa['energy_estimate_mj']:
                        baseline_avg_mw = pa['baseline']['avg_mw']
                        inference_duration_s = prompt['timing']['inference_duration_s']

                        # Energy if baseline continued for full inference duration
                        # FIXED: No division needed - mW × s = mJ directly
                        baseline_continuation_energy_mj = baseline_avg_mw * inference_duration_s

                        # Extra energy consumed by inference workload above baseline
                        incremental_energy_mj = pa['energy_estimate_mj'] - baseline_continuation_energy_mj

                        # Percentage of total energy that's incremental
                        incremental_pct = (incremental_energy_mj / pa['energy_estimate_mj'] * 100) if pa['energy_estimate_mj'] > 0 else 0

                        table.add_row(['Incremental Energy',
                                      f"{incremental_energy_mj:.1f} mJ ({incremental_pct:.1f}%)"])

                    table.add_hline()

                doc.append('\n')

                # Individual timeline chart
                try:
                    fig = create_prompt_timeline_chart(prompt, power_samples, i)
                    with doc.create(Figure(position='H')) as plot:
                        plot.add_plot(width=NoEscape(r'\textwidth'), dpi=300)
                        plot.add_caption(f'Power timeline for Prompt {i+1}')
                    plt.close(fig)
                except Exception as e:
                    doc.append(f'Error generating timeline for this prompt: {str(e)}\n')

    # Generate PDF
    print(f"Compiling LaTeX to PDF...")
    try:
        doc.generate_pdf(output_file, clean_tex=False, compiler='lualatex')
        print(f"\n{'='*70}")
        print(f"PDF report generated successfully!")
        print(f"{'='*70}")
        print(f"Output: {output_file}.pdf")
        print(f"LaTeX source: {output_file}.tex")
        print(f"{'='*70}")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: LaTeX compilation failed!", file=sys.stderr)
        print(f"Check {output_file}.log for details", file=sys.stderr)
        raise
    except Exception as e:
        print(f"\nERROR: PDF generation failed: {e}", file=sys.stderr)
        raise


def generate_comparison_pdf_report(
    enhanced_data: Dict,
    power_samples: List[Dict],
    output_file: str
):
    """
    Generate comparison-mode PDF report showing all prompt versions.

    Args:
        enhanced_data: Enhanced benchmark data with prompt_groups
        power_samples: All power samples
        output_file: Output PDF path (without extension)
    """
    # Create document
    geometry_options = {
        "tmargin": "2cm",
        "bmargin": "2cm",
        "lmargin": "2cm",
        "rmargin": "2cm"
    }
    doc = Document(geometry_options=geometry_options, page_numbers=True)

    # Add required packages
    doc.packages.append(Package('booktabs'))
    doc.packages.append(Package('float'))
    doc.packages.append(Package('longtable'))
    doc.packages.append(Package('array'))
    doc.packages.append(Package('listings'))

    # Configure listings
    doc.preamble.append(NoEscape(r'\lstset{basicstyle=\small\ttfamily, breaklines=true, breakatwhitespace=false, columns=flexible}'))

    # Title page
    doc.preamble.append(Command('title', 'LM Studio Power Consumption Analysis\\\\\\large{Comparison Mode}'))
    doc.preamble.append(Command('author', f'Model: {enhanced_data["model"]}'))
    doc.preamble.append(Command('date', f'Benchmark: {enhanced_data["benchmark_id"]}'))
    doc.append(NoEscape(r'\maketitle'))

    # Executive Summary
    with doc.create(Section('Executive Summary')):
        doc.append('Comparison of three prompt optimization strategies:\n\n')

        with doc.create(Tabular('|l|r|r|r|')) as table:
            table.add_hline()
            table.add_row([bold('Version'), bold('Total Energy (mJ)'), bold('Avg Power (mW)'), bold('Peak Power (mW)')])
            table.add_hline()

            summary = enhanced_data['summary']
            for version_name in ['original', 'rule_optimized', 'llm_optimized']:
                if version_name in summary['by_version'] and summary['by_version'][version_name]:
                    stats = summary['by_version'][version_name]
                    table.add_row([
                        version_name.replace('_', ' ').title(),
                        f"{stats['total_energy_mj']:.1f}" if stats['total_energy_mj'] else 'N/A',
                        f"{stats['avg_inference_power_mw']:.0f}" if stats['avg_inference_power_mw'] else 'N/A',
                        f"{stats['peak_power_mw']}" if stats['peak_power_mw'] else 'N/A'
                    ])
            table.add_hline()

        # Add LLM optimization overhead breakdown
        doc.append('\n\n')
        if summary['by_version']['llm_optimized']:
            llm_stats = summary['by_version']['llm_optimized']
            if llm_stats.get('optimization_overhead_mj'):
                doc.append(NoEscape(r'\textbf{LLM Self-Optimization Breakdown:}\\'))
                doc.append(f"Optimization overhead: {llm_stats['optimization_overhead_mj']:.1f} mJ\\\\")
                doc.append(f"Execution energy: {llm_stats['execution_only_energy_mj']:.1f} mJ\\\\")
                doc.append(f"Total (optimization + execution): {llm_stats['total_energy_mj']:.1f} mJ\\\\")
                doc.append('\n\n')

        # Calculate and show savings
        if summary['by_version']['original'] and summary['by_version']['rule_optimized']:
            orig_e = summary['by_version']['original']['total_energy_mj']
            rule_e = summary['by_version']['rule_optimized']['total_energy_mj']
            savings_pct = ((orig_e - rule_e) / orig_e * 100) if orig_e > 0 else 0
            doc.append(NoEscape(f"\\textbf{{Rule-based optimization:}} {savings_pct:+.1f}\\% energy change\\\\"))

        if summary['by_version']['original'] and summary['by_version']['llm_optimized']:
            orig_e = summary['by_version']['original']['total_energy_mj']
            llm_e_total = summary['by_version']['llm_optimized']['total_energy_mj']  # Includes overhead
            llm_e_exec = summary['by_version']['llm_optimized'].get('execution_only_energy_mj', llm_e_total)

            savings_with_overhead = ((orig_e - llm_e_total) / orig_e * 100) if orig_e > 0 else 0
            savings_exec_only = ((orig_e - llm_e_exec) / orig_e * 100) if orig_e > 0 else 0

            doc.append(NoEscape(f"\\textbf{{LLM self-optimization:}}\\\\"))
            doc.append(NoEscape(f"\\hspace{{1em}} Including overhead: {savings_with_overhead:+.1f}\\% energy change\\\\"))
            doc.append(NoEscape(f"\\hspace{{1em}} Execution only: {savings_exec_only:+.1f}\\% energy change\\\\"))

    # Version Comparison Charts
    with doc.create(Section('Version Comparison')):
        doc.append('Comparison of power and energy metrics across optimization strategies.\n\n')

        try:
            fig = create_version_comparison_charts(enhanced_data)
            with doc.create(Figure(position='H')) as plot:
                plot.add_plot(width=NoEscape(r'\textwidth'), dpi=300)
                plot.add_caption('Comparison of optimization strategies')
            plt.close(fig)
        except Exception as e:
            doc.append(f'Error generating comparison charts: {str(e)}\n\n')

    # Per-Group Analysis
    with doc.create(Section('Detailed Per-Group Analysis')):
        doc.append('Detailed breakdown of each prompt group showing all three versions.\n\n')

        for group_id, group in enumerate(enhanced_data['prompt_groups']):
            with doc.create(Subsection(f'Group {group_id + 1}: {group["category"].upper()}')):
                doc.append(NoEscape(r'\textbf{Base Prompt:}\\'))
                base_prompt = group['base_prompt']
                lstlisting_block = f"\\begin{{lstlisting}}\n{base_prompt}\n\\end{{lstlisting}}\n"
                doc.append(NoEscape(lstlisting_block))

                # Comparison table for this group
                doc.append(NoEscape(r'\textbf{Version Comparison:}'))
                doc.append('\n\n')

                with doc.create(Tabular('|l|r|r|r|r|')) as table:
                    table.add_hline()
                    table.add_row([bold('Version'), bold('Total Energy (mJ)'), bold('Opt. OH (mJ)'),
                                  bold('Avg Power (mW)'), bold('Duration (s)')])
                    table.add_hline()

                    for version in group['versions']:
                        pa = version['power_analysis']
                        timing = version['timing']

                        # Get optimization overhead if it exists
                        opt_overhead = ''
                        if version['prompt_version'] == 'llm_optimized' and pa.get('optimization_overhead_mj'):
                            opt_overhead = f"{pa['optimization_overhead_mj']:.1f}"

                        table.add_row([
                            version['prompt_version'].replace('_', ' ').title(),
                            f"{pa['energy_estimate_mj']:.1f}" if pa.get('energy_estimate_mj') else 'N/A',
                            opt_overhead,
                            f"{pa['inference']['avg_mw']:.0f}" if pa.get('inference') else 'N/A',
                            f"{timing['inference_duration_s']:.2f}"
                        ])
                    table.add_hline()

                # Show optimized prompts
                doc.append('\n\n')
                for version in group['versions']:
                    if version['prompt_version'] != 'original':
                        doc.append(NoEscape(f"\\textbf{{{version['prompt_version'].replace('_', ' ').title()} Prompt:}}\\\\"))
                        lstlisting_block = f"\\begin{{lstlisting}}\n{version['prompt']}\n\\end{{lstlisting}}\n"
                        doc.append(NoEscape(lstlisting_block))

    # Generate PDF
    print(f"Compiling LaTeX to PDF...")
    try:
        doc.generate_pdf(output_file, clean_tex=False, compiler='lualatex')
        print(f"\n{'='*70}")
        print(f"PDF report generated successfully!")
        print(f"{'='*70}")
        print(f"Output: {output_file}.pdf")
        print(f"LaTeX source: {output_file}.tex")
        print(f"{'='*70}")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: LaTeX compilation failed!", file=sys.stderr)
        print(f"Check {output_file}.log for details", file=sys.stderr)
        raise
    except Exception as e:
        print(f"\nERROR: PDF generation failed: {e}", file=sys.stderr)
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Correlate LM Studio benchmark timing with powermetrics data and generate PDF report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic correlation with PDF output
  python3 correlate_power_timing.py benchmark.json power.csv

  # Custom time windows
  python3 correlate_power_timing.py benchmark.json power.csv --baseline 10 --cooldown 10

  # Save to specific output file
  python3 correlate_power_timing.py benchmark.json power.csv --output my_report

Note:
  - Requires LaTeX installation (MacTeX or BasicTeX on macOS)
  - Output will be <filename>.pdf
  - Install dependencies: pip install pylatex matplotlib
        """
    )

    parser.add_argument(
        'benchmark_json',
        help='Path to benchmark JSON file'
    )

    parser.add_argument(
        'power_csv',
        help='Path to power CSV file'
    )

    parser.add_argument(
        '--baseline',
        type=float,
        default=5.0,
        metavar='SECONDS',
        help='Baseline window duration in seconds (default: 5)'
    )

    parser.add_argument(
        '--cooldown',
        type=float,
        default=5.0,
        metavar='SECONDS',
        help='Cooldown window duration in seconds (default: 5)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        metavar='FILE',
        help='Output PDF file path without extension (default: <benchmark>_power_report)'
    )

    args = parser.parse_args()

    # Generate output filename if not provided
    if args.output is None:
        base = os.path.splitext(args.benchmark_json)[0]
        args.output = f"{base}_power_report"
    else:
        # Remove .pdf extension if user provided it
        if args.output.endswith('.pdf'):
            args.output = args.output[:-4]

    try:
        # Correlate data
        print("Loading and correlating data...")
        enhanced_data = correlate_power_timing(
            args.benchmark_json,
            args.power_csv,
            args.baseline,
            args.cooldown
        )

        # Load power samples for visualization
        power_samples = load_power_data(args.power_csv)

        # Generate PDF report
        generate_pdf_report(enhanced_data, power_samples, args.output)

        print(f"\nSummary:")

        # Handle different summary structures for comparison vs standard mode
        if enhanced_data.get('comparison_mode'):
            print(f"  Mode: Comparison")
            print(f"  Total base prompts: {enhanced_data['summary']['total_base_prompts']}")
            print(f"  Total executions: {enhanced_data['summary']['total_executions']}")
            print(f"  Total energy (all versions): {enhanced_data['summary']['total_energy_all_versions_mj']:.1f} mJ")

            # Show per-version breakdown
            for version_name, stats in enhanced_data['summary']['by_version'].items():
                if stats:
                    print(f"  {version_name.replace('_', ' ').title()}: {stats['total_energy_mj']:.1f} mJ")
        else:
            print(f"  Mode: Standard")
            print(f"  Total prompts: {enhanced_data['summary']['total_prompts']}")
            print(f"  Total energy: {enhanced_data['summary']['total_energy_mj']:.1f} mJ")
            print(f"  Avg inference power: {enhanced_data['summary']['avg_inference_power_mw']:.0f} mW")
            print(f"  Peak power: {enhanced_data['summary']['peak_power_mw']} mW")

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
