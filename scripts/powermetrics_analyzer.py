#!/usr/bin/env python3
"""
PowerMetrics Output Analyzer

This script analyzes macOS powermetrics output and extracts power usage data
into a CSV format with timestamps and power measurements.
"""

import re
import csv
from datetime import datetime
from typing import List, Dict, Tuple


def parse_powermetrics_output(input_file: str, output_csv: str) -> None:
    """
    Parse powermetrics output file and extract power data to CSV.

    Args:
        input_file: Path to the powermetrics output text file
        output_csv: Path to the output CSV file
    """
    samples = []

    # Regular expressions for parsing
    timestamp_pattern = re.compile(r'\*\*\* Sampled system activity \((.*?)\) \(([\d.]+)ms elapsed\)')
    cpu_power_pattern = re.compile(r'CPU Power: (\d+) mW')
    gpu_power_pattern = re.compile(r'GPU Power: (\d+) mW')
    ane_power_pattern = re.compile(r'ANE Power: (\d+) mW')
    combined_power_pattern = re.compile(r'Combined Power \(CPU \+ GPU \+ ANE\): (\d+) mW')

    current_sample = {}

    with open(input_file, 'r') as f:
        for line in f:
            # Check for timestamp
            timestamp_match = timestamp_pattern.search(line)
            if timestamp_match:
                # Save previous sample if exists
                if current_sample:
                    samples.append(current_sample)

                # Start new sample
                timestamp_str = timestamp_match.group(1)
                elapsed_ms = float(timestamp_match.group(2))

                # Parse timestamp
                timestamp = datetime.strptime(timestamp_str, "%a %b %d %H:%M:%S %Y %z")

                current_sample = {
                    'timestamp': timestamp_str,
                    'datetime': timestamp,
                    'elapsed_ms': elapsed_ms,
                    'cpu_power_mw': None,
                    'gpu_power_mw': None,
                    'ane_power_mw': None,
                    'combined_power_mw': None
                }

            # Extract power measurements
            cpu_match = cpu_power_pattern.search(line)
            if cpu_match and current_sample:
                current_sample['cpu_power_mw'] = int(cpu_match.group(1))

            gpu_match = gpu_power_pattern.search(line)
            if gpu_match and current_sample:
                current_sample['gpu_power_mw'] = int(gpu_match.group(1))

            ane_match = ane_power_pattern.search(line)
            if ane_match and current_sample:
                current_sample['ane_power_mw'] = int(ane_match.group(1))

            combined_match = combined_power_pattern.search(line)
            if combined_match and current_sample:
                current_sample['combined_power_mw'] = int(combined_match.group(1))

    # Don't forget the last sample
    if current_sample and current_sample['combined_power_mw'] is not None:
        samples.append(current_sample)

    # Write to CSV
    write_to_csv(samples, output_csv)

    # Print summary statistics
    print(f"Processed {len(samples)} samples")
    if samples:
        print_statistics(samples)


def write_to_csv(samples: List[Dict], output_file: str) -> None:
    """Write samples to CSV file."""
    if not samples:
        print("No samples to write")
        return

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'elapsed_ms', 'cpu_power_mw', 'gpu_power_mw',
                      'ane_power_mw', 'combined_power_mw']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for sample in samples:
            row = {
                'timestamp': sample['timestamp'],
                'elapsed_ms': sample['elapsed_ms'],
                'cpu_power_mw': sample['cpu_power_mw'],
                'gpu_power_mw': sample['gpu_power_mw'],
                'ane_power_mw': sample['ane_power_mw'],
                'combined_power_mw': sample['combined_power_mw']
            }
            writer.writerow(row)


def print_statistics(samples: List[Dict]) -> None:
    """Print summary statistics of power usage."""
    cpu_powers = [s['cpu_power_mw'] for s in samples if s['cpu_power_mw'] is not None]
    gpu_powers = [s['gpu_power_mw'] for s in samples if s['gpu_power_mw'] is not None]
    combined_powers = [s['combined_power_mw'] for s in samples if s['combined_power_mw'] is not None]

    print("\nPower Usage Statistics:")
    print(f"CPU Power - Min: {min(cpu_powers)} mW, Max: {max(cpu_powers)} mW, Avg: {sum(cpu_powers)/len(cpu_powers):.1f} mW")
    print(f"GPU Power - Min: {min(gpu_powers)} mW, Max: {max(gpu_powers)} mW, Avg: {sum(gpu_powers)/len(gpu_powers):.1f} mW")
    print(f"Combined Power - Min: {min(combined_powers)} mW, Max: {max(combined_powers)} mW, Avg: {sum(combined_powers)/len(combined_powers):.1f} mW")


def analyze_powermetrics(input_file: str, output_csv: str = None) -> Tuple[List[Dict], str]:
    """
    Main function to analyze powermetrics output.

    Args:
        input_file: Path to the powermetrics output file
        output_csv: Path to output CSV file (optional, will be auto-generated if not provided)

    Returns:
        Tuple of (samples list, output csv path)
    """
    if output_csv is None:
        # Generate output filename based on input
        import os
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_csv = f"{base_name}_power_analysis.csv"

    parse_powermetrics_output(input_file, output_csv)

    # Return the samples for further processing if needed
    samples = []
    with open(output_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)

    return samples, output_csv


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python powermetrics_analyzer.py <input_file> [output_csv]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None

    analyze_powermetrics(input_file, output_csv)
