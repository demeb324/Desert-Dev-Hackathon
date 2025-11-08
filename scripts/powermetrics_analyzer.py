#!/usr/bin/env python3
"""
PowerMetrics Live Analyzer

This script invokes macOS powermetrics and analyzes power usage data in real-time,
extracting power measurements into CSV format with timestamps.
"""

import re
import csv
import sys
import subprocess
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Callable


class PowerMetricsParser:
    """Parser for powermetrics output."""

    def __init__(self):
        """Initialize regex patterns for parsing."""
        self.timestamp_pattern = re.compile(r'\*\*\* Sampled system activity \((.*?)\) \(([\d.]+)ms elapsed\)')
        self.cpu_power_pattern = re.compile(r'CPU Power: (\d+) mW')
        self.gpu_power_pattern = re.compile(r'GPU Power: (\d+) mW')
        self.ane_power_pattern = re.compile(r'ANE Power: (\d+) mW')
        self.combined_power_pattern = re.compile(r'Combined Power \(CPU \+ GPU \+ ANE\): (\d+) mW')

        self.current_sample = {}
        self.samples = []

    def parse_line(self, line: str) -> Optional[Dict]:
        """
        Parse a single line from powermetrics output.

        Args:
            line: Line from powermetrics output

        Returns:
            Complete sample dict if a sample is finished, None otherwise
        """
        # Check for timestamp (indicates new sample)
        timestamp_match = self.timestamp_pattern.search(line)
        if timestamp_match:
            # Save previous sample if it exists and has data
            completed_sample = None
            if self.current_sample and 'combined_power_mw' in self.current_sample:
                completed_sample = self.current_sample.copy()
                self.samples.append(completed_sample)

            # Start new sample
            timestamp_str = timestamp_match.group(1)
            elapsed_ms = float(timestamp_match.group(2))

            # Parse timestamp
            timestamp = datetime.strptime(timestamp_str, "%a %b %d %H:%M:%S %Y %z")

            self.current_sample = {
                'timestamp': timestamp_str,
                'datetime': timestamp,
                'elapsed_ms': elapsed_ms,
                'cpu_power_mw': None,
                'gpu_power_mw': None,
                'ane_power_mw': None,
                'combined_power_mw': None
            }

            return completed_sample

        # Extract power measurements
        cpu_match = self.cpu_power_pattern.search(line)
        if cpu_match and self.current_sample:
            self.current_sample['cpu_power_mw'] = int(cpu_match.group(1))

        gpu_match = self.gpu_power_pattern.search(line)
        if gpu_match and self.current_sample:
            self.current_sample['gpu_power_mw'] = int(gpu_match.group(1))

        ane_match = self.ane_power_pattern.search(line)
        if ane_match and self.current_sample:
            self.current_sample['ane_power_mw'] = int(ane_match.group(1))

        combined_match = self.combined_power_pattern.search(line)
        if combined_match and self.current_sample:
            self.current_sample['combined_power_mw'] = int(combined_match.group(1))

        return None

    def finalize(self) -> Optional[Dict]:
        """
        Finalize parsing and return the last sample if it exists.

        Returns:
            Last sample dict if it exists, None otherwise
        """
        if self.current_sample and 'combined_power_mw' in self.current_sample:
            self.samples.append(self.current_sample)
            return self.current_sample
        return None


def run_powermetrics_streaming(
    interval_ms: int = 1000,
    sample_count: int = 60,
    callback: Optional[Callable[[Dict], None]] = None
) -> List[Dict]:
    """
    Run powermetrics in streaming mode with real-time parsing.

    Args:
        interval_ms: Sampling interval in milliseconds
        sample_count: Number of samples to collect
        callback: Optional callback function called for each completed sample

    Returns:
        List of collected samples

    Raises:
        PermissionError: If not running with sudo privileges
        subprocess.CalledProcessError: If powermetrics fails
    """
    cmd = [
        "sudo", "powermetrics",
        "-i", str(interval_ms),
        "-n", str(sample_count),
        "--samplers", "cpu_power,gpu_power,ane_power",
        "-f", "text",
        "-b", "1"  # Line buffering for real-time output
    ]

    print(f"Starting powermetrics (streaming mode): {sample_count} samples at {interval_ms}ms intervals")
    print(f"This will take approximately {sample_count * interval_ms / 1000:.1f} seconds...")
    print()

    parser = PowerMetricsParser()
    samples_collected = 0

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )

        # Process output line by line
        for line in process.stdout:
            sample = parser.parse_line(line)
            if sample:
                samples_collected += 1
                print(f"Sample {samples_collected}/{sample_count}: "
                      f"CPU={sample['cpu_power_mw']}mW, "
                      f"GPU={sample['gpu_power_mw']}mW, "
                      f"Combined={sample['combined_power_mw']}mW")

                if callback:
                    callback(sample)

        # Wait for process to complete
        process.wait()

        # Check for errors
        if process.returncode != 0:
            stderr_output = process.stderr.read()
            if "superuser" in stderr_output.lower():
                raise PermissionError("powermetrics requires sudo privileges. Run with: sudo python3 powermetrics_analyzer.py")
            else:
                raise subprocess.CalledProcessError(process.returncode, cmd, stderr=stderr_output)

        # Finalize any remaining sample
        final_sample = parser.finalize()
        if final_sample and final_sample not in parser.samples:
            samples_collected += 1
            print(f"Sample {samples_collected}/{sample_count}: "
                  f"CPU={final_sample['cpu_power_mw']}mW, "
                  f"GPU={final_sample['gpu_power_mw']}mW, "
                  f"Combined={final_sample['combined_power_mw']}mW")
            if callback:
                callback(final_sample)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Terminating powermetrics...")
        process.terminate()
        process.wait()
        print(f"Collected {samples_collected} samples before interruption")

    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()

    return parser.samples


def run_powermetrics_batch(
    interval_ms: int = 1000,
    sample_count: int = 60
) -> List[Dict]:
    """
    Run powermetrics in batch mode, collecting all output before parsing.

    Args:
        interval_ms: Sampling interval in milliseconds
        sample_count: Number of samples to collect

    Returns:
        List of collected samples

    Raises:
        PermissionError: If not running with sudo privileges
        subprocess.CalledProcessError: If powermetrics fails
    """
    cmd = [
        "sudo", "powermetrics",
        "-i", str(interval_ms),
        "-n", str(sample_count),
        "--samplers", "cpu_power,gpu_power,ane_power",
        "-f", "text"
    ]

    print(f"Starting powermetrics (batch mode): {sample_count} samples at {interval_ms}ms intervals")
    print(f"This will take approximately {sample_count * interval_ms / 1000:.1f} seconds...")
    print("Waiting for powermetrics to complete...")
    print()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=sample_count * interval_ms / 1000 + 30,  # Add 30s buffer
            check=True
        )

        # Parse all output at once
        parser = PowerMetricsParser()
        for line in result.stdout.splitlines():
            parser.parse_line(line)

        # Finalize any remaining sample
        parser.finalize()

        print(f"Collected {len(parser.samples)} samples")

        return parser.samples

    except subprocess.CalledProcessError as e:
        if "superuser" in e.stderr.lower():
            raise PermissionError("powermetrics requires sudo privileges. Run with: sudo python3 powermetrics_analyzer.py")
        else:
            raise

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"powermetrics timed out after {sample_count * interval_ms / 1000 + 30} seconds")


def write_to_csv(samples: List[Dict], output_file: str) -> None:
    """
    Write samples to CSV file.

    Args:
        samples: List of power measurement samples
        output_file: Path to output CSV file
    """
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

    print(f"\nData written to: {output_file}")


def print_statistics(samples: List[Dict]) -> None:
    """
    Print summary statistics of power usage.

    Args:
        samples: List of power measurement samples
    """
    cpu_powers = [s['cpu_power_mw'] for s in samples if s['cpu_power_mw'] is not None]
    gpu_powers = [s['gpu_power_mw'] for s in samples if s['gpu_power_mw'] is not None]
    combined_powers = [s['combined_power_mw'] for s in samples if s['combined_power_mw'] is not None]

    if not combined_powers:
        print("No power data collected")
        return

    print("\n" + "="*60)
    print("Power Usage Statistics")
    print("="*60)
    print(f"CPU Power     - Min: {min(cpu_powers):4d} mW, Max: {max(cpu_powers):5d} mW, Avg: {sum(cpu_powers)/len(cpu_powers):6.1f} mW")
    print(f"GPU Power     - Min: {min(gpu_powers):4d} mW, Max: {max(gpu_powers):5d} mW, Avg: {sum(gpu_powers)/len(gpu_powers):6.1f} mW")
    print(f"Combined Power - Min: {min(combined_powers):4d} mW, Max: {max(combined_powers):5d} mW, Avg: {sum(combined_powers)/len(combined_powers):6.1f} mW")
    print("="*60)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Invoke macOS powermetrics and analyze power usage in real-time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stream 60 samples at 1s intervals (default)
  sudo python3 powermetrics_analyzer.py

  # Batch mode with custom parameters
  sudo python3 powermetrics_analyzer.py --mode batch --interval 500 --samples 120

  # Stream for 5 minutes with 2s intervals
  sudo python3 powermetrics_analyzer.py --interval 2000 --samples 150 --output long_run.csv

Note: This script requires sudo privileges to run powermetrics.
        """
    )

    parser.add_argument(
        '--mode',
        choices=['stream', 'batch'],
        default='stream',
        help='Processing mode: stream (real-time) or batch (wait then parse). Default: stream'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=1000,
        metavar='MS',
        help='Sampling interval in milliseconds. Default: 1000 (1 second)'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=60,
        metavar='N',
        help='Number of samples to collect. Default: 60 (1 minute at 1s intervals)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        metavar='FILE',
        help='Output CSV file path. Default: powermetrics_<timestamp>.csv'
    )

    args = parser.parse_args()

    # Generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"powermetrics_{timestamp}.csv"

    try:
        # Run powermetrics in selected mode
        if args.mode == 'stream':
            samples = run_powermetrics_streaming(
                interval_ms=args.interval,
                sample_count=args.samples
            )
        else:  # batch
            samples = run_powermetrics_batch(
                interval_ms=args.interval,
                sample_count=args.samples
            )

        # Write results
        if samples:
            write_to_csv(samples, args.output)
            print_statistics(samples)
        else:
            print("No samples collected")
            sys.exit(1)

    except PermissionError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        print("\nThis script must be run with sudo privileges:", file=sys.stderr)
        print(f"  sudo python3 {sys.argv[0]} {' '.join(sys.argv[1:])}", file=sys.stderr)
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nScript interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
