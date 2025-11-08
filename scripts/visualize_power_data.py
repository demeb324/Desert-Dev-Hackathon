#!/usr/bin/env python3
"""
PowerMetrics Data Visualization

This script visualizes power consumption data from the powermetrics analyzer CSV output
using matplotlib to create time series plots.
"""

import csv
import sys
from datetime import datetime
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def read_power_data(csv_file: str) -> Tuple[List[datetime], Dict[str, List[float]]]:
    """
    Read power data from CSV file.

    Args:
        csv_file: Path to the CSV file from powermetrics analyzer

    Returns:
        Tuple of (timestamps, power_data_dict)
    """
    timestamps = []
    cpu_power = []
    gpu_power = []
    combined_power = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse timestamp
            timestamp_str = row['timestamp']
            try:
                timestamp = datetime.strptime(timestamp_str, "%a %b %d %H:%M:%S %Y %z")
            except ValueError:
                # Try alternative format without timezone
                try:
                    timestamp = datetime.strptime(timestamp_str, "%a %b %d %H:%M:%S %Y")
                except ValueError:
                    print(f"Warning: Could not parse timestamp: {timestamp_str}")
                    continue

            timestamps.append(timestamp)
            cpu_power.append(float(row['cpu_power_mw']))
            gpu_power.append(float(row['gpu_power_mw']))
            combined_power.append(float(row['combined_power_mw']))

    power_data = {
        'cpu': cpu_power,
        'gpu': gpu_power,
        'combined': combined_power
    }

    return timestamps, power_data


def create_visualization(timestamps: List[datetime], power_data: Dict[str, List[float]],
                         output_file: str = None, show_plot: bool = True) -> None:
    """
    Create visualization with multiple subplots for power metrics.

    Args:
        timestamps: List of datetime objects
        power_data: Dictionary with 'cpu', 'gpu', 'combined' power data
        output_file: Path to save the plot (optional)
        show_plot: Whether to display the plot interactively
    """
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('PowerMetrics Power Consumption Over Time', fontsize=16, fontweight='bold')

    # Plot 1: Combined Power
    axes[0].plot(timestamps, power_data['combined'], color='#2E86AB', linewidth=1.5, label='Combined Power')
    axes[0].set_ylabel('Power (mW)', fontsize=12)
    axes[0].set_title('Combined Power (CPU + GPU + ANE)', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(loc='upper right')

    # Add statistics annotation for combined power
    avg_combined = sum(power_data['combined']) / len(power_data['combined'])
    max_combined = max(power_data['combined'])
    min_combined = min(power_data['combined'])
    axes[0].axhline(y=avg_combined, color='red', linestyle='--', alpha=0.5, linewidth=1, label=f'Avg: {avg_combined:.0f} mW')
    axes[0].text(0.02, 0.95, f'Max: {max_combined:.0f} mW\nMin: {min_combined:.0f} mW\nAvg: {avg_combined:.0f} mW',
                 transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: CPU Power
    axes[1].plot(timestamps, power_data['cpu'], color='#A23B72', linewidth=1.5, label='CPU Power')
    axes[1].set_ylabel('Power (mW)', fontsize=12)
    axes[1].set_title('CPU Power', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(loc='upper right')

    # Add statistics annotation for CPU power
    avg_cpu = sum(power_data['cpu']) / len(power_data['cpu'])
    max_cpu = max(power_data['cpu'])
    min_cpu = min(power_data['cpu'])
    axes[1].axhline(y=avg_cpu, color='red', linestyle='--', alpha=0.5, linewidth=1, label=f'Avg: {avg_cpu:.0f} mW')
    axes[1].text(0.02, 0.95, f'Max: {max_cpu:.0f} mW\nMin: {min_cpu:.0f} mW\nAvg: {avg_cpu:.0f} mW',
                 transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: GPU Power
    axes[2].plot(timestamps, power_data['gpu'], color='#F18F01', linewidth=1.5, label='GPU Power')
    axes[2].set_ylabel('Power (mW)', fontsize=12)
    axes[2].set_xlabel('Time', fontsize=12)
    axes[2].set_title('GPU Power', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].legend(loc='upper right')

    # Add statistics annotation for GPU power
    avg_gpu = sum(power_data['gpu']) / len(power_data['gpu'])
    max_gpu = max(power_data['gpu'])
    min_gpu = min(power_data['gpu'])
    axes[2].axhline(y=avg_gpu, color='red', linestyle='--', alpha=0.5, linewidth=1, label=f'Avg: {avg_gpu:.0f} mW')
    axes[2].text(0.02, 0.95, f'Max: {max_gpu:.0f} mW\nMin: {min_gpu:.0f} mW\nAvg: {avg_gpu:.0f} mW',
                 transform=axes[2].transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Format x-axis to show time nicely
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save to file if specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")

    # Show plot if requested
    if show_plot:
        plt.show()


def visualize_power_data(csv_file: str, output_file: str = None, show_plot: bool = True) -> None:
    """
    Main function to visualize power data from CSV.

    Args:
        csv_file: Path to the CSV file from powermetrics analyzer
        output_file: Path to save the visualization (optional)
        show_plot: Whether to display the plot interactively
    """
    print(f"Reading power data from: {csv_file}")
    timestamps, power_data = read_power_data(csv_file)

    print(f"Loaded {len(timestamps)} data points")
    print(f"Time range: {timestamps[0]} to {timestamps[-1]}")

    create_visualization(timestamps, power_data, output_file, show_plot)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_power_data.py <csv_file> [output_image]")
        print("Example: python visualize_power_data.py power_analysis.csv power_plot.png")
        sys.exit(1)

    csv_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    # Auto-generate output filename if not provided
    if output_file is None:
        import os
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        output_file = f"{base_name}_visualization.png"

    visualize_power_data(csv_file, output_file, show_plot=False)
