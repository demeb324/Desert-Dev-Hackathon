#!/usr/bin/env python3
"""
Test script to verify PowerMetricsParser works correctly
without requiring sudo access to powermetrics.
"""

import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(__file__))

from powermetrics_analyzer import PowerMetricsParser

# Sample powermetrics output
SAMPLE_OUTPUT = """*** Sampled system activity (Sat Nov  8 12:40:03 2025 -0700) (1012.83ms elapsed) ***

*** Running tasks ***
*** Kernel task ***
  CPU ms/s:   0.00
  Interrupt wakeups/s: 0.00

CPU Power: 9553 mW
GPU Power: 13 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 9566 mW

*** Sampled system activity (Sat Nov  8 12:40:04 2025 -0700) (1015.22ms elapsed) ***

*** Running tasks ***

CPU Power: 12420 mW
GPU Power: 25 mW
ANE Power: 5 mW
Combined Power (CPU + GPU + ANE): 12450 mW

*** Sampled system activity (Sat Nov  8 12:40:05 2025 -0700) (1001.45ms elapsed) ***

CPU Power: 8100 mW
GPU Power: 10 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 8110 mW
"""


def test_parser():
    """Test the PowerMetricsParser with sample data."""
    print("Testing PowerMetricsParser...")
    print("=" * 60)

    parser = PowerMetricsParser()

    # Parse sample output line by line
    for line in SAMPLE_OUTPUT.split('\n'):
        sample = parser.parse_line(line)
        if sample:
            print(f"Parsed sample: CPU={sample['cpu_power_mw']}mW, "
                  f"GPU={sample['gpu_power_mw']}mW, "
                  f"ANE={sample['ane_power_mw']}mW, "
                  f"Combined={sample['combined_power_mw']}mW")

    # Finalize to get the last sample
    final_sample = parser.finalize()
    if final_sample:
        print(f"Final sample: CPU={final_sample['cpu_power_mw']}mW, "
              f"GPU={final_sample['gpu_power_mw']}mW, "
              f"ANE={final_sample['ane_power_mw']}mW, "
              f"Combined={final_sample['combined_power_mw']}mW")

    print("=" * 60)
    print(f"Total samples collected: {len(parser.samples)}")

    # Verify we got 3 samples
    if len(parser.samples) == 3:
        print("✓ Parser test PASSED - collected expected 3 samples")

        # Verify first sample
        sample1 = parser.samples[0]
        assert sample1['cpu_power_mw'] == 9553
        assert sample1['gpu_power_mw'] == 13
        assert sample1['ane_power_mw'] == 0
        assert sample1['combined_power_mw'] == 9566
        print("✓ First sample values correct")

        # Verify second sample
        sample2 = parser.samples[1]
        assert sample2['cpu_power_mw'] == 12420
        assert sample2['gpu_power_mw'] == 25
        assert sample2['ane_power_mw'] == 5
        assert sample2['combined_power_mw'] == 12450
        print("✓ Second sample values correct")

        # Verify third sample
        sample3 = parser.samples[2]
        assert sample3['cpu_power_mw'] == 8100
        assert sample3['gpu_power_mw'] == 10
        assert sample3['ane_power_mw'] == 0
        assert sample3['combined_power_mw'] == 8110
        print("✓ Third sample values correct")

        print("\n✓ All parser tests PASSED!")
        return True
    else:
        print(f"✗ Parser test FAILED - expected 3 samples, got {len(parser.samples)}")
        return False


if __name__ == "__main__":
    success = test_parser()
    sys.exit(0 if success else 1)
