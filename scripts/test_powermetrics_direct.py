#!/usr/bin/env python3
"""
Debug script to test powermetrics output directly
"""
import subprocess
import sys
import time

print("Testing powermetrics directly...")
print("This script must be run with sudo")
print()

cmd = [
    "powermetrics",
    "-i", "1000",
    "-n", "3",
    "--samplers", "cpu_power,gpu_power,ane_power",
    "-f", "text",
]

print(f"Running command: {' '.join(cmd)}")
print("=" * 60)

try:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )

    print("Process started, PID:", process.pid)
    print("Waiting for output...")
    print("=" * 60)

    line_count = 0
    start_time = time.time()

    # Read with timeout
    import select
    while time.time() - start_time < 10:
        # Check if there's data to read
        if process.stdout in select.select([process.stdout], [], [], 0.1)[0]:
            line = process.stdout.readline()
            if line:
                line_count += 1
                print(f"[Line {line_count}] {line}", end='')
            else:
                break

        # Check if process ended
        if process.poll() is not None:
            print(f"\nProcess ended with return code: {process.returncode}")
            break

    print(f"\n{'=' * 60}")
    print(f"Total lines read: {line_count}")
    print(f"Time elapsed: {time.time() - start_time:.1f}s")

    # Get any stderr output
    process.terminate()
    process.wait()

    stderr_output = process.stderr.read()
    if stderr_output:
        print(f"\nStderr output:")
        print(stderr_output)

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
