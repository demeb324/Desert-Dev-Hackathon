# LM Studio Inference Benchmark with Power Monitoring

Tools for benchmarking LLM inference and correlating with power consumption using macOS powermetrics.

## Overview

This directory contains two main scripts that work together:

1. **`powermetrics_analyzer.py`** - Collects power consumption data from macOS powermetrics
2. **`lm_studio_power_benchmark.py`** - Runs LLM inference benchmarks and records detailed timing

The scripts run **separately** and results are correlated via timestamps.

## Requirements

### System Requirements
- macOS with Apple Silicon (M1/M2/M3/M4)
- sudo/root access (required only for powermetrics)
- LM Studio installed and running

### Python Dependencies
```bash
pip install lmstudio
```

## Quick Start - Two Terminal Workflow

### Terminal 1: Start Power Monitoring
```bash
# Start collecting power data (runs until Ctrl+C)
sudo python3 scripts/powermetrics_analyzer.py --samples 1000 --output power_data.csv
```

This will collect power samples at 1-second intervals and save to CSV.

### Terminal 2: Run Benchmark
```bash
# No sudo needed for benchmark!
python3 scripts/lm_studio_power_benchmark.py --output benchmark.json
```

This will:
- Connect to LM Studio
- Run 9 prompts (3 short, 3 medium, 3 long)
- Record detailed timing for each inference
- Generate a JSON report with timestamps

### Stop Power Monitoring
When the benchmark completes, press `Ctrl+C` in Terminal 1 to stop power collection.

### Correlate Results
Use the timestamps in `benchmark.json` to find corresponding power samples in `power_data.csv`.

## Why Two Separate Scripts?

Running power monitoring and benchmarks separately provides:
- **Simpler architecture** - No threading complexity
- **Better reliability** - powermetrics runs independently
- **Flexibility** - Can start power monitoring before benchmark
- **No sudo for benchmark** - Only powermetrics needs root access
- **Easier debugging** - Each component works independently

## Benchmark Suite

The benchmark runs 9 prompts across 3 categories:

### Short Prompts (~10-20 tokens)
- Quick factual questions
- Testing baseline inference overhead
- Examples: "What is 2+2?", "Name three colors."

### Medium Prompts (~50-100 tokens)
- Explanatory questions
- Typical chatbot interactions
- Examples: "Explain photosynthesis in simple terms."

### Long Prompts (~200-300 tokens)
- Complex analysis requests
- Longer context processing
- Examples: "Write a detailed analysis of renewable energy sources..."

## Output Formats

### Benchmark JSON Output
```json
{
  "benchmark_id": "2025-11-08T14:30:00.123456",
  "model": "qwen/qwen3-4b-2507",
  "summary": {
    "total_prompts": 9,
    "avg_tokens_per_second": 45.2,
    "avg_time_to_first_token_ms": 523.4,
    "total_duration_s": 125.3
  },
  "prompts": [
    {
      "category": "short",
      "prompt": "What is 2+2?",
      "response": "2+2 equals 4.",
      "timing": {
        "prompt_submit_time": "2025-11-08T14:30:01.234567",
        "first_token_time": "2025-11-08T14:30:01.756789",
        "response_complete_time": "2025-11-08T14:30:02.123456",
        "inference_duration_s": 0.889,
        "time_to_first_token_ms": 522.2
      },
      "inference_stats": {
        "prompt_tokens": 5,
        "completion_tokens": 6,
        "tokens_per_second": 42.5
      }
    }
  ]
}
```

### Power CSV Output
```csv
timestamp,elapsed_ms,cpu_power_mw,gpu_power_mw,ane_power_mw,combined_power_mw
Sat Nov  8 14:30:01 2025 -0700,1012.83,9553,13,0,9566
Sat Nov  8 14:30:02 2025 -0700,1015.22,12420,25,5,12450
...
```

## Usage Examples

### Basic Benchmark
```bash
# Terminal 1
sudo python3 scripts/powermetrics_analyzer.py --output power.csv

# Terminal 2
python3 scripts/lm_studio_power_benchmark.py --output benchmark.json
```

### Longer Monitoring Session
```bash
# Terminal 1 - collect 10 minutes of power data
sudo python3 scripts/powermetrics_analyzer.py --samples 600 --output power_10min.csv

# Terminal 2 - run benchmark with custom pause between prompts
python3 scripts/lm_studio_power_benchmark.py --pause 10 --output benchmark.json
```

### Specific Model
```bash
# Terminal 2 only needs to specify model
python3 scripts/lm_studio_power_benchmark.py --model qwen/qwen3-4b-2507
```

## Command Line Options

### powermetrics_analyzer.py
```bash
sudo python3 powermetrics_analyzer.py [OPTIONS]

Options:
  --mode {stream,batch}    Processing mode (default: stream)
  --interval MS            Sampling interval in ms (default: 1000)
  --samples N              Number of samples (default: 60)
  --output FILE            Output CSV path
```

### lm_studio_power_benchmark.py
```bash
python3 lm_studio_power_benchmark.py [OPTIONS]

Options:
  --model MODEL            LM Studio model identifier
  --pause SECONDS          Pause between prompts (default: 5)
  --output FILE            Output JSON path
```

## Correlating Timestamps

The benchmark JSON contains ISO 8601 timestamps with microsecond precision:
```
prompt_submit_time: "2025-11-08T14:30:01.234567"
response_complete_time: "2025-11-08T14:30:02.123456"
```

The power CSV contains human-readable timestamps:
```
Sat Nov  8 14:30:01 2025 -0700
```

To correlate:
1. Parse timestamps from both files
2. Find power samples between `prompt_submit_time` and `response_complete_time`
3. Calculate average/min/max power during inference

## Tips for Accurate Measurements

### Before Running
1. **Close unnecessary applications** - Minimize background activity
2. **Disable automatic updates** - Prevent system tasks during benchmark
3. **Let system stabilize** - Wait a minute after opening LM Studio
4. **Plugin power** - Use AC power for consistent performance

### During Measurement
1. **Start power monitoring first** - Get baseline before benchmark
2. **Don't interact with system** - Avoid mouse/keyboard during benchmark
3. **Monitor temperature** - High temp can affect power/performance
4. **Run multiple iterations** - Average results across runs

### After Measurement
1. **Verify sample counts** - Check power CSV has expected number of samples
2. **Check for anomalies** - Look for unexpected power spikes
3. **Document conditions** - Note model, quant, temp, etc.

## Troubleshooting

### "powermetrics must be invoked as the superuser"
- Run powermetrics_analyzer.py with `sudo`
- Benchmark script does not need sudo

### "Failed to connect to LM Studio"
- Verify LM Studio is running
- Ensure local server is started (green button)
- Check server is on port 1234

### "lmstudio package not found"
- Install: `pip install lmstudio`
- Verify: `python3 -c "import lmstudio"`

### No power samples in CSV
- Verify you ran powermetrics_analyzer.py with sudo
- Check that powermetrics is available: `which powermetrics`
- On non-Apple Silicon Macs, power metrics may be limited

### Timestamps don't align
- Ensure system clocks are synchronized
- Check timezone matches between files
- Verify both scripts ran on same machine

## Example Analysis Workflow

```bash
# 1. Start power monitoring
sudo python3 scripts/powermetrics_analyzer.py --samples 600 --output power.csv &
POWER_PID=$!

# 2. Wait for baseline
sleep 10

# 3. Run benchmark
python3 scripts/lm_studio_power_benchmark.py --output benchmark.json

# 4. Let cooldown complete
sleep 10

# 5. Stop power monitoring
kill $POWER_PID

# 6. Analyze results
python3 scripts/correlate_power_timing.py power.csv benchmark.json
```

## Advanced Usage

### Custom Prompts
Edit `BENCHMARK_PROMPTS` dict in `lm_studio_power_benchmark.py` to add your own test cases.

### Streaming vs Batch Power Collection
```bash
# Stream mode (default) - see real-time samples
sudo python3 scripts/powermetrics_analyzer.py --mode stream

# Batch mode - wait for all samples then process
sudo python3 scripts/powermetrics_analyzer.py --mode batch
```

### High Frequency Sampling
```bash
# Sample every 500ms for detailed power profile
sudo python3 scripts/powermetrics_analyzer.py --interval 500 --samples 1200
```

## License

MIT License - See main repository LICENSE file

## Contributing

Contributions welcome! Please test thoroughly on Apple Silicon hardware before submitting PRs.

## Related Tools

- **powermetrics** - macOS system utility for power measurement
- **LM Studio** - Local LLM inference platform
- **lmstudio-python** - Official Python SDK for LM Studio
