# LM Studio Power Consumption Benchmark

Tools for measuring power consumption during LLM inference using macOS powermetrics and LM Studio.

## Overview

This directory contains two main scripts:

1. **`powermetrics_analyzer.py`** - Standalone tool for collecting and analyzing macOS power metrics
2. **`lm_studio_power_benchmark.py`** - Benchmark suite that measures power consumption during LM Studio inference

## Requirements

### System Requirements
- macOS with Apple Silicon (M1/M2/M3/M4)
- sudo/root access (required for powermetrics)
- LM Studio installed and running

### Python Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- `lmstudio` - Official LM Studio Python SDK

## Quick Start

### 1. Install Dependencies
```bash
pip install lmstudio
```

### 2. Start LM Studio
1. Open LM Studio application
2. Download and load a model (e.g., qwen/qwen3-4b-2507)
3. Start the local server (green "Start Server" button)

### 3. Run Benchmark
```bash
sudo python3 scripts/lm_studio_power_benchmark.py
```

The script will:
- Connect to LM Studio
- Run 9 prompts (3 short, 3 medium, 3 long)
- Measure power consumption before, during, and after each inference
- Generate a JSON report with detailed metrics

## Usage

### Basic Usage
```bash
# Use currently loaded model with default settings (10s baseline/cooldown)
sudo python3 lm_studio_power_benchmark.py
```

### Specify Model
```bash
# Load and benchmark a specific model
sudo python3 lm_studio_power_benchmark.py --model qwen/qwen3-4b-2507
```

### Custom Timing
```bash
# Longer baseline and cooldown periods
sudo python3 lm_studio_power_benchmark.py --baseline 15 --cooldown 15
```

### Custom Output
```bash
# Specify output file location
sudo python3 lm_studio_power_benchmark.py --output results/my_benchmark.json
```

### All Options
```bash
sudo python3 lm_studio_power_benchmark.py \
  --model qwen/qwen3-4b-2507 \
  --baseline 15 \
  --cooldown 15 \
  --output benchmark_results.json
```

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

## Output Format

### JSON Report Structure
```json
{
  "benchmark_id": "2025-11-08T14:30:00.000000",
  "model": "qwen/qwen3-4b-2507",
  "summary": {
    "total_prompts": 9,
    "avg_baseline_power_mw": 950,
    "avg_inference_power_mw": 22500,
    "avg_cooldown_power_mw": 4500,
    "avg_tokens_per_second": 45.2
  },
  "prompts": [
    {
      "category": "short",
      "prompt": "What is 2+2?",
      "response": "2+2 equals 4.",
      "timing": {
        "baseline_duration_s": 10.0,
        "inference_duration_s": 0.8,
        "cooldown_duration_s": 10.0
      },
      "inference_stats": {
        "prompt_tokens": 5,
        "completion_tokens": 6,
        "tokens_per_second": 7.5,
        "time_to_first_token_ms": 120.5,
        "inference_duration_s": 0.8
      },
      "power_consumption": {
        "baseline": {
          "min_mw": 800,
          "max_mw": 1200,
          "avg_mw": 950,
          "sample_count": 10
        },
        "inference": {
          "min_mw": 15000,
          "max_mw": 28000,
          "avg_mw": 22500,
          "sample_count": 1
        },
        "cooldown": {
          "min_mw": 2000,
          "max_mw": 8000,
          "avg_mw": 4500,
          "sample_count": 10
        }
      }
    }
  ]
}
```

## How It Works

### Power Measurement Process

1. **Baseline Phase (10s default)**
   - Measures idle power consumption before inference
   - Establishes baseline system power usage
   - Helps isolate inference-specific power draw

2. **Inference Phase (variable duration)**
   - Submits prompt to LM Studio
   - Measures power during model computation
   - Tracks Time To First Token (TTFT) and generation speed
   - Duration varies based on prompt complexity and response length

3. **Cooldown Phase (10s default)**
   - Measures power consumption after inference completes
   - Captures system return to baseline
   - Shows GPU/CPU cooldown behavior

### Technical Implementation

- **Threading**: Runs powermetrics in background thread for continuous monitoring
- **Time Synchronization**: Uses relative timestamps to align power samples with inference phases
- **LM Studio SDK**: Direct integration with LM Studio's official Python SDK
- **Progress Tracking**: Callbacks for first token and completion events

## Power Metrics Explained

### Power Measurements (in milliwatts - mW)

- **CPU Power**: Main processor power consumption
- **GPU Power**: Graphics/Neural Engine power (critical for LLM inference)
- **ANE Power**: Apple Neural Engine (if utilized by model)
- **Combined Power**: Total of CPU + GPU + ANE

### Expected Power Ranges (Apple Silicon)

- **Baseline**: 500-2000 mW (idle system)
- **Inference**: 15,000-35,000 mW (varies by model size and batch)
- **Cooldown**: 2,000-10,000 mW (gradual return to baseline)

### Factors Affecting Power Consumption

- **Model Size**: Larger models (7B, 13B+) consume more power
- **Quantization**: Lower precision (Q4, Q5) reduces power vs FP16
- **Context Length**: Longer prompts increase processing power
- **Response Length**: More tokens generated = more power consumed
- **System Load**: Background processes affect baseline

## Troubleshooting

### "sudo: a password is required"
- The script must be run with sudo for powermetrics access
- Run: `sudo python3 lm_studio_power_benchmark.py`

### "Failed to connect to LM Studio"
- Verify LM Studio is running
- Ensure local server is started (green button in LM Studio)
- Check server is on default port 1234
- Try loading a model first in LM Studio GUI

### "lmstudio package not found"
- Install the SDK: `pip install lmstudio`
- Verify installation: `python3 -c "import lmstudio"`

### No power samples collected
- Verify you have sudo access
- Check that powermetrics is available: `which powermetrics`
- On non-Apple Silicon Macs, powermetrics may have limited functionality

### High baseline power
- Close background applications
- Wait for system to stabilize before running benchmark
- Consider increasing baseline duration: `--baseline 20`

## Standalone Powermetrics Tool

### Basic Power Monitoring
```bash
# Stream 60 samples at 1s intervals (default)
sudo python3 scripts/powermetrics_analyzer.py

# Custom duration and interval
sudo python3 scripts/powermetrics_analyzer.py --interval 500 --samples 120

# Batch mode (wait for completion)
sudo python3 scripts/powermetrics_analyzer.py --mode batch --samples 30
```

### Output
- CSV file with time-series power data
- Summary statistics (min/max/avg)
- Supports both streaming and batch modes

## Example Workflow

```bash
# 1. Install dependencies
pip install lmstudio

# 2. Start LM Studio and load a model

# 3. Run quick test (reduced timing for testing)
sudo python3 scripts/lm_studio_power_benchmark.py \
  --baseline 5 \
  --cooldown 5 \
  --output test_run.json

# 4. View results
cat test_run.json | python3 -m json.tool

# 5. Run full benchmark with production settings
sudo python3 scripts/lm_studio_power_benchmark.py \
  --baseline 15 \
  --cooldown 15 \
  --output production_benchmark.json
```

## Tips for Accurate Measurements

1. **Minimize Background Activity**
   - Close unnecessary applications
   - Disable automatic backups during testing
   - Stop browser and other heavy processes

2. **Thermal Considerations**
   - Allow system to cool between benchmarks
   - Monitor system temperature
   - Consider ambient temperature effects

3. **Consistency**
   - Use same power settings (plugged in vs battery)
   - Run multiple iterations and average results
   - Document system state (CPU usage, temp, etc.)

4. **Model Selection**
   - Test same model with different quantizations
   - Compare similar-sized models
   - Document model parameters (size, quant, context)

## License

MIT License - See main repository LICENSE file

## Contributing

Contributions welcome! Please test thoroughly on Apple Silicon hardware before submitting PRs.
