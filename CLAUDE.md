# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **LLM Power Consumption Research Tool** that measures and analyzes the energy efficiency of different prompt optimization strategies for local LLM inference on Apple Silicon Macs. The project investigates whether removing polite lexicon from prompts can reduce power consumption and improve inference efficiency.

The system compares three versions of each prompt:
1. **Original** - Prompts with polite language ("please", "thank you", etc.)
2. **Rule-Optimized** - Prompts processed with regex-based lexicon removal
3. **LLM-Optimized** - Prompts optimized by the LLM itself for succinctness

## Architecture

The project consists of three main Python scripts that form a complete benchmarking pipeline:

### Core Components

1. **`powermetrics_analyzer.py`** (410 lines)
   - Collects real-time power consumption data from macOS powermetrics utility
   - Measures CPU, GPU, and ANE (Apple Neural Engine) power draw
   - Outputs timestamped CSV files with power measurements at 1-second intervals
   - **Requires sudo** to access system power metrics

2. **`lm_studio_power_benchmark.py`** (768 lines)
   - Core benchmarking tool that runs LLM inference tests via LM Studio API
   - Executes benchmark suite across three prompt categories (short, medium, long)
   - Records microsecond-precision timing data for each inference
   - Supports comparison mode (`--compare` flag) to run all three prompt versions
   - Outputs detailed JSON reports with inference statistics and timing metadata

3. **`correlate_power_timing.py`** (1,582 lines)
   - Correlates benchmark timing data with power measurements using timestamps
   - Analyzes energy consumption for each prompt version (including LLM optimization overhead)
   - Generates comprehensive PDF reports with LaTeX visualization
   - Creates comparison charts and statistical analysis
   - Handles both standard and comparison mode reporting

### Supporting Modules

- **`optimizer.py`** - Python implementation of text optimization (lexicon removal, comma removal, whitespace normalization)
- **`visualize_power_data.py`** - Power data visualization utilities
- **`optimizer.ts`** - Legacy TypeScript implementation (still present but Python is primary)
- **`tokenwise/`** - React + TypeScript + Vite frontend application (web interface)

### Data Flow

```
Terminal 1: powermetrics_analyzer.py → power.csv
                                              ↘
Terminal 2: lm_studio_power_benchmark.py → comparison.json → correlate_power_timing.py → report.pdf
                                              ↗
Imported: optimizer.py (rule-based optimization)
```

## Development Commands

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- LM Studio installed and running with a model loaded
- Python 3.x with dependencies: `lmstudio`, `pylatex`, `matplotlib`
- sudo access for power metrics collection
- LuaLaTeX installed for PDF generation

### Standard Workflow

**Step 1: Start Power Collection (Terminal 1)**
```bash
# Collect power data for ~10 minutes (600 samples at 1-second intervals)
sudo python3 scripts/powermetrics_analyzer.py --samples 600 --output power.csv
```

**Step 2: Run Benchmark Suite (Terminal 2)**
```bash
# Run comparison benchmark with all three prompt versions
python3 scripts/lm_studio_power_benchmark.py --compare --output comparison.json

# Or run standard benchmark with original prompts only
python3 scripts/lm_studio_power_benchmark.py --output benchmark.json
```

**Step 3: Generate Analysis Report**
```bash
# Correlate timing and power data, generate PDF report
python3 scripts/correlate_power_timing.py comparison.json power.csv

# Specify custom output location
python3 scripts/correlate_power_timing.py comparison.json power.csv --output custom_report.pdf
```

### Additional Options

```bash
# Customize baseline and cooldown windows for power analysis
python3 scripts/correlate_power_timing.py comparison.json power.csv --baseline 2.0 --cooldown 3.0

# Use custom LLM optimization prompt
python3 scripts/lm_studio_power_benchmark.py --compare --optimization-prompt "Make this concise:"

# Adjust pause between benchmarks
python3 scripts/lm_studio_power_benchmark.py --compare --pause 3.0
```

## Project Structure

```
.
├── scripts/
│   ├── powermetrics_analyzer.py       # Power data collection
│   ├── lm_studio_power_benchmark.py   # LLM benchmarking tool
│   ├── correlate_power_timing.py      # Analysis and PDF generation
│   ├── optimizer.py                   # Text optimization module
│   ├── optimizer.ts                   # Legacy TypeScript version
│   ├── README_POWER_BENCHMARK.md      # Detailed usage guide
│   └── (other utility scripts)
├── tokenwise/                         # React frontend application
├── power-data/                        # Historical power measurement data
├── .gitignore                         # Excludes CSV, JSON, PDF, TEX, LOG files
└── CLAUDE.md                          # This file
```

## Key Technologies

- **Python Stack**: `lmstudio` SDK, `pylatex`, `matplotlib`, standard library (`re`, `subprocess`, `csv`, `json`)
- **System Tools**: macOS `powermetrics` utility (requires sudo), LuaLaTeX compiler
- **Frontend**: React + TypeScript + Vite + TailwindCSS (secondary component)
- **LM Studio**: Local LLM inference server

## Output Files

- **`*.csv`** - Timestamped power measurement data (CPU/GPU/ANE power in milliwatts)
- **`*.json`** - Benchmark results with timing metadata, inference statistics, and prompt versions
- **`*.pdf`** - Comprehensive analysis reports with visualizations, energy calculations, and comparisons
- **`*.tex`** - LaTeX source files for PDF reports

All output files are git-ignored to keep the repository clean.

## Detailed Documentation

For comprehensive usage instructions, troubleshooting, and technical details, see:
- **`scripts/README_POWER_BENCHMARK.md`** - Complete guide to the power benchmarking workflow

## Current Development Branch

The `live-powermetrics` branch contains the latest power measurement improvements and should be used for active development.
