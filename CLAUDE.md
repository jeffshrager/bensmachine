# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains a polarization simulation system that models the coupled dynamics between public and Supreme Court (SCOTUS) polarization over time. The system implements mathematical models to study how judicial term limits and other factors affect political polarization.

## Architecture

The codebase consists of three main Python scripts:

- **`model_sweep_termlimit.py`**: Primary simulator focused on sweeping SCOTUS term limits. Implements feedback loop between public and court polarization using differential equations with tanh activation.
- **`model_sweep_pretire.py`**: Minimal simulator focused on judicial attrition effects (retirement probabilities). Uses same core mathematical model with emphasis on stochastic retirement modeling.
- **`plot_rundir_public.py`**: Visualization tool that aggregates TSV results from simulation runs and plots public polarization trends.

### Core Mathematical Model

All simulators implement the same fundamental feedback system:
- Public polarization: `P_{t+1} = (1-λ)P_t + λ*tanh(γ S_t) + noise`
- Court appointments follow presidential party alignment
- Time step = 1 year, polarization values ∈ [-1, 1] where D=-1, R=+1

### Data Structures

- **`Justice`**: Represents individual justices with appointment year, term limits, and political lean
- **`Court`**: Manages justice lifecycle (appointments, expirations, retirements)
- **`SimParams`**: Configuration object for all simulation parameters
- **`SimResult`**: Output container with time series data

## Commands

### Development Environment
```bash
# Set up virtual environment (required for matplotlib)
python3 -m venv venv
source venv/bin/activate
pip install matplotlib
```

### Running Simulations

**Single run with term limits:**
```bash
source venv/bin/activate
python3 model_sweep_termlimit.py --start 1866 --end 2025 --term-limit 18 --alpha 0.15 --gain 1.5 --noise 0.03 --plot
```

**Sweep term limits using historical presidents:**
```bash
source venv/bin/activate
python3 model_sweep_termlimit.py --presidents-csv presidents.csv --sweep-term-limit "12,18,22,none" --outdir runs/term_sweep
```

**Sweep retirement probabilities:**
```bash
source venv/bin/activate
python3 model_sweep_pretire.py --presidents-csv presidents.csv --sweep-retire "0.00,0.02,0.05,0.08" --out runs/attrition
```

**Plot aggregated results:**
```bash
source venv/bin/activate
python3 plot_rundir_public.py --rundir runs/term_sweep --title "Term Limit Comparison"
```

### Input Data

- **`presidents.csv`**: Historical presidential timeline with columns `start_year,end_year,party` where party ∈ {D,R}
- Scripts can generate alternating president timelines if no CSV provided

### Output Format

Simulations generate TSV files with columns:
- `year`: Simulation year
- `president_party`: D or R  
- `public_polarization`: Public polarization value [-1,1]
- `scotus_polarization`: Court mean polarization [-1,1]

Results are typically saved to `runs/` subdirectories organized by sweep type.

## Key Parameters

- **`--alpha`**: Public response rate λ (how quickly public responds to court)
- **`--gain`**: Amplification factor γ inside tanh function  
- **`--noise`**: Gaussian noise standard deviation for public dynamics
- **`--term-limit`**: SCOTUS term limit in years (or "none" for lifetime tenure)
- **`--retire-prob`**: Annual stochastic retirement probability per justice

## Testing

No automated test suite present. Validation relies on:
1. Visual inspection of plots for reasonable behavior
2. Comparison across parameter sweeps
3. Historical plausibility checks against `presidents.csv` timeline