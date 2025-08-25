# UAV Wingspan Comparative Analysis

Analyzes how wingspan affects UAV flight performance using the FlightSword simulation framework.

## Overview

This project investigates the relationship between UAV wingspan and flight performance, comparing 6 wingspan configurations (0.7m - 2.0m) across different wind conditions.

## Key Files

- `DifferentWingSpan.py` - Generates aircraft model parameters for different wingspans
- `analysis/wingspan_comparative_analysis.py` - Main performance analysis and comparison tool
- `analysis/merge_wingspan_inertial_traces.py` - 3D trajectory visualization
- `analysis/generate_separate_ar_performance_plots.py` - Individual performance plots

## Usage

### Prerequisites

1. First, follow the FlightSword framework setup guide: https://github.com/FluffyCodeMonster/FlightSwordLite.git
2. Run the FlightSword simulations to generate flight data for all wingspan configurations (which provided by this project in data/models) and wind conditions
3. Ensure simulation results are saved in the expected data directories

### Analysis Workflow

```bash
# 1. Generate aircraft models for different wingspans
python DifferentWingSpan.py

# 2. Run FlightSword simulations (follow FlightSword documentation)
# This step generates the flight data required for analysis

# 3. Run performance analysis (requires simulation data)
python analysis/wingspan_comparative_analysis.py

# 4. Create trajectory plots
python analysis/merge_wingspan_inertial_traces.py

# 5. Generate performance plots
python analysis/generate_separate_ar_performance_plots.py
```

## Dependencies

- NumPy/Pandas - Data processing
- Matplotlib - Visualization
- FlightSword Framework - Core simulation engine
- JAX - Numerical computing

## Attribution

Based on FlightSword framework: https://github.com/FluffyCodeMonster/FlightSwordLite.git