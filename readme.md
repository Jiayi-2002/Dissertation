# UAV Wingspan Comparative Analysis

Code for analyzing how wingspan affects UAV flight performance, built upon the FlightSword simulation framework.

## Project Overview

This project investigates the relationship between UAV wingspan and flight performance metrics:
- Flight time optimization
- Lift-to-drag ratio efficiency  
- Path following accuracy
- Control activity requirements
- Energy utilization patterns

## Project Structure

### Core Analysis Files (Added by Jiayi)

#### `DifferentWingSpan.py`
Calculates aircraft model parameters for different wingspans:
- Generate inertia matrices and aerodynamic coefficients for 6 wingspan variants (0.7m - 2.0m)
- Wing inertia calculation using flat plate model
- Aileron effectiveness scaling with wingspan
- Optimal L/D ratio predictions
- Exports parameters for FlightSword model files

#### `analysis/wingspan_comparative_analysis.py` 
Main performance analysis and comparison tool:
- Analyze flight simulation results across wingspans and wind conditions
- Load and process 18 flight configurations (6 wingspans × 3 wind speeds)
- Extract performance metrics (flight time, L/D ratio, path efficiency, control activity)
- Generate comprehensive comparison plots
- Export data tables for further analysis

#### `analysis/merge_wingspan_inertial_traces.py`
3D trajectory visualization tool:
- Create detailed 3D flight path visualizations with energy analysis
- Plot flight trajectories for each wingspan configuration
- Color-code paths by inertial power (energy gain/loss)
- Compare multiple wind speeds on single plots
- Generate publication-quality 3D figures

#### `analysis/generate_separate_ar_performance_plots.py`
Individual performance plot generator:
- Create focused plots for each performance metric
- Generate 6 individual AR vs performance plots
- Wind speed comparison across all metrics
- High-resolution output for publications

### Simulation Framework (Original FlightSword)

The underlying simulation framework is based on FlightSword from:
```
https://github.com/FluffyCodeMonster/FlightSwordLite.git
```

FlightSword Components Used:
- `flight/simulator/` - Core UAV dynamics and simulation engine
- `flight/analysis/` - Base analysis tools and plotting utilities  
- `data/models/` - Aircraft model definitions (modified for different wingspans)
- `data/dataloggers/` - Simulation results storage
- All other foundational simulation infrastructure

## Experimental Setup

### Wingspan Configurations
- 0.7m (AR = 1.6) - Low aspect ratio configuration
- 0.9m (AR = 2.7) - Entry-level configuration  
- 1.206m (AR = 4.8) - Baseline reference
- 1.5m (AR = 7.5) - Medium aspect ratio
- 1.8m (AR = 10.8) - High aspect ratio
- 2.0m (AR = 13.3) - Maximum span configuration

### Wind Conditions
- 3 m/s - Low wind environment
- 5 m/s - Moderate wind (baseline)
- 7 m/s - High wind environment

### Fixed Parameters
- Wing area: 0.3 m²
- Total mass: 1.345 kg
- Flight mission: Optimized trajectory between waypoints

## Key Outputs

### Performance Analysis
1. AR vs Performance Curves - Comprehensive 6-metric comparison
2. Wind Speed Analysis - Environmental sensitivity assessment
3. Optimization Heatmaps - Best wingspan selection guidance

### Trajectory Analysis  
1. 3D Flight Paths - Detailed trajectory visualization
2. Energy Analysis - Inertial power distribution along flight paths
3. Multi-wind Comparison - Environmental adaptation patterns

### Data Export
1. Performance Metrics CSV - Quantitative analysis data
2. Individual Plots - Publication-ready figures
3. Model Parameters - Generated aircraft configurations

## Usage

### Generate Aircraft Models
```bash
python DifferentWingSpan.py
```

### Run Performance Analysis
```bash
python analysis/wingspan_comparative_analysis.py
```

### Create 3D Trajectory Plots
```bash
python analysis/merge_wingspan_inertial_traces.py
```

### Generate Individual Performance Plots
```bash
python analysis/generate_separate_ar_performance_plots.py
```

## Research Applications

This analysis framework supports research in:
- UAV Design Optimization - Wingspan selection for specific missions
- Environmental Adaptation - Wind sensitivity analysis
- Flight Performance Trade-offs - Multi-objective design decisions
- Control System Design - Stability and control authority requirements

## Technical Implementation

### Dependencies
- NumPy/Pandas - Data processing and analysis
- Matplotlib - Visualization and plotting
- FlightSword Framework - Core simulation engine
- JAX - High-performance numerical computing (FlightSword dependency)

### Analysis Pipeline
1. Model Generation → Aircraft parameters for each wingspan
2. Simulation Execution → FlightSword optimization runs  
3. Data Processing → Performance metric extraction
4. Visualization → Comprehensive analysis plots
5. Export → Results and publication materials

## Attribution

- Base Simulation Framework: FlightSword by FluffyCodeMonster
- Wingspan Analysis Extension: Custom analysis tools for comparative UAV performance research
- Original FlightSword: https://github.com/FluffyCodeMonster/FlightSwordLite.git

---

This project extends the FlightSword simulation framework with specialized tools for wingspan comparative analysis in UAV design research.