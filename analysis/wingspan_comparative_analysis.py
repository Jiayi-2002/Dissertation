import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# FlightSword imports
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from flight.analysis import results_plotting, analysis_calculations
from flight.simulator.utils import DataLogger
from flight.simulator.aircraft_model import AircraftModel
from flight.simulator import config

# =====================================================================================
# CONFIGURATION
# =====================================================================================

# Wingspan configurations and wind speeds to analyze
WINGSPAN_CONFIGS = {
    '0p7m': {'wingspan': 0.7, 'base_name': 'test_jiayi_wingspan_0p7m'},
    '0p9m': {'wingspan': 0.9, 'base_name': 'test_jiayi_wingspan_0p9m'},
    '1p206m_baseline': {'wingspan': 1.206, 'base_name': 'test_jiayi_wingspan_1p206m_baseline'},
    '1p5m': {'wingspan': 1.5, 'base_name': 'test_jiayi_wingspan_1p5m'},
    '1p8m': {'wingspan': 1.8, 'base_name': 'test_jiayi_wingspan_1p8m'},
    '2p0m': {'wingspan': 2.0, 'base_name': 'test_jiayi_wingspan_2p0m'},
}

# Wind speed configurations
WIND_SPEEDS = {
    'ws3': {'speed': 3, 'suffix': '_ws3', 'color': '#1f77b4', 'name': '3 m/s'},
    'ws5': {'speed': 5, 'suffix': '', 'color': '#ff7f0e', 'name': '5 m/s'},  # Original data
    'ws7': {'speed': 7, 'suffix': '_ws7', 'color': '#2ca02c', 'name': '7 m/s'},
}

# Fixed parameters
WING_AREA = 0.3  # m²
TOTAL_MASS = 1.345  # kg
WING_MASS = 0.4035  # kg

# Wind conditions
WIND_DIRECTION = 270  # degrees

# Analysis output configuration
OUTPUT_DIR = Path('./wingspan_analysis_results')
FIG_DPI = 600
FONT_SIZE = 12
plt.rcParams.update({'font.size': FONT_SIZE})


class WingspanPerformanceAnalyzer:
    """Main class for wingspan comparative analysis."""
    
    def __init__(self):
        self.dataloggers = {}
        self.aircraft_models = {}
        self.performance_metrics = {}
        self.optimization_results = {}
        
        # Create output directories
        self.setup_output_directories()
        
    def setup_output_directories(self):
        """Create necessary output directories."""
        dirs = ['basic_performance', 'data_tables']
        
        for dir_name in dirs:
            (OUTPUT_DIR / dir_name).mkdir(parents=True, exist_ok=True)
            
        print(f"Output directories created in: {OUTPUT_DIR}")

    def load_all_wingspan_data(self):
        """Load DataLoggers and aircraft models for all wingspan and wind speed configurations."""
        print("\n" + "="*60)
        print("LOADING ALL WINGSPAN DATA (ALL WIND SPEEDS)")
        print("="*60)
        
        successful_loads = {}
        failed_loads = {}
        
        # Nested loop: wingspan x wind speed
        for config_id, config in WINGSPAN_CONFIGS.items():
            for wind_id, wind_config in WIND_SPEEDS.items():
                # Create unique identifier for this configuration
                full_config_id = f"{config_id}_{wind_id}"
                
                try:
                    print(f"\nLoading {full_config_id} (wingspan: {config['wingspan']}m, wind: {wind_config['name']})...")
                    
                    # Construct DataLogger path
                    if config_id == '1p206m_baseline' and wind_id == 'ws5':
                        # Special case: baseline data at 5m/s doesn't have suffix
                        dl_name = config['base_name']
                    elif config_id == '1p206m_baseline':
                        # Special case: baseline data with wind speed suffix  
                        dl_name = f"test_jiayi_wingspan_1p206m{wind_config['suffix']}"
                    else:
                        # Regular case
                        dl_name = f"{config['base_name']}{wind_config['suffix']}"
                    
                    dl_path = Path(f"data/dataloggers/{dl_name}_optimiser")
                    if not dl_path.exists():
                        raise FileNotFoundError(f"DataLogger not found: {dl_path}")
                        
                    dl = DataLogger.load_from_path(dl_path)
                    self.dataloggers[full_config_id] = dl
                    
                    # Load Aircraft Model (same for all wind speeds of a wingspan)
                    if config_id == '1p206m_baseline':
                        model_path = Path(f"data/models/wot4_imav_v2.yaml")
                    else:
                        model_path = Path(f"data/models/wot4_imav_v2_wingspan_{config_id}.yaml")
                        if not model_path.exists():
                            # Try without 'p' in filename (e.g., 0p7m -> 0.7m)
                            wingspan_str = f"{config['wingspan']}m" 
                            model_path = Path(f"data/models/wot4_imav_v2_wingspan_{wingspan_str}.yaml")
                        if not model_path.exists():
                            model_path = Path(f"data/models/wot4_imav_v2.yaml")  # Fallback to default
                    
                    am = AircraftModel(model_path)
                    self.aircraft_models[full_config_id] = am
                    
                    # Check optimization status
                    opt_status = self.check_optimization_status(dl_name)
                    self.optimization_results[full_config_id] = opt_status
                    
                    successful_loads[full_config_id] = {
                        'wingspan': config['wingspan'],
                        'wind_speed': wind_config['speed'],
                        'wind_name': wind_config['name'],
                        'data_points': len(dl.times),
                        'flight_time': dl.times[-1],
                        'optimization_status': opt_status['status']
                    }
                    
                    print(f"   Success: {len(dl.times)} data points, "
                          f"flight time: {dl.times[-1]:.2f}s, status: {opt_status['status']}")
                    
                except Exception as e:
                    failed_loads[full_config_id] = str(e)
                    print(f"   Failed: {e}")
        
        # Summary
        total_expected = len(WINGSPAN_CONFIGS) * len(WIND_SPEEDS)
        print(f"\nLOADING SUMMARY:")
        print(f"   Successfully loaded: {len(successful_loads)}/{total_expected} configurations")
        print(f"   Wingspans: {len(WINGSPAN_CONFIGS)}, Wind speeds: {len(WIND_SPEEDS)}")
        
        if successful_loads:
            df_summary = pd.DataFrame.from_dict(successful_loads, orient='index')
            print(f"\n{df_summary}")

            
        if failed_loads:
            print(f"\nFailed loads:")
            for config_id, error in failed_loads.items():
                print(f"   {config_id}: {error}")
                
        return successful_loads, failed_loads

    def check_optimization_status(self, experiment_name):
        """Check the optimization convergence status from IPOPT output."""
        try:
            ipopt_path = Path(f"data/solver_output/{experiment_name}/IPOPT.out")
            if not ipopt_path.exists():
                return {'status': 'IPOPT file not found', 'objective': None, 'time': None}
            
            with open(ipopt_path, 'r') as f:
                content = f.read()
            
            # Extract key information
            lines = content.split('\n')
            
            # Find EXIT status
            exit_line = [line for line in lines if line.startswith('EXIT:')]
            status = exit_line[0].replace('EXIT: ', '') if exit_line else 'Unknown'
            
            # Find objective value
            obj_lines = [line for line in lines if 'Objective' in line and 'scaled' in line]
            objective = None
            if obj_lines:
                try:
                    obj_val = obj_lines[0].split(':')[1].split()[0]
                    objective = float(obj_val)
                except:
                    pass
            
            # Find computation time
            time_lines = [line for line in lines if 'Total seconds in IPOPT' in line]
            comp_time = None
            if time_lines:
                try:
                    comp_time = float(time_lines[0].split('=')[1].strip())
                except:
                    pass
            
            return {
                'status': status,
                'objective': objective,
                'computation_time': comp_time
            }
            
        except Exception as e:
            return {'status': f'Error reading IPOPT: {e}', 'objective': None, 'time': None}

    def extract_key_performance_metrics(self):
        """Extract key performance metrics for all loaded configurations."""
        print("\n" + "="*60)
        print("EXTRACTING KEY PERFORMANCE METRICS")
        print("="*60)
        
        metrics_data = {}
        
        for config_id, dl in self.dataloggers.items():
            print(f"\nAnalyzing {config_id}...")
            
            # Parse config_id to get wingspan and wind speed info
            # Format: "wingspan_id_wind_id" (e.g., "0p7m_ws3")
            wingspan_id = '_'.join(config_id.split('_')[:-1])  # Get everything except last part
            wind_id = config_id.split('_')[-1]  # Get last part (wind speed)
            
            am = self.aircraft_models[config_id]
            wingspan = WINGSPAN_CONFIGS[wingspan_id]['wingspan']
            wind_speed = WIND_SPEEDS[wind_id]['speed']
            
            try:
                # Basic flight metrics
                flight_time = dl.times[-1]
                total_distance = self.calculate_total_distance(dl)
                straight_line_distance = self.calculate_straight_line_distance(dl)
                path_efficiency = straight_line_distance / total_distance
                
                # Speed metrics
                groundspeed = np.sqrt(dl.us**2 + dl.vs**2 + dl.ws**2)
                avg_groundspeed = np.mean(groundspeed)
                avg_airspeed = np.mean(dl.vas)
                
                # Aerodynamic metrics
                lift_drag_ratio = self.calculate_lift_drag_ratio(dl)
                avg_lift_drag = np.mean(lift_drag_ratio[lift_drag_ratio > 0])  # Positive values only
                
                # Energy metrics
                initial_energy = self.calculate_specific_energy(dl, am, 0)
                final_energy = self.calculate_specific_energy(dl, am, -1)
                energy_change = final_energy - initial_energy
                
                # Control activity
                control_activity = self.calculate_control_activity(dl)
                
                # Load factor analysis
                avg_load_factor = np.mean(dl.load_factors)
                max_load_factor = np.max(dl.load_factors)
                
                # Aspect ratio
                aspect_ratio = wingspan**2 / WING_AREA
                
                metrics_data[config_id] = {
                    'wingspan_m': wingspan,
                    'wind_speed_ms': wind_speed,
                    'aspect_ratio': aspect_ratio,
                    'flight_time_s': flight_time,
                    'total_distance_m': total_distance,
                    'straight_line_distance_m': straight_line_distance,
                    'path_efficiency': path_efficiency,
                    'avg_groundspeed_ms': avg_groundspeed,
                    'avg_airspeed_ms': avg_airspeed,
                    'avg_lift_drag_ratio': avg_lift_drag,
                    'energy_change_J_kg': energy_change,
                    'control_activity': control_activity,
                    'avg_load_factor': avg_load_factor,
                    'max_load_factor': max_load_factor,
                    'optimization_status': self.optimization_results[config_id]['status']
                }
                
                print(f"   Metrics extracted successfully")
                
            except Exception as e:
                print(f"   Error extracting metrics: {e}")
                metrics_data[config_id] = {'error': str(e)}
        
        self.performance_metrics = metrics_data
        
        # Create and save performance metrics DataFrame
        df_metrics = pd.DataFrame.from_dict(metrics_data, orient='index')
        df_metrics = df_metrics.round(4)
        
        print(f"\nPERFORMANCE METRICS SUMMARY:")
        print(df_metrics[['wingspan_m', 'wind_speed_ms', 'aspect_ratio', 'flight_time_s', 'avg_lift_drag_ratio', 
                         'path_efficiency', 'energy_change_J_kg']].to_string())
        
        # Save metrics
        df_metrics.to_csv(OUTPUT_DIR / 'data_tables' / 'performance_metrics.csv')
        
        return df_metrics

    def calculate_total_distance(self, dl):
        """Calculate total flight path distance."""
        dx = np.diff(dl.ns)
        dy = np.diff(dl.es) 
        dz = np.diff(dl.ds)
        distances = np.sqrt(dx**2 + dy**2 + dz**2)
        return np.sum(distances)
    
    def calculate_straight_line_distance(self, dl):
        """Calculate straight-line distance from start to end."""
        dx = dl.ns[-1] - dl.ns[0]
        dy = dl.es[-1] - dl.es[0]
        dz = dl.ds[-1] - dl.ds[0]
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def calculate_lift_drag_ratio(self, dl):
        """Calculate lift-to-drag ratio."""
        # Avoid division by zero
        drag_safe = np.where(np.abs(dl.Ds) > 1e-6, dl.Ds, np.nan)
        return np.abs(dl.Ls) / np.abs(drag_safe)
    
    def calculate_specific_energy(self, dl, am, index):
        """Calculate specific energy at given index."""
        return -am.g * dl.ds[index] + 0.5 * (dl.us[index]**2 + dl.vs[index]**2 + dl.ws[index]**2)
    
    def calculate_control_activity(self, dl):
        """Calculate control surface activity (RMS of deflections)."""
        da_rms = np.sqrt(np.mean(dl.das**2))
        de_rms = np.sqrt(np.mean(dl.des**2))
        dr_rms = np.sqrt(np.mean(dl.drs**2))
        return da_rms + de_rms + dr_rms

    def plot_ar_vs_performance_curves(self):
        """Generate AR vs performance curves for key metrics with wind speed comparison."""
        print("\n" + "="*60)
        print("GENERATING AR vs PERFORMANCE CURVES WITH WIND SPEED COMPARISON")
        print("="*60)
        
        if not self.performance_metrics:
            print("No performance metrics available. Run extract_key_performance_metrics() first.")
            return
        
        # Prepare data
        df = pd.DataFrame.from_dict(self.performance_metrics, orient='index')
        df = df.dropna() 
        
        # Define key metrics to plot
        metrics_to_plot = [
            ('flight_time_s', 'Flight Time (s)'),
            ('avg_lift_drag_ratio', 'Average L/D Ratio'),
            ('path_efficiency', 'Path Efficiency'),
            ('avg_groundspeed_ms', 'Average Groundspeed (m/s)'),
            ('energy_change_J_kg', 'Energy Change (J/kg)'),
            ('control_activity', 'Control Activity (RMS)')
        ]
        
        # Create comprehensive plot with wind speed comparison
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, (metric, label) in enumerate(metrics_to_plot):
            ax = axes[i]
            
            # Plot separate lines for each wind speed
            for wind_id, wind_config in WIND_SPEEDS.items():
                # Filter data for this wind speed
                wind_data = df[df['wind_speed_ms'] == wind_config['speed']].copy()
                
                if len(wind_data) > 0:
                    # Sort by aspect ratio for smooth line
                    wind_data = wind_data.sort_values('aspect_ratio')
                    
                    x = wind_data['aspect_ratio']
                    y = wind_data[metric]
                    
                    # Plot line and points
                    ax.plot(x, y, color=wind_config['color'], linewidth=2.5, 
                           label=f"Wind {wind_config['name']}", marker='o', markersize=8,
                           alpha=0.8)
                    
                    # Add wingspan annotations (only for one wind speed to avoid clutter)
                    if wind_id == 'ws5':  # Only annotate for 5m/s wind speed
                        for ar, val, wingspan in zip(x, y, wind_data['wingspan_m']):
                            ax.annotate(f'{wingspan:.1f}m', (ar, val), xytext=(5, 5), 
                                       textcoords='offset points', fontsize=8, alpha=0.7)
            
            ax.set_xlabel('Aspect Ratio')
            ax.set_ylabel(label)
            ax.set_title(f'{label} vs Aspect Ratio')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'basic_performance' / 'ar_vs_performance_curves_wind_comparison.png', 
                   dpi=FIG_DPI, bbox_inches='tight')
        
        # Create wind speed specific analysis plots
        self.create_wind_speed_comparison_plots(df)
        
        print(f"AR vs Performance curves with wind speed comparison saved")
        return fig
    
    def create_wind_speed_comparison_plots(self, df):
        """Create detailed wind speed comparison plots."""
        print("\nCreating wind speed comparison plots...")
        
        # 1. Wind Speed Effect on Flight Time
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Flight Time vs Wind Speed for each wingspan
        ax = axes[0, 0]
        for wingspan_id in WINGSPAN_CONFIGS.keys():
            data_for_wingspan = []
            wind_speeds = []
            
            for wind_id, wind_config in WIND_SPEEDS.items():
                config_key = f"{wingspan_id}_{wind_id}"
                if config_key in df.index:
                    data_for_wingspan.append(df.loc[config_key, 'flight_time_s'])
                    wind_speeds.append(wind_config['speed'])
            
            if data_for_wingspan:
                wingspan = WINGSPAN_CONFIGS[wingspan_id]['wingspan']
                ax.plot(wind_speeds, data_for_wingspan, marker='o', linewidth=2,
                       label=f'{wingspan}m span')
        
        ax.set_xlabel('Wind Speed (m/s)')
        ax.set_ylabel('Flight Time (s)')
        ax.set_title('Flight Time vs Wind Speed by Wingspan')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # L/D Ratio vs Wind Speed
        ax = axes[0, 1]
        for wingspan_id in WINGSPAN_CONFIGS.keys():
            data_for_wingspan = []
            wind_speeds = []
            
            for wind_id, wind_config in WIND_SPEEDS.items():
                config_key = f"{wingspan_id}_{wind_id}"
                if config_key in df.index:
                    data_for_wingspan.append(df.loc[config_key, 'avg_lift_drag_ratio'])
                    wind_speeds.append(wind_config['speed'])
            
            if data_for_wingspan:
                wingspan = WINGSPAN_CONFIGS[wingspan_id]['wingspan']
                ax.plot(wind_speeds, data_for_wingspan, marker='o', linewidth=2,
                       label=f'{wingspan}m span')
        
        ax.set_xlabel('Wind Speed (m/s)')
        ax.set_ylabel('Average L/D Ratio')
        ax.set_title('L/D Ratio vs Wind Speed by Wingspan')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Path Efficiency vs Wind Speed
        ax = axes[1, 0]
        for wingspan_id in WINGSPAN_CONFIGS.keys():
            data_for_wingspan = []
            wind_speeds = []
            
            for wind_id, wind_config in WIND_SPEEDS.items():
                config_key = f"{wingspan_id}_{wind_id}"
                if config_key in df.index:
                    data_for_wingspan.append(df.loc[config_key, 'path_efficiency'])
                    wind_speeds.append(wind_config['speed'])
            
            if data_for_wingspan:
                wingspan = WINGSPAN_CONFIGS[wingspan_id]['wingspan']
                ax.plot(wind_speeds, data_for_wingspan, marker='o', linewidth=2,
                       label=f'{wingspan}m span')
        
        ax.set_xlabel('Wind Speed (m/s)')
        ax.set_ylabel('Path Efficiency')
        ax.set_title('Path Efficiency vs Wind Speed by Wingspan')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Energy Change vs Wind Speed
        ax = axes[1, 1]
        for wingspan_id in WINGSPAN_CONFIGS.keys():
            data_for_wingspan = []
            wind_speeds = []
            
            for wind_id, wind_config in WIND_SPEEDS.items():
                config_key = f"{wingspan_id}_{wind_id}"
                if config_key in df.index:
                    data_for_wingspan.append(df.loc[config_key, 'energy_change_J_kg'])
                    wind_speeds.append(wind_config['speed'])
            
            if data_for_wingspan:
                wingspan = WINGSPAN_CONFIGS[wingspan_id]['wingspan']
                ax.plot(wind_speeds, data_for_wingspan, marker='o', linewidth=2,
                       label=f'{wingspan}m span')
        
        ax.set_xlabel('Wind Speed (m/s)')
        ax.set_ylabel('Energy Change (J/kg)')
        ax.set_title('Energy Change vs Wind Speed by Wingspan')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'basic_performance' / 'wind_speed_comparison_plots.png', 
                   dpi=FIG_DPI, bbox_inches='tight')
        plt.close()
        
        # 2. Create heatmap showing optimal wingspan for each wind condition
        self.create_optimization_heatmap(df)
        
        print("Wind speed comparison plots saved")
    
    def create_optimization_heatmap(self, df):
        """Create heatmap showing optimal wingspan selection for different wind speeds."""
        print("\nCreating optimization heatmap...")
        
        # Prepare data for heatmap
        metrics = ['flight_time_s', 'avg_lift_drag_ratio', 'path_efficiency']
        wind_speeds = sorted([config['speed'] for config in WIND_SPEEDS.values()])
        wingspans = sorted([config['wingspan'] for config in WINGSPAN_CONFIGS.values()])
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Create matrix for heatmap
            heatmap_data = np.zeros((len(wind_speeds), len(wingspans)))
            
            for j, wind_speed in enumerate(wind_speeds):
                for k, wingspan in enumerate(wingspans):
                    # Find the data point for this wind speed and wingspan
                    matching_rows = df[(df['wind_speed_ms'] == wind_speed) & 
                                     (df['wingspan_m'] == wingspan)]
                    if len(matching_rows) > 0:
                        heatmap_data[j, k] = matching_rows[metric].iloc[0]
                    else:
                        heatmap_data[j, k] = np.nan
            
            # Create heatmap
            im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
            
            # Set labels
            ax.set_xticks(range(len(wingspans)))
            ax.set_xticklabels([f'{w:.1f}m' for w in wingspans])
            ax.set_yticks(range(len(wind_speeds)))
            ax.set_yticklabels([f'{ws:.0f} m/s' for ws in wind_speeds])
            
            # Add values to heatmap
            for j in range(len(wind_speeds)):
                for k in range(len(wingspans)):
                    if not np.isnan(heatmap_data[j, k]):
                        text = ax.text(k, j, f'{heatmap_data[j, k]:.2f}',
                                     ha="center", va="center", color="white", fontsize=10)
            
            ax.set_xlabel('Wingspan')
            ax.set_ylabel('Wind Speed')
            ax.set_title(f'{metric.replace("_", " ").title()}')
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'basic_performance' / 'optimization_heatmap.png', 
                   dpi=FIG_DPI, bbox_inches='tight')
        plt.close()
        
        print("Optimization heatmap saved")

    def create_detailed_performance_plots(self, df):
        """Create detailed individual performance plots."""
        
        # 1. Flight Time Analysis
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(df['wingspan_m'], df['flight_time_s'], 
                     color=plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(df))),
                     alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, time in zip(bars, df['flight_time_s']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{time:.1f}s', ha='center', va='bottom')
        
        ax.set_xlabel('Wingspan (m)')
        ax.set_ylabel('Optimal Flight Time (s)')
        ax.set_title('Flight Time vs Wingspan\\n(Optimization Objective)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'basic_performance' / 'flight_time_comparison.png', 
                   dpi=FIG_DPI, bbox_inches='tight')
        
        # 2. Aerodynamic Efficiency
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # L/D Ratio
        ax1.plot(df['aspect_ratio'], df['avg_lift_drag_ratio'], 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Aspect Ratio')
        ax1.set_ylabel('Average L/D Ratio')
        ax1.set_title('Lift-to-Drag Ratio vs Aspect Ratio')
        ax1.grid(True, alpha=0.3)
        
        # Path Efficiency
        ax2.plot(df['aspect_ratio'], df['path_efficiency'], 's-', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel('Aspect Ratio')
        ax2.set_ylabel('Path Efficiency')
        ax2.set_title('Path Efficiency vs Aspect Ratio')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'basic_performance' / 'aerodynamic_efficiency.png', 
                   dpi=FIG_DPI, bbox_inches='tight')
        
        print(f"Detailed performance plots saved")


def run_wingspan_analysis():
    """Run Wingspan Performance Analysis."""
    print("\nSTARTING WINGSPAN COMPARATIVE ANALYSIS")
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize analyzer
    analyzer = WingspanPerformanceAnalyzer()
    
    # Load data and extract metrics
    successful_loads, failed_loads = analyzer.load_all_wingspan_data()
    
    if len(successful_loads) < 2:
        print("Insufficient data loaded. Need at least 2 wingspan configurations.")
        return None
        
    # Extract performance metrics
    df_metrics = analyzer.extract_key_performance_metrics()
    
    # Generate AR vs Performance curves
    fig = analyzer.plot_ar_vs_performance_curves()
    
    print(f"Results saved in: {OUTPUT_DIR}")
    
    return analyzer

# =====================================================================================
# SCRIPT EXECUTION
# =====================================================================================

if __name__ == "__main__":
    # Run Wingspan Analysis
    analyzer = run_wingspan_analysis()
    
    if analyzer:
        print("\n" + "="*60)
        print("WINGSPAN ANALYSIS SUMMARY")
        print("="*60)
        
        # Display key findings
        if analyzer.performance_metrics:
            df = pd.DataFrame.from_dict(analyzer.performance_metrics, orient='index')
            df = df.dropna()
            
            if len(df) > 0:
                print(f"\nKEY FINDINGS:")
                
                # Best flight time
                best_time_config = df.loc[df['flight_time_s'].idxmin()]
                print(f"   Fastest flight time: {best_time_config['wingspan_m']}m wingspan "
                      f"({best_time_config['flight_time_s']:.2f}s)")
                
                # Best L/D ratio
                best_ld_config = df.loc[df['avg_lift_drag_ratio'].idxmax()]
                print(f"   Best L/D ratio: {best_ld_config['wingspan_m']}m wingspan "
                      f"({best_ld_config['avg_lift_drag_ratio']:.2f})")
                
                # Best path efficiency
                best_path_config = df.loc[df['path_efficiency'].idxmax()]
                print(f"   Best path efficiency: {best_path_config['wingspan_m']}m wingspan "
                      f"({best_path_config['path_efficiency']:.3f})")

    
    plt.show()
