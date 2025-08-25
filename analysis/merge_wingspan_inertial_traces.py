import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
import sys
import os

# Add the flight directory to the Python path
flight_dir = Path(__file__).parent.parent
sys.path.append(str(flight_dir))

# Import plotting functionality
from flight.analysis import results_plotting
from flight.analysis import analysis_calculations 
from flight.simulator.utils import DataLogger
from flight.simulator.aircraft_model import AircraftModel
from flight.simulator import config

# Configuration
OUTPUT_DIR = Path('./analysis/wingspan_analysis_results/inertial_trace_final')
FIG_DPI = 600
FONT_SIZE = 12

# Wingspan configurations
WINGSPAN_CONFIGS = {
    '0p7m': {'wingspan': 0.7, 'aspect_ratio': 1.6333, 'base_name': 'test_jiayi_wingspan_0p7m'},
    '0p9m': {'wingspan': 0.9, 'aspect_ratio': 2.7, 'base_name': 'test_jiayi_wingspan_0p9m'}, 
    '1p206m_baseline': {'wingspan': 1.206, 'aspect_ratio': 4.8481, 'base_name': 'test_jiayi_wingspan_1p206m_baseline'},
    '1p5m': {'wingspan': 1.5, 'aspect_ratio': 7.5, 'base_name': 'test_jiayi_wingspan_1p5m'},
    '1p8m': {'wingspan': 1.8, 'aspect_ratio': 10.8, 'base_name': 'test_jiayi_wingspan_1p8m'},
    '2p0m': {'wingspan': 2.0, 'aspect_ratio': 13.3333, 'base_name': 'test_jiayi_wingspan_2p0m'},
}

# Wind speed configurations
WIND_SPEEDS = {
    'ws3': {'speed': 3, 'suffix': '_ws3', 'name': '3 m/s'},
    'ws5': {'speed': 5, 'suffix': '', 'name': '5 m/s'}, 
    'ws7': {'speed': 7, 'suffix': '_ws7', 'name': '7 m/s'},
}

class WingspanInertialTraceOriginalStyle:
    """Generate inertial trace plots using original plotting style."""
    
    def __init__(self):
        self.dataloggers = {}
        self.aircraft_models = {}
        self.inertial_powers = {}
        
    def setup_output_directory(self):
        """Create output directory for results."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Output directory created: {OUTPUT_DIR}")
    
    def load_wingspan_data(self):
        """Load DataLogger and aircraft model data for all configurations."""
        print("\n" + "="*60)
        print("LOADING WINGSPAN DATA FOR ORIGINAL STYLE INERTIAL TRACES")
        print("="*60)
        
        successful_loads = 0
        total_configs = len(WINGSPAN_CONFIGS) * len(WIND_SPEEDS)
        
        # Load data for each wingspan and wind speed combination
        for wingspan_id, wingspan_config in WINGSPAN_CONFIGS.items():
            print(f"\nProcessing wingspan: {wingspan_config['wingspan']}m")
            
            # Load aircraft model (same for all wind speeds of this wingspan)
            if wingspan_id == '1p206m_baseline':
                model_path = Path("data/models/wot4_imav_v2.yaml")
            else:
                model_path = Path(f"data/models/wot4_imav_v2_wingspan_{wingspan_id}.yaml")
            
            if model_path.exists():
                am = AircraftModel(model_path)
                self.aircraft_models[wingspan_id] = am
                print(f"   Aircraft model loaded: {model_path.name}")
            else:
                print(f"   Aircraft model not found: {model_path}")
                continue
            
            # Load DataLoggers for each wind speed
            for wind_id, wind_config in WIND_SPEEDS.items():
                full_config_id = f"{wingspan_id}_{wind_id}"
                
                try:
                    # Construct DataLogger path
                    if wingspan_id == '1p206m_baseline' and wind_id == 'ws5':
                        # Special case: baseline data at 5m/s doesn't have suffix
                        dl_name = wingspan_config['base_name']
                    elif wingspan_id == '1p206m_baseline':
                        # Special case: baseline data with wind speed suffix  
                        dl_name = f"test_jiayi_wingspan_1p206m{wind_config['suffix']}"
                    else:
                        # Regular case
                        dl_name = f"{wingspan_config['base_name']}{wind_config['suffix']}"
                    
                    dl_path = Path(f"data/dataloggers/{dl_name}_optimiser")
                    
                    if dl_path.exists():
                        dl = DataLogger.load_from_path(dl_path)
                        self.dataloggers[full_config_id] = dl
                        
                        # Calculate inertial power for coloring
                        pow_i = analysis_calculations.calc_total_specific_inertial_energy_change(dl, am)
                        self.inertial_powers[full_config_id] = pow_i
                        
                        successful_loads += 1
                        print(f"   Loaded: {dl_name} (wind: {wind_config['name']})")
                    else:
                        print(f"   DataLogger not found: {dl_path}")
                        
                except Exception as e:
                    print(f"   Error loading {full_config_id}: {e}")
        
        print(f"\nLoaded {successful_loads}/{total_configs} configurations successfully")
        return successful_loads > 0
    
    def create_combined_inertial_trace_plots(self):
        """Create plots that combine multiple wind speeds on single axes using original style."""
        print("\n" + "="*60)
        print("GENERATING COMBINED INERTIAL TRACE PLOTS (ORIGINAL STYLE)")
        print("="*60)
        
        # Wind direction constant (from original code)
        wind_dir = 270  # degrees
        
        for wingspan_id, wingspan_config in WINGSPAN_CONFIGS.items():
            print(f"\nCreating combined plot for wingspan: {wingspan_config['wingspan']}m")
            
            # Check if we have data for all wind speeds
            available_winds = []
            for wind_id, wind_config in WIND_SPEEDS.items():
                full_config_id = f"{wingspan_id}_{wind_id}"
                if full_config_id in self.dataloggers and full_config_id in self.inertial_powers:
                    available_winds.append((wind_id, wind_config))
            
            if len(available_winds) == 0:
                print(f"   No data available for wingspan {wingspan_config['wingspan']}m")
                continue
            
            # Create figure with larger size and compact layout
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(projection='3d')
            ax.computed_zorder = False
            
            # Label axes using original style with larger fonts
            labelpad = 20
            fontsize_labels = 18
            ax.set_xlabel(r'$x_{\,[e]}$ ($m$)', labelpad=labelpad, fontsize=fontsize_labels, fontweight='bold')
            ax.set_ylabel('\n\n$y_{\\,[n]}$ ($m$)', labelpad=labelpad, fontsize=fontsize_labels, fontweight='bold')
            ax.set_zlabel(r'$z_{\,[-d]}$ ($m$)', labelpad=labelpad, fontsize=fontsize_labels, fontweight='bold')
            
            ax.grid(visible=False)
            
            # Colors for different wind speeds
            wind_colors = {'ws3': 'blue', 'ws5': 'green', 'ws7': 'red'}
            
            # Collect all trajectory data for axis limits
            all_xs, all_ys, all_zs = [], [], []
            
            # Plot each wind speed trajectory
            for wind_id, wind_config in available_winds:
                full_config_id = f"{wingspan_id}_{wind_id}"
                
                dl = self.dataloggers[full_config_id]
                pow_i = self.inertial_powers[full_config_id]
                
                # Convert from NED to XYZ (matplotlib coordinates)
                xs, ys, zs = results_plotting.ned_to_xyz(dl.ns, dl.es, dl.ds)
                
                # Collect for axis limits
                all_xs.extend(xs)
                all_ys.extend(ys) 
                all_zs.extend(zs)
                
                # Create custom coloring based on wind speed and power values
                # Each wind speed has its base color, with intensity showing energy gain/loss
                base_color = wind_colors.get(wind_id, 'black')
                
                # Convert base color name to RGB
                base_rgb = {
                    'blue': (0.0, 0.0, 1.0),
                    'green': (0.0, 0.8, 0.0), 
                    'red': (1.0, 0.0, 0.0)
                }.get(base_color, (0.5, 0.5, 0.5))
                
                colors = []
                max_power = max(abs(pow_i)) if max(abs(pow_i)) > 0 else 1.0
                
                for power_val in pow_i:
                    if power_val > 0:  # Energy gain - use very dark/saturated version of base color
                        intensity = 0.5 + 0.5 * min(abs(power_val) / max_power, 1.0)  # 0.5 to 1.0 (darker range)
                        colors.append((base_rgb[0] * intensity, base_rgb[1] * intensity, base_rgb[2] * intensity, 1.0))
                    else:  # Energy loss - use very light/pale version of base color
                        intensity = 0.15 + 0.35 * (1 - min(abs(power_val) / max_power, 1.0))  # 0.15 to 0.5 (much lighter)
                        colors.append((
                            base_rgb[0] * intensity + (1 - intensity) * 0.95,  # Mix more with white
                            base_rgb[1] * intensity + (1 - intensity) * 0.95,
                            base_rgb[2] * intensity + (1 - intensity) * 0.95,
                            0.9  # Slightly more transparent for lighter colors
                        ))
                
                # Create line collection with custom colors
                traj_lw = 3  # Thicker lines for better visibility
                
                # Create line segments
                points = np.array([xs, ys, zs]).T
                segments = []
                segment_colors = []
                for i in range(len(points) - 1):
                    segments.append([points[i], points[i + 1]])
                    segment_colors.append(colors[i])
                
                plane_trace = results_plotting.art3d.Line3DCollection(segments, colors=segment_colors, linewidths=traj_lw)
                ax.add_collection(plane_trace)
                
                # Add trajectory center line with wind speed color - thinner but visible
                ax.plot(xs, ys, zs, lw=1.5, color=wind_colors.get(wind_id, 'black'), 
                       alpha=0.8, ls='--', label=f'Wind: {wind_config["name"]}')
                
                print(f"   Plotted trace for wind speed: {wind_config['name']}")
            
            # Set axis limits using original style logic
            if all_xs and all_ys and all_zs:
                x_proj_coord = min(all_xs) - 50
                y_proj_coord = max(all_ys) + 20
                z_proj_coord = min(all_zs) - 5
                
                x_lims = (x_proj_coord, max(all_xs) + 20)
                y_lims = (min(all_ys) - 20, y_proj_coord)
                z_lims = (z_proj_coord, max(all_zs) + 5)
                
                ax.set_xlim(*x_lims)
                ax.set_ylim(*y_lims)
                ax.set_zlim(*z_lims)
                ax.set_aspect('equal')
                
                def set_custom_ticks(ax_method, lims):
                    """Set custom ticks with only 3 values and larger font"""
                    start, end = lims
                    middle = (start + end) / 2
                    ticks = [start, middle, end]
                    tick_labels = [f'{tick:.0f}' for tick in ticks]
                    ax_method(ticks)
                    return ticks, tick_labels
                
                # Apply custom ticks to all axes
                x_ticks, x_labels = set_custom_ticks(ax.set_xticks, x_lims)
                y_ticks, y_labels = set_custom_ticks(ax.set_yticks, y_lims)
                z_ticks, z_labels = set_custom_ticks(ax.set_zticks, z_lims)
                
                # Set tick labels with larger font
                ax.set_xticklabels(x_labels, fontsize=16)
                ax.set_yticklabels(y_labels, fontsize=16)
                ax.set_zticklabels(z_labels, fontsize=16)
                
                # Add ground projections like original
                ones_x = np.ones(len(all_xs)) * x_proj_coord
                ones_y = np.ones(len(all_ys)) * y_proj_coord  
                ones_z = np.ones(len(all_zs)) * z_proj_coord
                
                # Plot projections for each wind speed
                for wind_id, wind_config in available_winds:
                    full_config_id = f"{wingspan_id}_{wind_id}"
                    dl = self.dataloggers[full_config_id]
                    xs, ys, zs = results_plotting.ned_to_xyz(dl.ns, dl.es, dl.ds)
                    
                    proj_color = wind_colors.get(wind_id, 'grey')
                    proj_alpha = 0.3
                    proj_lw = 1
                    
                    # Add projections
                    ones = np.ones(len(xs))
                    ax.plot(ones*x_proj_coord, ys, zs, c=proj_color, alpha=proj_alpha, lw=proj_lw, ls='--', zorder=1)
                    ax.plot(xs, ones*y_proj_coord, zs, c=proj_color, alpha=proj_alpha, lw=proj_lw, ls='--', zorder=1)
                    ax.plot(xs, ys, ones*z_proj_coord, c=proj_color, alpha=proj_alpha, lw=proj_lw, ls='--', zorder=1)
            
            # Add title with larger font 
            ax.set_title(f'Inertial Power Traces - Multiple Wind Speeds\n'
                        f'Wingspan: {wingspan_config["wingspan"]}m (AR = {wingspan_config["aspect_ratio"]:.1f})',
                        fontsize=20, fontweight='bold', pad=35)
            
            # Add legend for wind speeds with larger font
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=16, frameon=True, 
                     fancybox=True, shadow=True, framealpha=0.9)
            

            # Adjust layout to prevent overlapping with larger fonts
            plt.tight_layout(pad=3.0)
            plt.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.08)
            
            # Save figure
            output_filename = f'wingspan_{wingspan_config["wingspan"]:.1f}m_combined_inertial_traces.png'
            plt.savefig(OUTPUT_DIR / output_filename, dpi=FIG_DPI, bbox_inches='tight')
            print(f"   Saved: {output_filename}")
            
            plt.close()
    

    # Summary report generation function removed
    def run_analysis(self):
        print("STARTING WINGSPAN INERTIAL TRACE ANALYSIS ")
        print("="*60)
        
        # Setup
        self.setup_output_directory()
        
        # Load data
        if not self.load_wingspan_data():
            print("Failed to load data. Exiting.")
            return
        
        # Generate combined plots only
        self.create_combined_inertial_trace_plots()
        
        print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    analyzer = WingspanInertialTraceOriginalStyle()
    analyzer.run_analysis()
