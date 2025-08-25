import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration matching the original analysis
WIND_SPEEDS = {
    'ws3': {'speed': 3, 'suffix': '_ws3', 'color': '#1f77b4', 'name': '3 m/s'},
    'ws5': {'speed': 5, 'suffix': '', 'color': '#ff7f0e', 'name': '5 m/s'},
    'ws7': {'speed': 7, 'suffix': '_ws7', 'color': '#2ca02c', 'name': '7 m/s'},
}

# Output configuration
OUTPUT_DIR = Path('./wingspan_analysis_results')
INDIVIDUAL_PLOTS_DIR = OUTPUT_DIR / 'basic_performance' / 'individual_plots'
FIG_DPI = 600
FONT_SIZE = 14
plt.rcParams.update({'font.size': FONT_SIZE})

def create_individual_ar_performance_plots():
    """Generate 6 individual AR vs performance plots."""
    
    print("CREATING INDIVIDUAL AR vs PERFORMANCE PLOTS")
    print("="*60)
    
    # Create individual plots directory
    INDIVIDUAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load performance data
    data_file = OUTPUT_DIR / 'data_tables' / 'performance_metrics.csv'
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        return
    
    df = pd.read_csv(data_file, index_col=0)
    df = df.dropna()  # Remove any failed configurations
    
    # Define metrics to plot (removed objective labels)
    metrics_to_plot = [
        ('flight_time_s', 'Flight Time (s)', 'flight_time'),
        ('avg_lift_drag_ratio', 'Average L/D Ratio', 'ld_ratio'),
        ('path_efficiency', 'Path Efficiency', 'path_efficiency'),
        ('avg_groundspeed_ms', 'Average Groundspeed (m/s)', 'groundspeed'),
        ('energy_change_J_kg', 'Energy Change (J/kg)', 'energy_change'),
        ('control_activity', 'Control Activity (RMS)', 'control_activity')
    ]
    
    for i, (metric, label, filename) in enumerate(metrics_to_plot):
        print(f"Creating plot {i+1}/6: {label}")
        
        # Create individual figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot separate lines for each wind speed
        for wind_id, wind_config in WIND_SPEEDS.items():
            # Filter data for this wind speed
            wind_data = df[df['wind_speed_ms'] == wind_config['speed']].copy()
            
            if len(wind_data) > 0:
                # Sort by aspect ratio for smooth line
                wind_data = wind_data.sort_values('aspect_ratio')
                
                x = wind_data['aspect_ratio']
                y = wind_data[metric]
                
                # Plot line and points with larger markers for individual plots
                ax.plot(x, y, color=wind_config['color'], linewidth=3, 
                       label=f"Wind {wind_config['name']}", marker='o', markersize=10,
                       alpha=0.9, markeredgewidth=1, markeredgecolor='white')
                
                # Add wingspan annotations (only for 5m/s wind speed to avoid clutter)
                if wind_id == 'ws5':
                    for ar, val, wingspan in zip(x, y, wind_data['wingspan_m']):
                        ax.annotate(f'{wingspan:.1f}m', (ar, val), xytext=(8, 8), 
                                   textcoords='offset points', fontsize=10, alpha=0.8,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                           alpha=0.7, edgecolor='none'))
        
        # Styling for individual plots
        ax.set_xlabel('Aspect Ratio', fontsize=FONT_SIZE+2, fontweight='bold')
        ax.set_ylabel(label, fontsize=FONT_SIZE+2, fontweight='bold')
        ax.set_title(f'{label} vs Aspect Ratio', 
                    fontsize=FONT_SIZE+4, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linewidth=0.8)
        ax.legend(fontsize=FONT_SIZE, framealpha=0.9, shadow=True)
        
        # Add some styling improvements
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        # Set background color
        ax.set_facecolor('#fafafa')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save individual plot
        output_filename = f'ar_vs_{filename}_comparison.png'
        plt.savefig(INDIVIDUAL_PLOTS_DIR / output_filename, 
                   dpi=FIG_DPI, bbox_inches='tight', facecolor='white')
        
        print(f"   Saved: {output_filename}")
        plt.close()
    
    print(f"\nALL INDIVIDUAL PLOTS CREATED!")
    print(f"Location: {INDIVIDUAL_PLOTS_DIR}")

# README generation function removed

if __name__ == "__main__":
    create_individual_ar_performance_plots()
