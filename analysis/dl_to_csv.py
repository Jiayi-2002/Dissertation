# FT 28/1/25

import pandas as pd
from pathlib import Path

# Add the flight directory to the Python path (when running from FlightSwordLite)
import sys, os
sys.path.append(os.getcwd())
sys.path.append(str(Path(os.getcwd()) / 'analysis'))

from flight.analysis import analysis_calculations
from flight.simulator.utils import ned_to_xyz, DataLogger
from flight.simulator.config import base_aircraft_model_path
from flight.simulator.aircraft_model import AircraftModel

def remove_trailing_0pt(val):
    return str(round(val)) if val % 1 == 0 else str(val)

# Creates a CSV file for a DataLogger
def dl_to_csv(dl, name, aircraft_model, output_folder_path):
    csv_path = output_folder_path / f"{name}.txt"
    if csv_path.exists():
        print(f"CSV file {name} already exists")
        return
    else:
        spec_energy_change_i = analysis_calculations.calc_total_specific_inertial_energy_change(dl, aircraft_model)
        spec_energy_change_a = analysis_calculations.calc_total_specific_air_relative_energy_change(dl, aircraft_model)
        air_rel_comps = analysis_calculations.calc_specific_air_relative_energy_change_components(dl, aircraft_model)
        drag_power = air_rel_comps['drag']
        throttle_power = air_rel_comps['throttle']
        static_power = air_rel_comps['static']
        gradient_power = air_rel_comps['gradient']
        aero_ib_proj = analysis_calculations.calc_aerodynamic_force_ib_projection(dl)
        aero_vi_proj = analysis_calculations.calc_aerodynamic_force_vi_unit_projection(dl)

        # Convert position to xyz system
        xs, ys, zs = ned_to_xyz(dl.ns, dl.es, dl.ds)

        data = {
            'time': dl.times,
            'x': xs,
            'y': ys,
            'z': zs,
            'n': dl.ns,
            'e': dl.es,
            'd': dl.ds,
            'u_traj': dl.us,    # This has to be named differently, so that it doesn't conflict with the wind field u in TecPlot.
            'v_traj': dl.vs,    # This has to be named differently, so that it doesn't conflict with the wind field v in TecPlot.
            'w_traj': dl.ws,    # This has to be named differently, so that it doesn't conflict with the wind field w in TecPlot.
            'phi': dl.phis,
            'theta': dl.thetas,
            'psi': dl.psis,
            'p': dl.ps,
            'q': dl.qs,
            'r': dl.rs,
            'va': dl.vas,
            'alpha': dl.alphas,
            'beta': dl.betas,
            'spec_energy_change_i': spec_energy_change_i,
            'spec_energy_change_a': spec_energy_change_a,
            'drag_power': drag_power,
            'throttle_power': throttle_power,
            'static_power': static_power,
            'gradient_power': gradient_power,
            'aero_ib_proj': aero_ib_proj,
            'aero_vi_proj': aero_vi_proj
        }

        df = pd.DataFrame(data)

        print(f"Creating CSV file: {csv_path}")
        # Create csv file
        df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    # Set this up to save the .csv files to a location of your choice
    output_folder_path = Path('.')

    # Load aircraft model file
    am = AircraftModel(base_aircraft_model_path / 'wot4_imav_v2.yaml')

    # Load DataLogger
    # The loaded DataLogger was for these conditions
    wind_dir = 270
    wind_speed = 5
    name = f'original__wd~{remove_trailing_0pt(wind_dir)}__ws~{remove_trailing_0pt(wind_speed)}'
    
    dl_path = Path('./analysis/example_datalogger') # Path to your DataLogger
    dl = DataLogger.load_from_path(dl_path)
    
    # Create .csv file for TecPlot
    dl_to_csv(dl, name, am, output_folder_path)