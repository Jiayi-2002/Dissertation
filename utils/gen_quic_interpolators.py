# FT 2/9/24

# Script to load wind field from QUIC and generate interpolators and sweeps.

# Process:
# - Find coordinates and associated values
# - Convert coordinates and values to NED
# - Set NED axis limits  
# - Generate interpolators and sweeps

import mat73
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gc

import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

# Add the flight directory to the Python path (when running from FlightSwordLite)
import sys, os
sys.path.append(os.getcwd())

from flight.simulator import config
from flight.simulator.utils import xyz_to_ned
from flight.simulator.wind_interpolators.natural_cubic_spline_interpax_interpolator import NaturalCubicSplineInterpolator


def load_wind_data(vel_data_filepath, windgrid_data_filepath):
    print("Loading wind velocity data")
    vel_data = mat73.loadmat(vel_data_filepath)
    print("Loading windgrid data")
    windgrid_data = mat73.loadmat(windgrid_data_filepath)

    # Looking down at the building in xy frame.
    wx = vel_data['velocity']['u'][0]
    wy = vel_data['velocity']['v'][0]
    wz = vel_data['velocity']['w'][0]
    # Dictionaries u, v and w are indexed with values which do not correspond exactly to the environment geometry. The
    # wind grid allows us to get the x, y and z coordinates corresponding to these indices.
    # {x/y/z}g = 'x/y/z grid'
    xg = windgrid_data['windgrid']['x']
    yg = windgrid_data['windgrid']['y']
    zg = windgrid_data['windgrid']['z']

    return wx, wy, wz, xg, yg, zg

def create_interpolator(parser_name, velocity_filepath, windgrid_filepath, interpolator_class, region_ranges_xyz, save_folderpath, create_sweeps, **extra_interpolator_kwargs):
        # Subsample to region of interest
        x_range, y_range, z_range = region_ranges_xyz

        wx, wy, wz, xg, yg, zg = load_wind_data(velocity_filepath, windgrid_filepath)

        # Flatten, to build interpolators.
        # 'f' for 'flattened'
        wxf = wx.flatten()
        wyf = wy.flatten()
        wzf = wz.flatten()
        xgf = xg.flatten()
        ygf = yg.flatten()
        zgf = zg.flatten()
        wind_coords = np.stack((xgf, ygf, zgf), axis=-1)
        wind_vals = np.stack((wxf, wyf, wzf), axis=-1)

        # Subsample to region of interest
        subsample_inds = np.where((wind_coords[:, 0] >= x_range[0]) & (wind_coords[:, 0] <= x_range[1]) &
                                (wind_coords[:, 1] >= y_range[0]) & (wind_coords[:, 1] <= y_range[1]) &
                                (wind_coords[:, 2] >= z_range[0]) & (wind_coords[:, 2] <= z_range[1]))
        wind_coords = wind_coords[subsample_inds]
        wind_vals = wind_vals[subsample_inds]

        # Convert to NED
        # TODO Test this!
        wind_coords_ned = xyz_to_ned(*wind_coords.T).T
        wind_vals_ned = xyz_to_ned(*wind_vals.T).T

        # Find range of validity
        ns = np.sort(np.unique(wind_coords_ned[:, 0]))
        es = np.sort(np.unique(wind_coords_ned[:, 1]))
        ds = np.sort(np.unique(wind_coords_ned[:, 2]))

        inds = np.array([0, -1])
        n_range = ns[inds]
        e_range = es[inds]
        d_range = ds[inds]

        # Generate plotting coords and values
        # Subsample
        n_subsample_coeff = 3
        e_subsample_coeff = 3
        d_subsample_coeff = 2
        ns_sub = ns[::n_subsample_coeff]
        es_sub = es[::e_subsample_coeff]
        ds_sub = ds[::d_subsample_coeff]
        subsample_mask = np.in1d(wind_coords_ned[:, 0], ns_sub) & np.in1d(wind_coords_ned[:, 1], es_sub) & np.in1d(wind_coords_ned[:, 2], ds_sub)
        plot_coords = wind_coords_ned[subsample_mask]
        plot_vals = wind_vals_ned[subsample_mask]

        # Build interpolator
        interpolator_type_name = interpolator_class.type_name
        name = f"{parser_name}__{interpolator_type_name}"
        # TODO Check that this is the right way to pass this argument dictionary
        interpolator = interpolator_class(name, n_range, e_range, d_range)
        print(f"Fitting {interpolator_type_name} interpolator")
        interpolator.fit(wind_coords_ned, wind_vals_ned[:, 0], wind_vals_ned[:, 1], wind_vals_ned[:, 2], **extra_interpolator_kwargs)
        if create_sweeps:
            print("Producing sweeps")
            folderpath = Path(config.sweeps_path) / name
            folderpath.mkdir(exist_ok=True)
            # TODO Is this a standard interface for all of them? Test this.
            interpolator.inspect_fit(plot_coords, plot_vals, folderpath, types=['x', 'y', 'z'])
        print(f"Saving {interpolator_type_name} interpolator")
        interpolator.save(save_folderpath)
        print("Done")

        # Info output
        print()
        print("==============================")
        print(f"Wind interpolator built and saved: {name}")
        print(f"n-range: {n_range}")
        print(f"e-range: {e_range}")
        print(f"d-range: {d_range}")
        print()


if __name__ == "__main__":
    # Default, but don't have to use this:
    save_folderpath = Path(config.cfd_data_folder_path)

    # Splitting wind field into four regions around (not intersecting) building. F (front), B (back), L (left, as seen from front side, of approaching wind), R (right, as seen from front side, of approaching wind)
    f_ranges_xyz = [(0, 100), (0, 500), (0, 70)]
    l_ranges_xyz = [(0, 400), (369, 500), (0, 70)]
    r_ranges_xyz = [(0, 400), (0, 131), (0, 70)]
    b_ranges_xyz = [(190, 400), (0, 500), (0, 70)]
    region_ranges_xyz = {
        'f': f_ranges_xyz,
        'l': l_ranges_xyz,
        'r': r_ranges_xyz,
        'b': b_ranges_xyz
    }

    quic_folderpath = Path('/mnt/c/Users/jx20025/Documents/FlightSword meeting with Jiayi/Raw QUIC files (for example)/original__wd~270__ws~4')
    save_folderpath = Path('/mnt/c/Users/jx20025/Documents/FlightSword meeting with Jiayi/Testing interpolator creation')
    # print(f"Filepath: {filepath}")
    filepath = quic_folderpath
    velocity_filepath = filepath / 'velocity.mat'
    windgrid_filepath = filepath / 'windgrid.mat'
    create_interpolator(f"orig_270_5__total", velocity_filepath, windgrid_filepath, NaturalCubicSplineInterpolator, [(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)], save_folderpath, False)
    gc.collect()