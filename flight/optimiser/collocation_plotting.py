# FT 4/6/24
# Plots the partial solutions while they're being solved

import openmdao.api as om
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation
from pathlib import Path
import os
# import time

# Add the flight directory to the Python path (when running from FlightSwordLite)
import sys, os
sys.path.append(os.getcwd())

from flight.simulator.utils import ned_to_xyz
from flight.optimiser import config
from flight.simulator import wind_models

# case_reader_paths is a list []
def create_trace(case_reader_paths, save_path, show_plot=False, save=True):
    design_var_data = {}
    exclude = ['t_duration']
    
    # Open recorder
    for case_reader_path in case_reader_paths:
        cr = om.CaseReader(case_reader_path)

        # Get values
        # Collect all design variable data through time
        cases = cr.get_cases()
        for case in cases:
            print(".", end="")
            for name, vals in case.outputs.items():
                # If this is a state or control... Using set intersection.
                split_name = re.split('[\.:]', name)
                if {'states', 'controls'} & set(split_name):
                    # Strip off just the end of the name
                    # Some name strings are delimited with '.', others with ':', so need to handle both.
                    name = split_name[-1]
                    if not name in exclude:
                        var_data = design_var_data.get(name, np.empty((0, vals.shape[0])))
                        # During automated grid refinement, the number of collocation points can
                        # increase. This accounts for this.
                        if vals.shape[0] > var_data.shape[1]:
                            expanded = np.full((var_data.shape[0], vals.shape[0]), np.nan)
                            expanded[:, :var_data.shape[1]] = var_data
                            var_data = expanded
                        var_data = np.append(var_data, vals.T, axis=0)
                        # Overwrite
                        design_var_data[name] = var_data

    # Close recorders
    # TODO How to do this?

    # Make animation
    fig = plt.figure()
    fig.suptitle("Collocation Knot Point Trace")
    # gs = gridspec.GridSpec(len(design_var_data), 2)

    # Add trajectory axis
    # x, y trace
    traj_ax = fig.add_subplot(1, 2, 1, projection='3d') # gs[:, 0])
    init_ax = fig.add_subplot(1, 2, 2, projection='3d')

    # Plot initial data, to be updated dynamically with set_data().
    traj, = traj_ax.plot([], [], [], marker='x')
    # Remove NaNs
    n_vals = design_var_data['n'][~np.isnan(design_var_data['n'])]
    e_vals = design_var_data['e'][~np.isnan(design_var_data['e'])]
    d_vals = design_var_data['d'][~np.isnan(design_var_data['d'])]

    # Scale - find min and max values of x and y
    n_lims = (np.min(n_vals), np.max(n_vals))
    n_buffer = (n_lims[1] - n_lims[0]) / 2
    e_lims = (np.min(e_vals), np.max(e_vals))
    e_buffer = (e_lims[1] - e_lims[0]) / 2
    d_lims = (np.min(d_vals), np.max(d_vals))
    d_buffer = (d_lims[1] - d_lims[0]) / 2

    # Update limits with buffer
    n_lims_buff = (n_lims[0] - n_buffer, n_lims[1] + n_buffer)
    e_lims_buff = (e_lims[0] - e_buffer, e_lims[1] + e_buffer)
    d_lims_buff = (d_lims[0] - d_buffer, d_lims[1] + d_buffer)

    # Convert from ned to xyz coordinate system for plotting
    x_lims, y_lims, z_lims = ned_to_xyz(n_lims_buff, e_lims_buff, d_lims_buff)

    traj_ax.set_xlim3d(np.sort(x_lims))
    traj_ax.set_ylim3d(np.sort(y_lims))
    traj_ax.set_zlim3d(np.sort(z_lims))
    traj_ax.set_xlabel('x')
    traj_ax.set_ylabel('y')
    traj_ax.set_zlabel('z')
    traj_ax.set_aspect('equal')
    traj_ax.set_title("Collocation points")

    # Convert data from ned to xyz frame
    # TODO Check
    xs, ys, zs = ned_to_xyz(design_var_data['n'], design_var_data['e'], design_var_data['d'])

    # Plot original configuration
    init_ax.plot(xs[0], ys[0], zs[0], marker='x')
    init_ax.set_xlabel('x')
    init_ax.set_ylabel('y')
    init_ax.set_zlabel('z')
    init_ax.set_title("Initial configuration")

    """
    # Add graph axes
    graph_axs = []
    # These are updated through time
    graph_plots = []
    names = list(design_var_data.keys())

    # Live plots for each of the states
    for i in range(len(names)):
        ax = fig.add_subplot(gs[i, 1])
        ax.set_title(names[i])
        graph_axs.append(ax)
        # Plot initial data, to be updated dynamically with set_data().
        graph_plots.append(ax.plot([], [])[0])
    fig.tight_layout()
    """

    # Animation functions
    # Initialise animation
    def init_anim():
        traj.set_data([], [])
        traj.set_3d_properties([], 'z')
        # map(lambda gp: gp.set_data([], []), graph_plots)

    # Run animation
    def anim_func(i):
        if i == 0:
            plt.pause(5)

        # Update trajectory
        traj.set_data(xs[i], ys[i])
        traj.set_3d_properties(zs[i], 'z')

        # Update state plots
        # TODO
    
    print()
    print("Creating animation")
    anim = animation.FuncAnimation(fig, anim_func, frames=len(xs), init_func=init_anim, interval=50, repeat_delay=1000)
    print("Animation creation complete")
    
    # Save animation
    # Used to save as a gif, but OneNote doesn't play them, so  now saving as a video instead.
    if save:
        print("Saving animation")
        anim.save(save_path / 'collocation_anim.gif', writer=animation.PillowWriter(fps=20))
        # anim.save(save_path / 'collocation_anim.mp4', writer=animation.FFMpegWriter(fps=30))
    
    if show_plot:
        plt.show(block=True)

if __name__ == "__main__":
    #log_folder_path = config.dymos_logs_folder_path
    #log_folders = [f for f in log_folder_path.iterdir() if f.is_dir()]
    #log_folders.sort(key=lambda x: os.path.getmtime(x))
    #log_path = log_folders[-1]
    #colloc_trace_sql_path = log_path / 'plane.sql'

    # colloc_trace_sql_path = Path('.') / 'dymos_logs' / '2024-09-08 20:38:55.139972' / 'plane.sql'
    # colloc_trace_sql_path = Path('.') / 'dymos_logs' / '2024-09-08 21:05:05.348866' / 'plane.sql'
    #colloc_trace_sql_path = config.dymos_logs_folder_path / 'front_on_5__b__spline__[410 205 -30]__more_pts_2' / 'plane.sql'

    # base = config.dymos_logs_folder_path / 'front_on_5__b__spline__(410, 205, -30)__new_14'
    # base = config.dymos_logs_folder_path / 'front_on_5__b__spline__(410, 205, -30)__test_wot4_imav'
    # base = config.dymos_logs_folder_path / 'linked_phases_test_16' # 'rayleigh_cycle_19.6' # 'sachs_validation_58' # 4'
    # base = config.dymos_logs_folder_path / 'linked_phases_test__link_as_objective_9' # 'rayleigh_cycle_19.6' # 'sachs_validation_58' # 4'
    # base = config.dymos_logs_folder_path / 'linked_phases_test__link_as_objective_8' # 'rayleigh_cycle_19.6' # 'sachs_validation_58' # 4'
    # base = config.dymos_logs_folder_path / 'linked_phases_test__4_phases__link_as_objective_3' # 'rayleigh_cycle_19.6' # 'sachs_validation_58' # 4'
    # base = config.dymos_logs_folder_path / 'long_building__front_on_5__total__cubic_spline__(930, 205, -20)' # 'rayleigh_cycle_19.6' # 'sachs_validation_58' # 4'
    # base = config.dymos_logs_folder_path / 'front_on_5__total__cubic_spline__(410, 205, -20)_poly_controls_test_4' # 'rayleigh_cycle_19.6' # 'sachs_validation_58' # 4'
    # base = config.dymos_logs_folder_path / 'baseline_min_time_descent_4' # 'rayleigh_cycle_19.6' # 'sachs_validation_58' # 4'
    # base = config.dymos_logs_folder_path / 'baseline_low_flex_dynamics_test_46' # 'rayleigh_cycle_19.6' # 'sachs_validation_58' # 4'
    # base = config.dymos_logs_folder_path / 'building_sweep__optimiser_new_testing__7' # 'rayleigh_cycle_19.6' # 'sachs_validation_58' # 4'
    # base = config.dymos_logs_folder_path / 'building_sweep__optimiser_new_testing__non_rates__15' # 'rayleigh_cycle_19.6' # 'sachs_validation_58' # 4'
    # base = config.dymos_logs_folder_path / 'long_building__front_on_4.6__(930, 205, -20)' # 'rayleigh_cycle_19.6' # 'sachs_validation_58' # 4'
    # base = config.dymos_logs_folder_path / 'linked_phases_round_building__final__16' # 'rayleigh_cycle_19.6' # 'sachs_validation_58' # 4'
    # base = config.dymos_logs_folder_path / 'building_sweep__optimiser_new_testing__non_rates__10' # 'rayleigh_cycle_19.6' # 'sachs_validation_58' # 4'
    # base = config.dymos_logs_folder_path / 'front_test__front_on_5__(90, 85, -20)_2' # 'rayleigh_cycle_19.6' # 'sachs_validation_58' # 4'
    #base = config.dymos_logs_folder_path / 'front_test__front_on_5__(131, 98.5, -33.5)__upthrust_region__2' # 'rayleigh_cycle_19.6' # 'sachs_validation_58' # 4'
    #base = config.dymos_logs_folder_path / 'back_of_building_test__wot4__high_drag'
    # base = config.dymos_logs_folder_path / 'back_of_building_test__wot4__new_constraints_3'
    # base = Path('/mnt/c/Users/fjtur/OneDrive - University of Bristol/Documents/PhD/Projects/Isolated building trajectory optimisation (chap 1)/Results/Tests/dymos_logs/') / 'back_of_building_sweep__dir270__wind5_(400, 205, -20)'
    # base = Path('/mnt/c/Users/fjtur/OneDrive - University of Bristol/Documents/PhD/Projects/Isolated building trajectory optimisation (chap 1)/Results/1. Wind speed and direction sweep - back of building/dymos_logs/') / 'back_of_building_sweep__dir270__wind5_(400, 205, -20)_4'
    # base = Path('/mnt/c/Documents and Settings/fjtur/OneDrive - University of Bristol/Documents/PhD/Projects/Isolated building trajectory optimisation (chap 1)/Results/0. Soaring examples/0. Dynamic/dymos_logs/2024-12-17 12:49:21.058912')
    # base = Path('/mnt/c/Documents and Settings/fjtur/OneDrive - University of Bristol/Documents/PhD/Projects/Isolated building trajectory optimisation (chap 1)/Results/0. Soaring examples/0. Dynamic/dymos_logs/dynamic_soaring')
    # base = Path('/mnt/c/Documents and Settings/fjtur/OneDrive - University of Bristol/Documents/PhD/Projects/Isolated building trajectory optimisation (chap 1)/Results/0. Soaring examples/0. Dynamic/dymos_logs/dynamic_soaring_upwind')
    # base = config.default_dymos_logs_folder_path / '2025-01-04 00:47:51.688497'
    # base = Path('/mnt/c/Documents and Settings/fjtur/OneDrive - University of Bristol/Documents/PhD/Projects/Isolated building trajectory optimisation (chap 1)/Results/4. Long building/dymos_logs/long_building_back_sweep__dir270__wind6_(930, 205, -20)')
    # base = Path('/mnt/c/Documents and Settings/fjtur/OneDrive - University of Bristol/Documents/PhD/Projects/Isolated building trajectory optimisation (chap 1)/Results/0. Soaring examples/1. Gust soaring space (sideways sine wind)/dymos_logs/sideways_sine_5__relaxed_angular_rates')
    # base = Path('/mnt/c/Users/fjtur/OneDrive - University of Bristol/Documents/PhD/Projects/Isolated building trajectory optimisation (chap 1)/Results/1.1. Wind speed and dir sweep - back - ei/baseline/dymos_logs/baseline__prawn_41')
    base = Path('//mnt/c/Users/fjtur/OneDrive - University of Bristol/DOCUME~1/PhD/Projects/ISOLAT~1/Results/5C085~1.URB/NEWRUN~1/DYMOS_~1/CANYON~1.0)_')
    colloc_trace_sql_paths = [
        base / 'plane_1.sql',
        # base / 'plane_2.sql',
        #base / 'plane_3.sql'
    ]

    # create_trace(colloc_trace_sql_paths, config.base_figure_path, show_plot=True, save=False)
    create_trace(colloc_trace_sql_paths, Path('.'), show_plot=True, save=False)