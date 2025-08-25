# FT 16/7/25
# This is an example script which re-loads a saved DataLogger and generates plots based on the information contained within it.
# FlightSword's plotting functions work on DataLoggers as a unified interface, so that optimisations don't have to be re-run to
# gain access to the results.

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Add the flight directory to the Python path (when running from FlightSwordLite)
# This needs doing so that all of the FlightSword imports work properly
import sys, os
sys.path.append(os.getcwd())

# Import plotting functionality
from flight.analysis import results_plotting
# Import functions for performing energy, work and power computations
from flight.analysis import analysis_calculations 
# Import DataLogger class
from flight.simulator.utils import DataLogger
# For loading aircraft model
from flight.simulator.aircraft_model import AircraftModel
from flight.simulator import config

# Save folder for figures
base_fig_save_folder = Path('./analysis/example_figs')
fig_dpi = 600   # Resolution of PNG figures - larger value means clearer images but greater file sizes

# Load the DataLogger
dl_path = Path('./analysis/example_datalogger') # Path to your DataLogger
dl = DataLogger.load_from_path(dl_path)

# Load the aircraft model (needed for generating some of the plots)
am = AircraftModel(config.base_aircraft_model_path / 'wot4_imav_v2.yaml')


# =========
# Re-run the code to generate all of the plots, and save them in the './analysis/example_figs' folder
# Note that these recreated images don't feature the collocation points, as this information is not
# stored in the DataLogger.
if True:    # Just here so that the code can be switched off
    results_plotting.generate_plots(base_fig_save_folder / 'plot_regeneration', dl, am, save=True)


# =========
# Create traces showing the different energy aspects of the trajectory

# Calculate...
# ...rate of change of inertial energy
pow_i = analysis_calculations.calc_total_specific_inertial_energy_change(dl, am)
# ...rate of change of air-relative energy
pow_a = analysis_calculations.calc_total_specific_air_relative_energy_change(dl, am)
# ...rate of change of air-relative energy due to...
pow_a_comps = analysis_calculations.calc_specific_air_relative_energy_change_components(dl, am)
# ...rising air (static soaring)
pow_stat = pow_a_comps['static']
# ...wind gradients (dynamic soaring)
pow_grad = pow_a_comps['gradient']

# The loaded DataLogger was for these conditions
wind_dir = 270
wind_speed = 5

fig_save_folder = base_fig_save_folder / 'coloured_traces'

if True:    # Just here so that the code can be switched off
    results_plotting.plot_trace(dl, wind_dir, wind_speed, title=None, plot_interval=500, traj_lw=1.25, plot_traj_centreline=False,
                                colouring_vals=None, colourbar_label=None,
                                plane_scale=1.5, wind_scale=0, save_folder=fig_save_folder, save_name='only', save_dpi=fig_dpi)
    print("Trace plotting complete")
    # Trajectory with wind
    results_plotting.plot_trace(dl, wind_dir, wind_speed, title=None, plot_interval=500, traj_lw=1.25, plot_traj_centreline=False,
                                colouring_vals=None, colourbar_label=None,
                                plane_scale=1.5, wind_scale=2, save_folder=fig_save_folder, save_name='wind', save_dpi=fig_dpi)
    print("Trace with wind plotting complete")
    # Inertial power
    results_plotting.plot_trace(dl, wind_dir, wind_speed, title=None, plot_interval=None, traj_lw=2, plot_traj_centreline=False,
                                colouring_vals=pow_i, colourbar_label='Specific inertial power ($J kg^{-1} s^{-1}$)',
                                plane_scale=1, wind_scale=0, save_folder=fig_save_folder, save_name='inertial', save_dpi=fig_dpi)
    print("Inertial trace plotting complete")
    # Air-relative power
    results_plotting.plot_trace(dl, wind_dir, wind_speed, title=None, plot_interval=None, traj_lw=2, plot_traj_centreline=False,
                                colouring_vals=pow_a, colourbar_label='Specific air-relative power ($J kg^{-1} s^{-1}$)',
                                plane_scale=1, wind_scale=0, save_folder=fig_save_folder, save_name='air_rel', save_dpi=fig_dpi)
    print("Air-relative trace plotting complete")
    # Static power
    results_plotting.plot_trace(dl, wind_dir, wind_speed, title=None, plot_interval=None, traj_lw=2, plot_traj_centreline=False,
                                colouring_vals=pow_stat, colourbar_label='Specific static power ($J kg^{-1} s^{-1}$)',
                                plane_scale=1, wind_scale=0, save_folder=fig_save_folder, save_name='static', save_dpi=fig_dpi)
    print("Static trace plotting complete")
    # Gradient power
    results_plotting.plot_trace(dl, wind_dir, wind_speed, title=None, plot_interval=None, traj_lw=2, plot_traj_centreline=False,
                                colouring_vals=pow_grad, colourbar_label='Specific gradient power ($J kg^{-1} s^{-1}$)',
                                plane_scale=1, wind_scale=0, save_folder=fig_save_folder, save_name='gradient', save_dpi=fig_dpi)
    print("Gradient trace plotting complete")

    # plt.show(block=True)


# =========
# Colour the traces by the wind experienced

if True:
    results_plotting.colour_trace_by_wind(dl, wind_dir, wind_speed, base_fig_save_folder / 'coloured_traces', save_dpi=fig_dpi)
    
    # plt.show(block=True)


# =========
# Create an animation of the trajectory

# This animation is coloured by air-relative energy, but pow_i, pow_stat or pow_grad from above could also be used.
if True:    # Just here so that the code can be switched off
    save_anim = False   # Animations can be saved, but it takes quite a long time.
    video_dpi = 350
    subsample_num = 30  # Animate 1 in every 30 points of the trajectory (and skip the other 29). This increases the rate of plotting, else it is very slow.
                        # This can be changed to suit your needs.
    
    # ==
    # This animation just shows the wind. Note that the size of the plane and the wind arrow can be scaled.
    if True:
        # The anim = ... must be present, else it doesn't work.
        anim = results_plotting.BuildingBackAnimatorBase().animate(dl, pow_a, 'Specific air-relative power ($Jkg^{-1}s^{-1}$)',
                                                                   base_fig_save_folder / 'animations' if save_anim else None, '_pow_a',
                                                                   plot_arrows=['wind'], wind_scale=3, plane_scale=1, anim_interval_ms=20,
                                                                   subsample_num=subsample_num, save_dpi=video_dpi)
        plt.show(block=True)

    # ==
    # This animation shows all of the forces acting
    if True:
        plot_arrows = ['all_forces']
        # The anim = ... must be present, else it doesn't work.
        anim = results_plotting.BuildingBackAnimatorBase().animate(dl, pow_a, 'Specific air-relative power ($Jkg^{-1}s^{-1}$)',
                                                                   base_fig_save_folder / 'animations' if save_anim else None, '_pow_a',
                                                                   plot_arrows=plot_arrows, plane_scale=1, force_scale=5, anim_interval_ms=20,
                                                                   subsample_num=subsample_num, save_dpi=video_dpi)

        # Create legend
        legend_fig = plt.figure()
        drag = Line2D([], [], c='r')
        sideforce = Line2D([], [], c='purple')
        lift = Line2D([], [], c='g')
        wind = Line2D([], [], c='orange', ls='--')
        gravity = Line2D([], [], c='pink')
        legend_fig.legend((drag, sideforce, lift, wind, gravity), ('drag', 'sideforce', 'lift', 'wind', 'gravity'), title="Arrow key") # prop={'size': 15})
        results_plotting.crop_and_save(legend_fig, base_fig_save_folder / 'animations' / 'forces_legend', dpi=fig_dpi)

        plt.show(block=True)

    # ==
    # This animation shows the wind, the airspeed and groundspeed vectors, and the projection of the aerodynamic force onto the direction of inertial motion.
    if True:
        # This trace is coloured by inertial power (pow_i, the rate of specific inertial energy change) rather than air-relative power, as this is positive (the aircraft's trajectory
        # trace is red) whenever the projection of the aircraft's total aerodynamic force onto its direction of inertial motion is positive. 
        plot_arrows = plot_arrows = ['lift', 'aero_force', 'aero_vi_force', 'wind_triangle', 'vi', 'va']
        # The anim = ... must be present, else it doesn't work.
        anim = results_plotting.BuildingBackAnimatorBase().animate(dl, pow_i, 'Specific inertial power ($Jkg^{-1}s^{-1}$)',
                                                                   base_fig_save_folder / 'animations' if save_anim else None, '_pow_i',
                                                                   plot_arrows=plot_arrows, wind_scale=3, plane_scale=1, force_scale=10, anim_interval_ms=20,
                                                                   subsample_num=subsample_num, save_dpi=video_dpi)
        
        # Create legend
        legend_fig = plt.figure()
        f_aero = Line2D([], [], c='indigo')
        f_aero_vi_proj = Line2D([], [], c='indigo', ls='--')
        lift = Line2D([], [], c='g')
        wind = Line2D([], [], c='orange', ls='--')
        vi = Line2D([], [], c='teal')
        va = Line2D([], [], c='mediumturquoise')
        legend_fig.legend((f_aero, f_aero_vi_proj, lift, wind, vi, va), ('aerodynamic force', 'aerodynamic force $v_i$ projection', 'lift', 'wind', 'groundspeed ($v_i$)', 'airspeed ($v_a$)'), title="Arrow key") # prop={'size': 15})
        results_plotting.crop_and_save(legend_fig, base_fig_save_folder / 'animations' / 'aero_projection_legend', dpi=fig_dpi)

        plt.show(block=True)


# =========
# Using the DataLoggers directly

# All of the data from the DataLoggers can be accessed to perform further calculations.
# Have a look at the DataLogger class in flight.simulator.utils to see what's recorded and saved.
# The methods chapter of my thesis details the modelling states and how to interpret them (conventions, etc).

if True:
    # Example 1: calculating and plotting groundspeed
    # -----------------------------------------------
    # u, v and w are the body-frame inertial velocities (forwards, sideways, below)
    vi = np.sqrt(dl.us**2 + dl.vs**2 + dl.ws**2)    # Calculate speed using Pythagoras' theorem

    # Plot
    fig, ax = plt.subplots()
    ax.plot(dl.times, vi, label='groundspeed ($v_i$)')
    ax.plot(dl.times, dl.vas, label='airspeed ($v_a$)')
    ax.set_xlabel('Time ($s$)')
    ax.set_ylabel('Velocity ($m s^{-1}$)')
    ax.legend()
    fig.suptitle("Velocity")

    # Example 2: calculating the final specific inertial energy
    # ---------------------------------------------------------

    def calc_specific_inertial_energy(dl, aircraft_model, ind):
        return -aircraft_model.g*dl.ds[ind] + 0.5*(dl.us[ind]**2 + dl.vs[ind]**2 + dl.ws[ind]**2)

    print(f"Final specific inertial energy >>> {calc_specific_inertial_energy(dl, am, -1)} J/kg")

    plt.show(block=True)