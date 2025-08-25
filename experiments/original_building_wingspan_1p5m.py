import jax
# Global flag to set a specific platform, must be used at startup.
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
# Checking that Jax is using the CPU (faster)
print(f"JAX default backend: {jax.devices()}")

import numpy as np
from itertools import product
from functools import reduce
import openmdao.api as om
import matplotlib.pyplot as plt
import gc
from pathlib import Path
from datetime import datetime

# Add the flight directory to the Python path (when running from FlightSwordLite)
import sys, os
sys.path.append(os.getcwd())

from flight.simulator.wind_models import WindQuic
from flight.simulator.config import base_aircraft_model_path
from flight.simulator.custom_exceptions import RatesCompException

from flight.optimiser.base_solver import TrajectorySolver # run_traj_solver, constrain_position
from flight.optimiser import config

# Debugging switch - name set to datetime-stamp for uniqueness, wind field set to 270_on_5.
debugging = False

# ==================================================
# Callback functions for optimiser

def set_objective(prob, control_phase, dynamics_phase, **kwargs):
    # Maximize inertial energy while minimizing control activity
    aircraft_model = kwargs['aircraft_model']
    control_rate_coeff = 1
    prob.model.add_subsystem('objective', om.ExecComp("E = -(-g*d + 0.5*(n_dot**2 + e_dot**2 + d_dot**2)) + control_rate_coeff*abs_control_surface_rate_int",
                                                      d={"shape":()},
                                                      n_dot={"shape":()},
                                                      e_dot={"shape":()},
                                                      d_dot={"shape":()},
                                                      abs_control_surface_rate_int={"shape":()},
                                                      g={'val': aircraft_model.g},
                                                      control_rate_coeff={'val': control_rate_coeff}))
    # prob.model.connect('traj.control_phase.timeseries.time', 'objective.time', src_indices=[-1])
    prob.model.connect('traj.dynamics_phase.timeseries.d', 'objective.d', src_indices=[-1])
    prob.model.connect('traj.dynamics_phase.timeseries.n_dot', 'objective.n_dot', src_indices=[-1])
    prob.model.connect('traj.dynamics_phase.timeseries.e_dot', 'objective.e_dot', src_indices=[-1])
    prob.model.connect('traj.dynamics_phase.timeseries.d_dot', 'objective.d_dot', src_indices=[-1])
    prob.model.connect('control_deriv_integrator.abs_control_rate_int', 'objective.abs_control_surface_rate_int')
    prob.model.add_objective('objective.E')


def set_additional_constraints(prob, control_phase, dynamics_phase):
    # Constrain initial and final position
    # Initial
    dynamics_phase.add_boundary_constraint('n', loc='initial', equals=start_coord[0])
    dynamics_phase.add_boundary_constraint('e', loc='initial', equals=start_coord[1])
    dynamics_phase.add_boundary_constraint('d', loc='initial', equals=start_coord[2])
    # Final
    dynamics_phase.add_boundary_constraint('n', loc='final', equals=end_coord[0])
    dynamics_phase.add_boundary_constraint('e', loc='final', equals=end_coord[1])

    # Constraints from wind model
    dynamics_phase.add_path_constraint('n', lower=n_lower, upper=n_upper)
    dynamics_phase.add_path_constraint('e', lower=e_lower, upper=e_upper)
    dynamics_phase.add_path_constraint('d', lower=d_lower, upper=d_upper)

    # Set initial groundspeed to be equal for all flights
    prob.model.add_subsystem('init_gs', om.ExecComp("vg = (u**2 + v**2 + w**2)**0.5",
                                                      u={"shape":()},
                                                      v={"shape":()},
                                                      w={"shape":()}))
    prob.model.connect('traj.dynamics_phase.timeseries.u', 'init_gs.u', src_indices=[0])
    prob.model.connect('traj.dynamics_phase.timeseries.v', 'init_gs.v', src_indices=[0])
    prob.model.connect('traj.dynamics_phase.timeseries.w', 'init_gs.w', src_indices=[0])
    prob.model.add_constraint('init_gs.vg', equals=20)


def set_initial_values(prob, control_phase, dynamics_phase):
    # Time
    prob.set_val('traj.control_phase.t_duration', control_times[-1])

    # States
    prob.set_val('traj.dynamics_phase.states:n', dynamics_phase.interp('n', xs=dynamics_times, ys=ns))
    prob.set_val('traj.dynamics_phase.states:e', dynamics_phase.interp('e', xs=dynamics_times, ys=es))
    prob.set_val('traj.dynamics_phase.states:d', dynamics_phase.interp('d', xs=dynamics_times, ys=ds))
    prob.set_val('traj.dynamics_phase.states:u', dynamics_phase.interp('u', xs=dynamics_times, ys=us))
    prob.set_val('traj.dynamics_phase.states:v', dynamics_phase.interp('v', xs=dynamics_times, ys=vs))
    prob.set_val('traj.dynamics_phase.states:w', dynamics_phase.interp('w', xs=dynamics_times, ys=ws))
    prob.set_val('traj.dynamics_phase.states:phi', dynamics_phase.interp('phi', xs=dynamics_times, ys=phis))
    prob.set_val('traj.dynamics_phase.states:theta', dynamics_phase.interp('theta', xs=dynamics_times, ys=thetas))
    prob.set_val('traj.dynamics_phase.states:psi', dynamics_phase.interp('psi', xs=dynamics_times, ys=psis))
    prob.set_val('traj.dynamics_phase.states:p', dynamics_phase.interp('p', xs=dynamics_times, ys=ps))
    prob.set_val('traj.dynamics_phase.states:q', dynamics_phase.interp('q', xs=dynamics_times, ys=qs))
    prob.set_val('traj.dynamics_phase.states:r', dynamics_phase.interp('r', xs=dynamics_times, ys=rs))

    # Controls on a different phase
    prob.set_val('traj.control_phase.controls:da', control_phase.interp('da', xs=control_times, ys=das))
    prob.set_val('traj.control_phase.controls:de', control_phase.interp('de', xs=control_times, ys=des))
    prob.set_val('traj.control_phase.controls:dr', control_phase.interp('dr', xs=control_times, ys=drs))
    prob.set_val('traj.control_phase.controls:dp', control_phase.interp('dp', xs=control_times, ys=dps))

    # Parameters
    # None in this model
    # Example: prob.set_val('traj.phase0.parameters:m', 1.)

    # Optimisable parameters
    # None in this model
    # Example: prob.set_val('traj.phase0.parameters:alpha', 1.)


# Useful when running with the linear interpolator, to reduce the tolerance required for a solution.
def set_driver_options(driver):
    pass

# ==================================================

def add_transcription_points_at_wind_boundary(prob, boundary_dist):
    # Code inspired by del's answer to: https://stackoverflow.com/questions/12926898/numpy-unique-without-sort
    def unique_maintain_order(arr):
        inds = np.unique(arr, return_index=True)[1]
        return np.array([arr[index] for index in sorted(inds)])

    # === Processing results ===
    # Get the x, y and z locations for each node - see when they're within a boundary of the edge  (this might have to be done via a callback)
    # If they are, get the times, map to the normalised time, and put a couple of collocation points in the vicinity.
    #   Do this by inserting the normalised time into tx_times_normalised in the right place, or appending it and then re-sorting.
    # Testing - running my own grid refinement here.
    # Get the initial transcription
    tx = prob.model.traj.phases.dynamics_phase.options['transcription']
    tx_times_normalised = tx.grid_data.segment_ends

    ns = unique_maintain_order(prob.get_val('traj.dynamics_phase.states:n')).flatten()
    es = unique_maintain_order(prob.get_val('traj.dynamics_phase.states:e')).flatten()
    ds = unique_maintain_order(prob.get_val('traj.dynamics_phase.states:d')).flatten()
    times = unique_maintain_order(prob.get_val('traj.dynamics_phase.t'))[::2]

    # Find times where there's a breach
    n_lims = (n_lower + boundary_dist, n_upper - boundary_dist)
    e_lims = (e_lower + boundary_dist, e_upper - boundary_dist)
    d_lims = (d_lower + boundary_dist, d_upper - boundary_dist)
    n_breaches = np.union1d(np.where(ns < n_lims[0])[0], np.where(ns > n_lims[1])[0])
    e_breaches = np.union1d(np.where(es < e_lims[0])[0], np.where(es > e_lims[1])[0])
    d_breaches = np.union1d(np.where(ds < d_lims[0])[0], np.where(ds > d_lims[1])[0])
    # Take the union of all of these indices, then find the times.
    all_breach_inds = reduce(np.union1d, (n_breaches, e_breaches, d_breaches))

    # For each value, make sure that the two around it are included.
    # Max and min are there to ensure that this stays in range.
    breach_triads = [unique_maintain_order([np.max((p-1, 0)), p, np.min((p+1, len(times)-1))]) for p in all_breach_inds]
    error_times = []
    for t in breach_triads:
        triad_times = times[t]
        for i in range(len(triad_times) - 1):
            error_times.append((triad_times[i] + triad_times[i+1])/2)
    # Remove duplicates
    error_times = unique_maintain_order(error_times)
    # Get the times which are causing an issue
    # error_times = times[all_breach_inds]
    # Convert into the normalised [-1, 1] interval
    # For now just take the ones around it.
    # TODO Incorporate the starting time too.
    # TODO Might want to add more times around the edges.
    # Scale into the range [-1, 1]
    error_times_normalised = (2*((error_times - times[0])) / (times[-1] - times[0])) - 1  # TODO Complete!
    
    # Add new times around this point (linear mapping - TODO is this the right way to do it?)
    new_tx_normalised_times = np.append(tx_times_normalised, error_times_normalised)
    new_tx_normalised_times.sort()

    # Return the new normalised transcription times
    return new_tx_normalised_times

    # Then restart everything again using this new transcription.

# ==================================================

# Setup solve parameters
solver_params = {
    'aircraft_model_path': base_aircraft_model_path / 'wot4_imav_v2_wingspan_1p5m.yaml',
    # 'wind_manager' added in loop
    'record_sim': True,
    'datalogger_dt': 0.001
}

run_params = {
    'num_dynamics_segments': 75,
    'dynamics_seg_order': 3,
    'num_control_segments': 15,
    'control_seg_order': 3,
    'num_refinements': 1,
    'max_iter': 5000,
    'time_duration_bounds': (1, 300),
    'enable_throttle_optimisation': False
}

start_coord = (400, 205, -20)
end_coord = (100, 205, -20)

valid_flight_region_n = (99., 401.)
valid_flight_region_e = (191., 399.)
valid_flight_region_d = (-69., -1.)

# False to simulate, True to optimise
solve = True
max_tries = 3
boundary_dist = 2 # m

results_folder_path = Path('.') / 'data'

if debugging:
    name = str(datetime.now())
else:
    name = f"test_jiayi_wingspan_1p5m"

#if name in existing_log_names:
#    continue

print(f"Running {name}")

# Load wind model
# @Jiayi: Set to path of folder containing wind field folders
wm = WindQuic(f'270_on_5__total__cubic_spline', Path('/home/cx24935/Downloads/flight/WindFields'))
# Add to solver_params dictionary
solver_params['wind_manager'] = wm

# Get wind field boundaries for constraints  (see set_additional_constraints method)
# Set region of validity for optimiser
eps = 0.5
n_wind_lower, n_wind_upper = (wm.valid_n_range[0] + eps, wm.valid_n_range[1] - eps)
e_wind_lower, e_wind_upper = (wm.valid_e_range[0] + eps, wm.valid_e_range[1] - eps)
d_wind_lower, d_wind_upper = (wm.valid_d_range[0] + eps, wm.valid_d_range[1] - eps)

# Find intersection of valid regions
n_lower, n_upper = (np.max((n_wind_lower, valid_flight_region_n[0])), np.min((n_wind_upper, valid_flight_region_n[1])))
e_lower, e_upper = (np.max((e_wind_lower, valid_flight_region_e[0])), np.min((e_wind_upper, valid_flight_region_e[1])))
d_lower, d_upper = (np.max((d_wind_lower, valid_flight_region_d[0])), np.min((d_wind_upper, valid_flight_region_d[1])))

solver = None   # So that garbage collection in finally clause has a reference to solver, even it it isn't assigned below.

# Start optimisation running. Everything should be saved automatically.
solved = False
try:
    solver = TrajectorySolver(solver_params, name, results_folder_path)

    # Set parameters for initial run
    dynamics_times = control_times = [0, 30.0]
    # States
    ns = [start_coord[0], end_coord[0]]
    es = [start_coord[1], end_coord[1]]
    ds = [start_coord[2], end_coord[2]]
    us = [20, 20]
    vs = [0, 0]
    ws = [0, 0]
    phis = [0, 0]
    thetas = [0, 0]
    psis = [np.pi, np.pi]
    ps = [0, 0]
    qs = [0, 0]
    rs = [0, 0]
    # Controls
    das = [0, 0]
    des = [0, 0]
    drs = [0, 0]
    dps = [0, 0]

    # First solve
    current_prob = solver.run_traj_solver(run_params, set_objective, set_additional_constraints, set_initial_values, set_driver_options, solve=solve, check_partials=True)

    for _ in range(max_tries):
        try:
            solver.post_process(plot=True)
            solved = True
            break       # Simulation successful
        except RatesCompException as rce:
            print("Simulation failed - rates computation exception. Adding points at boundary regions.")
        except Exception as e:
            print(f"An exception occurred: {e}")
            # TODO Does this work as intended?
            raise e

        # Add more points around boundary region and solve again
        # Calculate new initial values
        # Set parameters for initial run
        new_tx_normalised_times = add_transcription_points_at_wind_boundary(current_prob, boundary_dist)
        # Update run_params
        run_params['num_dynamics_segments'] = len(new_tx_normalised_times) - 1
        run_params['normalised_dynamics_segment_ends'] = new_tx_normalised_times
        # Run with updated transcription settings
        # Get previous time, state and control values for warm starting.
        dynamics_times = current_prob.get_val('traj.dynamics_phase.timeseries.time')
        ns = current_prob.get_val('traj.dynamics_phase.timeseries.n')
        es = current_prob.get_val('traj.dynamics_phase.timeseries.e')
        ds = current_prob.get_val('traj.dynamics_phase.timeseries.d')
        us = current_prob.get_val('traj.dynamics_phase.timeseries.u')
        vs = current_prob.get_val('traj.dynamics_phase.timeseries.v')
        ws = current_prob.get_val('traj.dynamics_phase.timeseries.w')
        phis = current_prob.get_val('traj.dynamics_phase.timeseries.phi')
        thetas = current_prob.get_val('traj.dynamics_phase.timeseries.theta')
        psis = current_prob.get_val('traj.dynamics_phase.timeseries.psi')
        ps = current_prob.get_val('traj.dynamics_phase.timeseries.p')
        qs = current_prob.get_val('traj.dynamics_phase.timeseries.q')
        rs = current_prob.get_val('traj.dynamics_phase.timeseries.r')
        control_times = current_prob.get_val('traj.control_phase.timeseries.time')
        das = current_prob.get_val('traj.control_phase.timeseries.da')
        des = current_prob.get_val('traj.control_phase.timeseries.de')
        drs = current_prob.get_val('traj.control_phase.timeseries.dr')
        dps = current_prob.get_val('traj.control_phase.timeseries.dp')
        current_prob = solver.run_traj_solver(run_params, set_objective, set_additional_constraints, set_initial_values, set_driver_options, solve=solve)
    
    if not solved:
        raise Exception("Maximum number of solve attempts exceeded")
except Exception as e:
    print(f"An error occurred - solve unsuccessful: {e}")
finally:
    # Preventing memory overuse
    plt.close('all')
    # Delete object and garbage collect
    del solver
    gc.collect()

del wm
gc.collect() 
