# FT 4/7/25
# Minimum time flight between two points

import jax
# Global flag to set a specific platform, must be used at startup.
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
# Checking that Jax is using the CPU (faster)
print(f"JAX default backend: {jax.devices()}")
import openmdao.api as om

# Add the flight directory to the Python path (when running from FlightSwordLite)
import sys, os
sys.path.append(os.getcwd())

from flight.simulator import wind_models
from flight.simulator.config import base_aircraft_model_path
from flight.optimiser.base_solver import TrajectorySolver 


# ==================================================
# Callback functions for optimiser

def set_objective(prob, control_phase, dynamics_phase, **kwargs):
    dynamics_phase.add_objective('time', loc='final')

def set_additional_constraints(prob, control_phase, dynamics_phase):
    # Constrain initial position
    TrajectorySolver.constrain_position(dynamics_phase, 'initial', *start_coord)
    TrajectorySolver.constrain_position(dynamics_phase, 'final', *end_coord)
    
    # Constrain initial speed
    dynamics_phase.add_boundary_constraint('u', 'initial', equals=20)
    dynamics_phase.add_boundary_constraint('v', 'initial', equals=0)
    dynamics_phase.add_boundary_constraint('w', 'initial', equals=0)

    # Constrain initial orientation
    dynamics_phase.add_boundary_constraint('phi', 'initial', equals=0)
    dynamics_phase.add_boundary_constraint('theta', 'initial', equals=0)
    dynamics_phase.add_boundary_constraint('psi', 'initial', equals=0)

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

# Setup solve parameters
solver_params = {
    'aircraft_model_path': base_aircraft_model_path / 'wot4_imav_v2.yaml',
    'wind_manager': wind_models.WindNone(),
    'record_sim': True,
    'datalogger_dt': 0.001
}

run_params = {
    'num_dynamics_segments': 10,
    'dynamics_seg_order': 3,
    'num_control_segments': 5,
    'control_seg_order': 3,
    'num_refinements': 0,
    'max_iter': 1000,
    'time_duration_bounds': (0, 30),
    'enable_throttle_optimisation': False
}

start_coord = (0, 0, -60)
end_coord = (100, 0, -20)

# TODO Need to add the folder path back in
name = 'baseline__min_time_descent'

# Set parameters for initial run
dynamics_times = control_times = [0, 30.0]
# States
ns = [start_coord[0], end_coord[0]]
es = [start_coord[1], end_coord[1]]
ds = [start_coord[2], end_coord[2]]
us = [20, 20]   # Not completely accurate, as it's the airspeed which starts at 20 m/s.
vs = [0, 0]
ws = [0, 0]
phis = [0, 0]
thetas = [0, 0]
psis = [0, 0]
ps = [0, 0]
qs = [0, 0]
rs = [0, 0]
# Controls
das = [0, 0]
des = [0, 0]
drs = [0, 0]
dps = [0, 0]

# False to simulate, True to optimise
solve = True

solver = TrajectorySolver(solver_params, name)
current_prob = solver.run_traj_solver(run_params, set_objective, set_additional_constraints, set_initial_values, set_driver_options, solve=solve, check_partials=False)
solver.post_process(plot=True)

import matplotlib.pyplot as plt
plt.show(block=True)