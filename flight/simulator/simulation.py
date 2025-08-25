# FT 30/8/24

import jax
jax.config.update("jax_enable_x64", True)
# Global flag to set a specific platform, must be used at startup.
jax.config.update('jax_platform_name', 'cpu')
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from functools import partial
import matplotlib.pyplot as plt

# Add the flight directory to the Python path (when running from FlightSwordLite)
import sys, os
sys.path.append(os.getcwd())

from flight.simulator.aircraft_model import AircraftModel
from flight.simulator.custom_exceptions import OutOfWorldException
from flight.simulator.dynamics import BeardDynamics
from flight.simulator.utils import StateDecoder, StateEncoder, DataLogger, ned_to_xyz
from flight.simulator.view import View, ViewBuildingBack, FlightGrapher
from flight.simulator.world import World, Building
from flight.simulator import config
from flight.simulator import controllers
from flight.simulator import integrators
from flight.simulator import wind_models

from flight.analysis.results_plotting import generate_plots

class Simulation:
    def __init__(self, aircraft_model_filepath, init_state, dynamics_class, controller, dt, num_steps, world=None, integrator=integrators.RK4, view=False):
        self.state = init_state
        self.sd = StateDecoder(self.state)
        self.controller = controller
        self.integrator = integrator

        self.dt = dt
        self.num_steps = num_steps

        # Reads aircraft parameter file and provides functions for calculating forces and moments
        aircraft_model = AircraftModel(aircraft_model_filepath)

        if world is None:
            wind_model = wind_models.NoWind()
            buildings = []
            world = World(wind_model, buildings)
        
        self.data_logger = DataLogger.create(dt, num_steps, False)
        self.dynamics = dynamics_class(aircraft_model, world, self.data_logger)

        if view:
            # View expects aircraft position in xyz rather than ned coordinate system (View is designed
            # # to be decoupled from the particular coordinate system used by the dynamics equations)
            x, y, z = ned_to_xyz(self.sd.n, self.sd.e, self.sd.d)
            self.view = View(world, x, y, z, self.sd.phi, self.sd.theta, self.sd.psi)
            # self.view = ViewBuildingBack(world, x, y, z, self.sd.phi, self.sd.theta, self.sd.psi)
        else:
            self.view = None
    
    # For debugging
    @staticmethod
    def print_state_values(sd, posns=True, vels=True, orients=True, ang_rates=True):
        print("|", end="")
        if posns:
            print(f" n: {sd.n}  \t|  e: {sd.e}  \t|  d:  {sd.d}  \t|", end="")
        if vels:
            print(f" va: {sd.va}  \t|  alpha: {sd.alpha}  \t|  beta:  {sd.beta}  \t|", end="")
        if orients:
            print(f" phi: {sd.phi}  \t|  theta: {sd.theta}  \t|  psi:  {sd.psi}  \t|", end="")
        if ang_rates:
            print(f" p: {sd.p}  \t|  q: {sd.q}  \t|  r:  {sd.r}  \t|", end="")
        print()
    
    # For debugging
    @staticmethod
    def print_control_values(ctrl_vec):
        print(f"da: {ctrl_vec[0]}  |  de: {ctrl_vec[1]}  |  dr:  {ctrl_vec[2]}  |  dp:  {ctrl_vec[3]}")

    def run(self, value_print_interval=10, stop_condition=(lambda sd: False)):
        t = 0.

        try:
            # Logs at the beginning of a step, so needs to do one more to accomplish the requested amount  (TODO improve this)
            for i in range(self.num_steps + 1):
                if stop_condition(self.sd):
                    print("Stop condition met")
                    # DataLogger length was pre-set on creation, but the simulator is being stopped early - need to truncate the DataLogger.
                    self.data_logger.end_logging()
                    break

                # Generate control signal
                # Control signal is a 4-tuple (da, de, dr, dp) - defined in Ana's
                # paper to be the deflection angles.
                # Pass the StateDecoder
                control_inputs = self.controller.gen_control_signal(sd=self.sd, t=t)
                if control_inputs is None:
                    print("Out of control inputs")
                    # DataLogger length was pre-set on creation, but the simulator is being stopped early - need to truncate the DataLogger.
                    self.data_logger.end_logging()
                    break
                
                # Update visualiser
                if self.view:
                    if i % 10 == 0:
                        # Transform position coordinates from NED frame to xyz frame used by Matplotlib for plotting.
                        # Returned xyz_posn_coords will be a (3,) Numpy array.
                        x, y, z = ned_to_xyz(self.sd.n, self.sd.e, self.sd.d)
                        self.view.update(t, x, y, z, self.sd.phi, self.sd.theta, self.sd.psi)

                if config.debugging:
                    if i % value_print_interval == 0:
                        # print_control_values(control_inputs)
                        self.print_state_values(self.sd, posns=True, vels=True, orients=True, ang_rates=True)

                # Integrates dynamics to update the state
                # This function needs the actual state, not the state decoder.
                # TODO Should be updating the state decoder rather than declaring a new one each time.
                # self.state, self.sd, self.data_logger = self.dynamics.step(self.state, t, control_inputs, self.dt, self.data_logger, self.integrator)
                self.state, self.sd = self.dynamics.step(self.state, t, control_inputs, self.dt, self.data_logger, self.integrator)

                # If state contains NaN values
                if np.isnan(self.state.sum()):
                    raise Exception("State contains NaN values")

                # Increment time (affects the wind field, and hence the dynamics)
                t += self.dt
                
        except OutOfWorldException as oowe:
            print(f"OutOfWorldException occurred: {oowe}")

            # End early
            self.data_logger.end_logging()
        
        except Exception as e:
            print(f"Exception occurred: {e}")

            # End early
            self.data_logger.end_logging()
        
        finally:
            return self.data_logger

if __name__ == "__main__":        
    # Create wind model
    wind = wind_models.WindNone()
    # wind = wind_models.WindLinear(1.)

    # Create World object - requires a wind model as argument
    world = World(wind)

    # Create state (this is just a Numpy array)
    init_state = StateEncoder.encode(d=-20., u=20.)

    dt = 0.005
    num_steps = 3000

    # Create a controller - this controller enables the aircraft to be controlled with
    # the keyboard arrow keys. 
    controller = controllers.ControllerManualDiscrete(world, dt)

    # Create simulation
    # Run simulation
    model_path = config.base_aircraft_model_path / 'wot4_imav_v2.yaml'
    # Create a simulation object. Requires the path to the aircraft model, the initial state,
    # the dynamic class, the controller, the integration timestep (dt), the number of steps to
    # integrate the dynamics for (num_steps), and the World object (which contains the wind field).
    sim = Simulation(model_path, init_state, BeardDynamics, controller, dt, num_steps, world, view=True)
    input("Press enter to start > ")
    # Run the simulation
    dl = sim.run()

    # Save DataLogger
    dl.save('test')

    # Generate the plots
    generate_plots(Path('/home/cx24935/Downloads/flight/data/simulator_figs'), dl, AircraftModel(model_path), save=True)
    grapher = FlightGrapher(100, plane_scale=2, wind_scale=5)
    grapher.add_flight(dl, 'black')
    plt.show(block=True)












"""
#dl = DataLogger.load('round_building_linked__no_wind__baseline_optimiser')

# Create wind model
# wind = wind_models.WindConstant(0., 10., 0.)
# wind = wind_models.WindTanh()
# wind = wind_models.WindNone()
# wind = wind_models.WindModel.create("cfd", fieldname="gaussian_sum_least_sq")
# wind = wind_models.WindModel.create("cfd", fieldname="gaussian_bump")
# wind = wind_models.WindModel.create("cfd", fieldname="gaussian_bump_linear")
# wind = wind_models.WindModel.create("cfd", fieldname="front_on_10__trilinear")
# Create world model
# wind = wind_models.WindUpdraft()
# wind = wind_models.WindSidewaysSine(3, 50)
# wind = WindSachs()
# wind = wind_models.WindFourierStationary([3, 1])

# Load wind model
wm_folder = Path('/mnt/c/Users/fjtur/OneDrive - University of Bristol/Documents/PhD/Projects/Isolated building trajectory optimisation (chap 1)/Wind field data/Simulator wind models/Original building/Speed and direction sweep')
wind_dir_deg = 270
wind_speed = 5
wm_name = f'{wind_dir_deg}_on_{wind_speed}__total__cubic_spline'
wind = wind_models.WindQuic(wm_name, wm_folder)

world = World(wind, [Building((131, 369), (100, 190), (-33, 0))])

init_state = StateEncoder.encode(360., 205., -20., u=20., theta=np.radians(40), psi=np.pi*1.2) # , n=-30.) # , theta=np.radians(30), psi=np.radians(30))

dt = 0.005 # dl.dt
num_steps = 5000 # 3000 # dl.num_steps

# controller = controllers.ControllerNull(world)
# controller = controllers.ControllerIterator(world, dl.das, dl.des, dl.drs, dl.dps)
controller = controllers.ControllerManualDiscrete(world, dt)
#controller = controllers.ControllerWaypoint(world, dt)
#controller.set_position_waypoint(131., 98.5, -33.5)
# controller.set_position_waypoint(131.5, 98, -33)

# Create simulation
# Run simulation
model_path = config.base_aircraft_model_path / 'wot4_imav_v2.yaml'
sim = Simulation(model_path, init_state, BeardDynamics, controller, dt, num_steps, world, view=True)
input("Press enter to start > ")
dl = sim.run() # stop_condition = lambda sd: (sd.e <= 191) or (sd.d <= 0))

# Save DataLogger
dl.save('manual_in_wind_8')

# generate_plots_back_of_building(None, dl, AircraftModel(model_path))
generate_plots_back_of_building(None, dl, AircraftModel(model_path), wind_dir_deg, wind_speed)
grapher = FlightGrapher(100, plane_scale=2, wind_scale=5) # , buildings=[Building((131, 369), (100, 190), (-33, 0))])
grapher.add_flight(dl, 'black')
import matplotlib.pyplot as plt
plt.show(block=True)
"""

"""
# v_ref_default = 5
# 
# # Define wind model, with a parameter.
# class WindSachs(wind_models.WindModel):
#     def __init__(self, h0=1, h_ref=10):
#         super().__init__()
# 
#         self.h0 = h0        # m
#         self.h_ref = h_ref  # m
#     
#     # Returns the wind Jacobian evaluated at the point (n, e, d).
#     # TODO Important! This doesn't handle time at all at the moment!
#     def jac_single(self, n, e, d, t, v_ref=v_ref_default):
#         return jnp.asarray(jax.jacobian(self.wind_single, argnums=[0, 1, 2])(n, e, d, t, v_ref)).T
#     
#     def time_deriv_single(self, n, e, d, t, v_ref=v_ref_default): 
#         return jax.jacobian(self.wind_single, argnums=[3])(n, e, d, t, v_ref)[0]
# 
#     @partial(jax.jit, static_argnums=(0,))
#     def wind_single(self, n, e, d, t, v_ref=v_ref_default):
#         wn = v_ref*jnp.log(-d/self.h0) / jnp.log(self.h_ref/self.h0)
#         return jnp.array([wn, 0., 0.])
"""
