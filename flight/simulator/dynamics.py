# FT 25/9/23

import abc
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt  # For model plotting functionality

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class 
from functools import partial

# Add the flight directory to the Python path (when running from FlightSwordLite)
import sys, os
sys.path.append(os.getcwd())

from flight.simulator import utils


# Base class for dynamics simulation (contains equations of motion in gradients function).
class DynamicsModelBase(metaclass=abc.ABCMeta):
    # @abc.abstractmethod
    def __init__(self, data_logger):
        self.data_logger = data_logger
        
    # Create plots of the dynamics model, save them in the folder with path folder_path.
    # Derived classes should implement this method and pass the required dynamics functions and arguments
    # to _plot_dynamics to perform the plotting.
    @abc.abstractmethod
    def plot(self, folder_path: Path):
        pass

    # This function creates plots of a dynamics model and saves them in the given 'folder_path' directory.
    # The function takes a list of Python functions which are salient to the dynamics (e.g. functions for computing lift, drag, moments, etc)
    # along with Numpy arrays of ranges for their arguments, and creates plots.
    # 
    # For each function, a plot is created showing the variation of that value (e.g. lift) with an argument (e.g. angle of attack). A
    # separate plot is created for each dynamics function - varied argument pair, for which all of the other arguments
    # (e.g. for lift: q, de and dp) are held at their defaults.
    # For this reason, all of the dynamics functions for plotting need default arguments for the arguments to be varied, where the default values will be
    # used whenever another parameter is being varied. It is okay to have arguments without default values, provided they are not to be varied
    # as parameters of interest (see the 'args' and 'required_args' argument information below).
    # 
    # This function also takes a 'sweep' argument. For each dynamics function - varied argument combination, sweep allows one variable to be swept
    # through and plotted for multiple values on the same graph. E.g. If the dynamics function is lift, the varied argument is alpha and the sweep
    # argument is airspeed, the plot would be of lift as a function of alpha, with multiple lines for different values of the airspeed.
    #   
    # The arguments for this function are:
    #  o folder_path : the path of the folder where the generated plots should be saved (Python Path object)
    #  o dynamics_functions : a Python list of functions, e.g. [L, C, D, T], where L, C, D, T are functions defined to calculate lift, sideforce,
    #    drag and thrust
    #  o args : a Python dictionary containing the function argument names as strings and their ranges (as Numpy arrays) as values. For example,
    #        args = {
    #            'va': np.array([0, 5, 10, 20, 30, 40]),
    #            'alpha': np.linspace(-40, 40, 50),
    #            'beta': np.linspace(-30, 30, 50)
    #        }
    #  o sweep : one argument can be swept through, plotting multiple curves on the same figure for different values (see the description above). Sweep
    #    is a string; the name of the argument to be swept through.
    #  o required_args : a Python dictionary containing arguments which don't have default values (for parameters which aren't to be varied). E.g. the lift
    #    function has the signature
    #       L(aircraft_model, va=0, alpha=0, q=0, de=0, dp=0)
    #    where aircraft_model is an object containing information and functions related to the flight model. This is a 'required_arg', and the rest have default
    #    values and will be varied (using the values passed in with 'args') to create the plots (in actual use, 'va' is swept through). The syntax for
    #    'required_args' would be:
    #       required_args = {'aircraft_model': self.aircraft_model}
    #
    # Key points
    # ==========
    #  o The dynamics functions supplied *must* have default values for the arguments to be varied. These values are used to hold the variables constant
    #    while the argument of interest is varied and its effect on the function value (e.g. lift) plotted.
    #  o 'args' *must* cover the complete set of all named arguments to all functions. E.g. if L is a function of alpha (only) and C is a function of beta (only),
    #    args must contain both alpha and beta. It doesn't matter that e.g. L doesn't take beta as an argument - the code handles this.
    #
    # Making a static method as it doesn't require access to the dynamics object - all information is passed in by parameter.
    @staticmethod
    def _plot_dynamics(folder_path: Path, dynamics_functions, args, sweep, required_args):
        print("Generating dynamics model plots...")

        sweep_vals = args[sweep]
        # Remove sweep from the other arguments for plotting
        args.pop(sweep)

        # For each function
        for func in dynamics_functions:
            # Iteratively for each argument (e.g. alpha), get the name and the corresponding array of values for plotting.
            for arg_name, arg_vals in args.items():
                skip_arg = False
                # Want a list of graphs
                graphs = []

                # Iterate: for each value of the sweep argument...
                #   Plot the function values against the x-axis values - func(arg_vals).
                #   A single plot will show this for each of the values of the sweep variable. E.g. if sweep is airspeed,
                #   func is lift (L) and arg_vals is an array of angle of attack values, e.g. np.array([-40, ..., 40]), the
                #   plot will show L(va=0, alpha=...), L(va=5, alpha=...), L(va=10, alpha=...), etc.
                for sweep_val in sweep_vals:
                    # Calculate
                    # All of the other arguments will automatically be at their defaults (e.g. zero, for the flight dynamics forces and moments equations).
                    # 'required_args' dictionary is merged into the dictionary of arguments provided to func.
                    try:
                        func_vals = func(**{sweep:sweep_val, arg_name:arg_vals, **required_args})
                        graphs.append(func_vals)

                    # This code branch will execute if arg_name is not a named argument of the function func.
                    #    I.e. if arg_name isn't a parameter of the function (e.g. lift doesn't depend on 'beta').
                    except TypeError as te:
                        # The argument is not used by func - skip to the next argument.
                        skip_arg = True
                        # Break out of inner loop
                        break
                
                if skip_arg:
                    continue   # If the argument is not used by func, skip to the next argument.
                               # No plot will be created for this function and argument combination.

                # Plot
                fig, ax = plt.subplots()
                for i in range(len(sweep_vals)):
                    ax.plot(arg_vals, graphs[i], label=str(sweep_vals[i]))
                
                # Label
                title = f"{func.__name__}({arg_name})"
                fig.suptitle(title)
                ax.set_xlabel(arg_name)
                ax.set_ylabel(func.__name__)
                ax.legend()
                
                # Save
                plt.savefig(folder_path / f"{title}.png")
                plt.close()
    
    # Calculates state gradients for propagating dynamics - this method is where the equations of motion go.
    # This method is called by the integrators in the integrators.py module.
    @abc.abstractmethod
    def gradients(self, state, t, controls):
        pass

    # Runs the dynamics to update the state
    # Argument 6 is the integrator
    # @partial(jax.jit, static_argnums=(6,))
    def step(self, state, t, controls, dt, data_logger, integrator):
        # Integrate state
        new_state, log_info = integrator.integrate(self.gradients, state, t, controls, dt)

        # Log flight data
        # Subclass specific actions - store forces and moments etc from log_info
        # State decoder
        # sd, data_logger = self.process_additional_dynamics_info(state, t, controls, log_info, data_logger)
        sd = self.process_additional_dynamics_info(state, t, controls, log_info, data_logger)

        # Return integrated state
        return new_state, sd
    
# Missing 'compute_gradients(...)' function, which calculates gradients of states. Two derived classes provide this - 
# AircraftDynamicsAna, containing Ana's dynamics equations, and AircraftDynamics, containing FT's.
@register_pytree_node_class
class AircraftDynamicsBase(DynamicsModelBase):
    def __init__(self, aircraft_model, world_model, data_logger):
        super().__init__(data_logger)
        self.aircraft_model = aircraft_model
        self.world = world_model
    
    def tree_flatten(self):
        children = (self.data_logger,)
        aux_data = {'aircraft_model': self.aircraft_model, 'world_model': self.world}
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data_logger = children[0]
        return cls(aux_data['aircraft_model'], aux_data['world_model'], data_logger)
        
    # Plotting flight dynamics model (with coefficients and coefficient functions defined in self.aircraft_model)
    # Plotting force and moment values (for a range of airspeeds), rather than coefficients. Function could easily be modified to plot coefficients.
    # folder_path should be a Python Path object
    def plot(self, folder_path: Path):
        # Defining range of inputs
        alphas = np.linspace(-40, 40, 1000)
        betas = np.linspace(-30, 30, 1000)

        # Max and min values gathered from flights
        # No guarantee that these are the extremes
        ps_normal = np.linspace(-4, 4, 1000)
        qs_normal = np.linspace(-4, 4, 1000)
        rs_normal = np.linspace(-4, 4, 1000)
        # I have seen values beyond these bounds
        ps_extreme = np.linspace(-400, 400, 1000)
        qs_extreme = np.linspace(-400, 400, 1000)
        rs_extreme = np.linspace(-400, 400, 1000)

        # Aileron, elevator and rudder deflection limits are from Ana's paper
        das = np.linspace(np.radians(-18), np.radians(18), 1000)
        des = np.linspace(np.radians(-15), np.radians(15), 1000)
        drs = np.linspace(np.radians(-29), np.radians(29), 1000)
        # This is just a guess
        dps = np.linspace(-10, 10, 1000)
        
        # For each graph (function and argument (e.g. alpha) combination), hold the other arguments constant, and plot it for a
        # variety of different airspeeds (airspeed is the sweep variable).
        vas = np.array([0, 5, 10, 20, 30, 40]) # , 60, 85, 110])
        
        # Functions to plot
        # Function arguments (which are varied for plotting) must all have default values. These are the values which are used when holding
        # them constant as other variable values are being changed to plot their effects.
        # E.g. L(aircraft_model, va=0, alpha=0, q=0, de=0, dp=0)
        # Note that aircraft_model doesn't have a default value as it's provided ('{'aircraft_model': self.aircraft_model}') in the required_args
        # argument to _plot_dynamics(...).
        dynamics_functions = [self.aircraft_model.L, self.aircraft_model.C, self.aircraft_model.D, self.aircraft_model.T, self.aircraft_model.L_lat, self.aircraft_model.M, self.aircraft_model.N]

        # Arguments for functions (dictionary of argument names and ranges for plotting)
        # TODO Have to find a way to do the extreme rate values separately.
        args = {
            'va': vas,
            'alpha': alphas,
            'beta': betas,
            'p': ps_normal,
            'q': qs_normal,
            'r': rs_normal,
            'da': das,
            'de': des,
            'dr': drs,
            'dp': dps
        }

        # Supplying arguments which are required for every function.
        required_args = {} # {'aircraft_model': self.aircraft_model}
        
        # Plot
        self._plot_dynamics(folder_path, dynamics_functions, args, 'va', required_args)


@register_pytree_node_class
class BeardDynamics(AircraftDynamicsBase):
    @staticmethod
    @jax.jit
    def calc_airstate(wind, u, v, w, phi, theta, psi):
        # Testing input values - make sure that none are NaN
        #jax.debug.print('(u, v, w, phi, theta, psi): {u}, {v}, {w}, {phi}, {theta}, {psi}', \
        #                  u=u, v=v, w=w, phi=phi, theta=theta, psi=psi)
        
        # No values come in as NaN, but NaN values are output.

        #jax.debug.print('phi:   {val}', val=phi)
        #jax.debug.print('theta: {val}', val=theta)
        #jax.debug.print('psi:   {val}', val=psi)

        # jax.debug.print('u:   {val}', val=u)
        # jax.debug.print('(n, e, d):   ({n}, {e}, {d}), ', n=n, e=e, d=d)

        body_to_inertial = utils.calc_body_to_inertial(phi, theta, psi)

        # These wind vectors have to be rotated into the body frame - Beard section 4.4.
        inertial_to_body = body_to_inertial.T                       # TODO [Dymos] Check that this still works as expected.

        # jax.debug.print('{val}', val=inertial_to_body)

        wind_in_body_frame = jnp.matmul(inertial_to_body, wind)      # TODO [Dymos] Check that this still works as expected.
        u_w, v_w, w_w = wind_in_body_frame
        
        # jax.debug.print('(u_w, v_w, w_w):   ({u_w}, {v_w}, {w_w}), ', u_w=u_w, v_w=v_w, w_w=w_w)

        u_r = u - u_w
        v_r = v - v_w
        w_r = w - w_w

        # jax.debug.print('(u_r, v_r, w_r):   ({u_r}, {v_r}, {w_r}), ', u_r=u_r, v_r=v_r, w_r=w_r)

        va = jnp.sqrt(jnp.square(u_r) + jnp.square(v_r) + jnp.square(w_r))
        # Do I need to think about using arctan2 here?
        alpha = jnp.arctan(w_r / u_r)
        beta = jnp.arcsin(v_r / va)

        #jax.debug.print('{val}', val=va)
        #jax.debug.print('{val}', val=alpha)
        #jax.debug.print('{val}', val=beta)
        
        return va, alpha, beta
    
    # Written this way (dynamics split between gradients() and compute_gradients()) so that it's compatable with both the simulator and
    # the Dymos trajectory optimiser.
    @jax.jit
    def gradients(self, state, t, controls):
        n, e, d, u, v, w, phi, theta, psi, p, q, r = state
        da, de, dr, dp = controls

        # grads, log_info
        return self.compute_gradients(self.aircraft_model, self.world.wind_at, self.world.wind_model.jac_single, self.world.wind_model.time_deriv_single, t, n, e, d, u, v, w, phi, theta, psi, p, q, r, da, de, dr, dp)
    
    # Written this way (dynamics split between gradients() and compute_gradients()) so that it's compatable with both the simulator and
    # the Dymos trajectory optimiser.
    # Dynamics go here
    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def compute_gradients(aircraft_model, wind_fn, jac_fn, time_deriv_fn, t, n, e, d, u, v, w, phi, theta, psi, p, q, r, da, de, dr, dp):
        # jax.debug.print("Computing gradients")

        # Testing input values - make sure that none are NaN
        #jax.debug.print('(n, e, d, u, v, w, phi, theta, psi, p, q, r, da, de, dr, dp): {n}, {e}, {d}, {u}, {v}, {w}, {phi}, {theta}, {psi}, {p}, {q}, {r}, {da}, {de}, {dr}, {dp}', \
        #                  n=n, e=e, d=d, u=u, v=v, w=w, phi=phi, theta=theta, psi=psi, p=p, q=q, r=r, da=da, de=dr, dr=dr, dp=dp)

        # Calculate the wind at the aircraft's location
        wind = wind_fn(n, e, d, t) # TODO [Dymos] Check that this is the right shape for Dymos
        w_n, w_e, w_d = wind
        # jax.debug.print("wind: {wind}", wind=wind)

        # Use the position (n, e, d) and wind to calculate va, alpha and beta.
        va, alpha, beta = BeardDynamics.calc_airstate(wind, u, v, w, phi, theta, psi)
        
        # jax.debug.print('(va, alpha, beta): ({va}, {alpha}, {beta})', va=va, alpha=alpha, beta=beta)

        # Use va, alpha and beta to calculate the forces and moments.
        L, C, D, T, L_lat, M, N = aircraft_model.calculate_forces_and_moments(va, alpha, beta, p, q, r, da, de, dr, dp)
        W = aircraft_model.m*aircraft_model.g # Weight

        # Translational kinematics
        # Check that this unpacking works correctly.
        body_to_inertial = utils.calc_body_to_inertial(phi, theta, psi)                  # TODO [Dymos] This needs checking
        n_dot, e_dot, d_dot = jnp.matmul(body_to_inertial, jnp.array([u, v, w]))        # TODO [Dymos] This needs checking

        # Translational dynamics
        # Force = aero + gravity + thrust
        # The aerodynamic forces are defined in the wind frame. These need to be rotated into the body frame, in which Beard's equations
        # are defined.
        # I think C should be positive, because the CY_beta coefficient in the 'wot4.yaml' file is negative.
        # This would mean that a positive sideslip angle (to the right), would result in a force to the left of the aircraft.
        f_aero_wind = jnp.array([-D, C, -L])
        f_aero_body = jnp.matmul(utils.calc_wind_to_body(alpha, beta), f_aero_wind)
        f_grav_body = jnp.matmul(body_to_inertial.T, jnp.array([0, 0, W]))
        f_thrust_body = jnp.array([T, 0, 0])
        f_x, f_y, f_z = f_aero_body + f_grav_body + f_thrust_body

        u_dot = r*v - q*w + (1/aircraft_model.m)*f_x
        v_dot = p*w - r*u + (1/aircraft_model.m)*f_y
        w_dot = q*u - p*v + (1/aircraft_model.m)*f_z

        # Rotational kinematics
        # Euler angle rates, calculated using p, q and r values (body axis rates)
        phi_dot   = p + (q*jnp.sin(phi) + r*jnp.cos(phi))*jnp.tan(theta)    # roll angle rate
        theta_dot = q*jnp.cos(phi) - r*jnp.sin(phi)                                 # pitch angle rate
        psi_dot   = (q*jnp.sin(phi) + r*jnp.cos(phi))/jnp.cos(theta)              # yaw angle rate
        
        # Rotational dynamics
        # These equations are the same as those in Beard, section 3.3
        # Apparently np.square() is slower but more accurate than **: https://stackoverflow.com/questions/29361856/python-numpy-square-vs
        p_dot = (aircraft_model.I_zz*L_lat + aircraft_model.I_xz*N - (aircraft_model.I_xz*(aircraft_model.I_yy - aircraft_model.I_xx - aircraft_model.I_zz)*p + \
                    (jnp.square(aircraft_model.I_xz) + aircraft_model.I_zz*(aircraft_model.I_zz - aircraft_model.I_yy))*r)*q)/aircraft_model.tau # roll rate
        q_dot = (M - (aircraft_model.I_xx - aircraft_model.I_zz)*p*r - aircraft_model.I_xz*(jnp.square(p) - jnp.square(r)))/aircraft_model.I_yy
        r_dot = (aircraft_model.I_xz*L_lat + aircraft_model.I_xx*N + (aircraft_model.I_xz*(aircraft_model.I_yy - aircraft_model.I_xx - aircraft_model.I_zz )*r + \
                    (jnp.square(aircraft_model.I_xz) + aircraft_model.I_xx*(aircraft_model.I_xx - aircraft_model.I_yy))*p)*q)/aircraft_model.tau # yaw rate
        
        # TODO Fictitious forces (calculated with wind Jacobian) - needs implementing.
        # TODO Looking at the boat code I think vi is meant to be in the ned frame, although I could be wrong.
        #    -> Do some analysis to figure it out.
        vi = jnp.array([n_dot, e_dot, d_dot])
        jac = jac_fn(n, e, d, t)
        time_deriv = time_deriv_fn(n, e, d, t)
        # TODO How to do this? This is not a force anymore, if we don't have the mass.
        # Or could we pass in the mass - would this break things?
        Fwn, Fwe, Fwd = -aircraft_model.m*(jnp.matmul(jac, vi) + time_deriv)
        
        # NOTE: This is defined in the body frame.
        # Gravity is a conservative force and doesn't change the total energy. Don't count its contribution. We are interested in energy change.
        # nog = 'no gravity'
        f_x_nog, f_y_nog, f_z_nog = f_aero_body + f_thrust_body
        lift_to_drag = jnp.dot(jnp.array([f_x_nog, f_y_nog, f_z_nog]), jnp.array([u_dot, v_dot, w_dot]))

        # Calculating load factor
        # n = -A|k^b / mg, where A|k^b is the component of the aerodynamic force parallel to k^b = f_aero_body[2]
        a_kb = f_aero_body[2]
        load_factor = -a_kb / (aircraft_model.m*aircraft_model.g)
        
        return jnp.array([n_dot, e_dot, d_dot, u_dot, v_dot, w_dot, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot]), jnp.array([L, C, D, T, W, L_lat, M, N, va, alpha, beta, w_n, w_e, w_d, Fwn, Fwe, Fwd, load_factor])
    
    def process_additional_dynamics_info(self, state, t, controls, log_info, data_logger):
        L, C, D, T, W, L_lat, M, N, va, alpha, beta, w_n, w_e, w_d, Fwn, Fwe, Fwd, load_factor = log_info

        sd = utils.StateDecoder(state, va, alpha, beta)
        
        # W: weight
        # Fwn, Fwe, Fwd: fictitious wind forces
        data_logger.log(t, state, va, alpha, beta, *controls, L, C, D, T, W, Fwn, Fwe, Fwd, L_lat, M, N, (w_n, w_e, w_d), load_factor)

        return sd