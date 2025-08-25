# FT 11/7/24
# Contains the OpenMDAO dynamics models used for simulation and optimisation

import openmdao.api as om
import dymos as dm
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from functools import partial

# Add the flight directory to the Python path (when running from FlightSwordLite)
import sys, os
sys.path.append(os.getcwd())

from flight.simulator.dynamics import BeardDynamics
from flight.simulator.custom_exceptions import RatesCompException

# from cfd_wind_field_parsers.quic_parser import QuicWindParser

# Define dynamics
class BeardDynamicsJax(om.ExplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._compute_primal_vec = jax.vmap(self._compute_primal)
        self._compute_partials_vec = jax.jit(jax.vmap(jax.jacfwd(self._compute_primal, argnums=np.arange(16))))
        # self._compute_partials_vec = jax.vmap(jax.jacfwd(self._compute_primal, argnums=np.arange(16)))

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('wind_model')
        self.options.declare('aircraft_model')

    def setup(self):
        nn = self.options['num_nodes']

        # States:
        # n, e, d           (inertial position)
        # u, v, w           (inertial speed, body-resolved)
        # phi, theta, psi   (Euler angles)
        # p, q, r           (angular rates, body resolved)
        self.add_input('n', shape=(nn,), desc='northerly displacement (ii direction)', units='m')
        self.add_input('e', shape=(nn,), desc='easterly displacement (ji direction)', units='m')
        self.add_input('d', shape=(nn,), desc='downwards displacement (ki direction)', units='m')
        self.add_input('u', shape=(nn,), desc='ib component of inertial velocity', units='m/s')
        self.add_input('v', shape=(nn,), desc='jb component of inertial velocity', units='m/s')
        self.add_input('w', shape=(nn,), desc='kb component of inertial velocity', units='m/s')
        # TODO These descriptions need improving.
        self.add_input('phi', shape=(nn,), desc='roll angle (see Beard chap. 2 & 3 for frame of reference)', units='rad')
        self.add_input('theta', shape=(nn,), desc='pitch angle, positive upwards from n-e plane', units='rad')
        self.add_input('psi', shape=(nn,), desc='heading (yaw) angle, clockwise from positive n (see Beard chap. 2 & 3 for frame of reference)', units='rad')
        #
        self.add_input('p', shape=(nn,), desc='body-frame roll rate, positive right wing down', units='rad/s')
        self.add_input('q', shape=(nn,), desc='body-frame pitch rate, positive nose up', units='rad/s')
        self.add_input('r', shape=(nn,), desc='body-frame yaw rate, positive nose right', units='rad/s')

        # Controls
        self.add_input('da', shape=(nn,), desc='aileron deflection', units='rad')
        self.add_input('de', shape=(nn,), desc='elevator deflection', units='rad')
        self.add_input('dr', shape=(nn,), desc='rudder deflection', units='rad')
        # TODO What are the units for this?
        # TODO Which direction does the throttle force act in?
        # self.add_input('dp', shape=(nn,), desc='throttle input', units=None)
        self.add_input('dp', shape=(nn,), desc='throttle input')

        # Parameters
        # None at the moment

        # Outputs
        self.add_output('n_dot', val=jnp.zeros(nn), desc='rate of change of n', units='m/s')
        self.add_output('e_dot', val=jnp.zeros(nn), desc='rate of change of e', units='m/s')
        self.add_output('d_dot', val=jnp.zeros(nn), desc='rate of change of d', units='m/s')
        self.add_output('u_dot', val=jnp.zeros(nn), desc='rate of change of u', units='m/s**2')
        self.add_output('v_dot', val=jnp.zeros(nn), desc='rate of change of v', units='m/s**2')
        self.add_output('w_dot', val=jnp.zeros(nn), desc='rate of change of w', units='m/s**2')
        self.add_output('phi_dot', val=jnp.zeros(nn), desc='rate of change of phi', units='rad/s')
        self.add_output('theta_dot', val=jnp.zeros(nn), desc='rate of change of theta', units='rad/s')
        self.add_output('psi_dot', val=jnp.zeros(nn), desc='rate of change of psi', units='rad/s')
        self.add_output('p_dot', val=jnp.zeros(nn), desc='rate of change of p', units='rad/s**2')
        self.add_output('q_dot', val=jnp.zeros(nn), desc='rate of change of q', units='rad/s**2')
        self.add_output('r_dot', val=jnp.zeros(nn), desc='rate of change of r', units='rad/s**2')

        # For logging only
        # Forces and moments
        self.add_output('L', val=jnp.zeros(nn), desc='lift force', units='N')
        self.add_output('C', val=jnp.zeros(nn), desc='sideforce', units='N')
        self.add_output('D', val=jnp.zeros(nn), desc='drag force', units='N')
        self.add_output('T', val=jnp.zeros(nn), desc='thrust force', units='N')
        self.add_output('W', val=jnp.zeros(nn), desc='weight force', units='N')
        # TODO Is Nm the correct unit for the moments?
        self.add_output('L_lat', val=jnp.zeros(nn), desc='roll moment', units='N*m')
        self.add_output('M', val=jnp.zeros(nn), desc='pitch moment', units='N*m')
        self.add_output('N', val=jnp.zeros(nn), desc='yaw moment', units='N*m')

        # Airstate
        self.add_output('va', val=jnp.zeros(nn), desc='airspeed', units='m/s')
        self.add_output('alpha', val=jnp.zeros(nn), desc='angle of attack', units='rad')
        self.add_output('beta', val=jnp.zeros(nn), desc='sideslip angle', units='rad')

        # Wind
        self.add_output('w_n', val=jnp.zeros(nn), desc='wind in northerly direction', units='m/s')
        self.add_output('w_e', val=jnp.zeros(nn), desc='wind in easterly direction', units='m/s')
        self.add_output('w_d', val=jnp.zeros(nn), desc='wind in downwards direction', units='m/s')
        
        # Fictitious forces
        self.add_output('Fwn', val=jnp.zeros(nn), desc='wind gradient fictitious force, ii component', units='N')
        self.add_output('Fwe', val=jnp.zeros(nn), desc='wind gradient fictitious force, ji component', units='N')
        self.add_output('Fwd', val=jnp.zeros(nn), desc='wind gradient fictitious force, ki component', units='N')

        # Load factor, used as a constraint
        self.add_output('load_factor', val=jnp.zeros(nn), desc='load factor')
        
        # Rate to compute sum of squares of control angles (squaring for positive values and smoothness)
        # Using 'effort' here but really this is just about deflection angles.
        #self.add_output('squared_control_effort_rate', val=np.zeros(nn), desc='rate of change of integral of sum of squares of control surface deflections', units='rad/s')
        #self.add_output('squared_throttle_effort_rate', val=np.zeros(nn), desc='rate of change of integral of squared throttle')

        # Not used, but for logging.
        #self.add_output('Fdi', val=np.zeros(nn), desc='longitudinal drag force, aligned with ib', units='N')
        #self.add_output('Fdj', val=np.zeros(nn), desc='lateral drag force, aligned with jb', units='N')
        #self.add_output('Fwx', val=np.zeros(nn), desc='wind gradient fictitious force, ii component', units='N')
        #self.add_output('Fwy', val=np.zeros(nn), desc='wind gradient fictitious force, ji component', units='N')
        #self.add_output('uf', val=np.zeros(nn), desc='flow-relative x velocity', units='m/s')
        #self.add_output('vf', val=np.zeros(nn), desc='flow-relative y velocity', units='m/s')

        # Partials declared analytically
        arange = jnp.arange(nn)
        self.declare_partials(of='*', wrt='*', method='exact', rows=arange, cols=arange)
        # self.declare_partials(of='*', wrt='*', method='fd', rows=arange, cols=arange)
    
    # Dynamics go here
    @partial(jax.jit, static_argnums=(0,))
    def _compute_primal(self, n, e, d, u, v, w, phi, theta, psi, p, q, r, da, de, dr, dp):
        # TODO Not including time at the moment - setting time to be equal to 0.
        t = 0.
        state, info = BeardDynamics.compute_gradients(self.options['aircraft_model'], self.options['wind_model'].wind_single, self.options['wind_model'].jac_single, self.options['wind_model'].time_deriv_single, t, n, e, d, u, v, w, phi, theta, psi, p, q, r, da, de, dr, dp)
        # Combine state and info into one Jax array to return
        # return jnp.append(state, info)
        return tuple(jnp.append(state, info))
    
    def print_vals_debugging(self, n_dot, e_dot, d_dot, u_dot, v_dot, w_dot, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot, L, C, D, T, W, L_lat, M, N, va, alpha, beta, w_n, w_e, w_d, Fwn, Fwe, Fwd, load_factor,
                             n, e, d, u, v, w, phi, theta, psi, p, q, r, da, de, dr, dp):
        print("Rates")
        print("=====")
        print(f"n_dot: {n_dot}\n")
        print(f"e_dot: {e_dot}\n")
        print(f"d_dot: {d_dot}\n")
        print(f"u_dot: {u_dot}\n")
        print(f"v_dot: {v_dot}\n")
        print(f"w_dot: {w_dot}\n")
        print(f"phi_dot: {phi_dot}\n")
        print(f"theta_dot: {theta_dot}\n")
        print(f"psi_dot: {psi_dot}\n")
        print(f"p_dot: {p_dot}\n")
        print(f"q_dot: {q_dot}\n")
        print(f"r_dot: {r_dot}\n")
        print(f"L: {L}\n")
        print(f"C: {C}\n")
        print(f"D: {D}\n")
        print(f"T: {T}\n")
        print(f"W: {W}\n")
        print(f"L_lat: {L_lat}\n")
        print(f"M: {M}\n")
        print(f"N: {N}\n")
        print(f"va: {va}\n")
        print(f"alpha: {alpha}\n")
        print(f"beta: {beta}\n")
        print(f"w_n: {w_n}\n")
        print(f"w_e: {w_e}\n")
        print(f"w_d: {w_d}\n")
        print(f"Fwn: {Fwn}\n")
        print(f"Fwe: {Fwe}\n")
        print(f"Fwd: {Fwd}\n")
        print(f"load_factor: {load_factor}\n")
        
        print("Inputs")
        print("======")
        print(f"n: {n}\n")
        print(f"e: {e}\n")
        print(f"d: {d}\n")
        print(f"u: {u}\n")
        print(f"v: {v}\n")
        print(f"w: {w}\n")
        print(f"phi: {phi}\n")
        print(f"theta: {theta}\n")
        print(f"psi: {psi}\n")
        print(f"p: {p}\n")
        print(f"q: {q}\n")
        print(f"r: {r}\n")
        print(f"da: {da}\n")
        print(f"de: {de}\n")
        print(f"dr: {dr}\n")
        print(f"dp: {dp}\n")
    
    def compute(self, inputs, outputs):

        # print("Running compute")
        rates = jnp.array(self._compute_primal_vec(*inputs.values()))
        # rates = self._compute_primal_vec(*inputs.values())

        # Check for nan values: https://stackoverflow.com/questions/6736590/fast-check-for-nan-in-numpy
        if jnp.isnan(jnp.sum(rates)):
            print("NaN values found in rates")
            self.print_vals_debugging(*rates, *inputs.values())
            raise RatesCompException("NaN values found in rates")
            # sys.exit(1)

        n_dot, e_dot, d_dot, u_dot, v_dot, w_dot, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot, L, C, D, T, W, L_lat, M, N, va, alpha, beta, w_n, w_e, w_d, Fwn, Fwe, Fwd, load_factor = rates

        outputs['n_dot'] = n_dot
        outputs['e_dot'] = e_dot
        outputs['d_dot'] = d_dot
        outputs['u_dot'] = u_dot
        outputs['v_dot'] = v_dot
        outputs['w_dot'] = w_dot
        outputs['phi_dot'] = phi_dot
        outputs['theta_dot'] = theta_dot
        outputs['psi_dot'] = psi_dot
        outputs['p_dot'] = p_dot
        outputs['q_dot'] = q_dot
        outputs['r_dot'] = r_dot
        
        # For output only
        outputs['L'] = L
        outputs['C'] = C
        outputs['D'] = D
        outputs['T'] = T
        outputs['W'] = W
        outputs['L_lat'] = L_lat
        outputs['M'] = M
        outputs['N'] = N

        outputs['va'] = va
        outputs['alpha'] = alpha
        outputs['beta'] = beta

        outputs['w_n'] = w_n
        outputs['w_e'] = w_e
        outputs['w_d'] = w_d

        outputs['Fwn'] = Fwn
        outputs['Fwe'] = Fwe
        outputs['Fwd'] = Fwd
        
        outputs['load_factor'] = load_factor

        """
        # Testing - for constraining controls and throttle
        da = inputs['da']
        de = inputs['de']
        dr = inputs['dr']
        dp = inputs['dp']

        outputs['squared_control_effort_rate'] = da**2 + de**2 + dr**2
        outputs['squared_throttle_effort_rate'] = dp**2
        """

    def compute_partials(self, inputs, partials):
        output_names = ['n_dot', 'e_dot', 'd_dot', 'u_dot', 'v_dot', 'w_dot', 'phi_dot', 'theta_dot', 'psi_dot', 'p_dot', 'q_dot', 'r_dot', 'L', 'C', 'D', 'T', 'W', 'L_lat', 'M', 'N', 'va', 'alpha', 'beta', 'w_n', 'w_e', 'w_d', 'Fwn', 'Fwe', 'Fwd', 'load_factor']
        input_names = ['n', 'e', 'd', 'u', 'v', 'w', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'da', 'de', 'dr', 'dp']

        # n, e, d, u, v, w, phi, theta, psi, p, q, r, da, de, dr, dp
        # jnp.array([n_dot, e_dot, d_dot, u_dot, v_dot, w_dot, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot])
        # jnp.array([L, C, D, T, L_lat, M, N, va, alpha, beta, w_n, w_e, w_d, Fwn, Fwe, Fwd, test])

        computed_partials = self._compute_partials_vec(*inputs.values())

        # Cycle through computed partials
        for out_ind, output_name in enumerate(output_names):
            for in_ind, input_name in enumerate(input_names):
                partials[output_name, input_name] = computed_partials[out_ind][in_ind]
        
        """
        # Testing - for constraining controls and throttle
        da = inputs['da']
        de = inputs['de']
        dr = inputs['dr']
        dp = inputs['dp']

        # Since all partial derivatives are declared (TODO This doesn't seem like the best way to do this)
        for input_name in (set(input_names) - {'da', 'de', 'dr'}):
            partials['squared_control_effort_rate', input_name] = 0
        
        for input_name in (set(input_names) - {'dp'}):
            partials['squared_throttle_effort_rate', input_name] = 0
        
        partials['squared_control_effort_rate', 'da'] = 2*da
        partials['squared_control_effort_rate', 'de'] = 2*de
        partials['squared_control_effort_rate', 'dr'] = 2*dr
        partials['squared_throttle_effort_rate', 'dp'] = 2*dp
        """