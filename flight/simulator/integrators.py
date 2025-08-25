# FT 3/8/23 (modified 30/8/24 for this simulator)

import abc
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial

# Add the flight directory to the Python path (when running from FlightSwordLite)
import sys, os
sys.path.append(os.getcwd())

# Integrator base class
class IntegratorBase(metaclass=abc.ABCMeta):
    # Return result (state) after one step of integration
    # gradients(state, t, controls) is a function which returns the gradients at state state and time t
    #  e.g. dx/dt = gradients(x, t (, controls))
    # state is the state at the start of the integration
    @staticmethod
    @abc.abstractmethod
    def integrate(gradients, state, t, controls, dt):
        pass

# Euler integrator
class Euler(IntegratorBase):
    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def integrate(gradients, state, t, controls, dt):
        grads, log_info = gradients(state, t, controls)
        return state + dt*grads, log_info

# Custom implementation of fourth-order Runge-Kutta integration,
# using RK equations. Time step is constant.
# Code from: http://faculty.washington.edu/sbrunton/me564/python/L18_simulateLORENZ.ipynb
class RK4(IntegratorBase):
    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def integrate(gradients, state, t, controls, dt):
        # Copying may not be required (see https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.copy.html) and may slow down the code, but doing it for safety.
        # Only capture for logging the forces, moments and wind at the original state point.
        f1, log_info = gradients(jnp.copy(state), t, controls)     # Copy used so that state is not modified
        f2, _ = gradients(jnp.copy(state) + (dt / 2) * f1, t + dt / 2, controls)
        f3, _ = gradients(jnp.copy(state) + (dt / 2) * f2, t + dt / 2, controls)
        f4, _ = gradients(jnp.copy(state) + dt * f3, t + dt, controls)
        offset = (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)

        return state + offset, log_info