# FT 30/8/24

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np
from functools import partial
from enum import IntEnum
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import copy

# Add the flight directory to the Python path (when running from FlightSwordLite)
import sys, os
sys.path.append(os.getcwd())

from flight.simulator import config


# Frame transformation matrix
# TODO Check that this description is correct. See Beard, chapter 2.
# Expresses a vector given in the body frame in the inertial frame, where the inertial frame is arrived at
# by three right-handed rotations, first by psi, then theta, then phi (about the corresponding axes). Alternatively,
# this matrix can be used within the fixed perspective of the inertial frame to perform a left-handed rotation(s) of a vector to the direction
# of the body frame. E.g. (1, 0, 0) = i_i (i in the inertial frame) is transformed by this matrix into i_b
# (i in the body frame). This is useful for orienting the aircraft in the visualiser.
#@jnp.vectorize(signature='(),(),()->(3,3)')
@partial(jnp.vectorize, signature='(),(),()->(3,3)')
@jax.jit
def calc_body_to_inertial(phi, theta, psi):
    return jnp.array([
        [jnp.cos(theta)*jnp.cos(psi),   jnp.sin(phi)*jnp.sin(theta)*jnp.cos(psi) - jnp.cos(phi)*jnp.sin(psi),   jnp.cos(phi)*jnp.sin(theta)*jnp.cos(psi) + jnp.sin(phi)*jnp.sin(psi)],
        [jnp.cos(theta)*jnp.sin(psi),   jnp.sin(phi)*jnp.sin(theta)*jnp.sin(psi) + jnp.cos(phi)*jnp.cos(psi),   jnp.cos(phi)*jnp.sin(theta)*jnp.sin(psi) - jnp.sin(phi)*jnp.cos(psi)],
        [-jnp.sin(theta),               jnp.sin(phi)*jnp.cos(theta),                                            jnp.cos(phi)*jnp.cos(theta)]
        ])

# Frame of reference transformation matrices
# Frame transformation matrix
# @jnp.vectorize(signature='(),()->(3,3)')
@partial(jnp.vectorize, signature='(),()->(3,3)')
@jax.jit
def calc_wind_to_body(alpha, beta):
    return jnp.array([
        [jnp.cos(alpha)*jnp.cos(beta),      -jnp.cos(alpha)*jnp.sin(beta),      -jnp.sin(alpha)],
        [jnp.sin(beta),                     jnp.cos(beta),                      0],
        [jnp.sin(alpha)*jnp.cos(beta),      -jnp.sin(alpha)*jnp.sin(beta),      jnp.cos(alpha)]
        ])

@jnp.vectorize
def _ned_to_xyz_vectorised(n, e, d):
    # return n, -e, -d
    return e, n, -d

def ned_to_xyz(n, e, d):
    # Call vectorised method
    transformed_coords = _ned_to_xyz_vectorised(n, e, d)
    return jnp.array(transformed_coords)

def xyz_to_ned(x, y, z):
    # In the current setup, can just reapply the ned_to_xyz transformation to invert.
    return ned_to_xyz(x, y, z)


# Previously (in the original simulator) the state was a subclass of numpy.ndarray. Because of uncertainty around
# how Jax jitting is perfored, I didn't want to subclass a Jax array. This StateDecoder
# class provides a bridge, allowing the state values to be addressed by their names.
@register_pytree_node_class
class StateDecoder:
    def __init__(self, init_beard_state=jnp.zeros(12), va=jnp.nan, alpha=jnp.nan, beta=jnp.nan):
        self.update(init_beard_state, va, alpha, beta)
    
    def update(self, beard_state, va=jnp.nan, alpha=jnp.nan, beta=jnp.nan):
        self.state = jnp.append(beard_state, jnp.array([va, alpha, beta]))
    
    # To make StateDecoder a PyTree, so that it works with Jax.
    def tree_flatten(self):
        children = (self.state,)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        state = children[0]
        beard_state = state[:12]
        va, alpha, beta = state[12:]
        return cls(beard_state, va, alpha, beta) # , **aux_data)
    
    @property
    def n(self):
        return self.state[0]
    
    @property
    def e(self):
        return self.state[1]
    
    @property
    def d(self):
        return self.state[2]
    
    @property
    def u(self):
        return self.state[3]
    
    @property
    def v(self):
        return self.state[4]
    
    @property
    def w(self):
        return self.state[5]
    
    @property
    def phi(self):
        return self.state[6]
    
    @property
    def theta(self):
        return self.state[7]
    
    @property
    def psi(self):
        return self.state[8]
    
    # Roll rate
    @property
    def p(self):
        return self.state[9]
    
    # Pitch rate
    @property
    def q(self):
        return self.state[10]
    
    # Yaw rate
    @property
    def r(self):
        return self.state[11]
    
    @property
    def va(self):
        # return self._va
        return self.state[12]
    
    @property
    def alpha(self):
        # return self._alpha
        return self.state[13]
    
    @property
    def beta(self):
        # return self._beta
        return self.state[14]


class StateEncoder:
    @staticmethod
    def encode(n=0, e=0, d=0, u=0, v=0, w=0, phi=0, theta=0, psi=0, p=0, q=0, r=0):
        return jnp.array([n, e, d, u, v, w, phi, theta, psi, p, q, r])


# =========


class CyclicBuffer:
    def __init__(self, size):
        self._size = size
        self._buffer = jnp.full(size, jnp.nan)
        # Index of the least recently filled position
        self._last = 0
        # Index of the most recently added value
        self._index = None
        self._num_elements = 0
        self._filled = False
    
    def get_oldest(self):
        if self._filled:
            return self._buffer[self._last]
        else:
            # if not config.suppress_warnings:
            print("Warning: buffer not yet filled (fill level {}/{}). Returning zero.".format(self._index + 1, self._size))
            return 0
    
    def diff(self):
        if self._filled:
            return self._buffer[self._index] - self._buffer[self._last]
        else:
            # TODO HACK! Really want this to be None.
            # if not config.suppress_warnings:
            print("Warning: buffer not yet filled (fill level {}/{})".format(self._index + 1, self._size))
            return 0
            # return None
    
    #def normalised_ave(self):
    #    if self._num_elements == 0:
    #        # TODO Change this to a more appropriate type of error.
    #        raise Exception("No values have yet been added to the buffer")
    #    
    #    return self._buffer[self._index] + self._buffer[self._last] / self._num_elements
    
    def push(self, val):
        if jnp.isnan(val):
            # TODO Change this to a more appropriate type of error.
            raise Exception("NaN values not accepted as input!")
        
        self._index = self._last
        self._last = (self._index + 1) % self._size
        self._buffer = self._buffer.at[self._index].set(val)

        if not self._filled:
            self._num_elements += 1
            # Check whether there are any NaN values remaining:
            if not jnp.any(jnp.isnan(self._buffer)):
                self._filled = True

    # For debugging
    def __repr__(self):
        return "Buffer: {}; last: {}; index: {}; filled: {}".format(self._buffer, self._last, self._index, self._filled)


# =========


@register_pytree_node_class
class DataLogger:

    # Written in a strange way so that it's compatable with Jax
    def __init__(self, dt, num_steps, log_arr, log_ptr, from_opt, opt_success):
        self.dt = dt
        self.num_steps = num_steps
        self.log_arr = log_arr
        self.log_ptr = log_ptr
        self.from_opt = from_opt
        self.opt_success = opt_success
        
        # Update this whenever any API changes are made, so that old
        # DataLoggers can be detected and converted. 
        self.datalogger_version = 2
    
    # Might need num_steps in the future if we choose to preallocate
    @classmethod
    def create(cls, dt, num_steps, from_optimisation, optimisation_successful=False):
        # TODO Could we face memory issues if this gets too large? 
        # return cls(dt, num_steps, jnp.empty(shape=(39, num_steps)), 0, from_optimisation, optimisation_successful)
        return cls(dt, num_steps, jnp.empty(shape=(35, num_steps)), 0, from_optimisation, optimisation_successful)
    
    # The length of the DataLogger is set on creation. This is called to truncate the DataLogger (log array and number of steps)
    # if a simulation has to be stopped early.
    # This just uses the internal log_ptr, which keeps a record of the number of times values have been logged.
    def end_logging(self):
        self.num_steps = self.log_ptr
        self.log_arr = self.log_arr[:, 0:self.log_ptr]
    
    # So that this class works with Jax
    def tree_flatten(self):
        children = (self.log_arr, self.log_ptr) # dynamic values
        aux_data = {'dt': self.dt,
                    'num_steps': self.num_steps,
                    'from_opt': self.from_opt,
                    'opt_success': self.opt_success} # static values
        return (children, aux_data)
    
    # So that this class works with Jax
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        log_arr, log_ptr = children
        return cls(aux_data['dt'], aux_data['num_steps'], log_arr, log_ptr, aux_data['from_opt'], aux_data['opt_success'])
    
    # Start and end coordinates are inclusive
    # Creates and returns a copy of this DataLogger between the given
    # start and end indices. 
    def create_sub_dl(self, start_ind, end_ind_inclusive):
        sub_dl = copy.deepcopy(self)

        sub_dl.log_arr = sub_dl.log_arr[:, start_ind:end_ind_inclusive+1]
        sub_dl.num_steps = sub_dl.log_arr.shape[1]
        sub_dl.log_ptr = sub_dl.num_steps

        # Unchanged:
        # o dt unchanged
        # o from_opt
        # o opt_success
        # o datalogger_version

        return sub_dl
    
    @property
    def times(self):
        return self.log_arr[0]
    
    @property
    def ns(self):
        return self.log_arr[1]
    
    @ns.setter
    def ns(self, val):
        self.log_arr[1, :] = val
    
    @property
    def es(self):
        return self.log_arr[2]
    
    @es.setter
    def es(self, val):
        self.log_arr[2, :] = val
    
    @property
    def ds(self):
        return self.log_arr[3]
    
    @ds.setter
    def ds(self, val):
        self.log_arr[3, :] = val
    
    @property
    def us(self):
        return self.log_arr[4]
    
    @property
    def vs(self):
        return self.log_arr[5]
    
    @property
    def ws(self):
        return self.log_arr[6]
    
    @property
    def phis(self):
        return self.log_arr[7]
    
    @property
    def thetas(self):
        return self.log_arr[8]
    
    @property
    def psis(self):
        return self.log_arr[9]
    
    @property
    def ps(self):
        return self.log_arr[10]
    
    @property
    def qs(self):
        return self.log_arr[11]
    
    @property
    def rs(self):
        return self.log_arr[12]
    
    @property
    def vas(self):
        return self.log_arr[13]
    
    @property
    def alphas(self):
        return self.log_arr[14]
    
    @property
    def betas(self):
        return self.log_arr[15]
    
    @property
    def das(self):
        return self.log_arr[16]
    
    @property
    def des(self):
        return self.log_arr[17]
    
    @property
    def drs(self):
        return self.log_arr[18]
    
    @property
    def dps(self):
        return self.log_arr[19]
    
    @property
    def Ls(self):
        return self.log_arr[20]

    @property
    def Cs(self):
        return self.log_arr[21]

    @property
    def Ds(self):
        return self.log_arr[22]

    @property
    def Ts(self):
        return self.log_arr[23]
    
    @property
    def Ws(self):
        return self.log_arr[24]
    
    @property
    def Fwns(self):
        return self.log_arr[25]
    
    @property
    def Fwes(self):
        return self.log_arr[26]
    
    @property
    def Fwds(self):
        return self.log_arr[27]

    @property
    def L_lats(self):
        return self.log_arr[28]

    @property
    def Ms(self):
        return self.log_arr[29]

    @property
    def Ns(self):
        return self.log_arr[30]

    @property
    def wns(self):
        return self.log_arr[31]

    @property
    def wes(self):
        return self.log_arr[32]

    @property
    def wds(self):
        return self.log_arr[33]
    
    @property
    def load_factors(self):
        return self.log_arr[34]
    
    # Check that the time difference is consistent
    def check_valid(self, dt):
        # Use self.dt here
        # Calculate the difference in times, check that it's unique, check that it matches the given dt
        #  -> Else throw an error
        pass

    #@jax.jit
    def log(self, time, state, va, alpha, beta, da, de, dr, dp, lift, sideforce, drag, thrust, weight, fictitious_wind_n, fictitious_wind_e, fictitious_wind_d, roll_mom, pitch_mom, yaw_mom, wind, load_factor):
        time_arr = jnp.array([time])
        airstate_arr = jnp.array([va, alpha, beta])
        control_arr = jnp.array([da, de, dr, dp])
        # Forces and moments
        fm_arr = jnp.array([lift, sideforce, drag, thrust, weight, fictitious_wind_n, fictitious_wind_e, fictitious_wind_d, roll_mom, pitch_mom, yaw_mom])
        wind_arr = jnp.array(wind)
        load_factor_arr = jnp.array([load_factor])
        # Currently these aren't obtained from the simulator - only the optimiser produces them. Just filling them with NaNs.
        # control_rates_arr = jnp.array([jnp.nan, jnp.nan, jnp.nan, jnp.nan])

        # combined_arr = jnp.array([jnp.concatenate((time_arr, state, airstate_arr, control_arr, fm_arr, wind_arr))])
        combined_arr = jnp.concatenate((time_arr, state, airstate_arr, control_arr, fm_arr, wind_arr, load_factor_arr)) # , control_rates_arr))

        self.log_arr = self.log_arr.at[:, self.log_ptr].set(combined_arr)
        # self.log_arr = jnp.append(self.log_arr, combined_arr.T, axis=1)
        # self.log_arr[:, self.log_ptr] = combined_arr
        self.log_ptr += 1
    

    # Pickles DataLogger
    def save(self, name):
        save_path = Path(config.datalogger_save_path) / f"{name}.pkl"
        self.save_to_path(save_path)
        
    
    # More flexibility over save location than save
    def save_to_path(self, save_path):
        dl_info = {
            'dt': self.dt,
            'num_steps': self.num_steps,
            'log_arr': self.log_arr,
            'log_ptr': self.log_ptr,
            'from_opt': self.from_opt,
            'opt_success': self.opt_success
        }

        print(f"Saving datalogger to {save_path}")

        with open(save_path, 'wb') as fh:
            pickle.dump(dl_info, fh)
    
    # Convenience method for loading from the config.datalogger_save_path folder - just provide the DataLogger name.
    @classmethod
    def load(cls, name):
        load_path = Path(config.datalogger_save_path) / f"{name}.pkl"
        return cls.load_from_path(load_path)
    
    @classmethod
    def load_from_path(cls, load_path, print_load_msg=True):
        if print_load_msg:
            print(f"Loading datalogger from {load_path}")

        with open(load_path, 'rb') as pickle_file:
            dl_info = pickle.load(pickle_file)
        
        # The DataLogger class was updated to include these, so some DataLoggers may not
        # have them. In this case, they're set as None.
        from_opt = dl_info.get('from_opt', None)
        opt_success = dl_info.get('opt_success', None)
        
        # Returns a DataLogger object
        return cls(dl_info['dt'], dl_info['num_steps'], dl_info['log_arr'], dl_info['log_ptr'], from_opt, opt_success)
    
    def graph_states_and_controls(self):
        fig, ax = plt.subplots(16)

        ax[0].plot(self.times, self.ns)
        ax[0].set_ylabel('n')
        
        ax[1].plot(self.times, self.es)
        ax[1].set_ylabel('e')
        
        ax[2].plot(self.times, self.ds)
        ax[2].set_ylabel('d')
        
        ax[3].plot(self.times, self.vas)
        ax[3].set_ylabel('Va')

        ax[4].plot(self.times, self.alphas)
        ax[4].set_ylabel('alpha')

        ax[5].plot(self.times, self.betas)
        ax[5].set_ylabel('beta')

        ax[6].plot(self.times, self.ps)
        ax[6].set_ylabel('p')
        
        ax[7].plot(self.times, self.qs)
        ax[7].set_ylabel('q')
        
        ax[8].plot(self.times, self.rs)
        ax[8].set_ylabel('r')
        
        ax[9].plot(self.times, self.phis)
        ax[9].set_ylabel('phi')

        ax[10].plot(self.times, self.thetas)
        ax[10].set_ylabel('theta')

        ax[11].plot(self.times, self.psis)
        ax[11].set_ylabel('psi')

        ax[12].plot(self.times, self.das)
        ax[12].set_ylabel('da')

        ax[13].plot(self.times, self.des)
        ax[13].set_ylabel('de')

        ax[14].plot(self.times, self.drs)
        ax[14].set_ylabel('dr')

        ax[15].plot(self.times, self.dps)
        ax[15].set_ylabel('dp')

        fig.suptitle("State and control values")
        plt.show(block=False)

    def graph_forces_and_moments(self):
        fig, ax = plt.subplots(7)

        ax[0].plot(self.times, self.Ls)
        ax[0].set_ylabel('Lift')
        
        ax[1].plot(self.times, self.Cs)
        ax[1].set_ylabel('Sideforce')
        
        ax[2].plot(self.times, self.Ds)
        ax[2].set_ylabel('Drag')
        
        ax[3].plot(self.times, self.Ts)
        ax[3].set_ylabel('Thrust')

        ax[4].plot(self.times, self.L_lats)
        ax[4].set_ylabel('Roll moment')

        ax[5].plot(self.times, self.Ms)
        ax[5].set_ylabel('Pitch moment')

        ax[6].plot(self.times, self.Ns)
        ax[6].set_ylabel('Yaw moment')
        
        fig.suptitle("Forces and moments")
        plt.show(block=False)

    # Testing
    def graph_wind_gradients(self):
        fig, ax = plt.subplots()

        ax.plot(self.times, self.dWxx)
        ax.plot(self.times, self.dWxy)
        ax.plot(self.times, self.dWxz)
        ax.plot(self.times, self.dWyx)
        ax.plot(self.times, self.dWyy)
        ax.plot(self.times, self.dWyz)
        ax.plot(self.times, self.dWzx)
        ax.plot(self.times, self.dWzy)
        ax.plot(self.times, self.dWzz)
        ax.plot(self.times, self.dWxt)
        ax.plot(self.times, self.dWyt)
        ax.plot(self.times, self.dWzt)

        fig.suptitle("Wind gradients")
        plt.show(block=False)

# All passed, 00:28 15/11/23
def log_lock_test():

    # Test 1: Normal
    def normal():
        data_logger.log_states_and_controls(0, sd, None, None, None, None)
        data_logger.log_forces_and_moments(0, None, None, None, None, None, None, None)
        data_logger.log_wind_gradients(0, None, None)
        data_logger.log_states_and_controls(1, sd, None, None, None, None)
        data_logger.log_forces_and_moments(1, None, None, None, None, None, None, None)
        data_logger.log_wind_gradients(1, None, None)

    # Test 2: Wind gradients missing
    def wind_grads_missing():
        data_logger.log_states_and_controls(0, sd, None, None, None, None)
        data_logger.log_forces_and_moments(0, None, None, None, None, None, None, None)
        data_logger.log_states_and_controls(1, sd, None, None, None, None)
        data_logger.log_forces_and_moments(1, None, None, None, None, None, None, None)
        data_logger.log_wind_gradients(1, None, None)

    # Test 3: Log with different time stamps
    def different_timestamps():
        data_logger.log_states_and_controls(0, sd, None, None, None, None)
        data_logger.log_forces_and_moments(0.5, None, None, None, None, None, None, None)
        data_logger.log_wind_gradients(0, None, None)

    state = jnp.zeros(12)
    sd = StateDecoder(state)
    data_logger = DataLogger()
    # normal()
    # wind_grads_missing()
    different_timestamps()

if __name__ == "__main__":
    log_lock_test()
