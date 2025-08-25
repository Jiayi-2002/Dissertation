# FT 30/8/24

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import abc
from functools import partial

# Add the flight directory to the Python path (when running from FlightSwordLite)
import sys, os
sys.path.append(os.getcwd())

from flight.simulator.utils import xyz_to_ned, ned_to_xyz
from flight.simulator.config import cfd_data_folder_path
from flight.simulator.wind_interpolators.base_wind_interpolator import BaseWindInterpolator

# Defining wind
# Wind base class - should not need modifying
class WindModel(metaclass=abc.ABCMeta):
    def __init__(self):
        # These can be overridden by subclasses.
        # They give the range of validity of the wind field,
        # which is used in World to ensure that the aircraft stays
        # within this region.
        # For the wind fields which are calculated analytically, this
        # defaults to an infinite region (wind model valid everywhere).
        self.valid_n_range = (-jnp.inf, jnp.inf)
        self.valid_e_range = (-jnp.inf, jnp.inf)
        self.valid_d_range = (-jnp.inf, jnp.inf)
    
    def wind(self, ns, es, ds, ts=None):
        # For convenience of working with stationary wind fields, if the time is
        # not provided, it is assumed to be zero.
        if ts is None:
            ts = jnp.zeros(len(ns))
        
        return jax.jit(jax.vmap(self.wind_single))(ns, es, ds, ts)
    
    # Overridden by subclasses
    @abc.abstractmethod
    def wind_single(self, n, e, d, t):
        pass
    
    # Returns the wind Jacobian evaluated at the point (n, e, d).
    def jac_single(self, n, e, d, t):
        return jnp.asarray(jax.jacobian(self.wind_single, argnums=[0, 1, 2])(n, e, d, t)).T
    
    def time_deriv_single(self, n, e, d, t): 
        return jax.jacobian(self.wind_single, argnums=[3])(n, e, d, t)[0]


# ====== Wind models ======


class WindNone(WindModel):
    @partial(jax.jit, static_argnums=(0,))
    def wind_single(self, n, e, d, t):
        return jnp.array([0., 0., 0.])


class WindConstant(WindModel):
    def __init__(self, wn, we, wd):
        super().__init__()
        self.wn = wn
        self.we = we
        self.wd = wd

    @partial(jax.jit, static_argnums=(0,))
    def wind_single(self, n, e, d, t):
        return jnp.array([self.wn, self.we, self.wd])


class WindLinear(WindModel):
    def __init__(self, k=1):
        super().__init__()
        self.k = k

    def wind_single(self, n, e, d, t):
        return jnp.array([self.k*-d, 0., 0.])


# Actual wind field from CFD
class WindQuic(WindModel):
    # windfield_name: e.g. 'front_on_10' - name of the folder containing the wind field pickle files
    # (or subclass) object.
    # TODO Check that kwargs works like this
    def __init__(self, windfield_name, wind_field_folder_path, **kwargs):
        super().__init__(**kwargs)

        self.interp = BaseWindInterpolator.load(wind_field_folder_path, windfield_name)  # Unpickle
        
        self.valid_n_range = self.interp.get_valid_n_range()
        self.valid_e_range = self.interp.get_valid_e_range()
        self.valid_d_range = self.interp.get_valid_d_range()

    def wind_single(self, n, e, d, t):
        return self.interp.interpolate_ned(jnp.array([[n, e, d]]))[0]


# ====== These have been left here from a previous version of the software; I'm not sure if they still work ======

"""
# Temp - not a zero wind, has some gradient.
class WindTiny(WindModel):
    @partial(jax.jit, static_argnums=(0,))
    def wind_single(self, n, e, d, t):
        return jnp.array([0.001*jnp.exp(-n**2), 0.001*jnp.exp(-e**2), 0.001*jnp.exp(-d**2)])
    
class WindUpdraft(WindModel):
    def wind_single(self, n, e, d, t):
        return jnp.array([0., 0., -1.])

class WindTanh(WindModel):
    def wind_single(self, n, e, d, t):
        a = 10 # 1.5
        k = 0.5
        n_0 = 0
        wd = -(a*jnp.tanh(k*(n - n_0)) + a)
        return jnp.array([0., 0., wd]) # 0.01])

class WindLinear(WindModel):
    def __init__(self, k=1):
        super().__init__()
        self.k = k

    def wind_single(self, n, e, d, t):
        return jnp.array([self.k*-d, 0., 0.])

class WindLog(WindModel):
    def __init__(self, vref, href, h0):
        super().__init__()
        self.vref = vref
        self.href = href
        self.h0 = h0

    def wind_single(self, n, e, d, t):
        return jnp.array([self.vref*(jnp.log(-d/self.h0)/jnp.log(self.href/self.h0)), 0., 0.])
        # return jnp.array([1., 0., 0.])

class WindGaussianBump(WindModel):
    # 'ne' = 'north-east' - shape (2,) array of n and e coordinates of Gaussian bump centre. 
    def __init__(self, amp=4, sigma=10, centre_ne=jnp.array([50, 0])):
        super().__init__()
        
        self.a = amp
        self.s = sigma
        self.c = centre_ne

    @partial(jax.jit, static_argnums=(0,))
    def wind_single(self, n, e, d, t):
        # Radially symmetric
        wd = -(self.a*jnp.exp(-0.5*jnp.square(jnp.linalg.norm(jnp.array([n, e]) - self.c)/self.s)))
        return jnp.array([0., 0., wd]) # 0.01])

# Can the optimiser fly along the ridge?
class WindGaussianRidge(WindModel):
    def wind_single(self, n, e, d, t):
        a = 5 # 30
        s = 7 # 0.1
        e_0 = 15 # 10
        wd = -(a*jnp.exp(-0.5*jnp.square((e - e_0)/s)))
        return jnp.array([0., 0., wd])

class WindGaussianUpdraftSum(WindModel):
    def __init__(self, amps, sigmas, ne_centres):
        super().__init__()

        num_gaussians = len(amps)   # TODO This isn't the most robust way of doing this.
        self.amps = amps
        self.sigmas = sigmas
        self.ne_centres = ne_centres

    @partial(jax.jit, static_argnums=(0,))
    def wind_single(self, n, e, d, t):
        wd = 0
        for a, s, c in zip(self.amps, self.sigmas, self.ne_centres):
            wd += -(a*jnp.exp(-0.5*jnp.square(jnp.linalg.norm(jnp.array([n, e]) - c)/s)))
        return jnp.array([0., 0., wd]) # 0.01])

# Gaussian sum in n, e and d winds
class WindGaussianSum(WindModel):
    # Centre coords arrays should have shape (n, 3) - e.g. an n-long list of [n, e, d] coordinates
    # providing the centres of the Gaussians.
    # Amps arrays should have length (n).
    # Sigma arrays should have length (n, 3) - e.g. separate n, e and d sigmas should be
    # provided for each point.
    def __init__(self,
                 wn_centre_coords, wn_amps, wn_sigmas,
                 we_centre_coords, we_amps, we_sigmas,
                 wd_centre_coords, wd_amps, wd_sigmas,
                 ):
        super().__init__()

        wn_num_gaussians = wn_centre_coords.shape[0]   # TODO This isn't the most robust way of doing this.
        we_num_gaussians = we_centre_coords.shape[0]
        wd_num_gaussians = wd_centre_coords.shape[0]

        self.wn_centre_coords = wn_centre_coords
        self.wn_amps = wn_amps
        self.wn_sigmas = wn_sigmas
        
        self.we_centre_coords = we_centre_coords
        self.we_amps = we_amps
        self.we_sigmas = we_sigmas

        self.wd_centre_coords = wd_centre_coords
        self.wd_amps = wd_amps
        self.wd_sigmas = wd_sigmas
    
    @staticmethod
    @jax.jit
    def _comp_gaussian_sum(n, e, d, centre_coords, amps, sigmas):
        val = 0
        for c, a, s in zip(centre_coords, amps, sigmas):
            x = jnp.array([n, e, d])
            expon = jnp.square((x - c) / s).sum()
            val += a*jnp.exp(-0.5*expon)
        return val

    @partial(jax.jit, static_argnums=(0,))
    def wind_single(self, n, e, d, t):
        wn = self._comp_gaussian_sum(n, e, d, self.wn_centre_coords, self.wn_amps, self.wn_sigmas)
        we = self._comp_gaussian_sum(n, e, d, self.we_centre_coords, self.we_amps, self.we_sigmas)
        wd = self._comp_gaussian_sum(n, e, d, self.wd_centre_coords, self.wd_amps, self.wd_sigmas)
        return jnp.array([wn, we, wd]) # 0.01])

# Can the optimiser turn to extract energy?
class WindSidewaysBump(WindModel):
    def wind_single(self, n, e, d, t):
        a = 2
        s = 1
        n_0 = 100
        we = a*jnp.exp(-0.5*jnp.square((n - n_0)/s))
        return jnp.array([0., we, 0.])

# Can the optimiser turn to extract energy?
class WindSidewaysTanh(WindModel):
    def wind_single(self, n, e, d, t):
        a = 2
        k = 3
        n_0 = 100
        we = (a/2)*jnp.tanh(k*(n - n_0)) + (a/2)
        return jnp.array([0., we, 0.])

class WindSidewaysSine(WindModel):
    def __init__(self, a, T):
        super().__init__()

        self.a = a  # Amplitude
        self.T = T  # Period

    def wind_single(self, n, e, d, t):
        return jnp.array([0., self.a*jnp.sin((2*jnp.pi*n)/self.T), 0.])

class WindHorizontalShear(WindModel):
    # s is the steepness of the gradient
    def __init__(self, ref_d, ref_wn, s):
        super().__init__()

        self.ref_d = ref_d
        self.ref_wn = ref_wn
        self.s = s
    
    def wind_single(self, n, e, d, t):
        wn = (self.ref_wn / (1 + jnp.exp(self.s*(d - self.ref_d)))) - (self.ref_wn / 2)
        # return jnp.array([wn, 0., 0.])
        # Temp
        return jnp.array([0., wn, 0.])

class WindTemporalGust(WindModel):
    def __init__(self, amp, sigma, t_centre):
        super().__init__()

        self.amp = amp
        self.sigma = sigma
        self.t_centre = t_centre

    # Sideways temporal Gaussian
    @partial(jax.jit, static_argnums=(0,))
    def wind_single(self, n, e, d, t):
        exp = -0.5*jnp.square((t - self.t_centre)/self.sigma)
        we = self.amp*jnp.exp(exp)
        return jnp.array([0., we, 0.])

class WindFourierStationary(WindModel):    
    def __init__(self, amplitudes, scaler=20):
        super().__init__()

        # Prevent wind from changing too quickly
        self.scaler = scaler

        self.num_components = len(amplitudes)
        # Spatial frequencies
        self.spatial_freqs = np.arange(self.num_components) + 1
        # Amplitudes (enveloped)
        self.amplitudes = amplitudes
        # Phase angles
        # TODO This should really be bounded
        self.phase_angles = np.random.random((3, self.num_components))

    def wind_single(self, n, e, d, t):
        wn = sum([a*jnp.sin(sf*(n/self.scaler) + p) for sf, a, p in zip(self.spatial_freqs, self.amplitudes, self.phase_angles[0])])
        we = sum([a*jnp.sin(sf*(e/self.scaler) + p) for sf, a, p in zip(self.spatial_freqs, self.amplitudes, self.phase_angles[1])])
        wd = sum([a*jnp.sin(sf*(d/self.scaler) + p) for sf, a, p in zip(self.spatial_freqs, self.amplitudes, self.phase_angles[2])])

        return jnp.array([wn, we, wd])
"""


"""
# Testing
if __name__ == "__main__":
    wf = WindModel.create('cfd', filename='front_on_10')
    print(wf.parser.interpolate_ned(np.array([[1, 2, 3, 4]])))
    print(wf.parser.interpolate_ned(np.array([[1, 2, 3, 4], [5, 6, 7, 8]])))

"""