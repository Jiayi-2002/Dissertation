# FT 8/9/24

# Natural cubic spline interpolation using interpax package

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pathlib import Path
from interpax import Interpolator3D

# Add the flight directory to the Python path (when running from FlightSwordLite)
import sys, os
sys.path.append(os.getcwd())

from flight.simulator.wind_interpolators.base_wind_interpolator import BaseWindInterpolator


# Uses interpax Jax interpolation package
class NaturalCubicSplineInterpolator(BaseWindInterpolator):
    type_name = "cubic_spline"

    def load(self, load_path):
        # Unpickle additional object data
        self.wn_interp = self.pickle_load(load_path / 'wn_interp.pkl')
        self.we_interp = self.pickle_load(load_path / 'we_interp.pkl')
        self.wd_interp = self.pickle_load(load_path / 'wd_interp.pkl')

    def save(self, pickle_dir):
        save_dir = super().save(pickle_dir)

        # Pickle additional object data
        self.pickle_save(self.wn_interp, save_dir / 'wn_interp.pkl')
        self.pickle_save(self.we_interp, save_dir / 'we_interp.pkl')
        self.pickle_save(self.wd_interp, save_dir / 'wd_interp.pkl')

    # TODO Surely the coordinates need to have been generated on a grid.
    def fit(self, coords, wn_vals, we_vals, wd_vals):
        # Order the points by x first, then y, then z.
        ordering_inds = jnp.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
        ordered_coords = coords[ordering_inds]
        # The ordering in the 'vals' arrays has to be as if they were generated in Numpy meshgrid 'ij'
        # order - e.g. axis 0 corresponds to the n coordinates and is the outermost iterated
        # over. E.g. subarray zero corresponds to all of the points with the lowest n value. 
        ordered_wn_vals = wn_vals[ordering_inds]
        ordered_we_vals = we_vals[ordering_inds]
        ordered_wd_vals = wd_vals[ordering_inds]

        n_coords = jnp.unique(ordered_coords[:, 0])
        e_coords = jnp.unique(ordered_coords[:, 1])
        d_coords = jnp.unique(ordered_coords[:, 2])
        reshape_shape = (n_coords.size, e_coords.size, d_coords.size)
        wns = ordered_wn_vals.reshape(*reshape_shape)
        wes = ordered_we_vals.reshape(*reshape_shape)
        wds = ordered_wd_vals.reshape(*reshape_shape)

        points = (n_coords, e_coords, d_coords)
        self.wn_interp = Interpolator3D(*points, wns, method="cubic2")
        self.we_interp = Interpolator3D(*points, wes, method="cubic2")
        self.wd_interp = Interpolator3D(*points, wds, method="cubic2")
    
    # Returns jnp.array([wn, we, wd])
    def interpolate_ned_single(self, coord):
        # coord_arr = jnp.array([coord])
        wn = self.wn_interp(*coord)
        we = self.we_interp(*coord)
        wd = self.wd_interp(*coord)

        return jnp.array([wn, we, wd])

