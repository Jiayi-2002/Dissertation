# FT 2/9/24

# Base class for wind interpolators.
# Required API:
# - load() a wind interpolator - provide folder and name
# - save() a wind interpolator - provide directory
# - get ranges of n, e and d validity
# - interpolate_ned_single() - provide a single coord
# - inspect_fit() - provide ground-truth coords and their values, plus viewing and saving parameters.
# - fit() abstract method for generating interpolation weights

import abc
import pickle
from pathlib import Path
import jax
jax.config.update("jax_enable_x64", True)
import importlib

# Add the flight directory to the Python path (when running from FlightSwordLite)
import sys, os
sys.path.append(os.getcwd())

from flight.simulator.utils import ned_to_xyz
from flight.simulator.wind_interpolators.sweeper import Sweeper


class BaseWindInterpolator(metaclass=abc.ABCMeta):
    def __init__(self, name, valid_n_range, valid_e_range, valid_d_range):
        # Name string
        self.name = name

        self.valid_n_range = valid_n_range
        self.valid_e_range = valid_e_range
        self.valid_d_range = valid_d_range

        self.interpolate_ned = jax.vmap(self.interpolate_ned_single, in_axes=[0])
    
    # 'interpolator_name' is the name of the folder containing the pickle files, e.g. 'front_on_10'.
    # Override (and call this base method) to load additional interpolator-specific items.
    @classmethod
    def load(cls, pickle_dir, interpolator_name):
        load_path = pickle_dir / interpolator_name
        print(f"Loading wind field interpolator from {load_path}")

        # Read dispatch file, to call load() on the appropriate subclass.
        dispatch_info = cls.pickle_load(load_path / 'dispatch_info.pkl')

        valid_n_range = dispatch_info['valid_n_range']
        valid_e_range = dispatch_info['valid_e_range']
        valid_d_range = dispatch_info['valid_d_range']

        # Get (load) subclass of interpolator
        interp_module_str = dispatch_info['interp_module']
        interp_class_str = dispatch_info['interp_class']
        interp_cls = getattr(importlib.import_module(interp_module_str), interp_class_str)

        # Create interpolator object
        loaded_interp = interp_cls(interpolator_name, valid_n_range, valid_e_range, valid_d_range)

        # Call load on subclass instance
        loaded_interp.load(load_path)

        return loaded_interp
    
    # Pickles the object
    def save(self, pickle_dir):
        # Make new directory for files
        save_dir = pickle_dir / self.name
        save_dir.mkdir()
        print(f"Saving wind field parser data to {save_dir}")

        # Create dispatch file, used on loading to generate the right type
        # of interpolator.
        dispatch_info = {
            'interp_module': self.__class__.__module__,     # For creating the right type of interpolator
            'interp_class': self.__class__.__name__,        # For creating the right type of interpolator
            'valid_n_range': self.valid_n_range,
            'valid_e_range': self.valid_e_range,
            'valid_d_range': self.valid_d_range
        }

        self.pickle_save(dispatch_info, save_dir / 'dispatch_info.pkl')

        return save_dir
    
    def get_valid_n_range(self):
        return self.valid_n_range
    
    def get_valid_e_range(self):
        return self.valid_e_range
    
    def get_valid_d_range(self):
        return self.valid_d_range

    @staticmethod
    def pickle_save(item, path):
        with open(path, 'wb') as fh:
            pickle.dump(item, fh)
    
    @staticmethod
    def pickle_load(path):
        with open(path, 'rb') as pickle_file:
            return pickle.load(pickle_file)
    
    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        pass
     
    # Returns jnp.array([wn, we, wd])
    @abc.abstractmethod
    def interpolate_ned_single(self, coord):
        pass
    
    # For checking fit
    def inspect_fit(self, plot_coords, plot_vals, folderpath='.', types=['x', 'y', 'z']):
        # Interpolate values
        interp_vals_ned = self.interpolate_ned(plot_coords)

        # NED should be mapped into XYZ for plotting.
        plot_coords = ned_to_xyz(*plot_coords.T).T
        plot_vals = ned_to_xyz(*plot_vals.T).T
        x_interp_vals, y_interp_vals, z_interp_vals = ned_to_xyz(*interp_vals_ned.T)

        if 'x' in types:
            x_sw = Sweeper(plot_coords, plot_vals[:, 0], x_interp_vals, 'x', {}, None, False, True, folderpath) # self.n_centres
        if 'y' in types:
            y_sw = Sweeper(plot_coords, plot_vals[:, 1], y_interp_vals, 'y', {}, None, False, True, folderpath) # self.e_centres
        if 'z' in types:
            z_sw = Sweeper(plot_coords, plot_vals[:, 2], z_interp_vals, 'z', {}, None, False, True, folderpath) # self.d_centres

        # plt.show(block=True)