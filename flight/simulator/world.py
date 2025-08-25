# FT 28/9/23

# This class is the interface to the environment. It provides access to information about the wind and 
# local topography (areas in which it's possible to fly - i.e. where there are no buildings, etc).

import jax
jax.config.update("jax_enable_x64", True)
from functools import partial
import numpy as np
import copy
from itertools import product, combinations
from enum import IntEnum

# Add the flight directory to the Python path (when running from FlightSwordLite)
import sys, os
sys.path.append(os.getcwd())

from flight.simulator import config
from flight.simulator.view import View
from flight.simulator.utils import ned_to_xyz
from flight.simulator.custom_exceptions import OutOfWorldException


class PositionCheckType(IntEnum):
    BUILDING = 0,
    ZERO_WIND = 1

# Represents a cuboidal building. More complex shapes can be built up using multiple Building objects.
class Building:
    # i_range = (i_min, i_max) 2-tuple
    # NOTE Be careful with these limits, particularly for z. E.g. a building from ground-level up to 20m should have d_range=(-20, 0)!
    # Min and max building extremes are inclusive
    # NED for north, east, down
    def __init__(self, n_range, e_range, d_range):
        self.n_range = n_range
        self.e_range = e_range
        self.d_range = d_range
        self.max_ned = np.array([n_range[1], e_range[1], d_range[1]])
        self.min_ned = np.array([n_range[0], e_range[0], d_range[0]])

    # Plot the building on the given 3d Matplotlib axis ax
    # Plotting is split at the split locations, so that things render in front of and behind each other correctly.
    def add_to_axis(self, ax, lw=1, ls='--', line_col='darkgrey', fill=False, n_splits=[], e_splits=[], d_splits=[], alpha=None, zorder=None):
    # def add_to_axis(self, ax, lw=1, ls='--', line_col='g', fill=False, n_splits=[], e_splits=[], d_splits=[]):
        # TODO This isn't using the coordinate transformation system very robustly
        # n splits -> y splits
        # e splits -> x splits
        # d splits -> -z splits
        y_splits = ned_to_xyz(n_splits, np.zeros(len(n_splits)), np.zeros(len(n_splits)))[1]
        x_splits = ned_to_xyz(np.zeros(len(e_splits)), e_splits, np.zeros(len(e_splits)))[0]
        z_splits = ned_to_xyz(np.zeros(len(d_splits)), np.zeros(len(d_splits)), d_splits)[2]

        # Draw building planes. Find all corner combinations, and plot the ones which share a coordinate.
        # TODO Might not be the best way to do this
        # for a, b, c, d in combinations(np.array(list(product(*View.ned_to_xyz(self.n_range, self.e_range, self.d_range)))), 4):
        #    X = np.array([[a[0], b[0]], [c[0], d[0]]])
        #    Y = np.array([[a[1], b[1]], [c[1], d[1]]])
        #    Z = np.array([[a[2], b[2]], [c[2], d[2]]])
        if fill:
            for four_corners in combinations(np.array(list(product(*ned_to_xyz(self.n_range, self.e_range, self.d_range)))), 4):
                four_corners_t = np.array(four_corners).transpose()
                X = four_corners_t[0].reshape(2, 2)
                Y = four_corners_t[1].reshape(2, 2)
                Z = four_corners_t[2].reshape(2, 2)
                # If all of these share one of the same coordinate, it's a plane to be plotted
                if (np.unique(X).size == 1) or (np.unique(Y).size == 1) or (np.unique(Z).size == 1):
                    # Plot the side
                    ax.plot_surface(X, Y, Z, color='lightgrey', alpha=config.building_alpha if alpha is None else alpha, zorder=zorder)
                    # ax.plot_surface(X, Y, Z, color='lightgreen', alpha=config.building_alpha)
        
        # Draw building boundary
        # Based on code given in HYRY's answer to https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector
        for s, e in combinations(np.array(list(product(*ned_to_xyz(self.n_range, self.e_range, self.d_range)))), 2):
            if np.count_nonzero(s != e) == 1:       # If the coordinates differ in only one axis
                # ax.plot3D(*zip(s, e), c='g', ls='--')
                # How to do this split? We know that they only differ in one axis. We can create the split point in this axis and iterate.
                # 1. Get the nonzero axis
                plot_axis = np.where((s - e) != 0)[0].item()   # TODO Is .item() depricated?
                # 2. Get range
                if plot_axis == 0:
                    plot_range = x_splits
                elif plot_axis == 1:
                    plot_range = y_splits
                else:
                    plot_range = z_splits
                
                plot_range.sort()

                # Find the indices which bound the array by the original building extremes.
                lower = np.min((s[plot_axis], e[plot_axis]))
                upper = np.max((s[plot_axis], e[plot_axis]))
                lower_insertion_ind = np.searchsorted(plot_range, lower)
                upper_insertion_ind = np.searchsorted(plot_range, upper)
                bounded_plot_range = plot_range[lower_insertion_ind:upper_insertion_ind]
                bounded_plot_range = np.append(bounded_plot_range, lower)
                bounded_plot_range = np.append(bounded_plot_range, upper)
                bounded_plot_range.sort()

                # Now replicate the other coords, with all of these coord subsegments.
                # Inspired by cs95's answer to: https://stackoverflow.com/questions/38163366/split-list-into-separate-but-overlapping-chunks
                # coord_segments = [plot_range[i:i+2] for i in range(0, len(plot_range))]

                # Now I want to add the other coordinates back in
                # Duplicate s len(coord_segments)-1 times, then sub in the values?
                linkage_coords = np.repeat([s], len(bounded_plot_range), axis=0)
                linkage_coords[:, plot_axis] = bounded_plot_range

                # colours = ['r', 'g', 'b', 'orange', 'cyan', 'purple', 'y', 'black', 'grey']
                for i in range(len(linkage_coords)-1):
                    ax.plot([linkage_coords[i,0], linkage_coords[i+1,0]],
                              [linkage_coords[i,1], linkage_coords[i+1,1]],
                              [linkage_coords[i,2], linkage_coords[i+1,2]],
                              # c=colours[i], #
                              c=line_col,
                              ls=ls,
                              lw=lw,
                              zorder=0)
                    
                # How should I bridge this?
                # (10, 30) - 22 == (10, 22), (22, 30)
                # Just insert the values and re-sort, depending on the order

    # Checks whether the aircraft is within the confines of the building
    # Returns True if the position is within the building
    def within_building(self, n, e, d):
        pos_arr = np.array([n, e, d])
        # TODO This needs testing!
        # If the given point (e.g. aircraft coordinates) are within the bounds of the building, return True, else return False.
        return ((self.min_ned <= pos_arr).all() and (pos_arr <= self.max_ned).all())

class World:
    # TODO Only takes wind_model for now - calculates presence of obstacles from extremes of wind model.
    # 'buildings' is a list of Building objects which should be plotted on the axis.
    def __init__(self, wind_model, buildings=[], print_range=True):
        self.wind_model = wind_model
        self.buildings = buildings

        # NOTE: If the epsilon derivative distance isn't accounted for, the differentiation
        # operation may query the wind field outside of its bounds of validity. The CFD wind models
        # raise their own error if this happens.
        self.wind_n_range = wind_model.valid_n_range
        self.wind_e_range = wind_model.valid_e_range
        # Ensure that negative heights (positive values of z, due to NED coord system) are not permitted
        self.wind_d_range = (wind_model.valid_d_range[0], np.min((wind_model.valid_d_range[1], 0)))

        self._set_valid_range(self.wind_n_range, self.wind_e_range, self.wind_d_range, print_range)

    def _set_valid_range(self, n_range_ned, e_range_ned, d_range_ned, print_range):
        self.valid_n_range = n_range_ned
        self.valid_e_range = e_range_ned
        self.valid_d_range = d_range_ned

        self.max_ned = np.array([self.valid_n_range[1], self.valid_e_range[1], self.valid_d_range[1]])
        self.min_ned = np.array([self.valid_n_range[0], self.valid_e_range[0], self.valid_d_range[0]])
        
        # Using n, e and d for north, east and down instead of x, y and z.
        if print_range:
            print(f"Region of world validity: n: {self.valid_n_range}, e: {self.valid_e_range}, d: {self.valid_d_range}")
    
    def get_n_extent(self):
        return self.valid_n_range

    def get_e_extent(self):
        return self.valid_e_range

    def get_d_extent(self):
        return self.valid_d_range
    
    """
    # Returns a new World object which is further constrained (e.g. it is 'contained' within this one). This is used by the RRT to
    # define a search region.
    def constrain(self, x_range_ned, y_range_ned, z_range_ned):
        # Checks that the 'inner' range lies within the 'outer' range.
        def range_contains(outer, inner):
            return (inner[0] >= outer[0]) and (inner[1] <= outer[1])
        
        # For the RRT, the regions have to be finite.
        def is_finite(range):
            return ((range[0] > -np.inf) and (range[1] < np.inf))
        
        # ===
        
        # Check that the given argument regions are finite
        if not (is_finite(x_range_ned) and is_finite(y_range_ned) and is_finite(z_range_ned)):
            raise ValueError("Ranges must be finite!")
             
        # Check that the given argument regions lie within the regions of wind model validity
        if not (range_contains(self.wind_x_range, x_range_ned) and \
                range_contains(self.wind_y_range, y_range_ned) and \
                    range_contains(self.wind_z_range, z_range_ned)):
            raise ValueError("Given ranges for x, y and z do not lie within the region of wind model validity")
            
        # Create a new copy of the World object
        world = World(copy.deepcopy(self.wind_model), copy.deepcopy(self.buildings), False)

        # Set ranges
        world._set_valid_range(x_range_ned, y_range_ned, z_range_ned, True)

        # Return new World object (this is a different object in memory to 'self') with constrained
        # x, y and z region.
        return world
    """
    
    def get_buildings(self):
        return self.buildings
    
    # Plot all buildings on the given 3d Matplotlib axis
    def plot_buildings(self, ax):
        for building in self.buildings:
            building.add_to_axis(ax)
    
    # Returns True if the given position is within the region of world validity, else False. This includes valid_x/y/z_range and buildings.
    # n, e and d for north, east and down.
    #
    # Tests conducted (and passed 19/3/24):
    # Building where there is wind, zero-wind patch where there is not a building.
    #  > Building mode: flies through zero-wind patch, stops at building.
    #  > Zero-wind mode: stops at zero-wind patch, flies through building.
    # [All tests passed]
    def valid_position(self, n, e, d):
        # Check if within the overall valid world extent
        pos_arr = np.array([n, e, d])
        # TODO This needs testing!
        if not ((self.min_ned <= pos_arr).all() and (pos_arr <= self.max_ned).all()):
            return False
        
        # If the position validity check is being done on building locations
        if config.position_validity_check_type == PositionCheckType.BUILDING:
            # Ensure that the location isn't within a building
            for building in self.buildings:
                # If within the building, position is invalid
                if building.within_building(n, e, d):
                    return False
        # If the position validity check is being done on locations with zero wind
        else:   # PositionCheckType.ZERO_WIND
            # If the wind is zero at the given location, position is invalid
            # NOTE Using zero for time - assuming that obstacles have zero wind for all time.
            if np.array_equal(self.wind_at(n, e, d, 0), np.zeros(3)):
                return False
        
        # Position is valid if this point has been reached
        return True
    
    @partial(jax.jit, static_argnums=(0,))
    def wind_at(self, x, y, z, t):
        # Returns a Numpy array (because wind_model.at(...) returns a Numpy array)
        return self.wind_model.wind_single(x, y, z, t)
    
    # TODO Example function for obtaining information about the environmental topography
    def lidar(self):
        pass