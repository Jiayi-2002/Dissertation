# Add the flight directory to the Python path (when running from FlightSwordLite)
import sys, os
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from matplotlib.cm import ScalarMappable
from pathlib import Path

from flight.simulator.world import Building

# Sweeps through 3D space to show the values - either of the interpolation or the error.
class Sweeper:
    # wind_dir = 'x'/'y'/'z'
    def __init__(self, plot_coords, plot_vals, interp_vals, wind_dir, plot_options={'rbf_centres', 'building'}, rbf_centre_coords=None, show=True, save=False, folderpath='.'):
        self.plot_coords = plot_coords
        self.plot_vals = plot_vals
        self.interp_vals = interp_vals

        # Calculate interpolation errors
        self.interpolation_errors = plot_vals - interp_vals

        # Define norms and colormaps used for colouring
        cmap = 'bwr'
        interp_norm = TwoSlopeNorm(0, vmin=np.min((np.min(self.interp_vals), -0.1)), vmax=np.max((np.max(self.interp_vals), 0.1)))
        err_norm = TwoSlopeNorm(0, vmin=np.min((np.min(self.interpolation_errors), -0.1)), vmax=np.max((np.max(self.interpolation_errors), 0.1)))

        # For adding building to plot
        building = Building((131, 369), (100, 190), (-33, 0))

        if show:
            self.fig = plt.figure()
            self.x_interp_ax = self.fig.add_subplot(2, 4, 1, projection='3d')
            self.y_interp_ax = self.fig.add_subplot(2, 4, 2, projection='3d')
            self.z_interp_ax = self.fig.add_subplot(2, 4, 3, projection='3d')
            self.x_err_ax = self.fig.add_subplot(2, 4, 5, projection='3d')
            self.y_err_ax = self.fig.add_subplot(2, 4, 6, projection='3d')
            self.z_err_ax = self.fig.add_subplot(2, 4, 7, projection='3d')
            # Colourbar axes
            self.interp_cbar_ax = self.fig.add_subplot(2, 4, 4, aspect=1.2)
            self.err_cbar_ax = self.fig.add_subplot(2, 4, 8, aspect=1.2)

            # Create sweeps
            print("Creating sweeps")
            self.interp_x_anim = self.create_anim(self.fig, self.x_interp_ax, self.plot_coords, self.interp_vals, 'x', f'Interpolation values (w{wind_dir})', rbf_centre_coords, cmap, interp_norm, building, plot_options)
            self.interp_y_anim = self.create_anim(self.fig, self.y_interp_ax, self.plot_coords, self.interp_vals, 'y', f'Interpolation values (w{wind_dir})', rbf_centre_coords, cmap, interp_norm, building, plot_options)
            self.interp_z_anim = self.create_anim(self.fig, self.z_interp_ax, self.plot_coords, self.interp_vals, 'z', f'Interpolation values (w{wind_dir})', rbf_centre_coords, cmap, interp_norm, building, plot_options)
            self.err_x_anim = self.create_anim(self.fig, self.x_err_ax, self.plot_coords, self.interpolation_errors, 'x', f'Interpolation errors (w{wind_dir})', rbf_centre_coords, cmap, err_norm, building, plot_options)
            self.err_y_anim = self.create_anim(self.fig, self.y_err_ax, self.plot_coords, self.interpolation_errors, 'y', f'Interpolation errors (w{wind_dir})', rbf_centre_coords, cmap, err_norm, building, plot_options)
            self.err_z_anim = self.create_anim(self.fig, self.z_err_ax, self.plot_coords, self.interpolation_errors, 'z', f'Interpolation errors (w{wind_dir})', rbf_centre_coords, cmap, err_norm, building, plot_options)

            # Add colourbars
            plt.colorbar(ScalarMappable(interp_norm, cmap), self.interp_cbar_ax) # = self.fig.add_subplot(2, 3, 5, projection='3d')
            plt.colorbar(ScalarMappable(err_norm, cmap), self.err_cbar_ax) # = self.fig.add_subplot(2, 3, 6, projection='3d')

        # Save the animations
        # TODO Have to regenerate the animations for each - this probably isn't the best way to do this.
        if save:
            self.save_anim(self.plot_coords, self.interp_vals, 'x', f'interpolation (w{wind_dir})', rbf_centre_coords, cmap, interp_norm, building, folderpath, plot_options)
            self.save_anim(self.plot_coords, self.interp_vals, 'y', f'interpolation (w{wind_dir})', rbf_centre_coords, cmap, interp_norm, building, folderpath, plot_options)
            self.save_anim(self.plot_coords, self.interp_vals, 'z', f'interpolation (w{wind_dir})', rbf_centre_coords, cmap, interp_norm, building, folderpath, plot_options)
            self.save_anim(self.plot_coords, self.interpolation_errors, 'x', f'errors (w{wind_dir})', rbf_centre_coords, cmap, err_norm, building, folderpath, plot_options)
            self.save_anim(self.plot_coords, self.interpolation_errors, 'y', f'errors (w{wind_dir})', rbf_centre_coords, cmap, err_norm, building, folderpath, plot_options)
            self.save_anim(self.plot_coords, self.interpolation_errors, 'z', f'errors (w{wind_dir})', rbf_centre_coords, cmap, err_norm, building, folderpath, plot_options)

    def save_anim(self, coords, vals, dir, title, rbf_centre_coords, cmap, norm, building, folderpath, plot_options={'building'}):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        anim = self.create_anim(fig, ax, coords, vals, dir, title, rbf_centre_coords, cmap, norm, building, plot_options)
        plt.colorbar(ScalarMappable(norm, cmap), ax=ax)
        anim.save(Path(folderpath) / f'{title} - {dir}.gif', writer=PillowWriter(fps=20))
        # To prevent excess memory usage buildup over time
        plt.close()
    
    @staticmethod
    def create_anim(fig, ax, coords, vals, dir, title, rbf_centre_coords, cmap, norm, building, plot_options={'building', 'rbf_centres'}):
        def init_func():
            pass

        def animate(coord):
            # Find the indices where the coords have the value given.
            if dir == 'x':
                # x coordinates
                slice_inds = np.where(coords[:, 0] == coord)
            elif dir == 'y':
                slice_inds = np.where(coords[:, 1] == coord)
            else: # z
                slice_inds = np.where(coords[:, 2] == coord)

            # Get those coordinates
            slice_coords = coords[slice_inds]
            # Get those values
            slice_vals = vals[slice_inds]
            # Update scatter plot and colour by the values
            scat._offsets3d = tuple(slice_coords.T)

            # Still need to do colour and colourbar
            scat.set_array(slice_vals)
            # return self.scat
        
        scat = ax.scatter([], [], [], c=[], cmap=cmap, norm=norm)

        if dir == 'x':
            # x coordinates
            dir_coords = coords[:, 0]
        elif dir == 'y':
            dir_coords = coords[:, 1]
        elif dir == 'z':
            dir_coords = coords[:, 2]
        else:
            raise Exception("dir argument invalid - should be 'x', 'y' or 'z'")
        frames = np.unique(dir_coords) # .sort()

        anim = FuncAnimation(fig, animate, frames, interval=20, blit=False)

        # Plot RBF centres
        if 'rbf_centres' in plot_options:
            ax.scatter(*rbf_centre_coords.T, marker='x', c='y')

        # Scale
        mins = np.min(coords, axis=0)
        maxs = np.max(coords, axis=0)
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # Add building to plot
        if 'building' in plot_options:
            building.add_to_axis(ax)

        ax.set_aspect('equal')
        ax.set_title(title + f"  [{dir}-sweep]")
        # fig.suptitle(title + f"  [{dir}-sweep]")
        # err_fig.colorbar(s)

        return anim


# Simpler version, which just plots the function given.
class SimpleSweeper:
    def __init__(self, plot_coords, plot_vals, title=None):
        plot_coords = plot_coords
        plot_vals = plot_vals

        # Define norms and colormaps used for colouring
        cmap = 'bwr'
        norm = TwoSlopeNorm(0, vmin=np.min((np.min(plot_vals), -0.1)), vmax=np.max((np.max(plot_vals), 0.1)))

        fig = plt.figure()
        x_sweep_ax = fig.add_subplot(1, 4, 1, projection='3d')
        y_sweep_ax = fig.add_subplot(1, 4, 2, projection='3d')
        z_sweep_ax = fig.add_subplot(1, 4, 3, projection='3d')
        # Colourbar axes
        cbar_ax = fig.add_subplot(1, 4, 4, aspect=1.2)

        # Create sweeps
        print("Creating sweeps")
        self.interp_x_anim = Sweeper.create_anim(fig, x_sweep_ax, plot_coords, plot_vals, 'x', '', None, cmap, norm, None, {})
        self.interp_y_anim = Sweeper.create_anim(fig, y_sweep_ax, plot_coords, plot_vals, 'y', '', None, cmap, norm, None, {})
        self.interp_z_anim = Sweeper.create_anim(fig, z_sweep_ax, plot_coords, plot_vals, 'z', '', None, cmap, norm, None, {})
        # Add colourbars
        plt.colorbar(ScalarMappable(norm, cmap), cbar_ax)

        if title is not None:
            fig.suptitle(title)