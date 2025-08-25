import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import FancyArrowPatch, Rectangle
import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d import Axes3D, art3d, proj3d
import numpy as np
from pathlib import Path

# Add the flight directory to the Python path (when running from FlightSwordLite)
import sys, os
sys.path.append(os.getcwd())

# TODO I don't know if it's still going to work with the Jax version?
from flight.simulator.utils import calc_body_to_inertial, ned_to_xyz, xyz_to_ned

# Fix to make 3D quiver plot work. Code from: https://github.com/matplotlib/matplotlib/issues/21688.
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    
    # FT modification - 3/11/23
    def update_3d_posns(self, xs, ys, zs):
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

class Plane:
    # '_xyz' indicates that the position coordinates are in the xyz frame
    def __init__(self, ax, scale=1, init_x_xyz=0, init_y_xyz=0, init_z_xyz=0, init_phi=0, init_theta=0, init_psi=0, colour=None, linewidth=None, zorder=1e5):
        # Original plane
        #self.verts = scale*np.array([
        #    [-1, 1, 0],
        #    [-0.5, 0, -0.5],
        #    [-1, -1, 0],
        #    [2, 0, 0],
        #])
        # https://stackoverflow.com/questions/4622057/plotting-3d-polygons
        # WOT4: wingspan 1.33m, length 1.2m
        v = scale*np.array([
            [1.5, 0, 0],
            [-1, 1, 0],
            [-0.5, 0, -0.3],
            [-1, -1, 0],
            # [-0.5, 0, -1]
        ])
        f = [[0, 1, 2], [0, 2, 3]] # [[0, 1, 2, 3]] # [[0, 1, 3], [0, 2, 3], [0, 3, 4]]
        # Done like this so that the code in set_pose can still rotate the vertices.
        self.verts = np.array([[v[i] for i in p] for p in f])
        self.verts_flat = self.verts.reshape(np.prod(self.verts.shape[:2]), 3)
        self.plane = art3d.Poly3DCollection(self.verts)
        self.plane.set_zorder(zorder)
        # Each plane will be a random colour if not specified
        self.plane.set_color(colour if colour else colors.rgb2hex(np.random.rand(3)))
        self.plane.set_edgecolor('k')
        if linewidth is not None:
            self.plane.set_linewidth(linewidth)

        ax.add_collection3d(self.plane)

        # Set initial position and orientation
        self.set_pose(init_x_xyz, init_y_xyz, init_z_xyz, init_phi, init_theta, init_psi)
    
    def transform_verts(self, verts):
        return verts.reshape(*self.verts.shape[:2], 3)

    def set_pose(self, posnx_xyz, posny_xyz, posnz_xyz, phi, theta, psi):
        # Orient
        # TODO Check this description
        ## Body to inertial matrix calculates the inertial frame coordinates of a vector expressed in the body
        ## frame. Alternatively, it can be used to rotate a vector in the fixed inertial perspective by the
        ## orientation of the body frame. This is what it is being used for here - to align the aircraft polygon
        ## with the body frame.
        # Using this matrix, we can treat the aircraft vertices as if they are in the body frame, and then find their inertial frame positions for plotting.
        rotat = calc_body_to_inertial(phi, theta, psi)
        # verts needs to be in the format...
        #  +-                    -+
        #  |   |    |    |        |
        #  |  pt1  pt2  pt3  ...  |
        #  |   |    |    |        |
        #  +-                    -+
        # so that each vertex point is rotated by the rotation matrix. This is why self.verts
        # is transposed, since this is the opposite orientation to that required by Poly3DCollection.
        oriented_verts = np.matmul(rotat, self.verts_flat.T)

        # Convert NED system coordinates (for *plane vertices* - position of plane already given in xyz coord system) to
        # xyz coord system used by Matplotlib for plotting.
        # (Need to flip E and D)
        # [TODO Is this true, following code modifications?] NOTE The oriented_verts array is changed by the ned_to_xyz() function, so it technically doesn't have to be returned and re-assigned, but it helps to make
        # the code more maintainable by making value changes more obvious.
        # TODO [Do a proper comment for this] Want to use the scalar version so that it comes out as a 2D Numpy array. Note that the expansion operator (*)
        # has to be used in the argument.
        oriented_verts = ned_to_xyz(*oriented_verts)

        # Translate
        # posn_x_xyz, posn_y_xyz and posn_z_xyz should be in the xyz coordinate system used by Matplotlib for plotting.
        shifted_verts = oriented_verts + np.array([[posnx_xyz], [posny_xyz], [posnz_xyz]])

        # .T to transpose back to the expected format.
        self.plane.set_verts(self.transform_verts(shifted_verts.T))

        # plt.draw()


# Flights using the same grapher should have the same wind field, else the visualisation could be misleading.
class FlightGrapher:
    # Initialises the plot
    # The orientation (a plane) is plotted every orientation_interval steps
    # If 'orientation_interval' is None, aircraft and wind are not plotted.
    # 'buildings' is a list of Building objects which should be plotted on the axis.
    def __init__(self, orientation_interval=None, plane_scale=1, wind_scale=1, buildings=[], time_markers=[]):
        self.dataloggers = []
        self.line_colours = []
        self.line_styles = []
        self.orientation_interval = orientation_interval
        self.plane_scale = plane_scale
        self.wind_scale = wind_scale

        self.trace_fig = plt.figure()
        self.trace_ax = self.trace_fig.add_subplot(projection='3d')

        # Plot buildings
        for building in buildings:
            building.add_to_axis(self.trace_ax)
        
        self.time_markers = time_markers

        # plt.show(block=False)
    
    # Returns grapher figure object for customisation
    def get_grapher_fig(self):
        return self.trace_fig
    
    # Returns grapher axis object for customisation
    def get_grapher_ax(self):
        return self.trace_ax
    
    # Plot the orientation (a plane) every self.orientation_interval steps
    def plot_orientations(self, datalogger, plane_colour):
        # Plot aircraft orientation after every self.orientation_interval steps
        step = 0
        while step < len(datalogger.ns):
            Plane(self.trace_ax, self.plane_scale, *ned_to_xyz(datalogger.ns[step], datalogger.es[step], datalogger.ds[step]),
                datalogger.phis[step], datalogger.thetas[step], datalogger.psis[step], plane_colour)
            step += self.orientation_interval
    
    # Plot the wind every self.orientation_interval steps (coincides with the planes, if they're plotted)
    def plot_wind(self, datalogger, colour):
        step = 0
        while step < len(datalogger.ns):
            # We know the wind NED. Need to convert it to xyz (for Matplotlib),
            # then add to position of the planes.
            wind_x, wind_y, wind_z = ned_to_xyz(datalogger.wns[step],
                                                     datalogger.wes[step],
                                                     datalogger.wds[step])
            # Convert plane positions from NED to xyz
            pos_x, pos_y, pos_z = ned_to_xyz(datalogger.ns[step], datalogger.es[step], datalogger.ds[step])
            wind_vec = Arrow3D([pos_x, pos_x + wind_x*self.wind_scale],
                               [pos_y, pos_y + wind_y*self.wind_scale],
                               [pos_z, pos_z + wind_z*self.wind_scale],
                               mutation_scale=10, lw=1, arrowstyle="-|>", color=colour)
            self.trace_ax.add_artist(wind_vec)
            step += self.orientation_interval

    def add_flight(self, datalogger, line_colour, line_style='-', wind_colour='orange', plane_colour=None):
        self.dataloggers.append(datalogger)
        self.line_colours.append(line_colour)
        self.line_styles.append(line_style)

        # Plot trajectory
        self.trace_ax.plot(*ned_to_xyz(datalogger.ns, datalogger.es, datalogger.ds), c=line_colour, ls=line_style)
        if self.orientation_interval:
            # Plot orientations
            self.plot_orientations(datalogger, plane_colour)
            # Plot wind vectors
            self.plot_wind(datalogger, wind_colour)

        self.trace_ax.set_aspect('equal')
    
    # Save trace as an image
    # save_folder can be a string or a Pathlib Path object.
    def save_trace(self, save_folder):
        self.trace_fig.savefig(Path(save_folder) / 'trace.png', dpi=600)
    
    # Takes a Matplotlib axis, an axis label and a selector function which dictates
    # the DataLogger parameter (e.g. xs) to be plotted. Iterates through all of the
    # DataLoggers plotting this parameter for each, with the same line colour and style
    # used in the flight trace. 
    def plot_iterator(self, ax, label, selector):
        for dl_ind in range(len(self.dataloggers)):
            # Plotting separately
            if ax is None:
                fig, ax = plt.subplots()
                
            dl = self.dataloggers[dl_ind]
            c = self.line_colours[dl_ind]
            ls = self.line_styles[dl_ind]

            ax.plot(dl.times, selector(dl), c=c, ls=ls)
            ax.set_ylabel(label)

        for t in self.time_markers:
            ax.axvline(x=t, c='grey')
    
    # Plot DataLogger graphs, but compound, with all flights present.
    # save_folder can be a string or a Pathlib Path object.
    def graph_states_and_controls(self, separate=False, save_folder=None):
        if separate:
            axs = [None]*16
        else:
            fig, axs = plt.subplots(16)
            fig.suptitle("State and control values")

        self.plot_iterator(axs[0], 'n (m)', lambda dl: dl.ns)
        self.plot_iterator(axs[1], 'e (m)', lambda dl: dl.es)
        self.plot_iterator(axs[2], 'd (m)', lambda dl: dl.ds)
        #self.plot_iterator(axs[3], 'Va (m/s)', lambda dl: dl.vas)
        #self.plot_iterator(axs[4], 'alpha (rad)', lambda dl: dl.alphas)
        #self.plot_iterator(axs[5], 'beta (rad)', lambda dl: dl.betas)
        self.plot_iterator(axs[3], 'u (m/s)', lambda dl: dl.us)
        self.plot_iterator(axs[4], 'v (m/s)', lambda dl: dl.vs)
        self.plot_iterator(axs[5], 'w (m/s)', lambda dl: dl.ws)
        self.plot_iterator(axs[6], 'phi (rad)', lambda dl: dl.phis)
        self.plot_iterator(axs[7], 'theta (rad)', lambda dl: dl.thetas)
        self.plot_iterator(axs[8], 'psi (rad)', lambda dl: dl.psis)
        # TODO Are p, q and r in rad/s?
        self.plot_iterator(axs[9], 'p (rad/s)', lambda dl: dl.ps)
        self.plot_iterator(axs[10], 'q (rad/s)', lambda dl: dl.qs)
        self.plot_iterator(axs[11], 'r (rad/s)', lambda dl: dl.rs)
        
        self.plot_iterator(axs[12], 'da (rad)', lambda dl: dl.das)
        self.plot_iterator(axs[13], 'de (rad)', lambda dl: dl.des)
        self.plot_iterator(axs[14], 'dr (rad)', lambda dl: dl.drs)
        self.plot_iterator(axs[15], 'dp (rad)', lambda dl: dl.dps)

        if save_folder is not None:
            fig.savefig(Path(save_folder) / 'states_and_controls.png', dpi=600)
        
        # plt.show(block=False)

    # save_folder can be a string or a Pathlib Path object.
    def graph_forces_and_moments(self, separate=False, save_folder=None):
        if separate:
            axs = [None]*7
        else:
            fig, axs = plt.subplots(7)
            fig.suptitle("Forces and moments")

        # TODO Are these forces in Newtons?
        self.plot_iterator(axs[0], 'Lift (N)', lambda dl: dl.Ls)
        self.plot_iterator(axs[1], 'Sideforce (N)', lambda dl: dl.Cs)
        self.plot_iterator(axs[2], 'Drag (N)', lambda dl: dl.Ds)
        self.plot_iterator(axs[3], 'Thrust (N)', lambda dl: dl.Ts)
        self.plot_iterator(axs[4], 'Roll moment (Nm)', lambda dl: dl.L_lats)
        self.plot_iterator(axs[5], 'Pitch moment (Nm)', lambda dl: dl.Ms)
        self.plot_iterator(axs[6], 'Yaw moment (Nm)', lambda dl: dl.Ns)

        if save_folder is not None:
            fig.savefig(Path(save_folder) / 'forces_and_moments.png', dpi=600)
        
        # plt.show(block=False)
    
    # Plot DataLogger airstate graphs, but compound, with all flights present.
    def graph_airstate(self, separate=False):
        if separate:
            axs = [None]*3
        else:
            fig, axs = plt.subplots(3)
            fig.suptitle("Airstate values")

        self.plot_iterator(axs[0], 'va (m/s)', lambda dl: dl.vas)
        self.plot_iterator(axs[1], 'alpha (rad)', lambda dl: dl.alphas)
        self.plot_iterator(axs[2], 'beta (rad)', lambda dl: dl.betas)
    
    # Plot DataLogger wind graphs, but compound, with all flights present.
    def graph_wind(self, separate=False):
        if separate:
            axs = [None]*3
        else:
            fig, axs = plt.subplots(3)
            fig.suptitle("Wind values")

        self.plot_iterator(axs[0], 'wn (m/s)', lambda dl: dl.wns)
        self.plot_iterator(axs[1], 'we (m/s)', lambda dl: dl.wes)
        self.plot_iterator(axs[2], 'wd (m/s)', lambda dl: dl.wds)


# This is a very cut down version for now - no wind plotting.
# See the old code to bring the wind plotting across - both in __init__(...)
# and update(...).
class View:
    # TODO This needs completing!
    def __init__(self, world, init_x, init_y, init_z, init_phi, init_theta, init_psi,
                 # plane_scale=1, x_range=(-50, 50), y_range=(-50, 50), z_range=(0, 80)):
                 plane_scale=1, x_range=(-10, 70), y_range=(-40, 40), z_range=(0, 60)):
        
        self.wind = world.wind_at

        self.world_fig = plt.figure(num="Simulator")
        self.world_ax = self.world_fig.add_subplot(projection='3d')

        self.world_ax.set_xlabel("x [ e] (m)")
        self.world_ax.set_ylabel("y [ n] (m)")
        self.world_ax.set_zlabel("z [-d] (m)")

        # Showing x, y and z axes at origin
        i_vec_scaling = 30
        x = Arrow3D([0, i_vec_scaling], [0, 0], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color="r")
        y = Arrow3D([0, 0], [0, i_vec_scaling], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color="g")
        z = Arrow3D([0, 0], [0, 0], [0, i_vec_scaling], mutation_scale=20, lw=1, arrowstyle="-|>", color="b")
        self.world_ax.add_artist(x)
        self.world_ax.add_artist(y)
        self.world_ax.add_artist(z)
        
        # Plot ground
        # TODO This seems like a messy way of doing this
        ground_x, ground_y = np.meshgrid(x_range, y_range)
        ground_z = np.zeros(ground_x.shape)
        self.world_ax.plot_surface(ground_x, ground_y, ground_z, color='gray', alpha=0.1)
        
        # Plot aircraft
        self.plane = Plane(self.world_ax, plane_scale, init_x, init_y, init_z, init_phi, init_theta, init_psi)

        # Create a wind arrow at the plane
        # Assuming time = 0 for start of simulation
        self.wind_at_plane = Arrow3D(*self.calc_wind_at_plane_vectors(init_x, init_y, init_z, 0), mutation_scale=20, lw=1, arrowstyle="-|>", color="r")
        self.world_ax.add_artist(self.wind_at_plane)

        # Aircraft position trace
        self.x_positions = []
        self.y_positions = []
        self.z_positions = []
        self.trace, = self.world_ax.plot([], [], [], '.-', c='orange', lw=1, ms=2)
        
        # Scale axes
        self.world_ax.set_xlim3d(*x_range)
        self.world_ax.set_ylim3d(*y_range)
        self.world_ax.set_zlim3d(*z_range)
        self.world_ax.set_aspect('equal')

        # Plot buildings
        for building in world.buildings:
            building.add_to_axis(self.world_ax)

        plt.show(block=False)
    
    # Update aircraft and wind visualisations
    # Expects positions in xyz (not ned) coordinate system. This class should be expected to be
    # decoupled from the coordinate system used by the flight dynamics.
    def update(self, t, x, y, z, phi, theta, psi):
        # Update aircraft position
        # posn_x, posn_y and posn_z should be in the xyz coordinate system used by Matplotlib for plotting.
        self.plane.set_pose(x, y, z, phi, theta, psi)

        # Update wind vector at aircraft position
        ### TODO Update doesn't seem to be working. For now, just deleting the arrow object each time and redrawing it.
        self.wind_at_plane.update_3d_posns(*self.calc_wind_at_plane_vectors(x, y, z, t))

        # Update aircraft position trace
        self.x_positions.append(x)
        self.y_positions.append(y)
        self.z_positions.append(z)
        self.trace.set_data(self.x_positions, self.y_positions)
        self.trace.set_3d_properties(self.z_positions, 'z')

        # TODO Hack, for now.
        plt.pause(0.03)
        plt.draw()
    
    # For updating the 'wind at plane' vector. Calculates the wind at the plane and returns the vector components as:
    # [plane x posn xyz frame, posn_x + wind_x], [plane y posn xyz frame, posn_y + wind_y], [plane z posn xyz frame, posn_z + wind_z]
    def calc_wind_at_plane_vectors(self, x, y, z, t, vec_scale=1):
        # Convert aircraft position to ned frame - need this to get wind
        n, e, d = xyz_to_ned(x, y, z)
        # Get wind (in ned frame) at aircraft ned position
        wn, we, wd = self.wind(n, e, d, t)
        # Convert wind to xyz frame
        wx, wy, wz = ned_to_xyz(wn, we, wd)

        return [x, x + wx*vec_scale], [y, y + wy*vec_scale], [z, z + wz*vec_scale]


class ViewBuildingBack(View):
    # TODO This needs completing!
    def __init__(self, world, init_x, init_y, init_z, init_phi, init_theta, init_psi,
                 # plane_scale=1, x_range=(-50, 50), y_range=(-50, 50), z_range=(0, 80)):
                 plane_scale=1): # , x_range=(-10, 70), y_range=(-40, 40), z_range=(0, 60)):
        
        self.wind = world.wind_at

        self.world_fig = plt.figure(num="Simulator")
        self.world_ax = self.world_fig.add_subplot(projection='3d')
        self.world_ax.computed_zorder = False

        labelpad = 10
        self.world_ax.set_xlabel(r'$x_{\,[e]}$ ($m$)', labelpad=labelpad)
        self.world_ax.set_ylabel(r'$y_{\,[n]}$ ($m$)', labelpad=labelpad)
        self.world_ax.set_zlabel(r'$z_{\,[-d]}$ ($m$)', labelpad=labelpad)

        # Add building projection
        b = Rectangle((131, 0), 238, 33, zorder=0, fc='lightgrey', ec='darkgrey')
        self.world_ax.add_patch(b)
        art3d.patch_2d_to_3d(b, z=190, zdir='x')

        self.world_ax.set_axis_off()
        self.world_ax.grid(visible=False)

        # Showing x, y and z axes at origin
        i_vec_scaling = 30
        x = Arrow3D([0, i_vec_scaling], [0, 0], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color="r")
        y = Arrow3D([0, 0], [0, i_vec_scaling], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color="g")
        z = Arrow3D([0, 0], [0, 0], [0, i_vec_scaling], mutation_scale=20, lw=1, arrowstyle="-|>", color="b")
        self.world_ax.add_artist(x)
        self.world_ax.add_artist(y)
        self.world_ax.add_artist(z)
        
        # Plot ground
        # TODO This seems like a messy way of doing this
        ground_x, ground_y = np.meshgrid((190, 400), (0, 500))
        ground_z = np.zeros(ground_x.shape)
        self.world_ax.plot_surface(ground_x, ground_y, ground_z, color='gray', alpha=0.1)
        
        # Plot aircraft
        self.plane = Plane(self.world_ax, plane_scale, init_x, init_y, init_z, init_phi, init_theta, init_psi)

        # Create a wind arrow at the plane
        # Assuming time = 0 for start of simulation
        self.wind_at_plane = Arrow3D(*self.calc_wind_at_plane_vectors(init_x, init_y, init_z, 0), mutation_scale=20, lw=1, arrowstyle="-|>", color="r")
        self.world_ax.add_artist(self.wind_at_plane)

        # Aircraft position trace
        self.x_positions = []
        self.y_positions = []
        self.z_positions = []
        self.trace, = self.world_ax.plot([], [], [], '.-', c='orange', lw=1, ms=2)

        # Aircraft ground projection trace
        self.ground_x = []
        self.ground_y = []
        self.ground_z = []
        self.ground_trace, = self.world_ax.plot([], [], [], '--', c='grey', lw=1)

        # Aircraft wall projection trace
        self.wall_x = []
        self.wall_y = []
        self.wall_z = []
        self.wall_trace, = self.world_ax.plot([], [], [], '--', c='grey', lw=1)
        
        # Scale axes
        self.world_ax.set_xlim(190, 400)
        self.world_ax.set_ylim(0, 500)
        self.world_ax.set_zlim(0, 70)
        self.world_ax.set_aspect('equal')

        ## Plot buildings
        #for building in world.buildings:
        #    building.add_to_axis(self.world_ax)

        plt.show(block=False)
    
    def update(self, t, x, y, z, phi, theta, psi):
        # Update aircraft position
        # posn_x, posn_y and posn_z should be in the xyz coordinate system used by Matplotlib for plotting.
        self.plane.set_pose(x, y, z, phi, theta, psi)

        # Update wind vector at aircraft position
        ### TODO Update doesn't seem to be working. For now, just deleting the arrow object each time and redrawing it.
        self.wind_at_plane.update_3d_posns(*self.calc_wind_at_plane_vectors(x, y, z, t))

        # Update aircraft position trace
        self.x_positions.append(x)
        self.y_positions.append(y)
        self.z_positions.append(z)
        self.trace.set_data(self.x_positions, self.y_positions)
        self.trace.set_3d_properties(self.z_positions, 'z')

        # Update aircraft ground projection trace
        self.ground_x.append(x)
        self.ground_y.append(y)
        self.ground_z.append(0)
        self.ground_trace.set_data(self.ground_x, self.ground_y)
        self.ground_trace.set_3d_properties(self.ground_z, 'z')
        
        # Update aircraft wall projection trace
        self.wall_x.append(190)
        self.wall_y.append(y)
        self.wall_z.append(z)
        self.wall_trace.set_data(self.wall_x, self.wall_y)
        self.wall_trace.set_3d_properties(self.wall_z, 'z')

        # TODO Hack, for now.
        plt.pause(0.03)
        plt.draw()


# Helper methods

def plot_streamline(ax, wind_model, n, e, d):
    pass