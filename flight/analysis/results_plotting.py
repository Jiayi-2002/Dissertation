# Here in this module I'm going to put some code from results plotting.
# This code will enable the analysis of the flight trajectories which come out of the optimiser.
# Specifically, this code will:
#  o Take a DataLogger
#  o Create a number of figures

# Create traces - coloured by:
#  o Inertial energy change, all in one
#  o Air-relative energy change, all in one
#  o Air-relative energy, separate traces: drag, static and dynamic
#  o Work done by the aerodynamic force (green for gain, red for loss?)
#  o Work done (?) by aerodynamic force in the direction of i^b (/amount of force in the direction i^b)
# Produce the work figure from a DataLogger
# Produce the energy figure from a DataLogger

import matplotlib.pyplot as plt
import matplotlib as mpl
#import imageio_ffmpeg
#mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
from matplotlib import animation
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import TwoSlopeNorm, Normalize
from mpl_toolkits.mplot3d import art3d
from matplotlib.patches import ArrowStyle, Rectangle
from matplotlib.lines import Line2D
import numpy as np
import math
from pathlib import Path
import jax
import jax.numpy as jnp
# from mayavi import mlab
import copy
from PIL import Image, ImageOps
import io
import shutil
from pypdf import PdfReader, PdfWriter

# Add the flight directory to the Python path (when running from FlightSwordLite)
import sys, os
sys.path.append(os.getcwd())

from flight.simulator import config
from flight.simulator.utils import ned_to_xyz, calc_body_to_inertial
from flight.simulator.aircraft_model import AircraftModel
from flight.simulator.view import Arrow3D, Plane
from flight.simulator.world import Building
from flight.analysis import analysis_calculations

# For testing
from flight.simulator.utils import DataLogger
from flight.simulator.wind_models import WindQuic


# Output quality
base_dpi = 600


# Scale plot
# padding_ratio: 0 for no padding, 0.5 for half padding, etc
# n/e/d_min_range are tuples: (min, max)
def scale(ax, padding_ratio, ns, es, ds, n_min_range=None, e_min_range=None, d_min_range=None):
    # Add min range values in
    if n_min_range is not None:
        ns = np.append(ns, n_min_range)
    if e_min_range is not None:
        es = np.append(es, e_min_range)
    if d_min_range is not None:
        ds = np.append(ds, d_min_range)
    
    n_range = sorted((min(ns), max(ns)))
    n_span = n_range[1] - n_range[0]
    e_range = sorted((min(es), max(es)))
    e_span = e_range[1] - e_range[0]
    d_range = sorted((min(ds), max(ds)))
    d_span = d_range[1] - d_range[0]

    n_min = n_range[0] - n_span*padding_ratio
    n_max = n_range[1] + n_span*padding_ratio
    n_lim = (n_min, n_max)
    e_min = e_range[0] - e_span*padding_ratio
    e_max = e_range[1] + e_span*padding_ratio
    e_lim = (e_min, e_max)
    d_min = d_range[0] - d_span*padding_ratio
    d_max = d_range[1] + d_span*padding_ratio
    d_lim = (d_min, d_max)

    # Convert to xyz coordinates for Matplotlib
    x_lim, y_lim, z_lim = ned_to_xyz(n_lim, e_lim, d_lim)

    ax.set_xlim3d(*np.sort(x_lim))
    ax.set_ylim3d(*np.sort(y_lim))
    ax.set_zlim3d(*np.sort(z_lim))
    ax.set_aspect('equal')

    # TODO Does this need to return anything?
    # return n_lim, e_lim, d_lim, x_lim, y_lim, z_lim
    return x_lim, y_lim, z_lim


def plot_orientations(ax, dl, plot_interval, plane_scale=1, forces=None, wind_scale=None, arrowhead_width=0.1):
# def plot_orientations(ax, dl, plot_interval, plane_scale=1, forces=None, wind_scale=1, arrowhead_width=0.1):       # Temp, until wind_scale is connected up.
    # Plot aircraft orientation after every plot_interval steps
    step = 0
    while step < dl.num_steps:
        # Get x, y and z coordinates
        # Matplotlib uses an xyz coordinate system. Convert from ned:
        n = dl.ns[step]
        e = dl.es[step]
        d = dl.ds[step]
        x, y, z = ned_to_xyz(n, e, d)

        # Plot vehicle
        Plane(ax, plane_scale, x, y, z, dl.phis[step], dl.thetas[step], dl.psis[step], 'gray')
        
        # Plot forces
        if forces is not None:
            print("WARNING: force plotting is currently restricted to inertial forces only - doesn't include fictitious force.")
            # TODO Temp hack
            force_scaler_func = lambda x: 1*x # TODO HARDCODED - fix this
            # TODO Test whether forces is in the right format
            ForcePlotter(ax, force_scaler_func, arrowhead_width, plot_inds=[2]).update(n, e, d, forces[:, step, :])

        # Plot wind (dotted line)
        # TODO Use 'is not None', or just set length to 0 to hide and add logic which hides it, like I've done previously?
        if wind_scale is not None:
            wn = dl.wns[step]
            we = dl.wes[step]
            wd = dl.wds[step]
            # Convert to xyz
            wx, wy, wz = ned_to_xyz(wn, we, wd)
            
            # print("NOTE: Plotted wind field is only correct in the inertial frame!")
            ax.add_artist(Arrow3D([x, x + wind_scale*wx], [y, y + wind_scale*wy], [z, z + wind_scale*wz], mutation_scale=10, lw=1, arrowstyle=ArrowStyle.CurveFilledB(head_width=arrowhead_width), color='orange', linestyle='--'))
        
        step += plot_interval


def gen_norm(colouring_vals, zero_centred_norm):
    min_val = np.min(np.concatenate((colouring_vals, [-0.01])))
    max_val = np.max(np.concatenate((colouring_vals, [0.01])))
    if zero_centred_norm:
        norm = TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)
    else:
        norm = Normalize(min_val, max_val)
    
    return norm, min_val, max_val


# Updates a trace for plotting. Done this way so that it can be used both with the static code and the
# animation.
def update_trace(plane_trace, xs, ys, zs, colouring_vals):
    points = np.array([xs, ys, zs]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Update trace
    plane_trace.set_segments(segments)
    # Colour trace
    plane_trace.set_array(colouring_vals)


# Code inspired by Lewis Morris' answer to: https://stackoverflow.com/questions/57316491/how-to-convert-matplotlib-figure-to-pil-image-object-without-saving-image
# First saves as a bitmap to get the cropping box, then saves as a PDF.
def crop_and_save(fig, save_path, save_png=True, dpi=400, pad=100, no_crop=False):
    if no_crop:
        fig.savefig(f'{str(save_path)}.pdf') # , bbox_inches="tight", pad_inches=3) # , bbox_inches='tight')
        if save_png:
            fig.savefig(f'{str(save_path)}.png', dpi=dpi) # , bbox_inches="tight", pad_inches=3) # , bbox_inches='tight')
    else:
        # print("PDF cropping still to be implemented")

        # Get cropping coordinates
        # Buffer for transparent image, for cropping.
        crop_buf = io.BytesIO()
        fig.savefig(crop_buf, transparent=True, dpi=dpi)
        crop_buf.seek(0)
        crop_im = Image.open(crop_buf)

        # Crop and save
        # Auto-cropping - get cropping box
        image_box = crop_im.getbbox()

        if save_png:
            # Convert figure to PIL Image
            # Buffer for image
            img_buf = io.BytesIO()
            fig.savefig(img_buf, dpi=dpi) # bbox_inches="tight", pad_inches=3) # , bbox_inches='tight')
            # fig.savefig(save_path, dpi=dpi)
            # sys.exit(0)
            img_buf.seek(0)
            im = Image.open(img_buf)

            # Crop original
            cropped_im = im.crop(image_box)

            # Add a border
            cropped_im = ImageOps.expand(cropped_im, pad, (255, 255, 255))
    
            cropped_im.save(f'{str(save_path)}.png')
        
        # ===
        # Save as a PDF
        # See https://pypdf.readthedocs.io/en/latest/user/cropping-and-transforming.html
        fig.savefig(f'{str(save_path)}.pdf')

        # Load PDF
        reader = PdfReader(f'{str(save_path)}.pdf')
        pdf_image = reader.pages[0]

        #"""
        # Calculate relative coordinates from PNG bounding box
        png_width, png_height = crop_im.size
        lower_left = (image_box[0] / png_width, 1 - (image_box[3] / png_height))
        upper_right = (image_box[2] / png_width, 1 - (image_box[1] / png_height))

        # Calculate PDF coordinates
        pdf_width = pdf_image.mediabox.right - pdf_image.mediabox.left
        pdf_height = pdf_image.mediabox.top - pdf_image.mediabox.bottom

        # Crop PDF
        eps = 1 # %
        pdf_image.mediabox.lower_left = ((lower_left[0] - eps/100)*pdf_width, (lower_left[1] - eps/100)*pdf_height)
        pdf_image.mediabox.upper_right = ((upper_right[0] + eps/100)*pdf_width, (upper_right[1]  + eps/100)*pdf_height)
        #"""

        # Re-write PDF
        writer = PdfWriter()
        writer.add_page(pdf_image)
        with open(f'{str(save_path)}.pdf', "wb") as fp:
            writer.write(fp)


projection_alpha = 0.3
projection_lw = 0.3
arrowhead_width = 0.1
projection_colour = 'grey'
text_colour = 'dimgrey'

def remove_trailing_0pt(val):
    return str(round(val)) if val % 1 == 0 else str(val)

def get_col_str(smap, val):
        return mpl.colors.to_hex(smap.to_rgba(val))

# New version of plot_trace - aesthetically nicer, with ground projections
# and 2D building projection.
# init_view_angles: None, or 3-tuple.
# NOTE Now replaced by the class below - remove this in time.
def plot_trace(dl, wind_dir_deg, wind_speed, title=None, plot_interval=None, traj_lw=1.25, plot_traj_centreline=False, init_view_angles=None,
               colouring_vals=None, cmap_name='bwr', norm=None, zero_centred_norm=True, colourbar_label=None, label_0=True, save_colourbar=False,
               plane_scale=1, forces=None, wind_scale=None,
               save_folder=None, save_name=None, save_dpi=600):
                  
    # Matplotlib uses an xyz coordinate system. Convert from ned:
    xs, ys, zs = ned_to_xyz(dl.ns, dl.es, dl.ds)

    # TODO Do we need the 'constrained_layout=True'
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection='3d')
    ax.computed_zorder = False

    # Label axes
    labelpad = 10
    ax.set_xlabel(r'$x_{\,[e]}$ ($m$)', labelpad=labelpad)
    ax.set_ylabel('\n\n$y_{\\,[n]}$ ($m$)', labelpad=labelpad)
    ax.set_zlabel(r'$z_{\,[-d]}$ ($m$)', labelpad=labelpad)

    # Add building projection
    b = Rectangle((131, 0), 238, 33, zorder=0, fc='lightgrey', ec='darkgrey')
    ax.add_patch(b)
    art3d.patch_2d_to_3d(b, z=190, zdir='x')

    ax.grid(visible=False)

    # ===============
    # Calculate axis limits
    x_proj_coord = 190
    y_proj_coord = np.max(ys) + 20
    z_proj_coord = 0

    x_lims = (x_proj_coord, max(xs)+20)
    y_lims = (np.min(ys)-20, y_proj_coord)
    z_lims = (z_proj_coord, max(zs)+5)

    # ===============
    # Plot trace

    # If colouring_vals isn't given, just plot a line rather than a sequence of line segments.

    if colouring_vals is None:
        ax.plot(xs, ys, zs, lw=traj_lw, color='lightseagreen', zorder=2)
    else:
        if norm is not None:    # If a norm to use has not already been provided..
            print("Using given norm - may not be zero-centred!")
        else:
            # Generate the norm
            norm, min_val, max_val = gen_norm(colouring_vals, zero_centred_norm)
        
        # Create colourmap - for colouring trace by 'colouring_vals'
        cmap = mpl.colormaps[cmap_name]
        smap = mpl.cm.ScalarMappable(norm, cmap)

        # Create empty trace, then populate.
        plane_trace = art3d.Line3DCollection([], cmap=cmap, norm=norm, lw=traj_lw)
        update_trace(plane_trace, xs, ys, zs, colouring_vals)       # Populate trace

        # Add trace to axis
        ax.add_collection(plane_trace)
        if plot_traj_centreline:
            ax.plot(xs, ys, zs, lw=0.2, color='grey', ls='--')

    # ===============
    # TODO
    # Plot aircraft, forces and wind
    if plot_interval is not None:
        step = plot_interval
        while step < dl.num_steps:
            # Get x, y and z coordinates
            # Matplotlib uses an xyz coordinate system. Convert from ned:
            n = dl.ns[step]
            e = dl.es[step]
            d = dl.ds[step]
            x, y, z = ned_to_xyz(n, e, d)

            # Plot vehicle
            Plane(ax, plane_scale, x, y, z, dl.phis[step], dl.thetas[step], dl.psis[step],
                  get_col_str(smap, colouring_vals[step]) if colouring_vals is not None else 'grey',
                  linewidth=0.5, zorder=2*dl.num_steps)
            
            # Plot forces
            if forces is not None:
                print("WARNING: force plotting is currently restricted to inertial forces only - doesn't include fictitious force.")
                # TODO Temp hack
                force_scaler_func = lambda x: 1*x # TODO HARDCODED - fix this
                # TODO Test whether forces is in the right format
                ForcePlotter(ax, force_scaler_func, arrowhead_width, plot_inds=[2]).update(n, e, d, forces[:, step, :])

            # Plot wind (dotted line)
            # TODO Use 'is not None', or just set length to 0 to hide and add logic which hides it, like I've done previously?
            if wind_scale is not None:
                wn = dl.wns[step]
                we = dl.wes[step]
                wd = dl.wds[step]
                # Convert to xyz
                wx, wy, wz = ned_to_xyz(wn, we, wd)
                
                # print("NOTE: Plotted wind field is only correct in the inertial frame!")
                ax.add_artist(Arrow3D([x, x + wind_scale*wx], [y, y + wind_scale*wy], [z, z + wind_scale*wz], mutation_scale=10, lw=0.5, arrowstyle=ArrowStyle.CurveFilledB(head_width=arrowhead_width), color='orange', linestyle='-', zorder=2*dl.num_steps+1))
            
            step += plot_interval

    # ===============
    # Plot projections
    ones = np.ones(dl.num_steps)
    ax.plot(ones*x_proj_coord, ys, zs, c=projection_colour, alpha=projection_alpha, lw=projection_lw, ls='--', zorder=1)
    ax.plot(xs, ones*y_proj_coord, zs, c=projection_colour, alpha=projection_alpha, lw=projection_lw, ls='--', zorder=1)
    ax.plot(xs, ys, ones*z_proj_coord, c=projection_colour, alpha=projection_alpha, lw=projection_lw, ls='--', zorder=1)

    # ===============
    # Plot starting and ending target positions
    wind_colour = 'orangered'
    ax.plot(*ned_to_xyz(400, 205, -20), c=wind_colour, marker='o', zorder=3*dl.num_steps, markersize=4)
    ax.plot([205, 205], [100, 100], z_lims, c=wind_colour, ls='--', zorder=3*dl.num_steps)

    # ===============
    # Add wind arrow
    head_width = 0.3
    lw = 1.5
    wind_arrow_scale = 30
    wind_anchor_x, wind_anchor_y, wind_anchor_z = (205, 100, 0)
    wind_arrow = Arrow3D([wind_anchor_x, wind_anchor_x], [wind_anchor_y, wind_anchor_y], [wind_anchor_z, wind_anchor_z], mutation_scale=10, lw=lw, arrowstyle=ArrowStyle.CurveFilledB(head_width=head_width), color='orange', zorder=0)
    ax.add_artist(wind_arrow)
    wind = np.array([-np.sin(np.radians(wind_dir_deg)), -np.cos(np.radians(wind_dir_deg)), 0])
    wind_arrow.update_3d_posns([wind_anchor_x, wind_anchor_x + wind_arrow_scale*wind[0]], [wind_anchor_y, wind_anchor_y + wind_arrow_scale*wind[1]], [wind_anchor_z, wind_anchor_z + wind_arrow_scale*wind[2]])
    
    # ===============
    # Set ticks
    ax.set_xticks((math.floor(min(xs)), math.ceil(max(xs))))
    ax.set_yticks(
        np.concatenate((
            np.array([round(v) for v in np.linspace(min(ys), max(ys), 10)])[1:-1],
            [math.floor(min(ys))],
            [math.ceil(max(ys))]
        ))
    )
    ax.set_zticks((math.floor(min(zs)), math.ceil(max(zs))))

    # ===============
    # Add colourbar
    # Use colourbar label as title
    #if colourbar_label is not None:
    #    fig.suptitle(colourbar_label,  y=0.85, fontsize='x-large', weight=500)

    if title is None:
        title = f'Wind speed: ${remove_trailing_0pt(wind_speed)}\\,ms^{{-1}}$\nWind direction: ${remove_trailing_0pt(wind_dir_deg)}^\\circ$'        
    ax.set_title(title, loc='left', y=0.7, color=text_colour)
    
    if colouring_vals is not None:
        cbax = ax.inset_axes((0.5, 0.17, 0.4, 0.04))
        if label_0:
            # FLoor and ceil to 1.d.p
            plt.colorbar(smap, cbax, label=colourbar_label, orientation='horizontal', ticks=[np.min(colouring_vals), 0, np.max(colouring_vals)], format="%.1f")
            # plt.colorbar(smap, cbax, label=colourbar_label, orientation='horizontal', ticks=[round(np.min(colouring_vals), 1), 0, round(np.max(colouring_vals), 1)])
        else:
            plt.colorbar(smap, cbax, label=colourbar_label, orientation='horizontal', ticks=[np.min(colouring_vals), np.max(colouring_vals)], format="%.1f")
            # plt.colorbar(smap, cbax, label=colourbar_label, orientation='horizontal', ticks=[round(np.min(colouring_vals), 1), round(np.max(colouring_vals), 1)])
        # plt.colorbar(smap, cbax, label=colourbar_label, orientation='horizontal', ticks=[np.min(colouring_vals), 0, np.max(colouring_vals)])
        # plt.colorbar(smap, cbax, label=colourbar_label, orientation='horizontal', ticks=[math.floor(np.min(colouring_vals)), 0, math.ceil(np.max(colouring_vals))])
    
    # ===============
    # Change the view angle
    if init_view_angles is not None:
        ax.view_init(init_view_angles[0], init_view_angles[1], init_view_angles[2])
    else:
        ax.view_init(30, -45, 0)
    
    # ===============
    # Scale plot
    # scale(ax, padding_ratio, dl.ns, dl.es, dl.ds) 
    ax.autoscale(False)
    # ax.set_box_aspect([x_lims[1] - x_lims[0], y_lims[1] - y_lims[0], z_lims[1] - z_lims[0]])
    ax.set_xlim(*x_lims)
    ax.set_ylim(*y_lims)
    ax.set_zlim(*z_lims)
    # TODO - do this custom
    ax.set_aspect('equal')
    # From critor's answer to https://stackoverflow.com/questions/11300650/how-to-scale-3d-axes
    # ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

    # ===============
    # Save
    if (save_folder is not None) and (save_name is not None):
        ## Trace and colourbar saved separately
        ## Save trace

        save_path = save_folder / f'{save_name}_trace'
        crop_and_save(fig, save_path, dpi=save_dpi)

        # Save colourbar separately
        if (colouring_vals is not None) and save_colourbar:
            cbar_fig, cbar_ax = plt.subplots()
            cbar_ax.set_aspect(1/15)
            cbar_fig.colorbar(smap, cbar_ax, label=colourbar_label, orientation='horizontal')
            crop_and_save(cbar_fig, save_folder / f'{save_name}_colourbar', dpi=save_dpi)


def create_traces(dl, aircraft_model, plot_interval, plane_scale=1, block_after_each=True, save_folder=Path('.')):
    # Calculate inertial forces
    inertial_forces = analysis_calculations.calc_inertial_forces_ned(dl)

    """
    # Groundspeed
    plot_trace(dl, analysis_calculations.calc_groundspeed(dl), plot_interval, cmap_name='viridis', zero_centred_norm=False, plot_colourbar=True, colourbar_label='Groundspeed (m/s)', plane_scale=plane_scale)
    plt.show(block=block_after_each)
    
    # Airspeed
    plot_trace(dl, dl.vas, plot_interval, cmap_name='viridis', zero_centred_norm=False, plot_colourbar=True, colourbar_label='Airspeed (m/s)', plane_scale=plane_scale)
    plt.show(block=block_after_each)
    """
    inertial_energy_change = analysis_calculations.calc_total_specific_inertial_energy_change(dl, aircraft_model)
    plot_trace(dl, inertial_energy_change, plot_interval, cmap_name='bwr', plot_colourbar=True, colourbar_label='Inertial specific power ($J s^{-1} kg^{-1}$)', plane_scale=plane_scale, save_folder=save_folder, save_name='d_ei')
    plt.show(block=block_after_each)

    # Air-relative energy change (power)
    air_rel_energy_change = analysis_calculations.calc_total_specific_air_relative_energy_change(dl, aircraft_model)
    plot_trace(dl, air_rel_energy_change, plot_interval, cmap_name='bwr', plot_colourbar=True, colourbar_label='Air-relative specific power ($J s^{-1} kg^{-1}$)', plane_scale=plane_scale, save_folder=save_folder, save_name='d_ea')
    plt.show(block=block_after_each)

    specific_air_rel_power_components = analysis_calculations.calc_specific_air_relative_energy_change_components(dl, aircraft_model)
    
    # Drag losses
    plot_trace(dl, specific_air_rel_power_components['drag'], plot_interval, cmap_name='bwr', plot_colourbar=True, colourbar_label='Specific drag power ($J s^{-1} kg^{-1}$)', plane_scale=plane_scale, save_folder=save_folder, save_name='drag_power')
    plt.show(block=block_after_each)

    # Throttle power
    plot_trace(dl, specific_air_rel_power_components['throttle'], plot_interval, cmap_name='bwr', plot_colourbar=True, colourbar_label='Specific power contribution from throttle ($J s^{-1} kg^{-1}$)', plane_scale=plane_scale, save_folder=save_folder, save_name='throttle_power')
    plt.show(block=block_after_each)
    
    # Static power
    plot_trace(dl, specific_air_rel_power_components['static'], plot_interval, cmap_name='bwr', plot_colourbar=True, colourbar_label='Specific static power ($J s^{-1} kg^{-1}$)', plane_scale=plane_scale, save_folder=save_folder, save_name='static_power')
    plt.show(block=block_after_each)

    # Dynamic (gradient) power
    plot_trace(dl, specific_air_rel_power_components['gradient'], plot_interval, cmap_name='bwr', plot_colourbar=True, colourbar_label='Specific gradient power ($J s^{-1} kg^{-1}$)', plane_scale=plane_scale, save_folder=save_folder, save_name='gradient_power')
    plt.show(block=block_after_each)

    # Aerodynamic work
    # Specific energy change due to aerodynamic force
    plot_trace(dl, analysis_calculations.calc_aerodynamic_specific_work_rate(dl, aircraft_model), plot_interval, cmap_name='bwr', plot_colourbar=True, colourbar_label='Aerodynamic specific work rate ($J s^{-1} kg^{-1}$)', plane_scale=plane_scale, save_folder=save_folder, save_name='spec_aero_work_rate')
    plt.show(block=block_after_each)

    # Aerodynamic force forward (i^b) projection
    plot_trace(dl, analysis_calculations.calc_aerodynamic_force_ib_projection(dl), plot_interval, cmap_name='bwr', plot_colourbar=True, colourbar_label='Aerodynamic force forward ($i^b$) projection (N)', plane_scale=plane_scale, save_folder=save_folder, save_name='aero_ib')
    # print(analysis_calculations.calc_aerodynamic_force_ib_projection(dl))
    plt.show(block=block_after_each)

    # Aerodynamic force inertial direction (vi unit) projection
    plot_trace(dl, analysis_calculations.calc_aerodynamic_force_vi_unit_projection(dl), plot_interval, cmap_name='bwr', plot_colourbar=True, colourbar_label='Aerodynamic force inertial direction ($\hat{v}_i$) projection (N)', plane_scale=plane_scale, save_folder=save_folder, save_name='aero_vi') # , forces=inertial_forces)
    plt.show(block=block_after_each)


# TODO Need to convert this to use the 'graph' function
def create_graphs(dl, aircraft_model):
    # Calculate values
    inertial_energy_change = analysis_calculations.calc_total_specific_inertial_energy_change(dl, aircraft_model)
    air_rel_energy_change = analysis_calculations.calc_total_specific_air_relative_energy_change(dl, aircraft_model)
    specific_air_rel_power_components = analysis_calculations.calc_specific_air_relative_energy_change_components(dl, aircraft_model)
    specific_aerodynamic_work = analysis_calculations.calc_aerodynamic_specific_work_rate(dl, aircraft_model)
    aerodynamic_force_ib_projection = analysis_calculations.calc_aerodynamic_force_ib_projection(dl)
    aerodynamic_force_vi_unit_projection = analysis_calculations.calc_aerodynamic_force_vi_unit_projection(dl)

    # Create graph
    graph_fig, graph_ax = plt.subplots(9, layout='constrained')

    # TODO [CRITICAL] Need to check the alignment on this
    graph_ax[0].plot(dl.times, inertial_energy_change)
    # TODO Need to check all of these titles and their units, re: the analysis yesterday.
    graph_ax[0].set_title("Inertial specific power ($J s^{-1} kg^{-1}$)") # Inertial energy change")

    graph_ax[1].plot(dl.times, air_rel_energy_change)
    graph_ax[1].set_title("Air-relative specific power ($J s^{-1} kg^{-1}$)")

    graph_ax[2].plot(dl.times, specific_air_rel_power_components['drag'])
    graph_ax[2].set_title("Specific drag power ($J s^{-1} kg^{-1}$)")

    graph_ax[3].plot(dl.times, specific_air_rel_power_components['throttle'])
    graph_ax[3].set_title("Specific power contribution from throttle ($J s^{-1} kg^{-1}$)")

    graph_ax[4].plot(dl.times, specific_air_rel_power_components['static'])
    graph_ax[4].set_title("Specific static power ($J s^{-1} kg^{-1}$)")
    
    graph_ax[5].plot(dl.times, specific_air_rel_power_components['gradient'])
    graph_ax[5].set_title("Specific gradient power ($J s^{-1} kg^{-1}$)")

    graph_ax[6].plot(dl.times, specific_aerodynamic_work)
    graph_ax[6].set_title("Aerodynamic specific work rate ($J s^{-1} kg^{-1}$)")

    graph_ax[7].plot(dl.times, aerodynamic_force_ib_projection)
    graph_ax[7].set_title("Aerodynamic force forward ($i^b$) projection (N)")

    graph_ax[8].plot(dl.times, aerodynamic_force_vi_unit_projection)
    graph_ax[8].set_title("Aerodynamic force inertial direction ($\hat{v}_i$) projection (N)")
    # plt.show()


# Can be used as a base class or on its own
# colours, line_styles: [list]s of colour and line_style strings. Both should contain the same number of elements.
# plot_inds: None to use all
class VecPlotterBase:
    def __init__(self, ax, colours, line_styles=None, plot_inds=None, scaler_func=lambda x: x, head_width=0.2, line_width=1, alpha=1, zorder=0):
        # Create list of vectors based on given colours
        self.vec_arrows = []
        if not line_styles:
            line_styles = ['-']*len(colours)
        for c, ls in zip(colours, line_styles):
            # CurveFilledB = "-|>"
            vec_arrow = Arrow3D([0, 0], [0, 0], [0, 0], mutation_scale=10, lw=line_width, arrowstyle=ArrowStyle.CurveFilledB(head_width=head_width), color=c, linestyle=ls, alpha=alpha)
            vec_arrow.set_zorder(zorder)
            self.vec_arrows.append(vec_arrow)
            ax.add_artist(vec_arrow)
        
        self.plot_inds = plot_inds
        self.scaler_func = scaler_func
    
    # Separated out in case there are multiple vectors to iterate through, as there are in the case of the forces.
    # vecs is a Numpy array of vector lengths - shape = (n, 3).
    def update(self, n, e, d, vecs):
        for i in range(len(vecs)):
            if (self.plot_inds is None) or (i in self.plot_inds):
                vec_data = vecs[i]
                vec_arrow = self.vec_arrows[i]

                # Hide arrows when the force is 0
                if np.linalg.norm(vec_data) == 0:
                    vec_arrow.set(visible=False)
                else:
                    vec_arrow.set(visible=True)

                ### ned -> xyz conversion has already been performed for x, y and z coordinates
                x, y, z = ned_to_xyz(n, e, d)
                vec_xyz = ned_to_xyz(*vec_data)
                vec_x_xyz, vec_y_xyz, vec_z_xyz = vec_xyz

                # Transform to spherical coordinates...
                # theta = np.arctan(vec_y_xyz / vec_x_xyz)
                theta = np.arctan2(vec_y_xyz, vec_x_xyz)
                # phi = np.arctan(vec_z_xyz / np.sqrt(vec_y_xyz**2 + vec_x_xyz**2))
                phi = np.arctan2(vec_z_xyz, np.sqrt(vec_y_xyz**2 + vec_x_xyz**2))
                
                # ...scale...
                # Scale on the norm of the vector, so that the x, y and z components are all scaled by the same amount.
                vec_len = self.scaler_func(np.linalg.norm(vec_xyz))

                # ...then transform back
                # sin(phi) = O/H
                vec_z_xyz_scaled = vec_len*np.sin(phi)
                vec_y_xyz_scaled = (vec_len*np.cos(phi))*np.sin(theta)
                vec_x_xyz_scaled = (vec_len*np.cos(phi))*np.cos(theta)

                vec_arrow.update_3d_posns([x, x + vec_x_xyz_scaled], [y, y + vec_y_xyz_scaled], [z, z + vec_z_xyz_scaled])


class WindPlotter(VecPlotterBase):
    def __init__(self, ax, **kwargs):
        super().__init__(ax, ['orange'], ['--'], **kwargs)

    # Call for single vector wind
    def update(self, n, e, d, wind):
        super().update(n, e, d, np.array([wind]))


# Used by animate(...).
class ForcePlotter(VecPlotterBase):
    # plot_inds is for only plotting a subset of the forces
    def __init__(self, ax, **kwargs):
        self.force_colours = ['r', 'purple', 'g', 'orange', 'pink', 'b']
        super().__init__(ax, self.force_colours, **kwargs)

    def update(self, n, e, d, forces):
        super().update(n, e, d, forces)


class VecPlotterSingle(VecPlotterBase):
    def __init__(self, ax, colour, **kwargs):
        super().__init__(ax, [colour], **kwargs)

    # Call for single vector vec
    def update(self, n, e, d, vec):
        super().update(n, e, d, np.array([vec]))



# Animation code
# ==============

# Animate the flight, showing the forces and the wind.
# dls - takes a list of DataLoggers
# colouring_fn only takes a DataLogger - hence it may have to be a closure over another function.
# def animate(dls, aircraft_model, save_folder=None, anim_interval_ms=20, plane_scale=1, force_scale=1, wind_scale=1, plot_building=True, show_axes=False, n_min_range=None, e_min_range=None, d_min_range=None):
def animate(dls, colouring_fn, cmap=mpl.colormaps['bwr'], save_folder=None, anim_interval_ms=20, plane_scale=1, force_scale=1, wind_scale=1, plot_building=True, show_axes=False, n_min_range=None, e_min_range=None, d_min_range=None, subsample_num=1):
    # Get the forces
    # Get the wind

    anim_fig = plt.figure()
    anim_ax = anim_fig.add_subplot(projection='3d')
    
    if not show_axes: 
        anim_ax.set_axis_off()

    # Add building
    if plot_building:
        building = Building((131, 369), (100, 190), (-33, 0))
        building.add_to_axis(anim_ax)

    anim_ax.set_xlabel('x [e]')
    anim_ax.set_ylabel('y [n]')
    anim_ax.set_zlabel('z [-d]')

    # Showing x, y and z axes at origin
    i_vec_scaling = 20
    x = Arrow3D([0, i_vec_scaling], [0, 0], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color="r")
    y = Arrow3D([0, 0], [0, i_vec_scaling], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color="g")
    z = Arrow3D([0, 0], [0, 0], [0, i_vec_scaling], mutation_scale=20, lw=1, arrowstyle="-|>", color="b")
    anim_ax.add_artist(x)
    anim_ax.add_artist(y)
    anim_ax.add_artist(z)
    
    # What do we need?
    colouring_vals = []
    xs = []
    ys = []
    zs = []
    phis = []
    thetas = []
    psis = []

    for dl in dls:
        # Subsampling after finding the colouring values, in case the colouring function values would be affected by the subsampling.
        colouring_vals.append(colouring_fn(dl)[::subsample_num])
        
        # Subsample dls
        dl = copy.deepcopy(dl)
        dl.log_arr = dl.log_arr[:, ::subsample_num]
        dl.num_steps = dl.log_arr.shape[1]
        
        # Matplotlib uses an xyz coordinate system. Convert from ned:
        x_vals, y_vals, z_vals = ned_to_xyz(dl.ns, dl.es, dl.ds)
        xs.append(x_vals)
        ys.append(y_vals)
        zs.append(z_vals)
        phis.append(dl.phis)
        thetas.append(dl.thetas)
        psis.append(dl.psis)

    # o Generate the norm (by iterating over all of the provided values)

    # o Create traces for each one, and add to plot
    # o Create planes for each one
    # o Get xyz coordinates for each one
    # o Get orientations for each one

    # Don't worry about the force and wind plotting for now

    # It would be useful to plot this for both inertial and air-relative space
    # I suppose this amount to just applying a filter to some of them to generate the air-relative coordinates

    # inertial_energy_change = analysis_calculations.calc_total_specific_inertial_energy_change(dl, aircraft_model)
    # air_rel_energy_change = analysis_calculations.calc_total_specific_air_relative_energy_change(dl, aircraft_model)
    #aero_vi_projn = analysis_calculations.calc_aerodynamic_force_vi_unit_projection(dl)
    #colouring_vals = aero_vi_projn # inertial_energy_change
    # colouring_vals = air_rel_energy_change
    # colouring_vals = dl.vas

    # Create colourmap - for colouring trace by 'colouring_vals'
    # cmap_name = 'bwr'
    # cmap = mpl.colormaps[cmap_name]

    all_colouring_vals = np.concatenate(colouring_vals) # None
    norm, _, _ = gen_norm(all_colouring_vals, True) # True)

    # Create empty trace
    traces = []
    planes = []
    for dl in dls:
        trace = art3d.Line3DCollection([], cmap=cmap, norm=norm, lw=1.5)
        # Add trace to axis
        anim_ax.add_collection(trace)
        
        # Add trace and aircraft
        traces.append(trace)
        planes.append(Plane(anim_ax, plane_scale, colour='grey'))

    """
    # Add forces
    # TODO Note that this is only inertial forces at the moment (since the plotting is in the inertial frame).
    forces = analysis_calculations.calc_inertial_forces_ned(dl)
    force_plotter = ForcePlotter(anim_ax, lambda x: force_scale*x)
    # force_plotter.update(xs[0], ys[0], zs[0], forces[:, 0, :])

    # Add wind
    wind = np.array([dl.wns, dl.wes, dl.wds]).T
    wind_plotter = WindPlotter(anim_ax, wind_scale)
    """

    # Scale
    all_ns = np.concatenate([dl.ns for dl in dls])
    all_es = np.concatenate([dl.es for dl in dls])
    all_ds = np.concatenate([dl.ds for dl in dls])
    x_lim, y_lim, z_lim = scale(anim_ax, 0.1, all_ns, all_es, all_ds, n_min_range, e_min_range, d_min_range)

    # Plot ground
    # TODO This seems like a messy way of doing this
    ground_x, ground_y = np.meshgrid(x_lim, y_lim)
    ground_z = np.zeros(ground_x.shape)
    anim_ax.plot_surface(ground_x, ground_y, ground_z, color='gray', alpha=0.1)

    # Surely this is just getting the line collection and plotting it a bit at a time,
    # with a plane and the force arrows.

    def init():
        for trace in traces:
            trace.set_segments([])

    def update_aircraft(i, frame, dl):
        # Since some trajectories are longer than others
        if frame < len(xs[i]):
            # print(f"{frame}  {len(xs[i])}")
            # Code from: https://stackoverflow.com/questions/21077477/animate-a-line-with-different-colors
            part_xs = xs[i][:frame]
            part_ys = ys[i][:frame]
            part_zs = zs[i][:frame]
            update_trace(traces[i], part_xs, part_ys, part_zs, colouring_vals[i])
    
            planes[i].set_pose(xs[i][frame], ys[i][frame], zs[i][frame], phis[i][frame], thetas[i][frame], psis[i][frame])
            #force_plotter.update(xs[frame], ys[frame], zs[frame], forces[:, frame, :])
            #wind_plotter.update(xs[frame], ys[frame], zs[frame], wind[frame])

    def animate(frame):
        for i, dl in enumerate(dls):
            update_aircraft(i, frame, dl)

    anim = FuncAnimation(anim_fig, animate, frames=dl.num_steps, init_func=init, interval=anim_interval_ms, repeat_delay=1000)

    # Save
    if save_folder is not None:
        print("Saving")
        # anim.save(save_folder / 'trace.gif', dpi=300)
        anim.save(filename=save_folder / 'trace.mp4', writer=FFMpegWriter(fps=5)) # 'ffmpeg')
        print("Done")

    plt.show()


# Temp
# Animate the flight, showing the forces and the wind.
# def animate_for_arthur(dl, aircraft_model, colouring_vals, save_folder=None, anim_interval_ms=20, plane_scale=1, force_scale=1, wind_scale=1, plot_building=True, show_axes=False, n_min_range=None, e_min_range=None, d_min_range=None, subsample_num=1):
def animate_for_arthur(dl, colouring_vals, save_folder=None, anim_interval_ms=20, plane_scale=1, force_scale=1, wind_scale=1, n_lims=(-np.inf, np.inf), plot_building=True, show_axes=False, n_min_range=None, e_min_range=None, d_min_range=None, subsample_num=1):
    # Get the forces
    # Get the wind

    anim_fig = plt.figure()
    anim_ax = anim_fig.add_subplot(projection='3d')
    
    if not show_axes: 
        anim_ax.set_axis_off()

    # Add building
    if plot_building:
        building = Building((131, 369), (100, 190), (-33, 0))
        # building = Building((131, 369), (160, 190), (-33, 0))
        building.add_to_axis(anim_ax)

    anim_ax.set_xlabel('x [e]')
    anim_ax.set_ylabel('y [n]')
    anim_ax.set_zlabel('z [-d]')

    # Showing x, y and z axes at origin
    i_vec_scaling = 20
    x = Arrow3D([0, i_vec_scaling], [0, 0], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color="r")
    y = Arrow3D([0, 0], [0, i_vec_scaling], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color="g")
    z = Arrow3D([0, 0], [0, 0], [0, i_vec_scaling], mutation_scale=20, lw=1, arrowstyle="-|>", color="b")
    anim_ax.add_artist(x)
    anim_ax.add_artist(y)
    anim_ax.add_artist(z)

    # Restrict to region (just n for now)
    dl = copy.deepcopy(dl)
    restrict_inds = np.where((dl.ns >= n_lims[0]) & (dl.ns < n_lims[1]))[0]
    dl.log_arr = dl.log_arr[:, restrict_inds]
    dl.num_steps = dl.log_arr.shape[1]
    colouring_vals = colouring_vals[restrict_inds]

    # Subsample dl
    dl.log_arr = dl.log_arr[:, ::subsample_num]
    dl.num_steps = dl.log_arr.shape[1]
    colouring_vals = colouring_vals[::subsample_num]

    # Matplotlib uses an xyz coordinate system. Convert from ned:
    xs, ys, zs = ned_to_xyz(dl.ns, dl.es, dl.ds)

    # inertial_energy_change = analysis_calculations.calc_total_specific_inertial_energy_change(dl, aircraft_model)
    # air_rel_energy_change = analysis_calculations.calc_total_specific_air_relative_energy_change(dl, aircraft_model)
    #colouring_vals = colouring_fn(dl, aircraft_model) # air_rel_energy_change # inertial_energy_change
    # colouring_vals = air_rel_energy_change
    # colouring_vals = dl.vas

    # Create colourmap - for colouring trace by 'colouring_vals'
    cmap_name = 'bwr'
    cmap = mpl.colormaps[cmap_name]
    norm, min_val, max_val = gen_norm(colouring_vals, True)

    # Create empty trace
    plane_trace = art3d.Line3DCollection([], cmap=cmap, norm=norm, lw=2.0)
    # Add trace to axis
    anim_ax.add_collection(plane_trace)
 
    # Add aircraft
    plane = Plane(anim_ax, plane_scale, colour='grey')
    
    # Add forces
    # TODO Note that this is only inertial forces at the moment (since the plotting is in the inertial frame).
    # Individual forces
    forces = analysis_calculations.calc_inertial_forces_ned(dl)
    force_plotter = ForcePlotter(anim_ax, scaler_func=lambda x: force_scale*x)
    # Total aerodynamic force
    f_aero_ned = forces[0] + forces[1] + forces[2]
    f_aero_plotter = VecPlotterSingle(anim_ax, 'green')

    # Add wind
    wind = np.array([dl.wns, dl.wes, dl.wds]).T
    wind_plotter = WindPlotter(anim_ax) # , wind_scale, head_width=0.3, line_width=1.25)

    # And inertial speed vector
    # Get u, v and w
    # Calculate normalised vi vectors (direction vectors of inertial travel)
    vi_b_vecs = np.array([dl.us, dl.vs, dl.ws]).T
    # Convert body frame velocity to inertial frame
    r_bi = calc_body_to_inertial(dl.phis, dl.thetas, dl.psis)
    vi_vecs_ned = np.squeeze(np.matmul(r_bi, vi_b_vecs[:, :, np.newaxis]))
    #vi_plotter = VecPlotterSingle(anim_ax, 'indigo')

    # Add airspeed vector
    va_vecs_ned = analysis_calculations.calc_va_vecs_i_ned(dl)
    #va_plotter = VecPlotterSingle(anim_ax, 'b')
    # Plot the projection onto the body ik plane
    # Resolve va_vector in body frame
    va_vecs_b_ik = np.matmul(np.transpose(r_bi, (0, 2, 1)), va_vecs_ned[:, :, np.newaxis])
    # Remove j component
    va_vecs_b_ik[:, 1, :] = 0
    # Rotate back into inertial frame
    va_vecs_b_ik_ned = np.squeeze(np.matmul(r_bi, va_vecs_b_ik))
    va_b_ik_plotter = VecPlotterSingle(anim_ax, 'purple', line_styles=['--'])

    # Plot body i vector
    ib_vecs = np.array([np.ones(dl.num_steps), np.zeros(dl.num_steps), np.zeros(dl.num_steps)]).T
    # Convert body frame velocity to inertial frame
    ib_vecs_ned = np.squeeze(np.matmul(r_bi, ib_vecs[:, :, np.newaxis]))
    ib_plotter = VecPlotterSingle(anim_ax, 'purple')

    # What's the other thing that I should plot?
    # Use the same colours, just different line styles?
    # Plot the aerodynamic force parallel and perpendicular

    # Scale
    x_lim, y_lim, z_lim = scale(anim_ax, 0.01, dl.ns, dl.es, dl.ds, n_min_range, e_min_range, d_min_range)

    # Plot ground
    # TODO This seems like a messy way of doing this
    ground_x, ground_y = np.meshgrid(x_lim, y_lim)
    ground_z = np.zeros(ground_x.shape)
    anim_ax.plot_surface(ground_x, ground_y, ground_z, color='gray', alpha=0.1)

    # Surely this is just getting the line collection and plotting it a bit at a time,
    # with a plane and the force arrows.

    ground_projn_trace = art3d.Line3DCollection([], lw=1, cmap='grey', alpha=0.1)
    anim_ax.add_collection(ground_projn_trace)

    def init():
        plane_trace.set_segments([])

    def animate(frame):
        # Code from: https://stackoverflow.com/questions/21077477/animate-a-line-with-different-colors
        part_xs = xs[:frame]
        part_ys = ys[:frame]
        part_zs = zs[:frame]
        update_trace(plane_trace, part_xs, part_ys, part_zs, colouring_vals)
        update_trace(ground_projn_trace, part_xs, part_ys, np.zeros(len(part_zs)), np.ones(len(colouring_vals)))

        plane.set_pose(xs[frame], ys[frame], zs[frame], dl.phis[frame], dl.thetas[frame], dl.psis[frame])
        force_plotter.update(dl.ns[frame], dl.es[frame], dl.ds[frame], forces[:, frame, :])
        wind_plotter.update(dl.ns[frame], dl.es[frame], dl.ds[frame], wind[frame])
        #f_aero_plotter.update(dl.ns[frame], dl.es[frame], dl.ds[frame], f_aero_ned[frame])
        #ib_plotter.update(dl.ns[frame], dl.es[frame], dl.ds[frame], ib_vecs_ned[frame])
        #vi_plotter.update(dl.ns[frame], dl.es[frame], dl.ds[frame], vi_vecs_ned[frame])
        #va_b_ik_plotter.update(dl.ns[frame], dl.es[frame], dl.ds[frame], va_vecs_b_ik_ned[frame])
        #va_plotter.update(dl.ns[frame], dl.es[frame], dl.ds[frame], va_vecs_ned[frame])

    anim = FuncAnimation(anim_fig, animate, frames=dl.num_steps, init_func=init, interval=anim_interval_ms, repeat_delay=1000)

    # Save
    from matplotlib import animation
    if save_folder is not None:
        anim.save(save_folder / 'trace.mp4', dpi=300, fps=20)

    plt.show()


class BuildingBackAnimatorBase:
    def __init__(self):
        self.surface_proj_col = 'turquoise' # 'crimson' # 'grey'
        self.surface_proj_alpha = 0.03
        self.building_depth = 30
        self.building_face_col = 'gainsboro'
        self.building_edge_col = 'darkgrey'

    # view = 'top', 'side', None (normal)
    def animate(self, dl, colouring_vals, title=None, save_folder=None, name_suffix="",
                            anim_interval_ms=20, plot_arrows = ['wind'],
                            plane_scale=1, force_scale=1, wind_scale=1, viva_scale=1,
                            n_lims=(-np.inf, np.inf), subsample_num=1, save_dpi=600):
        
        # ===============
        # Create axes
        anim_fig = plt.figure(figsize=(14, 8))
        anim_ax = anim_fig.add_subplot(projection='3d')
        anim_ax.computed_zorder = False

        # ===============
        # Restrict to region (just n for now)
        dl = copy.deepcopy(dl)
        restrict_inds = np.where((dl.ns >= n_lims[0]) & (dl.ns < n_lims[1]))[0]
        dl.log_arr = dl.log_arr[:, restrict_inds]
        dl.num_steps = dl.log_arr.shape[1]
        colouring_vals = colouring_vals[restrict_inds]

        # Subsample dl
        dl.log_arr = dl.log_arr[:, ::subsample_num]
        dl.num_steps = dl.log_arr.shape[1]
        colouring_vals = colouring_vals[::subsample_num]

        # Matplotlib uses an xyz coordinate system. Convert from ned:
        xs, ys, zs = ned_to_xyz(dl.ns, dl.es, dl.ds)

        # ===============
        # Calculate axis limits
        x_proj_coord = 190
        y_proj_coord = np.max(ys) + 20
        z_proj_coord = 0

        x_lims = (x_proj_coord-self.building_depth, np.max(xs)+20)
        y_lims = (np.min(ys)-20, y_proj_coord)
        z_lims = (z_proj_coord, np.max(zs)+5)

        ## ===============
        ## Plot ground
        ## TODO This seems like a messy way of doing this
        #ground_x, ground_y = np.meshgrid(x_lims, y_lims)
        #ground_z = np.zeros(ground_x.shape)
        #anim_ax.plot_surface(ground_x, ground_y, ground_z, color=surface_proj_col, alpha=surface_proj_alpha)
        #

        # ===============
        # Plot side surface
        # TODO This seems like a messy way of doing this
        ground_y, ground_z = np.meshgrid(y_lims, z_lims)
        ground_x = np.ones(ground_y.shape)*x_proj_coord
        anim_ax.plot_surface(ground_x, ground_y, ground_z, color=self.surface_proj_col, alpha=self.surface_proj_alpha, zorder=1)

        ## ===============
        ## Add building
        #if plot_building:
        #    # building = Building((131, 369), (100, 190), (-33, 0))
        #    building = Building((131, 369), (160, 190), (-33, 0))
        #    building.add_to_axis(anim_ax)
        
        #if not show_axes:
        #    anim_ax.set_axis_off()

        # ===============
        # Set ticks
        anim_ax.set_xticks((math.floor(min(xs)), math.ceil(max(xs))))
        anim_ax.set_yticks(
            np.concatenate((
                np.array([round(v) for v in np.linspace(min(ys), max(ys), 10)])[1:-1],
                [math.floor(min(ys))],
                [math.ceil(max(ys))]
            ))
        )
        anim_ax.set_zticks((math.floor(min(zs)), math.ceil(max(zs))))
        
        anim_ax.grid(visible=False)

        # ===============
        # Create colourmap - for colouring trace by 'colouring_vals'
        cmap_name = 'bwr'
        cmap = mpl.colormaps[cmap_name]
        norm, min_val, max_val = gen_norm(colouring_vals, True)

        # Create empty trace
        plane_trace = art3d.Line3DCollection([], cmap=cmap, norm=norm, lw=2.0, zorder=3)
        # Add trace to axis
        anim_ax.add_collection(plane_trace)

        # Surely this is just getting the line collection and plotting it a bit at a time,
        # with a plane and the force arrows.

        # Add projection traces
        xy_projn_trace = art3d.Line3DCollection([], lw=0.7, cmap='grey', alpha=0.07, zorder=2)
        yz_projn_trace = art3d.Line3DCollection([], lw=0.7, cmap='grey', alpha=0.07, zorder=2)
        xz_projn_trace = art3d.Line3DCollection([], lw=0.7, cmap='grey', alpha=0.07, zorder=2)
        anim_ax.add_collection(xy_projn_trace)
        anim_ax.add_collection(yz_projn_trace)
        anim_ax.add_collection(xz_projn_trace)

        # Add aircraft
        plane = Plane(anim_ax, plane_scale, colour='grey', zorder=4)

        # ===============
        # Plot force, velocity and wind arrows

        # =====
        # Calculate

        # Get/calculate forces
        forces = analysis_calculations.calc_inertial_forces_ned(dl)
        # Total aerodynamic force
        f_aero_ned = forces[0] + forces[1] + forces[2]
        # Aero vi projection
        f_aero_vi_proj_ned = analysis_calculations.calc_aerodynamic_force_vi_unit_projection_vecs_ned(dl)

        # Get wind
        wind = np.array([dl.wns, dl.wes, dl.wds]).T

        # Convert body frame velocity to inertial frame
        r_bi = calc_body_to_inertial(dl.phis, dl.thetas, dl.psis)

        # Calculate body i vector
        ib_vecs = np.array([np.ones(dl.num_steps), np.zeros(dl.num_steps), np.zeros(dl.num_steps)]).T
        # Convert body frame velocity to inertial frame
        ib_vecs_ned = np.squeeze(np.matmul(r_bi, ib_vecs[:, :, np.newaxis]))

        # Calculate inertial speed vector
        # Get u, v and w
        # Calculate normalised vi vectors (direction vectors of inertial travel)
        vi_b_vecs = np.array([dl.us, dl.vs, dl.ws]).T
        # Convert body frame velocity to inertial frame
        vi_vecs_ned = np.squeeze(np.matmul(r_bi, vi_b_vecs[:, :, np.newaxis]))

        # Calculate va vectors
        va_vecs_ned = analysis_calculations.calc_va_vecs_i_ned(dl)

        # # Plot the projection onto the body ik plane
        # # Resolve va_vector in body frame
        # va_vecs_b_ik = np.matmul(np.transpose(r_bi, (0, 2, 1)), va_vecs_ned[:, :, np.newaxis])
        # # Remove j component
        # va_vecs_b_ik[:, 1, :] = 0
        # # Rotate back into inertial frame
        # va_vecs_b_ik_ned = np.squeeze(np.matmul(r_bi, va_vecs_b_ik))
        # va_b_ik_plotter = VecPlotterSingle(anim_ax, 'purple', line_styles=['--'], zorder=5)

        # =====
        # Plot

        # Add forces
        # TODO Note that this is only inertial forces at the moment (since the plotting is in the inertial frame).
        # Individual forces
        if 'all_forces' in plot_arrows:
            # scaler_func=lambda x: force_scale*x
            # scaler_func=lambda x: force_scale*np.sign(x)*np.log(np.abs(x))
            # force_plotter = ForcePlotter(anim_ax, scaler_func=lambda x: force_scale*x, zorder=5)
            force_plotter = ForcePlotter(anim_ax, scaler_func=lambda vec_norm: force_scale*np.sqrt(vec_norm), zorder=5)
        if 'lift' in plot_arrows:
            lift_plotter = VecPlotterSingle(anim_ax, 'g', scaler_func=lambda vec_norm: force_scale*np.sqrt(vec_norm), zorder=5)
        if 'aero_force' in plot_arrows:
            # f_aero_plotter = VecPlotterSingle(anim_ax, 'lightcoral', scaler_func=lambda x: force_scale*x, zorder=5)
            f_aero_plotter = VecPlotterSingle(anim_ax, 'indigo', line_width=1.5, scaler_func=lambda vec_norm: force_scale*np.sqrt(vec_norm), zorder=5)
        if 'aero_vi_force' in plot_arrows:
            # f_aero_vi_proj_plotter = VecPlotterSingle(anim_ax, 'lightcoral', line_styles=['--'], scaler_func=lambda x: force_scale*x, zorder=5)
            f_aero_vi_proj_plotter = VecPlotterSingle(anim_ax, 'indigo', line_styles=['--'], scaler_func=lambda vec_norm: force_scale*np.sqrt(vec_norm), zorder=6)

        # Add wind
        if 'wind' in plot_arrows:
            wind_plotter = WindPlotter(anim_ax, scaler_func=lambda x: wind_scale*x, zorder=5) # , wind_scale, head_width=0.3, line_width=1.25)
        
        # Add wind triangle (wind vector translated to form triangle)
        if 'wind_triangle' in plot_arrows:
            wind_tri_plotter = WindPlotter(anim_ax, scaler_func=lambda x: viva_scale*x, zorder=5) # , wind_scale, head_width=0.3, line_width=1.25)
        
        # Plot body i vector
        if 'ib' in plot_arrows:
            ib_plotter = VecPlotterSingle(anim_ax, 'purple', zorder=5)

        # Plot groundspeed vector
        if 'vi' in plot_arrows:
            vi_plotter = VecPlotterSingle(anim_ax, 'teal', scaler_func=lambda x: viva_scale*x, zorder=5)
        
        # Plot airspeed vector
        if 'va' in plot_arrows:
            va_plotter = VecPlotterSingle(anim_ax, 'mediumturquoise', scaler_func=lambda x: viva_scale*x, zorder=5)
        
        """
        # Create inset axis for aircraft
        plane_ax = anim_ax.inset_axes((0.5, 0.17, 0.4, 0.2), projection='3d')
        plane_ax.set_axis_off()
        plane_ax_plane = Plane(plane_ax, 1, colour='grey', zorder=0)
        plane_ax.set_aspect('equal')

        vi_plotter_plane_ax = VecPlotterSingle(plane_ax, 'indigo', zorder=5)
        va_plotter_plane_ax = VecPlotterSingle(plane_ax, 'b', zorder=5)
        """

        # What's the other thing that I should plot?
        # Use the same colours, just different line styles?
        # Plot the aerodynamic force parallel and perpendicular

        # elif view == 'side':
        #     anim_ax.view_init(0, 0, 0)
        # else:
        #     raise Exception("An invalid condition occurred")
        
        # Scale
        # x_lim, y_lim, z_lim = scale(anim_ax, 0.01, dl.ns, dl.es, dl.ds, n_min_range, e_min_range, d_min_range)

        
        # ===============
        # Configure (plot) building and labels - depends on view angle, hence subclass.
        self.setup_building_and_axes(anim_ax, colouring_vals, cmap, norm, plot_arrows, title)

        anim_ax.set_xlim3d(*np.sort(x_lims))
        anim_ax.set_ylim3d(*np.sort(y_lims))
        anim_ax.set_zlim3d(*np.sort(z_lims))
        anim_ax.set_aspect('equal')
        
        def animate_init():
            plane_trace.set_segments([])

        def create_animation(frame):
            # Code from: https://stackoverflow.com/questions/21077477/animate-a-line-with-different-colors
            part_xs = xs[:frame]
            part_ys = ys[:frame]
            part_zs = zs[:frame]
            update_trace(plane_trace, part_xs, part_ys, part_zs, colouring_vals)
            update_trace(xy_projn_trace, part_xs, part_ys, np.ones(len(part_zs))*z_proj_coord, np.ones(len(colouring_vals)))
            update_trace(yz_projn_trace, np.ones(len(part_xs))*x_proj_coord, part_ys, part_zs, np.ones(len(colouring_vals)))
            update_trace(xz_projn_trace, part_xs, np.ones(len(part_ys))*y_proj_coord, part_zs, np.ones(len(colouring_vals)))
            
            plane.set_pose(xs[frame], ys[frame], zs[frame], dl.phis[frame], dl.thetas[frame], dl.psis[frame])

            if 'all_forces' in plot_arrows:
                force_plotter.update(dl.ns[frame], dl.es[frame], dl.ds[frame], forces[:, frame, :])
            if 'lift' in plot_arrows:
                lift_plotter.update(dl.ns[frame], dl.es[frame], dl.ds[frame], forces[2, frame, :])
            if 'aero_force' in plot_arrows:
                f_aero_plotter.update(dl.ns[frame], dl.es[frame], dl.ds[frame], f_aero_ned[frame])
            if 'aero_vi_force' in plot_arrows:
                f_aero_vi_proj_plotter.update(dl.ns[frame], dl.es[frame], dl.ds[frame], f_aero_vi_proj_ned[frame])
            
            if 'wind' in plot_arrows:
                wind_plotter.update(dl.ns[frame], dl.es[frame], dl.ds[frame], wind[frame])
            if 'wind_triangle' in plot_arrows:
                va_n, va_e, va_d = viva_scale*va_vecs_ned[frame]
                wind_tri_plotter.update(dl.ns[frame] + va_n, dl.es[frame] + va_e, dl.ds[frame] + va_d, wind[frame])
            
            if 'ib' in plot_arrows:
                ib_plotter.update(dl.ns[frame], dl.es[frame], dl.ds[frame], ib_vecs_ned[frame])
            
            if 'vi' in plot_arrows:
                vi_plotter.update(dl.ns[frame], dl.es[frame], dl.ds[frame], vi_vecs_ned[frame])
            if 'va' in plot_arrows:
                va_plotter.update(dl.ns[frame], dl.es[frame], dl.ds[frame], va_vecs_ned[frame])

            # plane_ax_plane.set_pose(0, 0, 0, dl.phis[frame], dl.thetas[frame], dl.psis[frame])
            # va_plotter_plane_ax.update(0, 0, 0, va_vecs_ned[frame])
            # vi_plotter_plane_ax.update(0, 0, 0, vi_vecs_ned[frame])
            
            #va_b_ik_plotter.update(dl.ns[frame], dl.es[frame], dl.ds[frame], va_vecs_b_ik_ned[frame])
        
        anim = FuncAnimation(anim_fig, create_animation, frames=dl.num_steps, init_func=animate_init, interval=anim_interval_ms, repeat_delay=1000)

        # Save
        if save_folder is not None:
            anim.save(save_folder / f'trace{name_suffix}.mp4', dpi=save_dpi)
            # anim.save(save_folder / 'trace.gif', dpi=30)
        
        print(f"Animation {name_suffix} complete")
        
        return anim

        # plt.show()
        
    def setup_building_and_axes(self, anim_ax, colouring_vals, cmap, norm, plot_arrows, title):
        # Label axes
        labelpad = 10
        anim_ax.set_xlabel(r'$x_{\,[e]}$ ($m$)', labelpad=labelpad)
        anim_ax.set_ylabel('\n\n$y_{\\,[n]}$ ($m$)', labelpad=labelpad)
        anim_ax.set_zlabel(r'$z_{\,[-d]}$ ($m$)', labelpad=labelpad)
        
        building_long_side = Rectangle((131, 0), 238, 33, zorder=0, fc=self.building_face_col, ec=self.building_edge_col)
        anim_ax.add_patch(building_long_side)
        art3d.patch_2d_to_3d(building_long_side, z=190, zdir='x')

        building_short_side = Rectangle((190-self.building_depth, 0), self.building_depth, 33, zorder=0, fc=self.building_face_col, ec=self.building_edge_col)
        anim_ax.add_patch(building_short_side)
        art3d.patch_2d_to_3d(building_short_side, z=131, zdir='y')

        building_top = Rectangle((190-self.building_depth, 131), self.building_depth, 238, zorder=0, fc=self.building_face_col, ec=self.building_edge_col)
        anim_ax.add_patch(building_top)
        art3d.patch_2d_to_3d(building_top, z=33, zdir='z')

        cbax = anim_ax.inset_axes((0.6, 0.2, 0.4, 0.03))    
        plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cbax, orientation='horizontal', ticks=[np.min(colouring_vals), 0, np.max(colouring_vals)])

        if len(set(['all_forces', 'aero_force', 'aero_vi_force']) & set(plot_arrows)) > 0:
            anim_ax.text2D(0, 0.1, "Force lengths are sqrt scaled", c='dimgrey', transform=anim_ax.transAxes)
        
        # title = f'Wind speed: ${remove_trailing_0pt(wind_speed)}\\,ms^{{-1}}$\nWind direction: ${remove_trailing_0pt(wind_dir_deg)}^\\circ$'        
        anim_ax.set_title(title, loc='left', y=0.73, color='dimgrey', fontsize=16)

        anim_ax.view_init(30, -45, 0)

class BuildingBackAnimatorPlain(BuildingBackAnimatorBase):
    def setup_building_and_axes(self, anim_ax, colouring_vals, cmap, norm, plot_arrows, title):
        # Label axes
        labelpad = 10
        anim_ax.set_xlabel(r'$x_{\,[e]}$ ($m$)', labelpad=labelpad)
        anim_ax.set_ylabel('\n\n$y_{\\,[n]}$ ($m$)', labelpad=labelpad)
        anim_ax.set_zlabel(r'$z_{\,[-d]}$ ($m$)', labelpad=labelpad)
        
        building_long_side = Rectangle((131, 0), 238, 33, zorder=0, fc=self.building_face_col, ec=self.building_edge_col)
        anim_ax.add_patch(building_long_side)
        art3d.patch_2d_to_3d(building_long_side, z=190, zdir='x')

        building_short_side = Rectangle((190-self.building_depth, 0), self.building_depth, 33, zorder=0, fc=self.building_face_col, ec=self.building_edge_col)
        anim_ax.add_patch(building_short_side)
        art3d.patch_2d_to_3d(building_short_side, z=131, zdir='y')

        building_top = Rectangle((190-self.building_depth, 131), self.building_depth, 238, zorder=0, fc=self.building_face_col, ec=self.building_edge_col)
        anim_ax.add_patch(building_top)
        art3d.patch_2d_to_3d(building_top, z=33, zdir='z')

        if len(set(['all_forces', 'aero_force', 'aero_vi_force']) & set(plot_arrows)) > 0:
            anim_ax.text2D(0, 0.1, "Force lengths are sqrt scaled", c='dimgrey', transform=anim_ax.transAxes)

        anim_ax.view_init(30, -45, 0)

class BuildingBackAnimatorTop(BuildingBackAnimatorBase):
    def setup_building_and_axes(self, anim_ax, colouring_vals, cmap, norm, plot_arrows, title):
        # Label axes
        labelpad = 10
        anim_ax.set_xlabel(r'$x_{\,[e]}$ ($m$)', labelpad=labelpad)
        anim_ax.set_ylabel('\n\n$y_{\\,[n]}$ ($m$)', labelpad=labelpad)
        anim_ax.get_zaxis().set_visible(False)
        anim_ax.set_zticks([])

        building_top = Rectangle((190-self.building_depth, 131), self.building_depth, 238, zorder=0, fc=self.building_face_col, ec=self.building_edge_col)
        anim_ax.add_patch(building_top)
        art3d.patch_2d_to_3d(building_top, z=0, zdir='z')

        anim_ax.set_title("Top down", loc='left', y=0.65, color='dimgrey', fontsize=16)

        # 89.99 < 90 so that the axis labels are on the correct side
        anim_ax.view_init(89.9, 0, 0)

class BuildingBackAnimatorSide(BuildingBackAnimatorBase):
    def setup_building_and_axes(self, anim_ax, colouring_vals, cmap, norm, plot_arrows, title):
        # Label axes
        labelpad = 10
        anim_ax.get_xaxis().set_visible(False)
        anim_ax.set_xticks([])
        anim_ax.set_ylabel('\n\n$y_{\\,[n]}$ ($m$)', labelpad=labelpad)
        anim_ax.set_zlabel(r'$z_{\,[-d]}$ ($m$)', labelpad=labelpad)

        building_long_side = Rectangle((131, 0), 238, 33, zorder=0, fc=self.building_face_col, ec=self.building_edge_col)
        anim_ax.add_patch(building_long_side)
        art3d.patch_2d_to_3d(building_long_side, z=190, zdir='x')

        anim_ax.set_title("Side on", loc='left', y=0.6, color='dimgrey', fontsize=16)

        # 89.99 < 90 so that the axis labels are on the correct side
        anim_ax.view_init(0, 0, 0)


def animate_air_rel(dl, aircraft_model, plane_scale=1, force_scale=1, wind_scale=1,
                    anim_interval_ms=20, subsample_num=1, savefolder=None, save_dpi=200):
    
    # [o]zorder
    # [o]Wind arrow
    # Solid arrow for the wind
    # [o]Side projections
    # [o]Colourbar

    surface_proj_col = 'turquoise' # 'crimson' # 'grey'
    building_face_col = 'gainsboro'
    building_edge_col = 'darkgrey'
    surface_proj_alpha = 0.03 # 1
    building_alpha = 0.3 # 1

    # ===============
    # Construct position information

    # Construct a vector/array of the inertial positions
    pis = np.array([dl.ns, dl.es, dl.ds]).T
    # Find air-relative positions
    pas = analysis_calculations.calc_air_rel_positions(dl)

    pas_xyz = ned_to_xyz(*pas.T).T
    pis_xyz = ned_to_xyz(*pis.T).T

    # Calculate colouring values
    air_rel_energy_change = analysis_calculations.calc_total_specific_air_relative_energy_change(dl, aircraft_model)
    inertial_energy_change = analysis_calculations.calc_total_specific_inertial_energy_change(dl, aircraft_model)

    # Calculate forces
    forces_i = analysis_calculations.calc_inertial_forces_ned(dl)
    forces_a = analysis_calculations.calc_air_relative_forces_ned(dl)
    
    # ===============
    # Subsample
    dl = copy.deepcopy(dl)
    dl.log_arr = dl.log_arr[:, ::subsample_num]
    dl.num_steps = dl.log_arr.shape[1]
    pas = pas[::subsample_num]
    pis = pis[::subsample_num]
    pas_xyz = pas_xyz[::subsample_num]
    pis_xyz = pis_xyz[::subsample_num]
    air_rel_energy_change = air_rel_energy_change[::subsample_num]
    inertial_energy_change = inertial_energy_change[::subsample_num]
    forces_i = forces_i[::subsample_num]
    forces_a = forces_a[::subsample_num]

    # ===============
    # Set up figure
    anim_fig = plt.figure(figsize=(12, 10))
    anim_ax = anim_fig.add_subplot(projection='3d')
    anim_ax.computed_zorder = False

    # Label axes
    labelpad = 10
    anim_ax.set_xlabel(r'$x_{\,[e]}$ ($m$)', labelpad=labelpad)
    anim_ax.set_ylabel('\n\n$y_{\\,[n]}$ ($m$)', labelpad=labelpad)
    anim_ax.set_zlabel(r'$z_{\,[-d]}$ ($m$)', labelpad=labelpad)

    # ===============
    # Calculate axis limits
    
    # Find maxima and minima across both inertial and air-relative data
    all_xs = np.concatenate((pas_xyz.T[0], pis_xyz.T[0]))
    all_ys = np.concatenate((pas_xyz.T[1], pis_xyz.T[1]))
    all_zs = np.concatenate((pas_xyz.T[2], pis_xyz.T[2]))

    x_data_lims = (np.min(all_xs), np.max(all_xs))
    y_data_lims = (np.min(all_ys), np.max(all_ys))
    z_data_lims = (np.min(all_zs), np.max(all_zs))

    x_lims = (np.min(all_xs)-20, np.max(all_xs)+20)
    y_lims = (np.min(all_ys)-10, np.max(all_ys)+10)
    z_lims = (np.min(np.concatenate((all_zs, [0]))), np.max(all_zs)+10)
    
    x_proj_coord = x_lims[0]
    y_proj_coord = y_lims[1]
    z_proj_coord = z_lims[0]

    # ===============
    # Set ticks
    anim_ax.set_xticks([round(v) for v in x_data_lims])
    anim_ax.set_yticks([round(v) for v in y_data_lims])
    anim_ax.set_zticks([round(v) for v in z_data_lims])

    anim_ax.grid(visible=False)

    # ===============
    # Plot side surface and building projection
    # TODO This seems like a messy way of doing this
    ground_y, ground_z = np.meshgrid(y_lims, z_lims)
    ground_x = np.ones(ground_y.shape)*190
    anim_ax.plot_surface(ground_x, ground_y, ground_z, color=surface_proj_col, alpha=surface_proj_alpha, zorder=2)

    building_long_side = Rectangle((131, 0), 238, 33, zorder=2, fc=building_face_col, ec=building_edge_col, alpha=building_alpha)
    anim_ax.add_patch(building_long_side)
    art3d.patch_2d_to_3d(building_long_side, z=190, zdir='x')

    # ============================
    # Create lines

    # Create colourmap - for colouring trace by 'colouring_vals'
    all_colouring_vals = np.concatenate((inertial_energy_change, air_rel_energy_change))
    cmap_name = 'bwr'
    cmap = mpl.colormaps[cmap_name]
    norm, min_val, max_val = gen_norm(all_colouring_vals, True)

    # Create empty traces
    plane_trace_a = art3d.Line3DCollection([], cmap=cmap, norm=norm, lw=2, zorder=1)
    plane_trace_i = art3d.Line3DCollection([], cmap=cmap, norm=norm, lw=2, zorder=3)
    # Add traces to axis
    anim_ax.add_collection(plane_trace_a) # , label='a')
    anim_ax.add_collection(plane_trace_i) # , label='i')

    # Plot projection traces
    x_projn_trace_i, = anim_ax.plot([], [], [], lw=1, c='black', alpha=0.1, zorder=0)
    x_projn_trace_a, = anim_ax.plot([], [], [], lw=1, c='black', alpha=0.1, ls='--', zorder=0)
    y_projn_trace_i, = anim_ax.plot([], [], [], lw=1, c='black', alpha=0.1, zorder=0)
    y_projn_trace_a, = anim_ax.plot([], [], [], lw=1, c='black', alpha=0.1, ls='--', zorder=0)
    z_projn_trace_i, = anim_ax.plot([], [], [], lw=1, c='black', alpha=0.1, zorder=0)
    z_projn_trace_a, = anim_ax.plot([], [], [], lw=1, c='black', alpha=0.1, ls='--', zorder=0)

    # Plot aircraft
    plane_a = Plane(anim_ax, plane_scale, colour='gray', zorder=1.1)
    plane_i = Plane(anim_ax, plane_scale, colour='gray', zorder=3.1)
    
    # Add distiguishing markers
    #ax.scatter(*pis_xyz[-1], marker='x', s=30, c='black')
    #ax.scatter(*pas_xyz[-1], marker='o', s=30, c='black')

    # ============================
    # Add forces and wind
    # TODO Note that this is only inertial forces at the moment (since the plotting is in the inertial frame).
    forces_i = analysis_calculations.calc_inertial_forces_ned(dl)
    forces_a = analysis_calculations.calc_air_relative_forces_ned(dl)
    force_plotter_i = ForcePlotter(anim_ax, scaler_func=lambda x: force_scale*x, plot_inds=[])
    force_plotter_a = ForcePlotter(anim_ax, scaler_func=lambda x: force_scale*x, plot_inds=[5], zorder=1.3)

    wind = np.array([dl.wns, dl.wes, dl.wds]).T
    wind_plotter_i = WindPlotter(anim_ax, scaler_func=lambda x: wind_scale*x, head_width=0.3, line_width=1.25, zorder=3.1)
    wind_plotter_a = WindPlotter(anim_ax, scaler_func=lambda x: wind_scale*x, head_width=0.3, line_width=1.25, zorder=1.1)

    # Just a thought - to make the wind easier to visualise, what if we had one big animated arrow?
    # head_width = 0.35
    # lw = 2.5
    # wind_arrow_scale = wind_scale # 1.5
    # wind_arrow_x, wind_arrow_y, wind_arrow_z = (x_lims[0], y_lims[1], z_lims[0])
    # wind_arrow = Arrow3D([wind_arrow_x, wind_arrow_x], [wind_arrow_y, wind_arrow_y], [wind_arrow_z, wind_arrow_z], mutation_scale=10, lw=lw, arrowstyle=ArrowStyle.CurveFilledB(head_width=head_width), color='orange') # , linestyle='--')
    # anim_ax.add_artist(wind_arrow)

    # ============================
    # Add colourbar
    colourbar_label = "Specific power ($J kg^{-1} s^{-1}$)"
    cbax = anim_ax.inset_axes((0.7, 0.2, 0.3, 0.03))
    plt.colorbar(mpl.cm.ScalarMappable(norm, cmap), cbax, label=colourbar_label, orientation='horizontal', ticks=[round(np.min(all_colouring_vals), 1), 0, round(np.max(all_colouring_vals), 1)])

    # ============================
    # Scale
    anim_ax.set_xlim(x_lims[0], x_lims[1])
    anim_ax.set_ylim(y_lims[0], y_lims[1])
    anim_ax.set_zlim(z_lims[0], z_lims[1])
    anim_ax.set_aspect('equal')

    def init():
        plane_trace_a.set_segments([])
        plane_trace_i.set_segments([])

    def animate(frame):
        # Code from: https://stackoverflow.com/questions/21077477/animate-a-line-with-different-colors
        update_trace(plane_trace_a, *pas_xyz[:frame].T, air_rel_energy_change)
        update_trace(plane_trace_i, *pis_xyz[:frame].T, inertial_energy_change)
        # update_trace(ground_projn_trace, part_xs, part_ys, np.zeros(len(part_zs)), np.ones(len(colouring_vals)))

        plane_a.set_pose(*pas_xyz[frame], dl.phis[frame], dl.thetas[frame], dl.psis[frame])
        plane_i.set_pose(*pis_xyz[frame], dl.phis[frame], dl.thetas[frame], dl.psis[frame])
        
        force_plotter_a.update(*pas[frame], forces_a[:, frame, :])
        force_plotter_i.update(*pis[frame], forces_i[:, frame, :])
        
        wind_plotter_a.update(*pas[frame], wind[frame])
        wind_plotter_i.update(*pis[frame], wind[frame])

        # Update ground projection traces
        xs_i, ys_i, zs_i = pis_xyz[:frame].T
        xs_a, ys_a, zs_a = pas_xyz[:frame].T

        # x-projections
        x_proj = np.ones(frame)*x_proj_coord
        x_projn_trace_i.set_data(x_proj, ys_i)
        x_projn_trace_i.set_3d_properties(zs_i, 'z')
        x_projn_trace_a.set_data(x_proj, ys_a)
        x_projn_trace_a.set_3d_properties(zs_a, 'z')
        
        # y-projections
        y_proj = np.ones(frame)*y_proj_coord
        y_projn_trace_i.set_data(xs_i, y_proj)
        y_projn_trace_i.set_3d_properties(zs_i, 'z')
        y_projn_trace_a.set_data(xs_a, y_proj)
        y_projn_trace_a.set_3d_properties(zs_a, 'z')

        # z-projections
        z_proj = np.ones(frame)*z_proj_coord
        z_projn_trace_i.set_data(xs_i, ys_i)
        z_projn_trace_i.set_3d_properties(z_proj, 'z')
        z_projn_trace_a.set_data(xs_a, ys_a)
        z_projn_trace_a.set_3d_properties(z_proj, 'z')

    anim = FuncAnimation(anim_fig, animate, frames=dl.num_steps, init_func=init, interval=anim_interval_ms, repeat_delay=1000)

    if savefolder is not None:
        # anim.save(savefolder / 'air_rel_trace.mp4', writer=FFMpegWriter(fps=20))
        anim.save(savefolder / 'air_rel_trace.mp4', dpi=save_dpi) # , fps=60)
    
    return anim

    # plt.show(block=True)

    # def plot_air_relative_trace(dl, aircraft_model, plot_interval=500, wind_scale=5, plot_fictitious_force=True, plot_lift_force=True, plot_wind=True, force_scale=1, view=None):
    # The air-relative position is inertial position minus the integral of the wind.

    # # Plot ground
    # # TODO This seems like a messy way of doing this
    # ground_x, ground_y = np.meshgrid(x_lims, y_lims)
    # ground_z = np.zeros(ground_x.shape)
    # anim_ax.plot_surface(ground_x, ground_y, ground_z, color='gray', alpha=0.1)

    """
    # Set viewing angle
    # Hide the label and ticks of the axis not being used
    # ax.set_zticks([])
    ax.view_init(90, -90, 0)
    """


# To do
# =====
# Forces
# Live velocity plot
# Increase plane scale
# Air-relative plotting
# Maybe it would make more sense if these were two plots stacked atop each other?
def animate_air_rel_old(dl, aircraft_model, plot_velocities=True, plane_scale=1, force_scale=1, wind_scale=1, anim_interval_ms=20, subsample_num=1, savefolder=None):
# def plot_air_relative_trace(dl, aircraft_model, plot_interval=500, wind_scale=5, plot_fictitious_force=True, plot_lift_force=True, plot_wind=True, force_scale=1, view=None):
    # The air-relative position is inertial position minus the integral of the wind.
    
    # Construct a vector/array of the inertial positions
    pis = np.array([dl.ns, dl.es, dl.ds]).T
    # Find air-relative positions
    pas = analysis_calculations.calc_air_rel_positions(dl)

    pas_xyz = ned_to_xyz(*pas.T).T
    pis_xyz = ned_to_xyz(*pis.T).T

    # Calculate colouring values
    air_rel_energy_change = analysis_calculations.calc_total_specific_air_relative_energy_change(dl, aircraft_model)
    inertial_energy_change = analysis_calculations.calc_total_specific_inertial_energy_change(dl, aircraft_model)

    # Calculate forces
    forces_i = analysis_calculations.calc_inertial_forces_ned(dl)
    forces_a = analysis_calculations.calc_air_relative_forces_ned(dl)
    
    # Subsample
    dl = copy.deepcopy(dl)
    dl.log_arr = dl.log_arr[:, ::subsample_num]
    dl.num_steps = dl.log_arr.shape[1]
    pas = pas[::subsample_num]
    pis = pis[::subsample_num]
    pas_xyz = pas_xyz[::subsample_num]
    pis_xyz = pis_xyz[::subsample_num]
    air_rel_energy_change = air_rel_energy_change[::subsample_num]
    inertial_energy_change = inertial_energy_change[::subsample_num]
    forces_i = forces_i[::subsample_num]
    forces_a = forces_a[::subsample_num]

    # Set up figure
    anim_fig = plt.figure(figsize=(12, 10))
    if plot_velocities:
        anim_ax = anim_fig.add_subplot(121, projection='3d')
        vel_ax = anim_fig.add_subplot(122)
    else:
        anim_ax = anim_fig.add_subplot(projection='3d')

    anim_ax.set_axis_off()

    # Add building
    #if plot_building:
    building = Building((131, 369), (100, 190), (-33, 0))
    # building = Building((131, 369), (160, 190), (-33, 0))
    building.add_to_axis(anim_ax)

    anim_ax.set_xlabel('x [e]')
    anim_ax.set_ylabel('y [n]')
    anim_ax.set_zlabel('z [-d]')

    # Showing x, y and z axes at origin
    i_vec_scaling = 20
    x = Arrow3D([0, i_vec_scaling], [0, 0], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color="r")
    y = Arrow3D([0, 0], [0, i_vec_scaling], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color="g")
    z = Arrow3D([0, 0], [0, 0], [0, i_vec_scaling], mutation_scale=20, lw=1, arrowstyle="-|>", color="b")
    anim_ax.add_artist(x)
    anim_ax.add_artist(y)
    anim_ax.add_artist(z)

    # Create colourmap - for colouring trace by 'colouring_vals'
    all_colouring_vals = np.concatenate((inertial_energy_change, air_rel_energy_change))
    cmap_name = 'bwr'
    cmap = mpl.colormaps[cmap_name]
    norm, min_val, max_val = gen_norm(all_colouring_vals, True)

    # Create empty traces
    plane_trace_a = art3d.Line3DCollection([], cmap=cmap, norm=norm, lw=2)
    plane_trace_i = art3d.Line3DCollection([], cmap=cmap, norm=norm, lw=2)
    # Add trace to axis
    anim_ax.add_collection(plane_trace_a) # , label='a')
    anim_ax.add_collection(plane_trace_i) # , label='i')
    # Plot aircraft
    plane_a = Plane(anim_ax, plane_scale, colour='gray')
    plane_i = Plane(anim_ax, plane_scale, colour='gray')
    
    # Add distiguishing markers
    #ax.scatter(*pis_xyz[-1], marker='x', s=30, c='black')
    #ax.scatter(*pas_xyz[-1], marker='o', s=30, c='black')

    # Scale
    all_coords_xyz = np.concatenate((pas_xyz, pis_xyz))
    mins = np.min(all_coords_xyz, axis=0)
    maxs = np.max(all_coords_xyz, axis=0)
    x_lims, y_lims, z_lims = np.concatenate(([mins], [maxs])).T
    eps = 2
    anim_ax.set_xlim(x_lims[0] - eps, x_lims[1] + eps)
    anim_ax.set_ylim(y_lims[0] - eps, y_lims[1] + eps)
    anim_ax.set_zlim(z_lims[0] - eps, z_lims[1] + eps)

    # Add forces and wind
    # TODO Note that this is only inertial forces at the moment (since the plotting is in the inertial frame).
    forces_i = analysis_calculations.calc_inertial_forces_ned(dl)
    forces_a = analysis_calculations.calc_air_relative_forces_ned(dl)
    force_plotter_i = ForcePlotter(anim_ax, scaler_func=lambda x: force_scale*x, plot_inds=[])
    force_plotter_a = ForcePlotter(anim_ax, scaler_func=lambda x: force_scale*x, plot_inds=[5])

    wind = np.array([dl.wns, dl.wes, dl.wds]).T
    wind_plotter_i = WindPlotter(anim_ax, scaler_func=lambda x: wind_scale*x, head_width=0.3, line_width=1.25)
    wind_plotter_a = WindPlotter(anim_ax, scaler_func=lambda x: wind_scale*x, head_width=0.3, line_width=1.25)

    # Plot ground
    # TODO This seems like a messy way of doing this
    ground_x, ground_y = np.meshgrid(x_lims, y_lims)
    ground_z = np.zeros(ground_x.shape)
    anim_ax.plot_surface(ground_x, ground_y, ground_z, color='gray', alpha=0.1)
    
    ground_projn_trace_i, = anim_ax.plot([], [], [], lw=1, c='black', alpha=0.1)
    ground_projn_trace_a, = anim_ax.plot([], [], [], lw=1, c='black', alpha=0.1, ls='--')

    # Just a thought - to make the wind easier to visualise, what if we had one big animated arrow?
    head_width = 0.3
    lw = 5
    wind_arrow_scale = 1.5
    wind_arrow_x, wind_arrow_y, wind_arrow_z = (x_lims[0], y_lims[0], 0)
    wind_arrow = Arrow3D([wind_arrow_x, wind_arrow_x], [wind_arrow_y, wind_arrow_y], [wind_arrow_z, wind_arrow_z], mutation_scale=10, lw=lw, arrowstyle=ArrowStyle.CurveFilledB(head_width=head_width), color='orange') # , linestyle='--')
    anim_ax.add_artist(wind_arrow)

    anim_ax.set_aspect('equal')

    if plot_velocities:
        vis = np.sqrt(dl.us**2 + dl.vs**2 + dl.ws**2)
        vas = dl.vas
        vel_range = (np.min((np.min(vis), np.min(vas))), np.max((np.max(vis), np.max(vas))))

        # Set up velocity graph
        vi_line, = vel_ax.plot([], label='$v_i$')
        va_line, = vel_ax.plot([], label='$v_a$')
        # Plot stall speed
        stall_speed = aircraft_model.calc_stall_speed()
        vel_ax.plot([dl.times[0], dl.times[-1]], [stall_speed, stall_speed], label='stall (max $C_L$) speed', c='r', ls='--')
        vel_ax.set_xlim(dl.times[0], dl.times[-1])
        vel_ax.set_xlabel("Time (s)")
        vel_ax.set_ylim(0, vel_range[1] + 1)
        vel_ax.set_ylabel("Velocity (m/s)")
        vel_ax.legend()
        vel_ax.set_title("Velocity")

    """
    # Set viewing angle
    # Hide the label and ticks of the axis not being used
    # ax.set_zticks([])
    ax.view_init(90, -90, 0)
    """

    def update_wind_arrow(wind):
        # Hide arrow when the wind is 0
        if np.linalg.norm(wind) < 1:
            wind_arrow.set(visible=False)
        else:
            wind_arrow.set(visible=True)

            w_x, w_y, w_z = ned_to_xyz(*wind)

            wind_arrow.update_3d_posns([wind_arrow_x, wind_arrow_x + wind_arrow_scale*w_x], [wind_arrow_y, wind_arrow_y + wind_arrow_scale*w_y], [wind_arrow_z, wind_arrow_z + wind_arrow_scale*w_z])

    def init():
        plane_trace_a.set_segments([])
        plane_trace_i.set_segments([])

    def animate(frame):
        # Code from: https://stackoverflow.com/questions/21077477/animate-a-line-with-different-colors
        update_trace(plane_trace_a, *pas_xyz[:frame].T, air_rel_energy_change)
        update_trace(plane_trace_i, *pis_xyz[:frame].T, inertial_energy_change)
        # update_trace(ground_projn_trace, part_xs, part_ys, np.zeros(len(part_zs)), np.ones(len(colouring_vals)))

        plane_a.set_pose(*pas_xyz[frame], dl.phis[frame], dl.thetas[frame], dl.psis[frame])
        plane_i.set_pose(*pis_xyz[frame], dl.phis[frame], dl.thetas[frame], dl.psis[frame])
        
        force_plotter_a.update(*pas[frame], forces_a[:, frame, :])
        force_plotter_i.update(*pis[frame], forces_i[:, frame, :])
        
        wind_plotter_a.update(*pas[frame], wind[frame])
        wind_plotter_i.update(*pis[frame], wind[frame])

        # Update ground projection traces
        xs_i, ys_i, _ = pis_xyz[:frame].T
        xs_a, ys_a, _ = pas_xyz[:frame].T
        zs = np.zeros(frame)
        ground_projn_trace_i.set_data(xs_i, ys_i)
        ground_projn_trace_i.set_3d_properties(zs, 'z')
        ground_projn_trace_a.set_data(xs_a, ys_a)
        ground_projn_trace_a.set_3d_properties(zs, 'z')

        # Update wind arrow
        update_wind_arrow((dl.wns[frame], dl.wes[frame], dl.wds[frame]))

        if plot_velocities:
            vi_line.set_data(dl.times[:frame], vis[:frame])
            va_line.set_data(dl.times[:frame], vas[:frame])

    anim = FuncAnimation(anim_fig, animate, frames=dl.num_steps, init_func=init, interval=anim_interval_ms, repeat_delay=1000)

    if savefolder is not None:
        # anim.save(savefolder / 'air_rel_trace.mp4', writer=FFMpegWriter(fps=20))
        anim.save(savefolder / 'air_rel_trace.gif') # , fps=60)

    plt.show(block=True)

# Define cropping ratios (left, upper, right, lower)
def gen_stills(dl, aircraft_model, colouring_fn, frames, constrain_to_frames=True, cropping_ratios=(0, 0.19, 1, 1-0.22), force_plot_inds=[], arrow_alpha=1, init_view=(None, None, None), plane_scale=1, force_scale=1, wind_scale=1, vel_scale=1, arrow_lw=1, wind_ground_projection_scale=1.5, savefolder=None, time_label_x=0.1, time_label_y=0.25):

    frames = np.array(frames)
                      
    # Subsample to the region between the first and last frames (if requested)
    if constrain_to_frames:
        # Subsample dl
        dl = dl.create_sub_dl(frames[0], frames[-1])
        rel_frames = frames - frames[0]
    else:
        rel_frames = frames
    
    colouring_vals = colouring_fn(dl)

    # This needs modifying:
    # o I only want the base-plate drawn only for the bit that I'm drawing
    # o I also want to plot the airspeed, angle of attack and sideslip for the specific part that I'm plotting
        
    plt.ioff()

    ## Construct a vector/array of the inertial positions
    pis = np.array([dl.ns, dl.es, dl.ds]).T
    ## Find air-relative positions
    #pas = analysis_calculations.calc_air_rel_positions(dl)
    #
    #pas_xyz = ned_to_xyz(*pas.T).T
    pis_xyz = ned_to_xyz(*pis.T).T

    count = 1
    for frame_orig, frame in zip(frames, rel_frames):
        # Set up figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(projection='3d', computed_zorder=False)

        ax.set_axis_off()

        ax.set_xlabel('x [e]')
        ax.set_ylabel('y [n]')
        ax.set_zlabel('z [-d]')

        # Showing x, y and z axes at origin
        # i_vec_scaling = 20
        # x = Arrow3D([0, i_vec_scaling], [0, 0], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color="r")
        # y = Arrow3D([0, 0], [0, i_vec_scaling], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color="g")
        # z = Arrow3D([0, 0], [0, 0], [0, i_vec_scaling], mutation_scale=20, lw=1, arrowstyle="-|>", color="b")
        # ax.add_artist(x)
        # ax.add_artist(y)
        # ax.add_artist(z)

        # Create colourmap - for colouring trace by 'colouring_vals'
        cmap_name = 'bwr'
        cmap = mpl.colormaps[cmap_name]
        norm, min_val, max_val = gen_norm(colouring_vals, True)

        # Create empty traces
        plane_trace_i = art3d.Line3DCollection([], cmap=cmap, norm=norm, lw=4)
        # Add trace to axis
        ax.add_collection(plane_trace_i) # , label='i')
        # Plot aircraft
        plane_i = Plane(ax, plane_scale, colour='gray', zorder=1e100)

        # Scale
        all_coords_xyz = pis_xyz # np.concatenate((pas_xyz, pis_xyz))
        mins = np.min(all_coords_xyz, axis=0)
        maxs = np.max(all_coords_xyz, axis=0)
        x_lims, y_lims, z_lims = np.concatenate(([mins], [maxs])).T
        eps = 2
        ax.set_xlim(x_lims[0] - eps, x_lims[1] + 15 + eps)
        ax.set_ylim(y_lims[0] - eps, y_lims[1] + eps)
        ax.set_zlim(z_lims[0] - eps, z_lims[1] + eps)

        # Add forces and wind
        # TODO Note that this is only inertial forces at the moment (since the plotting is in the inertial frame).
        forces_i = analysis_calculations.calc_inertial_forces_ned(dl)
        force_plotter_i = ForcePlotter(ax, scaler_func=lambda x: force_scale*x, plot_inds=force_plot_inds, alpha=arrow_alpha, line_width=arrow_lw, zorder=1e101)
        # Total aerodynamic force
        f_aero_ned = forces_i[0] + forces_i[1] + forces_i[2]
        f_aero_plotter = VecPlotterSingle(ax, 'lightcoral', scaler_func=lambda x: force_scale*x, alpha=arrow_alpha, line_width=arrow_lw, zorder=1e101)

        wind = np.array([dl.wns, dl.wes, dl.wds]).T
        wind_plotter_i = WindPlotter(ax, scaler_func=lambda x: wind_scale*x, line_width=arrow_lw, zorder=1e101)

        # Generate air-relative and inertial speed vectors
        # Calculate/get va vectors
        va_vecs_ned = analysis_calculations.calc_va_vecs_i_ned(dl)

        # Calculate vi vectors
        vi_b_vecs = np.array([dl.us, dl.vs, dl.ws]).T
        # Convert body frame velocity to inertial frame
        r_bi = calc_body_to_inertial(dl.phis, dl.thetas, dl.psis)
        vi_vecs_ned = np.squeeze(np.matmul(r_bi, vi_b_vecs[:, :, np.newaxis]))

        vi_plotter = VecPlotterSingle(ax, 'teal', scaler_func=lambda x: vel_scale*x, alpha=arrow_alpha, line_width=arrow_lw, zorder=1e101)
        va_plotter = VecPlotterSingle(ax, 'orchid', scaler_func=lambda x: vel_scale*x, alpha=arrow_alpha, line_width=arrow_lw, zorder=1e101)

        # Plot ground
        # TODO This seems like a messy way of doing this
        ground_x, ground_y = np.meshgrid(x_lims, y_lims)
        ground_z = np.zeros(ground_x.shape)
        ax.plot_surface(ground_x, ground_y, ground_z, color='gray', alpha=0.1)
        
        ground_projn_trace_i, = ax.plot([], [], [], lw=1, c='black', alpha=0.1)

        # Just a thought - to make the wind easier to visualise, what if we had one big animated arrow?
        head_width = 0.3
        lw = 5
        wind_arrow_scale = wind_ground_projection_scale
        wind_arrow_x, wind_arrow_y, wind_arrow_z = (x_lims[0], y_lims[0], 0)
        wind_arrow = Arrow3D([wind_arrow_x, wind_arrow_x], [wind_arrow_y, wind_arrow_y], [wind_arrow_z, wind_arrow_z], mutation_scale=10, lw=lw, arrowstyle=ArrowStyle.CurveFilledB(head_width=head_width), color='orange') # , linestyle='--')
        ax.add_artist(wind_arrow)

        ax.set_aspect('equal')

        # Set viewing angle
        ax.view_init(*init_view)

        """
        # Hide the label and ticks of the axis not being used
        # ax.set_zticks([])
        """

        plane_trace_i.set_segments([])

        def update_wind_arrow(wind):
            # Hide arrow when the wind is 0
            if np.linalg.norm(wind) < 1:
                wind_arrow.set(visible=False)
            else:
                wind_arrow.set(visible=True)

                w_x, w_y, w_z = ned_to_xyz(*wind)

                wind_arrow.update_3d_posns([wind_arrow_x, wind_arrow_x + wind_arrow_scale*w_x], [wind_arrow_y, wind_arrow_y + wind_arrow_scale*w_y], [wind_arrow_z, wind_arrow_z + wind_arrow_scale*w_z])

        def comp_frame(frame):
            # Code from: https://stackoverflow.com/questions/21077477/animate-a-line-with-different-colors
            update_trace(plane_trace_i, *pis_xyz[:frame].T, colouring_vals)
            # update_trace(ground_projn_trace, part_xs, part_ys, np.zeros(len(part_zs)), np.ones(len(colouring_vals)))

            plane_i.set_pose(*pis_xyz[frame], dl.phis[frame], dl.thetas[frame], dl.psis[frame])

            force_plotter_i.update(*pis[frame], forces_i[:, frame, :])

            # Total aerodynamic force
            f_aero_plotter.update(*pis[frame], f_aero_ned[frame])

            wind_plotter_i.update(*pis[frame], wind[frame])

            # Update speed plots
            vi_plotter.update(*pis[frame], vi_vecs_ned[frame])
            va_plotter.update(*pis[frame], va_vecs_ned[frame])

            # Update ground projection traces
            xs_i, ys_i, _ = pis_xyz[:frame].T
            zs = np.zeros(frame)
            ground_projn_trace_i.set_data(xs_i, ys_i)
            ground_projn_trace_i.set_3d_properties(zs, 'z')

            # Update wind arrow
            update_wind_arrow((dl.wns[frame], dl.wes[frame], dl.wds[frame]))

        comp_frame(frame)
        #ax.text2D(0.1, 0.25, count, transform=ax.transAxes, c='darkgrey', fontsize='xx-large')
        ax.text2D(time_label_x, time_label_y, f"t={np.round(frame_orig*dl.dt, 1)}s", transform=ax.transAxes, c='darkgrey', fontsize='xx-large')

        if frame_orig == frames[-1]:
            # Add a legend to the final one
            vi_leg = Line2D([], [], c='teal') # marker='', )
            va_leg = Line2D([], [], c='orchid') # marker='', )
            f_aero_leg = Line2D([], [], c='lightcoral') # marker='', )
            wind_leg = Line2D([], [], ls='--', c='orange') # marker='', )
            # ax.legend((vi_leg, va_leg, f_aero_leg, wind_leg), (r'$\vec{v}_i$', r'$\vec{v}_a$', 'aerodynamic force', 'wind'), title="Arrow key", loc='center', bbox_to_anchor=(0.7, 0.4), bbox_transform=fig.transFigure, prop={'size': 15})
            # ax.legend((vi_leg, va_leg, f_aero_leg, wind_leg), (r'$\vec{v}_i$', r'$\vec{v}_a$', 'aerodynamic force', 'wind'), title="Arrow key", loc='center', bbox_to_anchor=(0.5, 0.5), bbox_transform=fig.transFigure, prop={'size': 15})
            # lg = ax.legend((vi_leg, va_leg, f_aero_leg, wind_leg), (r'$\vec{v}_i$', r'$\vec{v}_a$', 'aerodynamic force', 'wind'), loc='center', bbox_to_anchor=(0.5, 0.5), bbox_transform=fig.transFigure, prop={'size': 15})
            lg = ax.legend((vi_leg, va_leg, f_aero_leg, wind_leg), (r'$\vec{v}_i$', r'$\vec{v}_a$', 'aerodynamic force', 'wind'), loc='center', bbox_to_anchor=(0.68, 0.35), bbox_transform=fig.transFigure, prop={'size': 15})
            lg.set_title("Arrow key", prop={'size':15})

        if savefolder is not None:
            path = savefolder / f'{frame_orig}.png'
            fig.savefig(path, dpi=1000, bbox_inches='tight', pad_inches=0)
            
            # Crop
            # Open an image file
            im = Image.open(path)
            
            # Define the cropping box (left, upper, right, lower)
            w, h = im.size
            # print(w, h)
            # h_frac = 0.17
            # im.crop((0, h*h_frac, w, h*(1-h_frac))).save(path)
            cropping_box = (cropping_ratios[0]*w, cropping_ratios[1]*h, cropping_ratios[2]*w, cropping_ratios[3]*h)
            im.crop(cropping_box).save(path)
                
        print(f'Frame {frame_orig} saved')

        count += 1
    
    return dl

# All of this code needs unifying under a single framework
# This function plots a comparison of inertial and air-relative trajectories, plotting multiple aircraft
# at the same time.
# Coloured by air-relative energy
def inertial_air_rel_comparison_trace(dl, aircraft_model, smap, plot_interval,
                                      top_down=False, extra_ticks=[],
                                      plane_scale=1, plane_linewidth=None, force_scale=1, wind_scale=1,
                                      vel_scale=1, arrow_lw=1.25, plot_inertial=True,
                                      savefolder=None, save_name=None):

    # 1. Calculate the air-relative positions
    # 2. Plot the trajectories
    # 3. Plot the aircraft
    # 4. Plot the forces

    arrow_alpha = 1

    frame_inds = np.arange(0, dl.num_steps, plot_interval)

    # Construct position information

    # Construct a vector/array of the inertial positions
    pis = np.array([dl.ns, dl.es, dl.ds]).T
    # Find air-relative positions
    pas = analysis_calculations.calc_air_rel_positions(dl)

    pas_xyz = ned_to_xyz(*pas.T).T
    pis_xyz = ned_to_xyz(*pis.T).T

    # Calculate colouring values
    air_rel_energy_change = analysis_calculations.calc_total_specific_air_relative_energy_change(dl, aircraft_model)
    
    # Calculate forces
    forces_a = analysis_calculations.calc_air_relative_forces_ned(dl)

    # ===============
    # Set up figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')
    ax.computed_zorder = False

    # ===============
    # Calculate axis limits
    
    # Find maxima and minima across both inertial and air-relative data
    all_xs = np.concatenate((pas_xyz.T[0], pis_xyz.T[0]))
    all_ys = np.concatenate((pas_xyz.T[1], pis_xyz.T[1]))
    all_zs = np.concatenate((pas_xyz.T[2], pis_xyz.T[2]))

    x_data_lims = (np.min(all_xs), np.max(all_xs))
    y_data_lims = (np.min(all_ys), np.max(all_ys))
    z_data_lims = (np.min(all_zs), np.max(all_zs))

    x_lims = (np.min(all_xs)-5, np.max(all_xs)+2)
    # x_lims = (np.min(all_xs)-20, np.max(all_xs)+20)
    y_lims = (np.min(all_ys)-10, np.max(all_ys)+5)
    # y_lims = (np.min(all_ys)-10, np.max(all_ys)+10)
    z_lims = (np.min(np.concatenate((all_zs, [0]))), np.max(all_zs)+10)
    
    x_proj_coord = x_lims[0]
    y_proj_coord = y_lims[1]
    z_proj_coord = z_lims[0]

    # ===============
    # Label axes and set ticks
    ax.set_xticks([round(v) for v in x_data_lims])
    ax.set_yticks([round(v) for v in y_data_lims] + extra_ticks)
    ax.tick_params(axis='y', pad=10)
    ax.set_zticks([round(v) for v in z_data_lims])
    
    labelpad = 10
    ax.set_xlabel(r'$x_{\,[e]}$ ($m$)', labelpad=labelpad)
    ax.set_ylabel('\n\n$y_{\\,[n]}$ ($m$)', labelpad=labelpad)
    if top_down:
        ax.set_zticks([])
        ax.zaxis.line.set_color((0,0,0,0))
    else:
        ax.set_zlabel(r'$z_{\,[-d]}$ ($m$)', labelpad=labelpad)
    
    ax.grid(visible=False)

    # ===============
    # Create lines
    
    # Create empty traces
    plane_trace_a = art3d.Line3DCollection([], cmap=smap.cmap, norm=smap.norm, lw=2, zorder=1)
    # Code from: https://stackoverflow.com/questions/21077477/animate-a-line-with-different-colors
    update_trace(plane_trace_a, *pas_xyz.T, air_rel_energy_change)
    # Add traces to axis
    ax.add_collection(plane_trace_a)

    # Plot projection traces
    ones = np.ones(dl.num_steps)
    xs_i, ys_i, zs_i = pis_xyz.T
    xs_a, ys_a, zs_a = pas_xyz.T
    # ax.plot(ones*x_proj_coord, ys_i, zs_i, lw=1, c='black', alpha=0.1, ls='--', zorder=0)
    # ax.plot(ones*x_proj_coord, ys_a, zs_a, lw=1, c='black', alpha=0.1, zorder=0)
    # ax.plot(xs_i, ones*y_proj_coord, zs_i, lw=1, c='black', alpha=0.1, ls='--', zorder=0)
    # ax.plot(xs_a, ones*y_proj_coord, zs_a, lw=1, c='black', alpha=0.1, zorder=0)
    ax.plot(xs_i, ys_i, ones*z_proj_coord, lw=1.5, c='darkgrey', alpha=1, ls='--', zorder=0)
    # ax.plot(xs_a, ys_a, ones*z_proj_coord, lw=1, c='black', alpha=0.1, zorder=0)

    for frame_ind in frame_inds:
        # Create plane
        Plane(ax, plane_scale, *pas_xyz[frame_ind], dl.phis[frame_ind], dl.thetas[frame_ind], dl.psis[frame_ind], get_col_str(smap, air_rel_energy_change[frame_ind]), linewidth=plane_linewidth)
        
        # Add wind, forces and velocites

        # Wind
        wind = np.array([dl.wns, dl.wes, dl.wds]).T
        WindPlotter(ax, scaler_func=lambda x: wind_scale*x, head_width=0.3, line_width=arrow_lw, alpha=arrow_alpha).update(*pas[frame_ind], wind[frame_ind]) # , zorder=1.1)

        # Forces
        forces_a = analysis_calculations.calc_air_relative_forces_ned(dl)
        # Total force
        f_total_ned_i = forces_a[0] + forces_a[1] + forces_a[2] + forces_a[3] + forces_a[4] # + forces_a[5]
        if plot_inertial:
            VecPlotterSingle(ax, 'dimgrey', scaler_func=lambda x: (1/3)*force_scale*x, alpha=arrow_alpha, line_width=arrow_lw).update(*pas[frame_ind], f_total_ned_i[frame_ind])
        VecPlotterSingle(ax, 'blue', scaler_func=lambda x: force_scale*x, alpha=arrow_alpha, line_width=arrow_lw).update(*pas[frame_ind], forces_a[5, frame_ind, :]) # , zorder=1.3)

        # # And inertial speed vector
        # # Get u, v and w
        # # Calculate normalised vi vectors (direction vectors of inertial travel)
        # vi_b_vecs = np.array([dl.us, dl.vs, dl.ws]).T
        # # Convert body frame velocity to inertial frame
        # r_bi = calc_body_to_inertial(dl.phis, dl.thetas, dl.psis)
        # vi_vecs_ned = np.squeeze(np.matmul(r_bi, vi_b_vecs[:, :, np.newaxis]))
        # VecPlotterSingle(ax, 'teal', scaler_func=lambda x: vel_scale*x).update(*pas[frame_ind], vi_vecs_ned[frame_ind])

        # Add airspeed vector
        va_vecs_ned = analysis_calculations.calc_va_vecs_i_ned(dl)
        VecPlotterSingle(ax, 'orchid', scaler_func=lambda x: vel_scale*x, alpha=arrow_alpha, line_width=arrow_lw).update(*pas[frame_ind], va_vecs_ned[frame_ind])
    
    # ===============
    # Add colourbar
    #colourbar_label = "Specific air-relative power ($J kg^{-1} s^{-1}$)"
    #cbax = ax.inset_axes((0.7, 0.6, 0.2, 0.03))
    #plt.colorbar(mpl.cm.ScalarMappable(norm, cmap), cbax, label=colourbar_label, orientation='horizontal', ticks=[np.min(air_rel_energy_change), 0, np.max(air_rel_energy_change)])
    # plt.colorbar(mpl.cm.ScalarMappable(norm, cmap), ax=ax, label=colourbar_label, orientation='vertical', ticks=[np.min(air_rel_energy_change), 0, np.max(air_rel_energy_change)])

    # Legend now created elsewhere as its own separate plot
    """
    # Add a legend to the final one
    # vi_leg = Line2D([], [], c='teal') # marker='', )
    va_leg = Line2D([], [], c='orchid') # marker='', )
    f_inert_leg = Line2D([], [], c='dimgrey') # marker='', )
    f_fict_leg = Line2D([], [], c='blue') # marker='', )
    wind_leg = Line2D([], [], ls='--', c='orange') # marker='', )
    air_rel_track_leg = Line2D([], [], c='black', alpha=0.1)
    inertial_track_leg = Line2D([], [], ls='--', c='black', alpha=0.1)
    # ax.legend((vi_leg, va_leg, f_aero_leg, wind_leg), (r'$\vec{v}_i$', r'$\vec{v}_a$', 'aerodynamic force', 'wind'), title="Arrow key", loc='center', bbox_to_anchor=(0.7, 0.4), bbox_transform=fig.transFigure, prop={'size': 15})
    ax.legend((va_leg, f_inert_leg, f_fict_leg, wind_leg, air_rel_track_leg, inertial_track_leg), (r'airspeed ($v_a$)', r'inertial force ($1/3$ length)', 'fictitious wind force', 'wind', 'air-relative track', 'inertial track')) # ,
    # loc='lower left', bbox_to_anchor=(0.7, 0.85), bbox_transform=fig.transFigure) # title="Key", prop={'size': 15})
    """

    # Set view orientation
    if top_down:
        ax.view_init(90.1, -90, 0)
    
    # ===============
    # Scale
    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(y_lims[0], y_lims[1])
    ax.set_zlim(z_lims[0], z_lims[1])
    ax.set_aspect('equal')

    # Save
    if savefolder is not None:
        crop_and_save(fig, savefolder / f'{save_name}.png', 600)


# 3D wind contour slice
def plot_contour(ax, wind_fn_vec, plot_dir, val, wind_dir, norm, cmap='viridis', alpha=1):
    # For the contour plot window
    # Plot at a particular n
    n_range = (90, 410)
    e_range = (189, 300)
    # e_range = (150, 300)
    d_range = (-50, 1)
    intv = 1

    # n_vals = jnp.array([n]) # 250
    n_vals = jnp.array([val]) if plot_dir == 'n' else jnp.arange(*n_range, intv) # 250
    e_vals = jnp.array([val]) if plot_dir == 'e' else jnp.arange(*e_range, intv)
    d_vals = jnp.array([val]) if plot_dir == 'd' else jnp.arange(*d_range, intv)
    time = 0

    # Create all coordinates
    # Create from meshgrid, so that it can be undone.
    # N, E, D = jnp.meshgrid(e_vals, d_vals)
    N, E, D = jnp.meshgrid(n_vals, e_vals, d_vals)
    # Stack
    num_pts = E.size    # TODO Calculate this in a more methodical way
    # all_coords = jnp.stack((jnp.ones(num_pts)*n_vals, E.flatten(), D.flatten(), jnp.ones(num_pts)*time), axis=1)
    all_coords = jnp.stack((N.flatten(), E.flatten(), D.flatten(), jnp.ones(num_pts)*time), axis=1)
    wind_vals = wind_fn_vec(*all_coords.T)

    if wind_dir == 'n':
        w_vals = wind_vals[0]
    elif wind_dir == 'e':
        w_vals = wind_vals[1]
    elif wind_dir == 'd':
        w_vals = wind_vals[2]
    else:
        raise Exception("'wind_dir' argument invalid, should be 'n', 'e' or 'd'!")

    # Reshape into meshgrid shape
    w_vals_reshaped = w_vals.reshape(E.squeeze().shape)  # TODO Get the shape in a more methodical way - don't use E.

    # Convert from ned into xyz
    X, Y, Z = ned_to_xyz(N, E, D)

    # Plot contour - will still need to convert to XYZ before we do this properly - I don't know how to do this.
    # ax.contourf(N, E, D, zdir='x', offset=n_vals[0], levels=30)
    if plot_dir == 'n':
        ctr = ax.contourf(X.squeeze(), w_vals_reshaped, Z.squeeze(), zdir='y', offset=n_vals[0], levels=30, cmap=cmap, norm=norm, alpha=alpha)
    elif plot_dir == 'e':
        ctr = ax.contourf(w_vals_reshaped, Y.squeeze(), Z.squeeze(), zdir='x', offset=e_vals[0], levels=30, cmap=cmap, norm=norm, alpha=alpha)
    else:   # plot_dir == 'd'
        ctr = ax.contourf(X.squeeze(), Y.squeeze(), w_vals_reshaped, zdir='z', offset=-d_vals[0], levels=30, cmap=cmap, norm=norm, alpha=alpha)
    # ax.contour(N, E, D, zdir='x', offset=n_vals[0], levels=30, colors='black', zorder=100000)

    #print("X", X.squeeze())
    #print()
    #print("Y", w_vals_reshaped)
    #print()
    #print("Z", Z.squeeze())
    
    return ctr


# =================================
# Working on new graphing code

# Get the values from the Problem and from the simulation
# Code adapted from https://openmdao.github.io/dymos/getting_started/intro_to_dymos/intro_ivp.html
def graph(folderpath, title, times, value_dict, prob=None, phase_names=['dynamics_phase'], colloc_pt_colour=None, save=False):
    # Get and plot states against time points
    num_subplots = len(value_dict)
    
    fig, axes = plt.subplots(num_subplots, 1, layout='constrained')
    # In case num_subplots == 1, else the code below breaks.
    if num_subplots == 1:
        axes = [axes]
    # Keys have to match the state/control names in the OpenMDAO timeseries
    for i, key in enumerate(value_dict):
        vals, label = value_dict[key]
        # Plot state discretisation and collocation nodes first so that they don't obscure simulation line
        if prob is not None:
            for phase_name in phase_names:
                # TODO This might not be the most reliable, since it depends on the name of the phase.
                t_sol = prob.get_val(f'traj.{phase_name}.timeseries.time')
                sol = axes[i].plot(t_sol, prob.get_val(f'traj.{phase_name}.timeseries.{key}'), 'o', color=colloc_pt_colour)
        sim = axes[i].plot(times, vals, '-', color='#1f77b4')
        # Plotting zero line
        axes[i].plot([0, times[-1]], [0, 0], '--', c='r')
        axes[i].set_ylabel(label) # , rotation='horizontal')
        # axes[i].yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel('Time (s)')

    # Align y-axis labels
    fig.align_ylabels(axes)

    # Add legend
    if prob is not None:
        fig.legend((sol[0], sim[0]), ('solution', 'simulation'), loc='lower right', ncol=2)
    else:
        fig.legend((sim[0],), ('simulation',), loc='lower right', ncol=1)
    
    fig.suptitle(title)
    # fig.tight_layout()

    if save:
        # plt.savefig(folderpath / f"{title.replace(' ', '_').replace('-', '_').lower()}.png", dpi=base_dpi)
        crop_and_save(fig, folderpath / f"{title.replace(' ', '_').replace('-', '_').lower()}", dpi=base_dpi)


# u, v and w in the background, to remember that not all energy is beneficial.
# TODO Also want to plot air-relative energy
# 'end_ind' is inclusive
def plot_inertial_energy(folderpath, dl, aircraft_model, save=False, ax=None, start_ind=None, end_ind=None, name_suffix=None):
    # e/m = gh + 0.5*(u**2 + v**2 + w**2)
    ek_spec_u = 0.5*np.square(dl.us)
    ek_spec_v = 0.5*np.square(dl.vs)
    ek_spec_w = 0.5*np.square(dl.ws)
    inertial_specific_kinetic_energy = ek_spec_u + ek_spec_v + ek_spec_w
    specific_potential_energy = -aircraft_model.g*dl.ds
    total_inertial_specific_energy = inertial_specific_kinetic_energy + specific_potential_energy

    if ax is None:
        fig_created = True
        fig, ax = plt.subplots()
        fig.suptitle("Specific inertial energy")
    else: fig_created = False

    vel_energies_alpha = 0.5

    # ax.plot(dl.times, total_inertial_specific_energy, label='total inertial specific energy', linewidth=3, c='black', zorder=4)
    ax.plot(dl.times, total_inertial_specific_energy, label='total', linewidth=3, c='black', zorder=4)
    # ax.plot(dl.times, specific_potential_energy, label='specific potential energy', linewidth=1.5, c='red', zorder=2)
    ax.plot(dl.times, specific_potential_energy, label='potential', linewidth=1.5, c='red', zorder=2)
    # ax.plot(dl.times, inertial_specific_kinetic_energy, label='inertial specific kinetic energy', linewidth=1.5, c='green', zorder=2)
    ax.plot(dl.times, inertial_specific_kinetic_energy, label='kinetic', linewidth=1.5, c='green', zorder=2)
    # ax.plot(dl.times, ek_spec_u, label=r'$\frac{1}{2}u_i^2$  ($\vec{i}^b$ inertial specific kinetic energy)', c='orange', ls='--', linewidth=1, alpha=vel_energies_alpha, zorder=0)
    ax.plot(dl.times, ek_spec_u, label=r'$\frac{1}{2}u_i^2$  ($\vec{i}^b$ kinetic)', c='orange', ls='--', linewidth=1, alpha=vel_energies_alpha, zorder=0)
    # ax.plot(dl.times, ek_spec_v, label=r'$\frac{1}{2}v_i^2$  ($\vec{j}^b$ inertial specific kinetic energy)', c='purple', ls='--', linewidth=1, alpha=vel_energies_alpha, zorder=0)
    ax.plot(dl.times, ek_spec_v, label=r'$\frac{1}{2}v_i^2$  ($\vec{j}^b$ kinetic)', c='purple', ls='--', linewidth=1, alpha=vel_energies_alpha, zorder=0)
    # ax.plot(dl.times, ek_spec_w, label=r'$\frac{1}{2}w_i^2$  ($\vec{k}^b$ inertial specific kinetic energy)', c='blue', ls='--', linewidth=1, alpha=vel_energies_alpha, zorder=0)
    ax.plot(dl.times, ek_spec_w, label=r'$\frac{1}{2}w_i^2$  ($\vec{k}^b$ kinetic)', c='blue', ls='--', linewidth=1, alpha=vel_energies_alpha, zorder=0)

    ax.set_xlabel('Time ($s$)')
    ax.set_ylabel('Specific inertial energy ($Jkg^{-1}$)')
    legend = ax.legend()
    legend.get_frame().set_zorder(10)

    if fig_created:
        fig.tight_layout()

    if save:
        crop_and_save(fig, folderpath / ('spec_inertial_energy' + (f'_{name_suffix}' if name_suffix else ''))) # , dpi=base_dpi)


# u, v and w in the background, to remember that not all air-relative energy is beneficial (e.g. large sideslip is not useful).
def plot_air_relative_energy(folderpath, dl, aircraft_model, save=False, name_suffix=None):
    from flight.simulator.utils import calc_body_to_inertial

    body_to_inertial_mats = calc_body_to_inertial(dl.phis, dl.thetas, dl.psis)
    inertial_to_body_mats = np.transpose(body_to_inertial_mats, [0, 2, 1])
    
    # Find ib, jb and kb projections of the airspeed vector
    # Get inertial body frame vectors
    vi_b_vecs = np.array([dl.us, dl.vs, dl.ws]).T
    # Convert body frame velocity to inertial frame
    # TODO This needs checking
    vi_vecs = np.squeeze(np.matmul(body_to_inertial_mats, vi_b_vecs[:, :, np.newaxis]))
    # Subtract wind, to get air-relative velocity (axis vectors are parallel to those of inertial frame)
    wind = np.array([dl.wns, dl.wes, dl.wds]).T
    # Calculate airspeed vectors in inertial frame
    va_vecs = vi_vecs - wind
    # Convert these airspeed vectors back into the body frame
    va_b_vecs = np.squeeze(np.matmul(inertial_to_body_mats, va_vecs[:, :, np.newaxis]))

    # e/m = gh + 0.5*(u**2 + v**2 + w**2)
    va_ib, va_jb, va_kb = va_b_vecs.T
    ek_spec_ib = 0.5*np.square(va_ib)
    ek_spec_jb = 0.5*np.square(va_jb)
    ek_spec_kb = 0.5*np.square(va_kb)
    air_rel_specific_kinetic_energy = ek_spec_ib + ek_spec_jb + ek_spec_kb

    specific_potential_energy = -aircraft_model.g*dl.ds
    total_air_rel_specific_energy = air_rel_specific_kinetic_energy + specific_potential_energy

    fig, ax = plt.subplots()
    
    vel_energies_alpha = 0.5

    ax.plot(dl.times, total_air_rel_specific_energy, label='total', linewidth=3, c='black', zorder=4)
    ax.plot(dl.times, specific_potential_energy, label='potential', linewidth=1.5, c='red', zorder=2)
    ax.plot(dl.times, air_rel_specific_kinetic_energy, label='kinetic', linewidth=1.5, c='green', zorder=2)
    ax.plot(dl.times, ek_spec_ib, label=r'$\frac{1}{2}u_a^2$  ($\vec{i}^b$ kinetic)', c='orange', ls='--', linewidth=1, alpha=vel_energies_alpha, zorder=0)
    ax.plot(dl.times, ek_spec_jb, label=r'$\frac{1}{2}v_a^2$  ($\vec{j}^b$ kinetic)', c='purple', ls='--', linewidth=1, alpha=vel_energies_alpha, zorder=0)
    ax.plot(dl.times, ek_spec_kb, label=r'$\frac{1}{2}w_a^2$  ($\vec{k}^b$ kinetic)', c='blue', ls='--', linewidth=1, alpha=vel_energies_alpha, zorder=0)

    ax.set_xlabel('Time ($s$)')
    ax.set_ylabel('Specific air-relative energy ($Jkg^{-1}$)')
    legend = ax.legend()
    legend.get_frame().set_zorder(10)
    fig.suptitle("Specific air-relative energy")

    fig.tight_layout()
    if save:
        crop_and_save(fig, folderpath / ('spec_air_rel_energy' + (f'_{name_suffix}' if name_suffix else ''))) # , dpi=base_dpi)


def plot_inertial_work_rates(folderpath, dl, aircraft_model, save=False, ax=None, name_suffix=None):
    # Get the work rates and the total specific inertial energy change
    inertial_work_rates = analysis_calculations.calc_inertial_work_rates(dl)
    total_specific_inertial_energy_change = analysis_calculations.calc_total_specific_inertial_energy_change(dl, aircraft_model)

    line_alpha = 0.8

    if ax is None:
        fig_created = True
        fig, ax = plt.subplots()
        # fig.suptitle('Work Rate (Power)  [Energy Change]')
        fig.suptitle('Specific inertial energy change contributions: work rate (power) of forces')
    else: fig_created = False

    # ax.plot(dl.times, total_specific_inertial_energy_change, label='Total specific inertial energy change', linewidth=2)
    ax.plot(dl.times, total_specific_inertial_energy_change, label='total', linewidth=2)
    # dw = 'work rate'
    # Varying dash spacing so that different lines can be made out even when they overlap
    # ax.plot(dl.times, inertial_work_rates['dw_d'] / aircraft_model.m, label='Rate of inertial work by drag (mass normalised)', c='r', ls=(0, (4, 3)), alpha=line_alpha)
    ax.plot(dl.times, inertial_work_rates['dw_d'] / aircraft_model.m, label='drag', c='r', ls=(0, (4, 3)), alpha=line_alpha)
    # ax.plot(dl.times, inertial_work_rates['dw_c'] / aircraft_model.m, label='Rate of inertial work by sideforce (mass normalised)', c='purple', ls=(0, (4, 4)), alpha=line_alpha)
    ax.plot(dl.times, inertial_work_rates['dw_c'] / aircraft_model.m, label='sideforce', c='purple', ls=(0, (4, 4)), alpha=line_alpha)
    # ax.plot(dl.times, inertial_work_rates['dw_l'] / aircraft_model.m, label='Rate of inertial work by lift (mass normalised)', c='g', ls=(0, (4, 5)), alpha=line_alpha)
    ax.plot(dl.times, inertial_work_rates['dw_l'] / aircraft_model.m, label='lift', c='g', ls=(0, (4, 5)), alpha=line_alpha)
    # ax.plot(dl.times, inertial_work_rates['dw_t'] / aircraft_model.m, label='Rate of inertial work by thrust (mass normalised)', c='orange', ls=(0, (4, 6)), alpha=line_alpha)
    ax.plot(dl.times, inertial_work_rates['dw_t'] / aircraft_model.m, label='thrust', c='orange', ls=(0, (4, 6)), alpha=line_alpha)
    # This is dotted as it doesn't contribute to the total energy change (gravity is a conservative force).
    # ax.plot(dl.times, inertial_work_rates['dw_g'] / aircraft_model.m, label='Rate of inertial work by gravity (mass normalised)', c='pink', ls=':', alpha=line_alpha)
    ax.plot(dl.times, inertial_work_rates['dw_g'] / aircraft_model.m, label='gravity', c='pink', ls=':', alpha=line_alpha)
    ax.set_xlabel('Time ($s$)')
    ax.set_ylabel('Specific inertial work rate (power) ($Js^{-1}kg^{-1}$)')
    ax.legend()

    if fig_created:
        fig.tight_layout()
    
    if save:
        crop_and_save(fig, folderpath / ('spec_inertial_work' + (f'_{name_suffix}' if name_suffix else ''))) # , dpi=base_dpi)


def plot_air_relative_work_rates(folderpath, dl, aircraft_model, save=False, ylims=(None, None), name_suffix=None):
    # Get the work rates and the total specific air-relative energy change
    air_rel_work_rates = analysis_calculations.calc_air_relative_work_rates(dl)
    total_specific_air_rel_energy_change = analysis_calculations.calc_total_specific_air_relative_energy_change(dl, aircraft_model)
    # Checking against work-rate based calculation (which requires correction term - see thesis)
    #work_rates = analysis_calculations.calc_air_relative_work_rates(dl)
    #work_rate_based_specific_total_energy_change = (work_rates['dw_d'] + work_rates['dw_c'] + work_rates['dw_l'] + work_rates['dw_t'] + work_rates['dw_fw']) / aircraft_model.m
    #work_sum_corrected = work_rate_based_specific_total_energy_change - aircraft_model.g*dl.wds

    line_alpha = 0.8

    fig, ax = plt.subplots()

    ax.plot(dl.times, total_specific_air_rel_energy_change, label='total', linewidth=2)
    #ax.plot(dl.times, work_sum_corrected, label='Total specific air-relative energy change - work_sum_corrected', linewidth=2, ls='--')
    # dw = 'work rate'
    # Varying dash spacing so that different lines can be made out even when they overlap
    ax.plot(dl.times, air_rel_work_rates['dw_d'] / aircraft_model.m, label='drag', c='r', ls=(0, (4, 3)), alpha=line_alpha)
    ax.plot(dl.times, air_rel_work_rates['dw_c'] / aircraft_model.m, label='sideforce', c='purple', ls=(0, (4, 3.1)), alpha=line_alpha)
    ax.plot(dl.times, air_rel_work_rates['dw_l'] / aircraft_model.m, label='lift', c='g', ls=(0, (4, 3.2)), alpha=line_alpha)
    ax.plot(dl.times, air_rel_work_rates['dw_t'] / aircraft_model.m, label='thrust', c='orange', ls=(0, (4, 3.3)), alpha=line_alpha)
    # ax.plot(dl.times, air_rel_work_rates['dw_fw'] / aircraft_model.m, label='wind fictitious\nforce', c='darkorchid', ls=(0, (4, 3)), alpha=line_alpha)
    ax.plot(dl.times, air_rel_work_rates['dw_fw'] / aircraft_model.m, label='wind fictitious\nforce', c='deeppink', ls=(0, (4, 3.4)), alpha=line_alpha)
    ax.plot(dl.times, -aircraft_model.g*dl.wds, label='vertical wind', c='deepskyblue', ls=(0, (4, 3.5)), alpha=line_alpha)
    # This is dotted as it doesn't contribute to the total energy change (gravity is a conservative force).
    ax.plot(dl.times, air_rel_work_rates['dw_g'] / aircraft_model.m, label='gravity', c='pink', ls=':', alpha=line_alpha)
    ax.set_xlabel('Time ($s$)')
    ax.set_ylabel('Specific air-relative work rate ($Js^{-1}kg^{-1}$)')
    ax.legend()
    ax.set_ylim(ylims[0], ylims[1])

    # fig.suptitle('Work Rate (Power)  [Energy Change]')
    fig.suptitle('Specific air-relative energy change contributions: work rate (power) of forces')

    fig.tight_layout()
    if save:
        crop_and_save(fig, folderpath / ('spec_air_rel_work' + (f'_{name_suffix}' if name_suffix else ''))) # , dpi=base_dpi)
    
    return fig, ax


# For creating the collective set of plots for an optimisation run
# prob: OpenMDAO Problem instance - for plotting the collocation and state discretisation nodes
# TODO May need to take the wind in the future
# TODO At the moment, doesn't include the collocation trace
# 'colloc_pt_colour': if excluded, the points will all be the normal colour, regardless of the number of phases. If present and None, the phases will all have points of differing colours, which will
# make it easier to spot drift between the phases.
def generate_plots(folderpath, dl, aircraft_model, prob=None, dynamics_phase_names=['dynamics_phase'], control_phase_names=['control_phase'], save=False, colloc_pt_colour='#ff7f0e', create_traces=True, trace_plot_interval=3000, label_0=True):
    # If prob is used, the dictionary keys have to match the state/control names in the OpenMDAO timeseries.

    # States
    graph(folderpath, 'Translational states', dl.times, {'n': (dl.ns, '$n$\n$(m)$'),
                                                         'e': (dl.es, '$e$\n$(m)$'),
                                                         'd': (dl.ds, '$d$\n$(m)$'),
                                                         'u': (dl.us, '$u$\n$(m s^{-1})$'),
                                                         'v': (dl.vs, '$v$\n$(m s^{-1})$'),
                                                         'w': (dl.ws, '$w$\n$(m s^{-1})$')},
                                                         prob, dynamics_phase_names, colloc_pt_colour, save=save)
    
    graph(folderpath, 'Rotational states', dl.times, {'phi': (np.degrees(dl.phis), '$\\phi$\n$(^\circ)$'),
                                                      'theta': (np.degrees(dl.thetas), '$\\theta$\n$(^\circ)$'),
                                                      'psi': (np.degrees(dl.psis), '$\\psi$\n$(^\circ)$'),
                                                      'p': (np.degrees(dl.ps), '$p$\n$(^\circ s^{-1})$'),
                                                      'q': (np.degrees(dl.qs), '$q$\n$(^\circ s^{-1})$'),
                                                      'r': (np.degrees(dl.rs), '$r$\n$(^\circ s^{-1})$')},
                                                      prob, dynamics_phase_names, colloc_pt_colour, save=save)
    
    graph(folderpath, 'Air-relative states', dl.times, {'va': (dl.vas, '$v_a$\n$(m s^{-1})$'),
                                                        'alpha': (np.degrees(dl.alphas), '$\\alpha$\n$(^\circ)$'),
                                                        'beta': (np.degrees(dl.betas), '$\\beta$\n$(^\circ)$')},
                                                        prob, dynamics_phase_names, colloc_pt_colour, save=save)
    # Controls
    graph(folderpath, 'Controls', dl.times, {'da': (np.degrees(dl.das), '$\delta_a$\n$(^\circ)$'),
                                             'de': (np.degrees(dl.des), '$\delta_e$\n$(^\circ)$'),
                                             'dr': (np.degrees(dl.drs), '$\delta_r$\n$(^\circ)$'),
                                             # TODO What are the units of this?
                                             'dp': (dl.dps, '$\delta_p$')},
                                             prob, control_phase_names, colloc_pt_colour, save=save)
    # Forces and moments
    graph(folderpath, 'Forces and moments', dl.times, {'L': (dl.Ls, '$L$\n$(N)$'),
                                                       'C': (dl.Cs, '$C$\n$(N)$'),
                                                       'D': (dl.Ds, '$D$\n$(N)$'),
                                                       'T': (dl.Ts, '$T$\n$(N)$'),
                                                       'W': (dl.Ws, '$W$\n$(N)$'),
                                                       'L_lat': (dl.L_lats, '$\\hat{L}$\n$(Nm)$'),
                                                       'M': (dl.Ms, '$M$\n$(Nm)$'),
                                                       'N': (dl.Ns, '$N$\n$(Nm)$')},
                                                       prob, dynamics_phase_names, colloc_pt_colour, save=save)
    # Wind
    graph(folderpath, 'Wind', dl.times, {'w_n': (dl.wns, '$w_n$\n$(m s^{-1})$'),
                                         'w_e': (dl.wes, '$w_e$\n$(m s^{-1})$'),
                                         'w_d': (dl.wds, '$w_d$\n$(m s^{-1})$')},
                                         prob, dynamics_phase_names, save=save)
    # Load factor
    graph(folderpath, 'Load factor', dl.times, {'load_factor': (dl.load_factors, 'Load factor')}, prob, dynamics_phase_names, colloc_pt_colour, save=save)

    # Control rates
    #graph(folderpath, 'Control Rates', dl.times, {'da_dot': (dl.da_dots, '$\dot{\delta}_a$\n$(rad/s)$'),
    #                                              'de_dot': (dl.de_dots, '$\dot{\delta}_e$\n$(rad/s)$'),
    #                                              'dr_dot': (dl.dr_dots, '$\dot{\delta}_r$\n$(rad/s)$'),
    #                                              'dp_dot': (dl.dp_dots, '$\dot{\delta}_p$')},
    #                                              prob, 'control_phase', save=save)

    # graph(folderpath, 'Test', dl.times, {'test': dl.COMPLETE}, prob, save)

    # Plot energy
    plot_inertial_energy(folderpath, dl, aircraft_model, save)
    plot_air_relative_energy(folderpath, dl, aircraft_model, save)
    # plot_energy(figure_path, sim_out, aircraft_model)
    
    # Plot force rate of work contributions  (TODO Check this title)
    plot_inertial_work_rates(folderpath, dl, aircraft_model, save)
    plot_air_relative_work_rates(folderpath, dl, aircraft_model, save)

    # Create traces (TODO complete - still needs more)    
    if create_traces:
        # cbar_orient = 'vertical'
        inertial_energy_change = analysis_calculations.calc_total_specific_inertial_energy_change(dl, aircraft_model)
        # plot_trace(dl, inertial_energy_change, trace_plot_interval, cmap_name='bwr', plot_colourbar=True, colourbar_label='Inertial specific power ($J s^{-1} kg^{-1}$)', cbar_orientation=cbar_orient, plane_scale=1, save_folder=folderpath if save else None, save_name='trace_inertial')
        TracePlotterSingle.plot_trace(BaseEnv, dl, inertial_energy_change, trace_plot_interval, plot_traj_centreline=True, zero_centred_norm=True, colourbar_label="Specific inertial power ($J kg^{-1} s^{-1}$)", label_0=label_0, save_folder=folderpath, save_name='inertial')
        # TracePlotterSingle.plot_trace(OrigIsolEnv, dl, inertial_energy_change, trace_plot_interval, plot_traj_centreline=True, zero_centred_norm=True, colourbar_label="Specific inertial power ($J kg^{-1} s^{-1}$)", label_0=label_0, save_folder=folderpath, save_name='inertial')

        # Air-relative energy change (power)
        air_rel_energy_change = analysis_calculations.calc_total_specific_air_relative_energy_change(dl, aircraft_model)
        # plot_trace(dl, air_rel_energy_change, trace_plot_interval, cmap_name='bwr', plot_colourbar=True, colourbar_label='Air-relative specific power ($J s^{-1} kg^{-1}$)', cbar_orientation=cbar_orient, plane_scale=1, save_folder=folderpath if save else None, save_name='trace_air_rel')
        TracePlotterSingle.plot_trace(BaseEnv, dl, air_rel_energy_change, trace_plot_interval, plot_traj_centreline=True, zero_centred_norm=True, colourbar_label="Specific air-relative power ($J kg^{-1} s^{-1}$)", label_0=label_0, save_folder=folderpath, save_name='air_relative')
        # TracePlotterSingle.plot_trace(OrigIsolEnv, dl, air_rel_energy_change, trace_plot_interval, plot_traj_centreline=True, zero_centred_norm=True, colourbar_label="Specific air-relative power ($J kg^{-1} s^{-1}$)", label_0=label_0, save_folder=folderpath, save_name='air_relative')
        
        # Animate traces (TODO complete)
        # dl, aircraft_model, anim_interval_ms=20, plane_scale=1, force_scale=1, wind_scale=1, plot_building=True, show_axes=False, n_min_range=None, e_min_range=None, d_min_range=None
        # anim = animate(dl, aircraft_model, (folderpath if save else None))

# Specialist code for back of buildling plots
def generate_plots_back_of_building(folderpath, dl, aircraft_model, wind_dir_deg, wind_speed, prob=None, dynamics_phase_names=['dynamics_phase'], control_phase_names=['control_phase'], save=False, colloc_pt_colour='#ff7f0e', create_traces=True, trace_plot_interval=3000):
    # If prob is used, the dictionary keys have to match the state/control names in the OpenMDAO timeseries.

    # States
    graph(folderpath, 'Translational states', dl.times, {'n': (dl.ns, '$n$\n$(m)$'),
                                                         'e': (dl.es, '$e$\n$(m)$'),
                                                         'd': (dl.ds, '$d$\n$(m)$'),
                                                         'u': (dl.us, '$u$\n$(m s^{-1})$'),
                                                         'v': (dl.vs, '$v$\n$(m s^{-1})$'),
                                                         'w': (dl.ws, '$w$\n$(m s^{-1})$')},
                                                         prob, dynamics_phase_names, colloc_pt_colour, save=save)
    
    graph(folderpath, 'Rotational states', dl.times, {'phi': (dl.phis, '$\\phi$\n$(rad)$'),
                                                      'theta': (dl.thetas, '$\\theta$\n$(rad)$'),
                                                      'psi': (dl.psis, '$\\psi$\n$(rad)$'),
                                                      'p': (dl.ps, '$p$\n$(rad s^{-1})$'),
                                                      'q': (dl.qs, '$q$\n$(rad s^{-1})$'),
                                                      'r': (dl.rs, '$r$\n$(rad s^{-1})$')},
                                                      prob, dynamics_phase_names, colloc_pt_colour, save=save)
    
    graph(folderpath, 'Air-relative states', dl.times, {'va': (dl.vas, '$v_a$\n$(m s^{-1})$'),
                                                        'alpha': (dl.alphas, '$\\alpha$\n$(rad)$'),
                                                        'beta': (dl.betas, '$\\beta$\n$(rad)$')},
                                                        prob, dynamics_phase_names, colloc_pt_colour, save=save)
    # Controls
    graph(folderpath, 'Controls', dl.times, {'da': (dl.das, '$\delta_a$\n$(rad)$'),
                                             'de': (dl.des, '$\delta_e$\n$(rad)$'),
                                             'dr': (dl.drs, '$\delta_r$\n$(rad)$'),
                                             # TODO What are the units of this?
                                             'dp': (dl.dps, '$\delta_p$')},
                                             prob, control_phase_names, colloc_pt_colour, save=save)
    # Forces and moments
    graph(folderpath, 'Forces and moments', dl.times, {'L': (dl.Ls, '$L$\n$(N)$'),
                                                       'C': (dl.Cs, '$C$\n$(N)$'),
                                                       'D': (dl.Ds, '$D$\n$(N)$'),
                                                       'T': (dl.Ts, '$T$\n$(N)$'),
                                                       'W': (dl.Ws, '$W$\n$(N)$'),
                                                       'L_lat': (dl.L_lats, '$\\hat{L}$\n$(Nm)$'),
                                                       'M': (dl.Ms, '$M$\n$(Nm)$'),
                                                       'N': (dl.Ns, '$N$\n$(Nm)$')},
                                                       prob, dynamics_phase_names, colloc_pt_colour, save=save)
    # Wind
    graph(folderpath, 'Wind', dl.times, {'w_n': (dl.wns, '$w_n$\n$(m s^{-1})$'),
                                         'w_e': (dl.wes, '$w_e$\n$(m s^{-1})$'),
                                         'w_d': (dl.wds, '$w_d$\n$(m s^{-1})$')},
                                         prob, dynamics_phase_names, save=save)
    # Load factor
    graph(folderpath, 'Load factor', dl.times, {'load_factor': (dl.load_factors, 'Load factor')}, prob, dynamics_phase_names, colloc_pt_colour, save=save)

    # Control rates
    #graph(folderpath, 'Control Rates', dl.times, {'da_dot': (dl.da_dots, '$\dot{\delta}_a$\n$(rad/s)$'),
    #                                              'de_dot': (dl.de_dots, '$\dot{\delta}_e$\n$(rad/s)$'),
    #                                              'dr_dot': (dl.dr_dots, '$\dot{\delta}_r$\n$(rad/s)$'),
    #                                              'dp_dot': (dl.dp_dots, '$\dot{\delta}_p$')},
    #                                              prob, 'control_phase', save=save)

    # graph(folderpath, 'Test', dl.times, {'test': dl.COMPLETE}, prob, save)

    # Plot energy
    plot_inertial_energy(folderpath, dl, aircraft_model, save)
    plot_air_relative_energy(folderpath, dl, aircraft_model, save)
    # plot_energy(figure_path, sim_out, aircraft_model)
    
    # Plot force rate of work contributions  (TODO Check this title)
    plot_inertial_work_rates(folderpath, dl, aircraft_model, save)
    plot_air_relative_work_rates(folderpath, dl, aircraft_model, save)

    # Create traces (TODO complete - still needs more)    
    if create_traces:
        cbar_orient = 'vertical'
        inertial_energy_change = analysis_calculations.calc_total_specific_inertial_energy_change(dl, aircraft_model)
        plot_trace(dl, wind_dir_deg, wind_speed, inertial_energy_change, trace_plot_interval, cmap_name='bwr',
                   colourbar_label='Inertial specific power ($J s^{-1} kg^{-1}$)', plane_scale=1, save_folder=folderpath if save else None, save_name='trace_inertial')

        # Air-relative energy change (power)
        air_rel_energy_change = analysis_calculations.calc_total_specific_air_relative_energy_change(dl, aircraft_model)
        plot_trace(dl, wind_dir_deg, wind_speed, air_rel_energy_change, trace_plot_interval, cmap_name='bwr',
                   colourbar_label='Air-relative specific power ($J s^{-1} kg^{-1}$)', plane_scale=1, save_folder=folderpath if save else None, save_name='trace_air_rel')

        # Animate traces (TODO complete)
        # dl, aircraft_model, anim_interval_ms=20, plane_scale=1, force_scale=1, wind_scale=1, plot_building=True, show_axes=False, n_min_range=None, e_min_range=None, d_min_range=None
        # anim = animate(dl, aircraft_model, (folderpath if save else None))


# ========================================================================
# New trace plotting code, with building projections.

# TODO I don't really like the fact that this requires object instantiation, but it's the best way that
# I could think to do it at the moment.

# ============
# Environments

# TODO Passing the plotter in is messy
class BaseEnv:
    building_n_range = None

    def plot_buildings(plotter, **kwargs):
        pass

    def set_axis_lims(plotter, all_xs, all_ys, all_zs):
        # Calculate axis limits
        x_lims = (np.min(all_xs) - 20, np.max(all_xs) + 20)
        y_lims = (np.min(all_ys) - 20, np.max(all_ys) + 20)
        z_lims = (np.min(all_zs) - 5, np.max(all_zs) + 5)

        return x_lims, y_lims, z_lims

# Original isolated building
class OrigIsolEnv(BaseEnv):
    building_n_range = (131, 369)

    def plot_buildings(plotter, **kwargs):
        b = Rectangle((131, 0), 238, 33, zorder=0, fc='lightgrey', ec='darkgrey')
        plotter.ax.add_patch(b)
        art3d.patch_2d_to_3d(b, z=190, zdir='x')
    
    def set_axis_lims(plotter, all_xs, all_ys, all_zs):
        # Calculate axis limits
        x_lims = (190, np.max(all_xs) + 20)
        y_lims = (np.min(all_ys) - 20, np.max(all_ys) + 20)
        z_lims = (0, np.max(all_zs) + 5)

        return x_lims, y_lims, z_lims

class CanyonEnv(BaseEnv):
    building_n_range = (100, 500)

    def plot_buildings(plotter, **kwargs):
        # Add buildings
        # Canyon far side
        b = Rectangle((100, 0), 400, 15, zorder=0, fc='lightgrey', ec='darkgrey')
        plotter.ax.add_patch(b)
        art3d.patch_2d_to_3d(b, z=140, zdir='x')

        if plotter.sep > 0:     # Not isolated case
            # Canyon near side
            # Side
            b = Rectangle((100, 0), 400, 15, zorder=10*kwargs['num_steps']-1, fc='lightgrey', ec=None, fill=True, alpha=0.3)
            plotter.ax.add_patch(b)
            art3d.patch_2d_to_3d(b, z=140 + plotter.sep, zdir='x')
            # Boundary
            b = Rectangle((100, 0), 400, 15, zorder=10*kwargs['num_steps'], fc='lightgrey', ec='darkgrey', fill=False) # alpha=0.1)
            plotter.ax.add_patch(b)
            art3d.patch_2d_to_3d(b, z=140 + plotter.sep, zdir='x')
    
    def set_axis_lims(plotter, all_xs, all_ys, all_zs):
        # Calculate axis limits
        if plotter.sep == 0:    # Isolated case
            x_lims = (140, np.max(all_xs + 20))
        else:
            x_lims = (140, 140 + plotter.sep)
        y_lims = (np.min(all_ys) - 20, np.max(all_ys) + 20)
        z_lims = (0, np.max(all_zs) + 10)

        return x_lims, y_lims, z_lims

class CanyonWindwardSideEnv(BaseEnv):
    building_n_range = (100, 500)

    def plot_buildings(plotter, **kwargs):
        # Add buildings
        # Canyon far side
        b = Rectangle((100, 0), 400, 15, zorder=0, fc='lightgrey', ec='darkgrey')
        plotter.ax.add_patch(b)
        art3d.patch_2d_to_3d(b, z=140, zdir='x')
    
    def set_axis_lims(plotter, all_xs, all_ys, all_zs):
        # Calculate axis limits
        x_lims = (140, np.max(all_xs) + 20)
        y_lims = (np.min(all_ys) - 20, np.max(all_ys) + 20)
        z_lims = (np.min(all_zs) - 5, np.max(all_zs) + 5)

        return x_lims, y_lims, z_lims

class TracePlotterBase:
    # What is going to be the structure of this?
    # 1) Create the basic frame - figure, title, etc

# dl, wind_dir_deg, wind_speed, title=None, plot_interval=None, traj_lw=1.25, plot_traj_centreline=False, show_grid=False, init_view_angles=None,
# colouring_vals=None, cmap_name='bwr', norm=None, zero_centred_norm=True, colourbar_label=None, label_0=True, save_colourbar=False,
# plane_scale=1, forces=None, wind_scale=None,
# save_folder=None, save_name=None, save_dpi=600):
    def __init__(self, env=BaseEnv, figsize=None, title=None, traj_lw=1.25, show_grid=False, projection_alpha=0.3, projection_lw=0.3, init_view_angles=None,
                 plane_scale=1,
                 save_folder=None, save_name=None, save_dpi=600):
        self.env = env
        
        self.figsize = figsize
        self.title = title

        self.traj_lw = traj_lw
        self.projection_alpha = projection_alpha
        self.projection_lw = projection_lw

        self.show_grid = show_grid
        self.init_view_angles = init_view_angles
        
        self.plane_scale = plane_scale

        self.save_folder = save_folder
        self.save_name = save_name
        self.save_dpi = save_dpi

        # For scaling and trajectory projections
        self.all_xs = np.array([])
        self.all_ys = np.array([])
        self.all_zs = np.array([])
    
    # Call to make plot, for any of the subclasses.
    # Override
    def plot_trace(self):
        pass

    def _setup(self, **kwargs):
        # TODO Do we need the 'constrained_layout=True'
        self.fig = plt.figure(figsize=self.figsize) # figsize=(12, 8))
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.computed_zorder = False

        # Label axes
        labelpad = 10
        self.ax.set_xlabel(r'$x_{\,[e]}$ ($m$)', labelpad=labelpad)
        self.ax.set_ylabel('\n\n$y_{\\,[n]}$ ($m$)', labelpad=labelpad)
        self.ax.set_zlabel(r'$z_{\,[-d]}$ ($m$)', labelpad=labelpad)

        # Set grid visibility
        self.ax.grid(visible=self.show_grid)

        # Plot buildings
        self.env.plot_buildings(self, **kwargs)

        # Set title
        self._set_title()

        # Set view angle
        if self.init_view_angles is not None:
            self.ax.view_init(self.init_view_angles[0], self.init_view_angles[1], self.init_view_angles[2])
        else:
            self.ax.view_init(30, -45, 0)
    
    # Baseclass functionality
    def _add_flight(self, xs, ys, zs):
        # Add to ranges
        self.all_xs = np.concatenate((self.all_xs, xs))
        self.all_ys = np.concatenate((self.all_ys, ys))
        self.all_zs = np.concatenate((self.all_zs, zs))

        # Need to know:
        #  o Whether to plot as a line or a line segment. This will depend on whether it is single or multiple.
        #  o Single - plot as a line if colouring_vals is None, plot as a line segment if they're given.

        # Will also do the potential centreline plotting here.

        # I need to be careful about the zorders here
        # How does this work when there is only one flight vs when there are many?
        # Maybe each flight has its own individual colourbar? But this doesn't seem like a good way of doing it.

        # Also add the aircraft plots here

        # Every time an aircraft is plotted, adjust the minimum and maximum values. This will be useful
        # when determining the plotting ranges in the scalings, and also for determining the plotting
        # coordinates.

        # Will also require custom legends, depending on the type.

        # Colouring_vals will only be provided in the single case.
        # What about the norm? The norm maps the value - whatever it is - wind speed or direction
        # to a colour. Is it easier if I pass in a norm, or just a ScalarMappable?
        # If it is a single, it can take a norm, which might be required if I am comparing
        # multiple flights.
        # If a norm isn't provided, it will want to calculate one itself. So it is important
        # to provide both separately.
    
    def _add_projection(self, dl, proj_colour, zorder):
        xs, ys, zs = ned_to_xyz(dl.ns, dl.es, dl.ds)
        
        ones = np.ones(dl.num_steps)
        self.ax.plot(ones*self.x_proj_coord, ys, zs, c=proj_colour, alpha=self.projection_alpha, lw=self.projection_lw, ls='--', zorder=zorder)
        self.ax.plot(xs, ones*self.y_proj_coord, zs, c=proj_colour, alpha=self.projection_alpha, lw=self.projection_lw, ls='--', zorder=zorder)
        self.ax.plot(xs, ys, ones*self.z_proj_coord, c=proj_colour, alpha=self.projection_alpha, lw=self.projection_lw, ls='--', zorder=zorder)

    # Sets scale, colourbar etc - depends on flight data added
    def _post_formatting(self):
        # Set ticks
        self._set_ticks(self.all_xs, self.all_ys, self.all_zs)

        x_lims, y_lims, z_lims = self.env.set_axis_lims(self, self.all_xs, self.all_ys, self.all_zs)

        # Set projection coordinates
        self.x_proj_coord = x_lims[0]
        self.y_proj_coord = y_lims[1]
        self.z_proj_coord = z_lims[0]

        # Add colourbar, direction rose, etc
        self._add_cbar_or_legend()

        # Decorate (e.g. add start and end point/line, wind arrow)
        self._decorate(x_lims, y_lims, z_lims)

        # Scale plot
        self.ax.autoscale(False)
        self.ax.set_xlim(*x_lims)
        self.ax.set_ylim(*y_lims)
        self.ax.set_zlim(*z_lims)
        self.ax.set_aspect('equal')

    # TODO Functionality to save colourbar separately not currently implemented
    def _save(self):
        if (self.save_folder is not None) and (self.save_name is not None):
            # Save trace
            self.save_path = self.save_folder / f'{self.save_name}_trace'
            crop_and_save(self.fig, self.save_path, save_png=True)
            # crop_and_save(self.fig, self.save_path, 'png', self.save_dpi)
    
    # TODO Needs testing
    # Save a copy of the image to another location
    def make_copy(self, copy_dir):
        if self.save_path is None:
            raise Exception("Image wasn't saved - can't produce copy")
        
        # Copy
        shutil.copy(f'{str(self.save_path)}.png', str(copy_dir))
    
    def _set_title(self):
        self.ax.set_title(self.title, loc='left', y=0.7, color=text_colour)
    
    def _set_ticks(self, all_xs, all_ys, all_zs):
        self.ax.set_xticks((math.floor(np.min(all_xs)), math.ceil(np.max(all_xs))))
        self.ax.set_yticks(
            np.concatenate((
                np.array([round(v) for v in np.linspace(np.min(all_ys), np.max(all_ys), 10)])[1:-1],
                [math.floor(np.min(all_ys))],
                [math.ceil(np.max(all_ys))]
            ))
        )
        self.ax.set_zticks((math.floor(np.min(all_zs)), math.ceil(np.max(all_zs))))
    
    def _decorate(self, x_lims, y_lims, z_lims):
        pass
    
    def _add_cbar_or_legend(self):
        pass

    def _plot_start_pos(self, col, x, y, z, zorder=1e6):
        self.ax.plot(x, y, z, c=col, marker='o', zorder=zorder, markersize=4)
    
    # z_lims is a 2-tuple
    def _plot_end_line(self, col, x, y, z_lims, zorder=1e6):
        # TODO Get the values from the arguments
        self.ax.plot([x, x], [y, y], z_lims, c=col, ls='--', zorder=zorder)
    
    def _plot_wind_arrow(self, wind_arrow_anchor_x, wind_arrow_anchor_y, wind_arrow_anchor_z, wd):
        head_width = 0.3
        lw = 1.5
        wind_arrow_scale = 30
        wind_arrow = Arrow3D([wind_arrow_anchor_x, wind_arrow_anchor_x], [wind_arrow_anchor_y, wind_arrow_anchor_y], [wind_arrow_anchor_z, wind_arrow_anchor_z], mutation_scale=10, lw=lw, arrowstyle=ArrowStyle.CurveFilledB(head_width=head_width), color='orange', zorder=0)
        self.ax.add_artist(wind_arrow)
        wind = np.array([-np.sin(np.radians(wd)), -np.cos(np.radians(wd)), 0])
        wind_arrow.update_3d_posns([wind_arrow_anchor_x, wind_arrow_anchor_x + wind_arrow_scale*wind[0]], [wind_arrow_anchor_y, wind_arrow_anchor_y + wind_arrow_scale*wind[1]], [wind_arrow_anchor_z, wind_arrow_anchor_z + wind_arrow_scale*wind[2]])


class TracePlotterSingle(TracePlotterBase):
    def __init__(self, env, dl, colouring_vals=None, plot_interval=None, plot_traj_centreline=False,
                 cmap_name='bwr', norm=None, zero_centred_norm=True, colourbar_label=None, label_0=True,
                 forces=None, wind_scale=None,
                 **kwargs):
        
        if not 'figsize' in kwargs:
            kwargs['figsize'] = (12, 8)

        super().__init__(env, **kwargs)

        self.dl = dl
        self.colouring_vals = colouring_vals
        self.plot_interval = plot_interval
        self.plot_traj_centreline = plot_traj_centreline
        
        self.cmap_name = cmap_name
        self.norm = norm
        self.zero_centred_norm = zero_centred_norm
        self.colourbar_label = colourbar_label
        self.label_0 = label_0

        self.forces = forces
        self.wind_scale = wind_scale

        self.smap = None
    
    @classmethod
    def plot_trace(cls, env, dl, colouring_vals=None, plot_interval=None, **kwargs):
        plotter = cls(env, dl, colouring_vals, plot_interval, **kwargs)
        plotter._plot_trace()
    
    def _plot_trace(self, **kwargs):
        self._setup(**kwargs)
        
        # Plot trajectory for this specific flight
        self._add_flight(self.dl)

        self._plot_aircraft(self.dl)
        
        self._post_formatting()
        
        # Plot trajectory projection
        self._add_projection(self.dl, 'grey', 1)

        self._save()

    def _add_flight(self, dl):
        xs, ys, zs = ned_to_xyz(dl.ns, dl.es, dl.ds)

        super()._add_flight(xs, ys, zs)
        
        if self.colouring_vals is None:
            self.ax.plot(xs, ys, zs, lw=self.traj_lw, color='lightseagreen', zorder=2)
        else:
            if self.norm is not None:    # If a norm to use has not already been provided..
                norm = self.norm
                print("Using given norm - may not be zero-centred!")
            else:
                # Generate the norm
                norm, _, _ = gen_norm(self.colouring_vals, self.zero_centred_norm)
            
            # Create colourmap - for colouring trace by 'colouring_vals'
            cmap = mpl.colormaps[self.cmap_name]
            self.smap = mpl.cm.ScalarMappable(norm, cmap)

            # Create empty trace, then populate.
            plane_trace = art3d.Line3DCollection([], cmap=cmap, norm=norm, lw=self.traj_lw)
            update_trace(plane_trace, xs, ys, zs, self.colouring_vals)       # Populate trace
            # Add trace to axis
            self.ax.add_collection(plane_trace)
            
            if self.plot_traj_centreline:
                self.ax.plot(xs, ys, zs, lw=0.2, color='grey', ls='--')

    def _plot_aircraft(self, dl):
        if self.plot_interval is not None:
            step = self.plot_interval
            while step < self.dl.num_steps:
                # Get x, y and z coordinates
                # Matplotlib uses an xyz coordinate system. Convert from ned:
                n = self.dl.ns[step]
                e = self.dl.es[step]
                d = self.dl.ds[step]
                x, y, z = ned_to_xyz(n, e, d)

                # Plot vehicle
                Plane(self.ax, self.plane_scale, x, y, z, self.dl.phis[step], self.dl.thetas[step], self.dl.psis[step],
                    'grey' if self.colouring_vals is None else get_col_str(self.smap, self.colouring_vals[step]),
                    linewidth=0.5, zorder=2*self.dl.num_steps)
                
                # Plot forces
                if self.forces is not None:
                    print("WARNING: force plotting is currently restricted to inertial forces only - doesn't include fictitious force.")
                    # TODO Temp hack
                    force_scaler_func = lambda x: 1*x # TODO HARDCODED - fix this
                    # TODO Test whether forces is in the right format
                    ForcePlotter(self.ax, force_scaler_func, arrowhead_width, plot_inds=[2]).update(n, e, d, self.forces[:, step, :])

                # Plot wind (dotted line)
                # TODO Use 'is not None', or just set length to 0 to hide and add logic which hides it, like I've done previously?
                if self.wind_scale is not None:
                    wn = self.dl.wns[step]
                    we = self.dl.wes[step]
                    wd = self.dl.wds[step]
                    # Convert to xyz
                    wx, wy, wz = ned_to_xyz(wn, we, wd)
                    
                    # print("NOTE: Plotted wind field is only correct in the inertial frame!")
                    self.ax.add_artist(Arrow3D([x, x + self.wind_scale*wx], [y, y + self.wind_scale*wy], [z, z + self.wind_scale*wz], mutation_scale=10, lw=0.5, arrowstyle=ArrowStyle.CurveFilledB(head_width=arrowhead_width), color='orange', linestyle='-', zorder=2*self.dl.num_steps+1))
                
                step += self.plot_interval
    
    def _add_cbar_or_legend(self):
        if self.colouring_vals is not None:
            cbax = self.ax.inset_axes((0.5, 0.17, 0.4, 0.04))
            if self.label_0:
                # FLoor and ceil to 1.d.p
                plt.colorbar(self.smap, cbax, label=self.colourbar_label, orientation='horizontal', ticks=[np.min(self.colouring_vals), 0, np.max(self.colouring_vals)], format="%.1f")
            else:
                plt.colorbar(self.smap, cbax, label=self.colourbar_label, orientation='horizontal', ticks=[np.min(self.colouring_vals), np.max(self.colouring_vals)], format="%.1f")


# This is complicated - sometimes they need to all be the same colour (e.g. as in the clustering),
# and sometimes they need to be different colours (e.g. when the wind speed is increasing).
# I suppose they are still getting their colours from the colourmap/smap in both instances, it's just that the number
# remains the same if it is the same cluster.

class TracePlotterMultipleBase(TracePlotterBase):
    def __init__(self, env, smap, **kwargs):
        
        #if not 'figsize' in kwargs:
        #    kwargs['figsize'] = (12, 8)

        super().__init__(env, **kwargs)

        # self.dls = dls
        self.smap = smap

    # flight_ind is from 1.
    def _add_flight(self, dl, wd, ws, colour, flight_ind, num_flights, zorder_base):
        xs, ys, zs = ned_to_xyz(dl.ns, dl.es, dl.ds)

        super()._add_flight(xs, ys, zs)

        self.ax.plot(xs, ys, zs, lw=self.traj_lw, color=colour, zorder=zorder_base,
                     label=f"({remove_trailing_0pt(wd)}, {remove_trailing_0pt(ws)})",
                     ls='-' if dl.opt_success else '--')
        
        # Plot aircraft along the trajectories
        # Find the n locations along the trajectory
        if self.env.building_n_range is not None:
            try:
                # Ratio of progress through the set of values to be plotted
                ratio = (flight_ind - 1) / (num_flights - 1)
            except ZeroDivisionError:    # If there is only one flight
                ratio = 0.5
            n_target = (self.env.building_n_range[1] - 10) - ratio*((self.env.building_n_range[1] - 10) - (self.env.building_n_range[0] + 10))
            n_idx = np.argmin(np.abs(dl.ns - n_target))
            Plane(self.ax, self.plane_scale, xs[n_idx], ys[n_idx], zs[n_idx], dl.phis[n_idx], dl.thetas[n_idx], dl.psis[n_idx], colour=colour, linewidth=0.5, zorder=zorder_base+1)
    
    # Default to base-class function
    #def _add_projection(self, dl, proj_colour, zorder):
    #    return super()._add_projection(dl, proj_colour, zorder)

class TracePlotterMultiple(TracePlotterMultipleBase):
    # col_vals are for colouring the trajectories
    def __init__(self, env, dls, wds, wss, seps, col_vals, colourbar_label=None, **kwargs):
        
        if not 'figsize' in kwargs:
            kwargs['figsize'] = (12, 8)

        super().__init__(env, **kwargs)

        self.dls = dls
        self.wds = wds
        self.wss = wss
        self.seps = seps
        # if len(seps == 1):
        # TODO Made a change - is it right?
        if len(np.unique(seps)) == 1:
            self.sep = seps[0]
        self.col_vals = col_vals
        self.colourbar_label = colourbar_label
    
    # col_vals are for colouring the trajectories
    @classmethod
    def plot_trace(cls, env, dls, wds, wss, seps, col_vals, num_steps=1e20, **kwargs):
        plotter = cls(env, dls, wds, wss, seps, col_vals, **kwargs)
        plotter._plot_trace(num_steps=num_steps)
    
    def _plot_trace(self, **kwargs):
        self._setup(**kwargs)
        
        # Plot trajectory for this specific flight
        zorder_base = 10
        for i in range(len(self.dls)):
            if self.smap is None:
                # This simple example doesn't use the smap to generate a colour
                col = "lightseagreen"
            else:
                col = get_col_str(self.smap, self.col_vals[i])
            self._add_flight(self.dls[i], self.wds[i], self.wss[i], col, i+1, len(self.dls), zorder_base)
            zorder_base += 2
                
        self._post_formatting()
        
        # Plot trajectory projection
        for i in range(len(self.dls)):
            if self.smap is None:
                # This simple example doesn't use the smap to generate a colour
                col = "lightseagreen"
            else:
                col = get_col_str(self.smap, self.col_vals[i])
            self._add_projection(self.dls[i], col, 1)

        self._save()
    
    # TODO This is hacky
    def _add_cbar_or_legend(self):
        # if self.colouring_vals is not None:
        cbax = self.ax.inset_axes((0.5, 0.17, 0.4, 0.04))
        #if self.label_0:
        #    # FLoor and ceil to 1.d.p
        #    plt.colorbar(self.smap, cbax, label=self.colourbar_label, orientation='horizontal', ticks=[np.min(self.colouring_vals), 0, np.max(self.colouring_vals)], format="%.1f")
        #else:
        plt.colorbar(self.smap, cbax, label=self.colourbar_label, orientation='horizontal', ticks=[self.smap.norm.vmin, self.smap.norm.vmax], format="%.1f")
    


# wd = wind direction
# ws = wind speed
class TracePlotterSingleOrigBld(TracePlotterSingle):
    def __init__(self, dl, wd, ws, **kwargs):
        super().__init__(OrigIsolEnv, dl, **kwargs)
        
        self.wd = wd
        self.ws = ws
    
    @classmethod
    def plot_trace(cls, dl, wd, ws, **kwargs):
        plotter = cls(dl, wd, ws, **kwargs)
        plotter._plot_trace()

    def _set_title(self):
        if self.title is None:
            self.title = f'Wind speed: ${remove_trailing_0pt(self.ws)}\\,ms^{{-1}}$\nWind direction: ${remove_trailing_0pt(self.wd)}^\\circ$'        
        super()._set_title()
    
    #def _decorate(self, x_lims, y_lims, z_lims):
    #    # Plot starting and ending target positions
    #    bc_colour = 'orangered'
    #    start_xyz = ned_to_xyz(400, 205, -20)
    #    # NOTE These take x, y, z; not n, e, d.
    #    self._plot_start_pos(bc_colour, *start_xyz, 3*self.dl.num_steps)
    #    self._plot_end_line(bc_colour, 205, 100, z_lims, 3*self.dl.num_steps)
    
    def _decorate(self, x_lims, y_lims, z_lims):
        # Plot start and end positions
        # super()._decorate(x_lims, y_lims, z_lims)
        
        # Plot starting and ending target positions
        bc_colour = 'orangered'
        start_xyz = ned_to_xyz(400, 205, -20)
        # NOTE These take x, y, z; not n, e, d.
        self._plot_start_pos(bc_colour, *start_xyz, 3*self.dl.num_steps)
        self._plot_end_line(bc_colour, 205, 100, z_lims, 3*self.dl.num_steps)
        
        # Plot wind arrow
        self._plot_wind_arrow(205, 100, 0, self.wd)

class TracePlotterCanyon(TracePlotterSingle):
    def __init__(self, dl, wd, ws, sep, start_and_end_e_fixed, **kwargs):
        super().__init__(CanyonEnv, dl, **kwargs)

        self.wd = wd
        self.ws = ws
        self.sep = sep
        self.start_and_end_e_fixed = start_and_end_e_fixed

    @classmethod
    def plot_trace(cls, dl, wd, ws, sep, start_and_end_e_fixed, **kwargs):
        plotter = cls(dl, wd, ws, sep, start_and_end_e_fixed, **kwargs)
        plotter._plot_trace(num_steps=dl.num_steps)
        return plotter      # In case it's needed by the calling code
    
    # Include the buildling separation
    #def _set_title(self):
    #    self.title = '\n'.join([
    #        f'Wind speed: ${remove_trailing_0pt(self.ws)}\\,ms^{{-1}}$',
    #        f'Wind direction: ${remove_trailing_0pt(self.wd)}^\\circ$',
    #        f'Building separation: ${self.sep}\\,m$'
    #        ])
    #    # ax.set_title(self.title, loc='left', y=0.7, color=text_colour)
    #    super()._set_title()
    
    def _set_ticks(self, all_xs, all_ys, all_zs):
        self.ax.set_xticks((math.floor(np.min(all_xs)), math.ceil(np.max(all_xs)), 140 + self.sep))
        self.ax.set_yticks(
            np.concatenate((
                np.array([round(v) for v in np.linspace(np.min(all_ys), np.max(all_ys), 10)])[1:-1],
                [math.floor(np.min(all_ys))],
                [math.ceil(np.max(all_ys))]
            ))
        )
        self.ax.set_zticks((math.floor(np.min(all_zs)), math.ceil(np.max(all_zs))))
    
    def _decorate(self, x_lims, y_lims, z_lims):
        if self.start_and_end_e_fixed:
            # Plot starting and ending target positions
            bc_colour = 'orangered'
            start_xyz = ned_to_xyz(530, 140 + self.sep/2, -15)
            # NOTE These take x, y, z; not n, e, d.
            self._plot_start_pos(bc_colour, *start_xyz)
            end_x = 140 + self.sep/2
            end_y = 70
            self._plot_end_line(bc_colour, end_x, end_y, z_lims)

            self._plot_wind_arrow(end_x, end_y, 0, self.wd)
        else:
            self._plot_wind_arrow(x_lims[1], y_lims[0], 0, self.wd)


class TracePlotterMultipleCanyon(TracePlotterMultiple):
    def __init__(self, dls, sep, **kwargs):
        super().__init__(CanyonEnv, dls)

        self.sep = sep

    @classmethod
    def plot_trace(cls, dls, sep):
        plotter = cls(dls, sep)
        # Hack
        plotter._plot_trace(num_steps=dls[0].num_steps)


# ==============

def colour_trace_by_wind(dl, wind_dir_deg, wind_speed, save_folder, save_dpi=600):
    vals = [dl.wns, dl.wes, dl.wds]
    titles = ['$W_n$ (wind in north direction)', '$W_e$ (wind in east direction)', '$W_d$ (wind in down direction)']
    cbar_labels = ['$W_n$ ($m s^{-1}$)', '$W_e$ ($m s^{-1}$)', '$W_d$ ($m s^{-1}$)']
    names = ['wn', 'we', 'wd']

    cmap = 'bwr' # 'PiYG'

    for colouring_vals, title, cbar_label, name in zip(vals, titles, cbar_labels, names):
        plot_trace(dl, wind_dir_deg, wind_speed, title=title, plot_interval=None, traj_lw=2, plot_traj_centreline=True, init_view_angles=None,
                    colouring_vals=colouring_vals, cmap_name=cmap, norm=None, zero_centred_norm=True, colourbar_label=cbar_label, label_0=True,
                    plane_scale=1, forces=None, wind_scale=0,
                    save_folder=save_folder, save_name=name, save_dpi=save_dpi)


if __name__ == "__main__":
    # Load DataLogger
    dl = DataLogger.load_from_path(Path(r'C:\Users\y3178\Desktop\MsDissertation\FlightSwordLite\data\dataloggers\test_jiayi_2_optimiser')) # Path to DataLogger here...
    aircraft_model = AircraftModel(config.base_aircraft_model_path / 'wot4_imav_v2.yaml')
    generate_plots(None, dl, aircraft_model)

    colouring_vals_air_rel = analysis_calculations.calc_total_specific_air_relative_energy_change(dl, aircraft_model)
    # Plot trace, coloured by air-relative power
    wind_dir = 270
    wind_speed = 5 # This will depend on the case run
    plot_trace(dl, wind_dir, wind_speed, title=None, plot_interval=None, traj_lw=2, plot_traj_centreline=False,
                                colouring_vals=colouring_vals_air_rel, colourbar_label='Specific air-relative power ($J kg^{-1} s^{-1}$)',
                                plane_scale=1, wind_scale=0, save_folder=None, save_name=None, save_dpi=None)
    
    plt.show(block=True)