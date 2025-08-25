# FT 12/9/24

# Calculations for work and energy

# Calculate inertial energy
# Calculate air-relative energy

import numpy as np
import matplotlib.pyplot as plt

# Add the flight directory to the Python path (when running from FlightSwordLite)
import sys, os
sys.path.append(os.getcwd())

from flight.simulator.utils import calc_body_to_inertial, calc_wind_to_body

# Returns the forces in the NED frame
# Note: this assumes that the air-relative reference frame has i^a, j^a and k^a
# axis vectors which are aligned with the inertial ned frame of reference. 
def calc_forces_ned(dl):
    # L, C and D are calculated in the air-relative reference frame
    # T is calculated in the body frame - it's aligned with i^b
    # We want to resolve the forces in the inertial frame (or even if in the air-relative
    # frame, parallel to the inertial frame ned axis vectors). We start by finding
    # the representation of each wind frame axis vector (i^w, j^w and k^w) in the
    # inertial frame.
    # 's' because this is an array of rotation matrices - one for each point in time depending
    # on the aircraft's orientation at that point.
    r_bi_s = calc_body_to_inertial(dl.phis, dl.thetas, dl.psis)
    r_wb_s = calc_wind_to_body(dl.alphas, dl.betas)
    # TODO Check the matmul broadcasting rules
    r_wi_s = np.matmul(r_bi_s, r_wb_s)
    
    # For the D, C and L forces, find the inertial frame representations
    # of the wind frame axis vectors. 
    # TODO Need to check the matmul broadcasting on these
    iws = np.matmul(r_wi_s, np.array([1, 0, 0]))
    jws = np.matmul(r_wi_s, np.array([0, 1, 0]))
    kws = np.matmul(r_wi_s, np.array([0, 0, 1]))
    
    # Body frame axis vectors - find the representation of the i^bs in the inertial frame.
    # TODO Check that this matmul works as anticipated
    ibs = np.matmul(r_bi_s, np.array([1, 0, 0]))

    # For weight force - inertial k (down) vector
    # TODO The shape of this needs checking
    kis = np.repeat([[0, 0, 1]], dl.num_steps, axis=0)

    # Calculate forces in ned frame.
    # TODO The shapes of all of these need changing
    drag_forces_ned = -dl.Ds[:, np.newaxis]*iws
    side_forces_ned = dl.Cs[:, np.newaxis]*jws
    lift_forces_ned = -dl.Ls[:, np.newaxis]*kws
    thrust_forces_ned = dl.Ts[:, np.newaxis]*ibs
    # TODO Check this - it could be completely wrong.
    # TODO And check the shape of this.
    # TODO All of these forces still need putting into the DataLogger 
    gravity_forces_ned = dl.Ws[:, np.newaxis]*kis
    wind_fic_forces_ned = np.array([dl.Fwns, dl.Fwes, dl.Fwds]).T

    # If we don't have this information immediately, we can simulate it to get it.

    # Note: can calculate the resultant aerodynamic force by adding the components together. This would be easier on the eye
    # the plotting all of them.

    # Stacked in this way because it enables the dot products to be calculated later. See the custom dot()
    # function.
    forces_i = np.stack((drag_forces_ned, side_forces_ned, lift_forces_ned, thrust_forces_ned, gravity_forces_ned), axis=0)

    # Wind fictitious forces are returned separately
    return forces_i, wind_fic_forces_ned

def calc_inertial_forces_ned(dl):
    forces_i, _ = calc_forces_ned(dl)
    return forces_i

def calc_air_relative_forces_ned(dl):
    forces_i, wind_fic_forces_ned = calc_forces_ned(dl)
    return np.concatenate((forces_i, [wind_fic_forces_ned]), axis=0)

def dot(a, b):
    if a.shape != b.shape:
        raise Exception("Arrays must be of the same shape")
    
    return np.sum(a*b, axis=-1)

# def calc_work_rates(vel_vec, forces):
#     # TODO Check the shapes of this.
#     # This should work.
#     return dot(vel_vec, forces)

def calc_inertial_work_rates(dl):
    # Calculate the work rate of each force (including the fictitious force)
    # For this, we need each force
    # Then we just calculate the dot product with the speed in the appropriate reference frame to get the result.
    # These are the forces in the inertial ned reference frame
    forces = calc_inertial_forces_ned(dl)

    # Work is the integral of the dot product of force and velocity
    # So work rate is the dot product of force and velocity
    # 1. Get the (inertial, in body frame) velocity vector at each point
    vi_b_vecs = np.array([dl.us, dl.vs, dl.ws]).T
    # Convert body frame velocity to inertial frame
    # This needs checking
    vi_vecs = np.squeeze(np.matmul(calc_body_to_inertial(dl.phis, dl.thetas, dl.psis), vi_b_vecs[:, :, np.newaxis]))
    
    # 2. Take the dot product with each force to get the work rate
    fds, fcs, fls, fts, fgs = forces
    # dw...i = 'work rate' (inertial)
    # Drag, sideforce, lift, thrust, gravity work rate
    # dw_d_i, dw_c_i, dw_l_i, dw_t_i, dw_g_i = calc_work_rates(vi_vecs, forces)
    dw_d_i = dot(vi_vecs, fds)
    dw_c_i = dot(vi_vecs, fcs)
    dw_l_i = dot(vi_vecs, fls)
    dw_t_i = dot(vi_vecs, fts)
    dw_g_i = dot(vi_vecs, fgs)

    return {
        'dw_d': dw_d_i,
        'dw_c': dw_c_i,
        'dw_l': dw_l_i,
        'dw_t': dw_t_i,
        'dw_g': dw_g_i
    }

def calc_air_relative_work_rates(dl):
    # Calculate the work rate of each force (including the fictitious force)
    # For this, we need each force
    # Then we just calculate the dot product with the speed in the appropriate reference frame to get the result.
    # These are the forces in the inertial ned reference frame
    forces = calc_air_relative_forces_ned(dl)

    # Work is the integral of the dot product of force and velocity
    # So work rate is the dot product of force and velocity
    # 1. Get the (air_relative) velocity vector at each point
    # Get inertial body frame vector
    vi_b_vecs = np.array([dl.us, dl.vs, dl.ws]).T
    # Convert body frame velocity to inertial frame
    # This needs checking
    vi_vecs = np.squeeze(np.matmul(calc_body_to_inertial(dl.phis, dl.thetas, dl.psis), vi_b_vecs[:, :, np.newaxis]))
    # Subtract wind, to get air-relative velocity (axis vectors are parallel to those of inertial frame)
    wind = np.array([dl.wns, dl.wes, dl.wds]).T
    va_vecs = vi_vecs - wind

    # How to do the translation to air-relative velocity? Add on the wind.
    
    # 2. Take the dot product with each force to get the work rate
    fds, fcs, fls, fts, fgs, ffws = forces  # Last one is the fictitious wind force
    # dw...a = 'work rate' (air-relative)
    # Drag, sideforce, lift, thrust, gravity, fictitious wind work rate
    # dw_d_a, dw_c_a, dw_l_a, dw_t_a, dw_g_a, dw_fw_a = calc_work_rates(v_a_vecs, forces)
    dw_d_a = dot(va_vecs, fds)
    dw_c_a = dot(va_vecs, fcs)
    dw_l_a = dot(va_vecs, fls)
    dw_t_a = dot(va_vecs, fts)
    dw_g_a = dot(va_vecs, fgs)
    dw_fw_a = dot(va_vecs, ffws)     # fictitious wind

    return {
        'dw_d': dw_d_a,
        'dw_c': dw_c_a,
        'dw_l': dw_l_a,
        'dw_t': dw_t_a,
        'dw_g': dw_g_a,
        'dw_fw': dw_fw_a
    }

def calc_total_specific_inertial_energy_change(dl, aircraft_model):
    # Energy change is just the sum of the work rates excluding gravity, since gravity cancels out.
    # TODO Write up workings from notebook
    work_rates = calc_inertial_work_rates(dl)
    return (work_rates['dw_d'] + work_rates['dw_c'] + work_rates['dw_l'] + work_rates['dw_t']) / aircraft_model.m

def calc_total_specific_air_relative_energy_change(dl, aircraft_model):
    # "Energy change is just the sum of the work rates excluding gravity, since gravity cancels out."
    # Actually, I don't think this is true in the air-relative frame (only in the inertial one) - 
    # there is an extra term left over. See the workings in my thesis.
    # Can we validate this - is there an easier way to get thet total change?
    # Yes - can sum up the drag, throttle, static and gradient terms.
    # See the function '' below for validation of this.
    # TODO Write up workings from notebook
    #work_rates = calc_air_relative_work_rates(dl)
    #total_spec_de_a_work_sum = (work_rates['dw_d'] + work_rates['dw_c'] + work_rates['dw_l'] + work_rates['dw_t'] + work_rates['dw_fw']) / aircraft_model.m
    # return total_spec_de_a_work_sum, total_spec_de_a_comps
    
    comps = calc_specific_air_relative_energy_change_components(dl, aircraft_model)
    total_spec_air_rel_energy_change = comps['drag'] + comps['throttle'] + comps['static'] + comps['gradient']
    return total_spec_air_rel_energy_change 


def calc_specific_air_relative_energy_change_components(dl, aircraft_model):
    m = aircraft_model.m
    g = aircraft_model.g

    drag_losses = -(dl.Ds*dl.vas)/m
    static_power = -g*dl.wds

    # Calculate va vectors
    vi_b_vecs = np.array([dl.us, dl.vs, dl.ws]).T
    # Convert body frame velocity to inertial frame
    vi_vecs = np.squeeze(np.matmul(calc_body_to_inertial(dl.phis, dl.thetas, dl.psis), vi_b_vecs[:, :, np.newaxis]))
    # Subtract wind, to get air-relative velocity (axis vectors are parallel to those of inertial frame)
    wind = np.array([dl.wns, dl.wes, dl.wds]).T
    va_vecs = vi_vecs - wind

    # Construct T vectors
    # TODO Assumes that thrust acts in the body frame - is this assumption correct?!
    # Get ib unit vector in the ned inertial coordinate system
    r_bi_s = calc_body_to_inertial(dl.phis, dl.thetas, dl.psis)
    ibs = np.matmul(r_bi_s, np.array([1, 0, 0]))
    T_vecs = dl.Ts[:, np.newaxis]*ibs
    # Take dot product and divide by m
    throttle_power = (T_vecs*va_vecs).sum(axis=1)/m

    # Calculate gradient power
    # '-' (negative) is subsumed in the definition
    # Fictitious wind force
    wind_fic_force = np.array([dl.Fwns, dl.Fwes, dl.Fwds]).T
    # Take dot product and divide by m (to see why, see the dynamics code).
    # From dynamics code: Fwn, Fwe, Fwd = -aircraft_model.m*(jnp.matmul(jac, vi) + time_deriv)
    gradient_power = (va_vecs*wind_fic_force).sum(axis=1)/m

    # # Testing only
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(3)
    # axs[0].plot(va_vecs)
    # axs[1].plot(wind_fic_force)
    # axs[2].plot(gradient_power)
    # plt.show(block=True)

    return {
        'drag': drag_losses,
        'throttle': throttle_power,
        'static': static_power,
        'gradient': gradient_power
    }

def calc_aerodynamic_specific_work_rate(dl, aircraft_model):
    work_rates = calc_inertial_work_rates(dl)
    # Work done by aerodynamic force
    return (work_rates['dw_d'] + work_rates['dw_c'] + work_rates['dw_l']) / aircraft_model.m

# How to project aerodynamic force onto direction of inertial travel?  (ib = body axis i-vector)
def calc_aerodynamic_force_ib_projection(dl):
    # Get aerodynamic force vector
    # These forces are expressed in the ned inertial coordinate system
    fds, fcs, fls, _, _ = calc_inertial_forces_ned(dl)
    f_aero_ned = fds + fcs + fls
    # Get ib unit vector in the ned inertial coordinate system
    r_bi_s = calc_body_to_inertial(dl.phis, dl.thetas, dl.psis)
    ibs = np.matmul(r_bi_s, np.array([1, 0, 0]))
    # Get the projection
    # Taking the dot product of the pairs
    forward_aerodynamic = (f_aero_ned*ibs).sum(axis=1)

    return forward_aerodynamic

def calc_aerodynamic_force_vi_unit_projection(dl):
    # Get aerodynamic force vector
    # These forces are expressed in the ned inertial coordinate system
    fds, fcs, fls, _, _ = calc_inertial_forces_ned(dl)
    f_aero_ned = fds + fcs + fls

    # Calculate normalised vi vectors (direction vectors of inertial travel)
    vi_b_vecs = np.array([dl.us, dl.vs, dl.ws]).T
    # Convert body frame velocity to inertial frame
    vi_vecs = np.squeeze(np.matmul(calc_body_to_inertial(dl.phis, dl.thetas, dl.psis), vi_b_vecs[:, :, np.newaxis]))
    # Normalise
    norms = np.linalg.norm(vi_vecs, axis=1)
    vi_vecs_norm = (1/norms)[:, np.newaxis]*vi_vecs

    # Get the projection
    # Taking the dot product of the pairs
    vi_aerodynamic = (f_aero_ned*vi_vecs_norm).sum(axis=1)

    return vi_aerodynamic

def calc_aerodynamic_force_vi_unit_projection_vecs_ned(dl):
    # Get aerodynamic force vector
    # These forces are expressed in the ned inertial coordinate system
    fds, fcs, fls, _, _ = calc_inertial_forces_ned(dl)
    f_aero_ned = fds + fcs + fls

    # Calculate normalised vi vectors (direction vectors of inertial travel)
    vi_b_vecs = np.array([dl.us, dl.vs, dl.ws]).T
    # Convert body frame velocity to inertial frame
    vi_vecs = np.squeeze(np.matmul(calc_body_to_inertial(dl.phis, dl.thetas, dl.psis), vi_b_vecs[:, :, np.newaxis]))
    # Normalise
    norms = np.linalg.norm(vi_vecs, axis=1)
    vi_vecs_norm = (1/norms)[:, np.newaxis]*vi_vecs

    # Get the projection
    # Taking the dot product of the pairs
    vi_aerodynamic = (f_aero_ned*vi_vecs_norm).sum(axis=1)

    return vi_aerodynamic[:, np.newaxis]*vi_vecs_norm

def calc_groundspeed(dl):
    return np.sqrt(np.square([dl.us, dl.vs, dl.ws]).sum(axis=0))

# ==========

# TODO The code in this section needs testing

# TODO What is the format (e.g. shape) of the return values?
# Calculate normalised vi vectors (direction vectors of inertial travel)
def calc_aero_ib_projection_vecs_ned(dl):
    vi_b_vecs = np.array([dl.us, dl.vs, dl.ws]).T
    # Convert body frame velocity to inertial frame
    vi_vecs_ned = np.squeeze(np.matmul(calc_body_to_inertial(dl.phis, dl.thetas, dl.psis), vi_b_vecs[:, :, np.newaxis]))
    # Normalise
    norms_ned = np.linalg.norm(vi_vecs_ned, axis=1)
    vi_vecs_ned_norm = (1/norms_ned)[:, np.newaxis]*vi_vecs_ned

    # Get aerodynamic force i^b projection
    aero_ib_proj = calc_aerodynamic_force_ib_projection(dl)

    # Scale inertial direction unit vectors by aerodynamic force i^b projection
    return aero_ib_proj[:, np.newaxis]*vi_vecs_ned_norm

# How to calculate the airspeed vector in the ned inertial system?
# This is just the vector [1, 0, 0] in the wind axis system.
# Get this vector in the inertial NED reference frame and scale by va.
def calc_va_vecs_i_ned(dl):
    r_bi_s = calc_body_to_inertial(dl.phis, dl.thetas, dl.psis)
    r_wb_s = calc_wind_to_body(dl.alphas, dl.betas)
    # TODO Check the matmul broadcasting rules
    r_wi_s = np.matmul(r_bi_s, r_wb_s)
    iws = np.matmul(r_wi_s, np.array([1, 0, 0]))
    # Multiply these by the airspeed
    # return va_vecs_ned
    return dl.vas[:, np.newaxis]*iws

# Calculate vi vectors in NED reference system
def calc_vi_vecs_ned(dl):
    vi_b_vecs = np.array([dl.us, dl.vs, dl.ws]).T
    # Convert body frame velocity to inertial frame
    vi_vecs_ned = np.squeeze(np.matmul(calc_body_to_inertial(dl.phis, dl.thetas, dl.psis), vi_b_vecs[:, :, np.newaxis]))
    return vi_vecs_ned

# Calculate the air-relative n, e, d positions for a given DataLogger
def calc_air_rel_positions(dl):
    # Construct a vector/array of the inertial positions
    pis = np.array([dl.ns, dl.es, dl.ds]).T
    # Construct a vector/array of the integrated wind
    ws = np.array([dl.wns, dl.wes, dl.wds])
    # Adding 0 at the start
    ws = np.concatenate(([np.zeros(3)], ws.T))[:-1].T
    # 'integrate'
    ws_integrated = np.cumsum(ws.T, axis=0)*dl.dt

    pas = pis - ws_integrated

    return pas

# ==========

# For testing. Want to show that the total air-relative energy change is not just the sum of the
# air-relative work rates, as it is in the inertial case.
def air_relative_energy_change_calculation_comparison(dl, aircraft_model):
    # Calculate the total specific air-relative energy change as the sum of the specific
    # air-relative energy change components (drag, throttle, static and gradient).
    comp_based_specific_total_energy_change = calc_total_specific_air_relative_energy_change(dl, aircraft_model)
    # Calculate based on work rates
    work_rates = calc_air_relative_work_rates(dl)
    work_rate_based_specific_total_energy_change = (work_rates['dw_d'] + work_rates['dw_c'] + work_rates['dw_l'] + work_rates['dw_t'] + work_rates['dw_fw']) / aircraft_model.m

    # Correct with correction term (see thesis).
    work_rate_based_corrected = work_rate_based_specific_total_energy_change - aircraft_model.g*dl.wds

    # Plot
    fig, ax = plt.subplots()
    ax.plot(dl.times, comp_based_specific_total_energy_change, label='component-based (correct)', c='g')
    ax.plot(dl.times, work_rate_based_specific_total_energy_change, label='work-rate-based', c='r')
    ax.plot(dl.times, work_rate_based_corrected, label='work-rate-based corrected', ls='--', c='orange')
    ax.legend()
    plt.show()