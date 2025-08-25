# FT 25/9/23

import jax
jax.config.update("jax_enable_x64", True)
import abc
import numpy as np
import copy
import sys
import pickle
import threading
from sshkeyboard import listen_keyboard

# For testing
import matplotlib.pyplot as plt

# Add the flight directory to the Python path (when running from FlightSwordLite)
import sys, os
sys.path.append(os.getcwd())

from flight.simulator.utils import calc_body_to_inertial, CyclicBuffer

class ControllerBase(metaclass=abc.ABCMeta):
    # TODO Get these from FlightEnvelope class
    # For envelope() function
    max_a = np.radians(18)
    min_a = -np.radians(18)
    max_e = np.radians(15)
    min_e = -np.radians(15)
    max_r = np.radians(29)
    min_r = -np.radians(29)
    # TODO [CRITICAL] What is the maximum power?? Minimum power taken to be zero.
    max_p = 1
    min_p = 0

    # Should be overridden (and this superclass constructor called as part of the implementation) to
    # provide additional functionality.
    def __init__(self, world_model, *args):
        self.world_model = world_model

    @abc.abstractmethod
    def gen_control_signal(self, **kwargs):
        pass

    # Constrains commanded control values to the permitted range (see Ana's paper)
    def envelope(self, da, de, dr, dp):
        da = np.max((-self.max_a, np.min((self.max_a, da))))
        de = np.max((-self.max_e, np.min((self.max_e, de))))
        dr = np.max((-self.max_r, np.min((self.max_r, dr))))
        dp = np.max((0, np.min((self.max_p, dp))))
        return da, de, dr, dp


# Returns zero control signal
class ControllerNull(ControllerBase):
    def gen_control_signal(self, **kwargs):
        return 0, 0, 0, 0


# Control surfaces are either deflected or flush/off
# Used to use the 'keyboard' package, but this doesn't work on Ubuntu WSL.
# Now uses 'sshkeyboard', although this only allows one key to be pressed at a time,
# and it's a bit of a hack to get it to detect whether keys are currently pressed
# or not.
class ControllerManualDiscrete(ControllerBase):
    def __init__(self, world_model, dt):
        super().__init__(world_model)

        self.dt = dt
        
        # TODO [Critical] What should these be?
        self.fixed_aileron_defection_angle = np.radians(10)
        self.fixed_elevator_deflection_angle = np.radians(10)
        self.fixed_rudder_deflection_angle = np.radians(10)
        self.fixed_power_on = self.max_p

        self.key_state = {}
        
        self.kp_a = 0.02
        self.kp_e = 0.02
        self.kp_r = 0.02
        self.kp_p = 0.02

        self.kd_a = 0.0012
        self.kd_e = 0.0012
        self.kd_r = 0.0012
        self.kd_p = 0.0012

        self.e_prev_a = CyclicBuffer(5)
        self.e_prev_e = CyclicBuffer(5)
        self.e_prev_r = CyclicBuffer(5)
        self.e_prev_p = CyclicBuffer(5)

        self.da = 0
        self.da_dot = 0
        self.de = 0
        self.de_dot = 0
        self.dr = 0
        self.dr_dot = 0
        self.dp = 0
        self.dp_dot = 0

        # Start the keyboard listener in a separate thread
        self.listener = threading.Thread(target=lambda: listen_keyboard(on_press=self.press, on_release=self.release))
        self.listener.daemon = True
        self.listener.start()

    def press(self, key):
        self.key_state[key] = True

    def release(self, key):
        self.key_state[key] = False
    
    def is_pressed(self, key):
        return self.key_state.get(key, False)
    
    def gen_control_signal(self, **kwargs):        
        # TODO [CRITICAL] Need to check the polarities of these
        # Keypad left arrow
        if self.is_pressed('left'):
            a_target = self.fixed_rudder_deflection_angle
        # Keypad right arrow
        elif self.is_pressed('right'):
            a_target = -self.fixed_rudder_deflection_angle
        else:
            a_target = 0

        # Keypad down arrow
        if self.is_pressed('down'):
            e_target = -self.fixed_elevator_deflection_angle
        # Keypad up arrow
        elif self.is_pressed('up'):
            e_target = self.fixed_elevator_deflection_angle
        else:
            e_target = 0
        
        # Keypad 'Enter'
        if self.is_pressed('enter'):
            r_target = -self.fixed_rudder_deflection_angle
        # Keypad 0
        elif self.is_pressed('0'):
            r_target = self.fixed_rudder_deflection_angle
        else:
            r_target = 0

        if self.is_pressed('pageup'):
            p_target  = self.fixed_power_on
        else:
            p_target = 0
        
        if self.is_pressed('q'):
            return None
        
        # Adjust control surfaces smoothly towards target
        err_a = a_target - self.da
        err_e = e_target - self.de
        err_r = r_target - self.dr
        err_p = p_target - self.dp

        self.e_prev_a.push(err_a)
        self.e_prev_e.push(err_e)
        self.e_prev_r.push(err_r)
        self.e_prev_p.push(err_p)

        d_err_a = self.e_prev_a.diff() / (self.dt*self.e_prev_a._size)
        d_err_e = self.e_prev_e.diff() / (self.dt*self.e_prev_e._size)
        d_err_r = self.e_prev_r.diff() / (self.dt*self.e_prev_r._size)
        d_err_p = self.e_prev_p.diff() / (self.dt*self.e_prev_p._size)
        
        self.da_dot += self.kp_a*err_a + self.kd_a*d_err_a
        self.da = np.max((np.min((self.da + self.da_dot, self.max_a)), self.min_a))
        self.de_dot += self.kp_e*err_e + self.kd_a*d_err_e
        self.de = np.max((np.min((self.de + self.de_dot, self.max_e)), self.min_e))
        self.dr_dot += self.kp_r*err_r + self.kd_a*d_err_r
        self.dr = np.max((np.min((self.dr + self.dr_dot, self.max_r)), self.min_r))
        self.dp_dot += self.kp_p*err_p + self.kd_a*d_err_p
        self.dp = np.max((np.min((self.dp + self.dp_dot, self.max_p)), self.min_p))

        return self.da, self.de, self.dr, self.dp

    def gen_control_signal__discontinuous(self, **kwargs):        
        # TODO [CRITICAL] Need to check the polarities of these
        # Keypad left arrow
        if self.is_pressed('left'):
            a_commanded = self.fixed_rudder_deflection_angle
        # Keypad right arrow
        elif self.is_pressed('right'):
            a_commanded = -self.fixed_rudder_deflection_angle
        else:
            a_commanded = 0

        # Keypad down arrow
        if self.is_pressed('down'):
            e_commanded = -self.fixed_elevator_deflection_angle
        # Keypad up arrow
        elif self.is_pressed('up'):
            e_commanded = self.fixed_elevator_deflection_angle
        else:
            e_commanded = 0
        
        # Keypad 'Enter'
        if self.is_pressed('enter'):
            r_commanded = -self.fixed_rudder_deflection_angle
        # Keypad 0
        elif self.is_pressed('0'):
            r_commanded = self.fixed_rudder_deflection_angle
        else:
            r_commanded = 0

        if self.is_pressed('pageup'):
            p_commanded  = self.fixed_power_on
        else:
            p_commanded = 0
        
        if self.is_pressed('q'):
            sys.exit(0)
        
        return a_commanded, e_commanded, r_commanded, p_commanded


# Iterates through a given sequence of control inputs
class ControllerIterator(ControllerBase):
    def __init__(self, world_model, das=None, des=None, drs=None, dps=None):
        super().__init__(world_model)

        def not_none(x):
            return x is not None
        
        if not_none(das) or not_none(des) or not_none(drs) or not_none(dps):
            # If any of the control lists is present, they should all be.
            if not (not_none(das) and not_none(des) and not_none(drs) and not_none(dps)):
                raise ValueError("If any control list is present, control lists for da, de, dr and dp must be provided")
            
            # Check that they all have the same length
            if not (len(das) == len(des) == len(drs) == len(dps)):
                raise ValueError("Control lists must all have the same length")
            
            self.das = das
            self.des = des
            self.drs = drs
            self.dps = dps
        else:   # None of the control lists are present
            self.das = []
            self.des = []
            self.drs = []
            self.dps = []
        
        self.control_len = len(self.das)
        self.return_indx = 0
    
    def set(self, das, des, drs, dps):
        # Check that they all have the same length
        if not (len(das) == len(des) == len(drs) == len(dps)):
            raise ValueError("Control lists must all have the same length")
            
        self.das = das
        self.des = des
        self.drs = drs
        self.dps = dps

        self.control_len = len(self.das)
    
    # Return one set of control signals at a time (each time the function is called).
    def gen_control_signal(self, **kwargs):
        if self.return_indx == self.control_len:
            return None     # Out of control inputs
        else:
            da = self.das[self.return_indx]
            de = self.des[self.return_indx]
            dr = self.drs[self.return_indx]
            dp = self.dps[self.return_indx]
            
            self.return_indx += 1

            return da, de, dr, dp


class PID:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.error_int = 0
        buff_size = 10
        self.error_buff = CyclicBuffer(buff_size)
        self.diff_denom = buff_size * dt
    
    def set_target(self, target):
        self.target = target
    
    def calc_control(self, measured):
        err = self.target - measured

        self.error_int += err
        self.error_buff.push(err)

        control_sig = self.kp*err + self.ki*self.error_int + self.kd*(self.error_buff.diff() / self.diff_denom)
        return control_sig


# These values need updating to be 'n, e, d' instead of 'x, y, z'.
class ControllerWaypoint(ControllerBase):
    def __init__(self, world_model, dt):
        super().__init__(world_model)
        
        #kp_h = 0.0005
        #ki_h = 0.00001
        #kd_h = 0 #0.05
        #kp_att = 0.0005
        #ki_att = 0.00001
        #kd_att = 0 #0.05
        #self.height_controller = PID(kp_h, ki_h, kd_h)
        #self.attitude_controller = PID(kp_att, ki_att, kd_att, dt)
        kp_c = 1 # 0.1
        ki_c = 0.00001 # 0
        kd_c = 0
        self.climb_rate_controller = PID(kp_c, ki_c, kd_c, dt)

        # How is this going to work? The attitude controller commands a certain attitude towards the target - this is just a straight line - and this is achieved 
        #.. Wait, this is basically the sink rate I wanted earlier?

        # Freddie - think about this straight - everyone is doing well apart from you. You are not managing your time well - you are just one big ball of (creative and intelligent) chaos.
        # So, if you want to write a PID controller, write a PID controller. But do it properly. And test your assumptions. Let's start. A negative de should pull the nose up - does it? Yes, it does.

        # Now, the previous idea was good, but let's improve it. Split into two controllers. The roll control should only be a D controller.
        # The pitch control should be a PD controller. Maybe a PID.
        # Let's do the roll control first - we're going to need some kind of cyclic buffer. I've written one a thousand times before.
        # For now let's try it without, then I can pick it up on my other computer.
        # Get the difference in relative course angle.

        self.p_a = []
        self.d_a = []
        self.p_e = []
        self.i_e = []

        self.dt = dt

        # self.da_max = np.radians(8)
        # self.da_rate = 0.00001
        # self.de_rate = 0.00001

        # Right angle rotation to the right
        self.right_right = np.array([[0, 1],
                                     [-1, 0]])
        
        # self.course_angle_buffer_size = 10
        self.buffer_size = 10
        self.relative_course_angle_buffer = CyclicBuffer(self.buffer_size) # self.course_angle_buffer_size)
        # self.height_closure_rate_buffer_size = 10
        self.height_closure_rate_buffer = CyclicBuffer(self.buffer_size) # self.height_closure_rate_buffer_size)
        self.height_err_int = 0
        
        self.x_buffer = CyclicBuffer(self.buffer_size)
        self.y_buffer = CyclicBuffer(self.buffer_size)
        self.z_buffer = CyclicBuffer(self.buffer_size)
        #self.sink_rate_err_sum = 0

    def set_position_waypoint(self, n, e, d):
        self.wp_n = n
        self.wp_e = e
        self.wp_d = d

    def set_orientation_waypoint(self, phi, theta, psi):
        pass

    def gen_control_signal(self, **kwargs):
        # 'sd' for utils.StateDecoder
        sd = kwargs['sd']

        # Get relative bearing to waypoint
        # Get vector between waypoint and aircraft position, and convert it into aircraft body frame
        current_coords = np.array([sd.n, sd.e, sd.d])
        to_waypoint = np.array([self.wp_n, self.wp_e, self.wp_d]) - current_coords
        to_waypoint_planar = to_waypoint[:2]
        
        # Calculate relative course angle between this point and the aircraft
        aircraft_bearing_planar = np.matmul(calc_body_to_inertial(sd.phi, sd.theta, sd.psi), np.array([1, 0, 0]))[:2]
        relative_course_angle = np.arccos(np.dot(aircraft_bearing_planar, to_waypoint_planar) / (np.linalg.norm(aircraft_bearing_planar)*np.linalg.norm(to_waypoint_planar)))
        
        # To get orientation, take dot product of planar waypoint bearing vector with rotated (to right) planar i^b vector. This will be positive if waypoint is to the right of the
        # aircraft, and negative if it's to the left.
        aircraft_heading_rotated = np.matmul(self.right_right, to_waypoint_planar)
        direction_sgn = np.sign(np.dot(aircraft_heading_rotated, aircraft_bearing_planar))
        # In case the target is directly behind
        if direction_sgn == 0:
            direction_sgn = 1
        relative_course_angle = direction_sgn*relative_course_angle
        # Save angle, for calculating difference
        self.relative_course_angle_buffer.push(relative_course_angle)

        # print(f"Relative course angle: {np.degrees(relative_course_angle)}")

        self.x_buffer.push(sd.n)
        self.y_buffer.push(sd.e)
        # Altitude
        self.z_buffer.push(sd.d)
        
        """
        # Convert to aircraft body frame
        to_waypoint_body = np.matmul(calc_body_to_inertial(sd.phi, sd.theta, sd.psi).T, to_waypoint)

        # If waypoint vector is to left, roll left
        if to_waypoint_body[1] < 0:  # j^b: positive-right
            self.da += self.da_rate
        # If to right, roll right
        else:
            self.da -= self.da_rate

        # If waypoint vector is above, pitch up (might not be the best thing to do)
        if to_waypoint_body[2] < 0: # k^b: positive-down
        
            self.de -= self.de_rate
        # If below, pitch down (might not be the best thing to do)
        else:
            self.de += self.de_rate
        
        # return self.da, self.de, self.dr, self.dp
        """

        # Roll control
        # Get course angle closure rate
        # If target is to the right, roll right (negative aileron)
        kp_r = -0.05 # -0.01
        kd_r = -0.2 # -0.1 # -2 # -5
        # Relative course angle = heading error
        # Closure rate = d/dt(heading error)
        # Derivative is scaled by time
        closure_rate = self.relative_course_angle_buffer.diff() / (self.buffer_size*self.dt)
        # Want kd contribution to diminish as the target is approached, to avoid sudden deflections.
        da = kp_r*relative_course_angle + kd_r*np.min((1, np.linalg.norm(to_waypoint_planar)))*closure_rate

        # Pitch control
        # Calculate sink rate
        comparison_pt_planar = np.array([self.x_buffer.get_oldest(), self.y_buffer.get_oldest()])
        planar_dist = np.linalg.norm(current_coords[:2] - comparison_pt_planar)
        # Positive if the aircraft has descended (e.g. if it was at -100 and it's now at -80, height_diff = -100 - -80 = -20).
        height_diff = -(self.z_buffer.get_oldest() - current_coords[2])
        # Positive if the aircraft has descended
        current_sink_rate = 0 if planar_dist == 0 else (height_diff / planar_dist)

        height_surplus = to_waypoint[2]
        planar_dist_to_target = np.linalg.norm(to_waypoint_planar)
        # If this is positive, need to sink
        target_sink_rate = 0 if planar_dist_to_target == 0 else (height_surplus / planar_dist_to_target)

        #print(f"Height surplus: {height_surplus}, horizontal dist: {np.linalg.norm(to_waypoint_planar)}")
        #print(f"Target sink rate: {target_sink_rate}")

        self.climb_rate_controller.set_target(target_sink_rate)
        # Generate elevator control signal
        de = self.climb_rate_controller.calc_control(current_sink_rate)

        return self.envelope(da, de, 0, 0)
    
    def graph(self):
        fig, axs = plt.subplots(2)
        axs[0].plot(self.p_a, label='p')
        axs[0].plot(self.d_a, label='d')
        axs[0].legend()
        axs[1].plot(self.p_e, label='p')
        axs[1].plot(self.i_e, label='i')
        axs[1].legend()
        plt.show(block=False)

#########

# For testing only - creates random outputs
class ControllerRandom(ControllerBase):
    # When gen_control_signal is called, it returns an n-tuple of values based on
    # normal distributions. 'n' should be the number of control inputs expected by the dynamics
    # model (as used in its gradients function), and is set by the number of elements in
    # the 'mean' and 'stds' arrays passed to the constructor of this class (which should both
    # have the same length).
    # 'inits' is an n-dim vector of the initial values of the control signals.
    # 'means' is an n-dim vector of the means of the normal distributions from which the random
    # control value augmentations are drawn.
    # 'stds' gives the standard deviations of these distributions.
    # 'mins' gives the minimum permitted values of the control signals.
    # 'maxs' gives the maximum permitted values of the control signals.
    # NOTE: gen_control_signal doesn't return the sampled values directly as control inputs, as they might
    # change too quickly and upset the dynamics. It instead appends them to the previous values, and returns
    # those as the control values.
    # 'world_model' is the World object - it gives access to local wind and environment topography information.
    def __init__(self, world_model, inits, means, stds, mins, maxs):
        super().__init__(world_model)

        if not (len(inits) == len(means) == len(stds) == len(mins) == len(maxs)):
            raise ValueError("Lengths of initial values, means, standard deviations, mins and maxs arrays must match")
        
        self.num_control_vals = len(means)
        self.means = means
        self.stds = stds
        self.mins = mins
        self.maxs = maxs

        # Initial values of control signals
        self.control_vals = inits
    
    # Generated control signals are random walks (i.e. a random value is added to the
    # previous value each time). This is to prevent the commanded values from jumping
    # around too much.
    # Flexible length input-output - for the actual aircraft, use gen_control_signal(), which is specific for
    # da, de, dr, dp since it includes the enveloping.
    def gen_control_signal_flex_len(self, **kwargs):
        for i in range(self.num_control_vals):
            # Random walks capped by the maximum and minimum allowed values (e.g. would always want throttle to be greater than zero).
            self.control_vals[i] = np.max((
                np.min((
                    self.control_vals[i] + np.random.normal(self.means[i], self.stds[i]),
                    self.maxs[i])),
                self.mins[i]))

        return self.control_vals
    
    # Only works with a 4D input for da, de, dr, dp (because of the enveloping)
    def gen_control_signal(self, **kwargs):
        return self.envelope(*self.gen_control_signal_flex_len())


# Testing controllers
def controller_random_test():
    l = 10000

    inits = [10, 0, 0, 5]
    means = [0, 0, 0, 0]
    stds = [0.4, 0.3, 0.2, 0.1]
    mins = [-np.inf, -np.inf, -np.inf, 3]
    maxs = [np.inf, np.inf, np.inf, 6]
    n = len(means)
    controller = ControllerRandom(inits, means, stds, mins, maxs)

    control_vals = np.empty((n, l))

    for i in range(l):
        control_vals[:,i] = controller.gen_control_signal()
    
    # Plot
    fig, axs = plt.subplots(n)
    for i in range(n):
        axs[i].plot(control_vals[i, :])
    plt.show(block=True)

if __name__ == "__main__":
    controller_random_test()