import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from functools import partial
import yaml
from pathlib import Path

# Add the flight directory to the Python path (when running from FlightSwordLite)
import sys, os
sys.path.append(os.getcwd())

class AircraftModel:
    def __init__(self, path):
        # Load model from yaml file
        self.load_model(Path(path))
        
        # Calculate and set derived values
        self.set_derived_values()

    def load_model(self, path):
        with open(path, 'r') as flight_model_yaml:
            model_data = yaml.safe_load(flight_model_yaml)

            drag_model_name = model_data['drag_model_name']
            if drag_model_name == 'cl_derived':
                self.CD = self.CD_cl_derived
            elif drag_model_name == 'freddie':
                self.CD = self.CD_freddie
            else:
                raise Exception("Invalid drag model name in flight model .yaml file!")

            # Comments are Ana GL's
            # Physical constants
            self.g = model_data['g']
            self.rho = model_data['rho']
            # Non-dimensional constant, q_const = c/(2*V_bar)
            self.q_const = model_data['q_const']
            # Non-dimensional constant, b_2V
            self.b_2V = model_data['b_2V']

            # Aircraft Model Parameters
            # Wing area
            self.S = model_data['S']
            # Wing Mean-Aerodynamic-Chord (MAC)
            self.c = model_data['c']
            # Wing span
            self.b = model_data['b']
            # Aircraft/UAV mass
            self.m = model_data['m']
            # Aircraft/UAV moments of inertia - ESTIMATED WITH TRIFILAR PENDULUM
            self.I_xx = model_data['I_xx']
            self.I_yy = model_data['I_yy']
            self.I_zz = model_data['I_zz']
            # Aircraft/UAV moment of inertia - Sergio
            self.I_xz = model_data['I_xz']
            # Oswald factor
            self.e_oswald = model_data['e_oswald']

            # Aerodynamic coefficients
            self.Ct_k = model_data['Ct_k']
            self.Ct_alpha = model_data['Ct_alpha']
            self.Ct_q = model_data['Ct_q']
            self.Ct_dt = model_data['Ct_dt']
            self.Ct_dt2 = model_data['Ct_dt2']

            self.Cd_0 = model_data['Cd_0']
            self.Cd_alpha = model_data['Cd_alpha']
            self.Cd_alpha2 = model_data['Cd_alpha2']
            # For my custom modified version (not in Ana's)
            # Not all models have this, so adding a default value.
            self.Cd_q = model_data.get('Cd_q', jnp.nan)
            self.Cd_q2 = model_data.get('Cd_q2', jnp.nan)

            self.alpha_0 = model_data['alpha_0']
            self.Cl_alpha = model_data['Cl_alpha']
            self.Cl_q = model_data['Cl_q']
            self.Cl_de = model_data['Cl_de']
            self.Cl_dt = model_data['Cl_dt']

            self.Cm_0 = model_data['Cm_0']
            self.Cm_alpha = model_data['Cm_alpha']
            self.Cm_q = model_data['Cm_q']
            self.Cm_de = model_data['Cm_de']
            self.Cm_dt = model_data['Cm_dt']

            # Aerodynamic coefficients - lateral
            self.CY_0 = model_data['CY_0']
            self.CY_beta = model_data['CY_beta']
            self.CY_p = model_data['CY_p']
            self.CY_r = model_data['CY_r']
            self.CY_da = model_data['CY_da']
            self.CY_dr = model_data['CY_dr']

            self.Cl_0_lat = model_data['Cl_0_lat']
            self.Cl_beta = model_data['Cl_beta']
            self.Cl_p = model_data['Cl_p']
            self.Cl_r = model_data['Cl_r']
            self.Cl_da = model_data['Cl_da']
            self.Cl_dr = model_data['Cl_dr']

            self.Cn_0 = model_data['Cn_0']
            self.Cn_beta = model_data['Cn_beta']
            self.Cn_p = model_data['Cn_p']
            self.Cn_r = model_data['Cn_r']
            self.Cn_da = model_data['Cn_da']
            self.Cn_dr = model_data['Cn_dr']

            # Enveloping parameters
            self.max_cl = model_data['max_cl']
            self.min_va = model_data['min_va']
            self.max_va = model_data['max_va']
            self.min_alpha = model_data['min_alpha']
            self.max_alpha = model_data['max_alpha']
            self.min_beta = model_data['min_beta']
            self.max_beta = model_data['max_beta']
            self.min_p = model_data['min_p']
            self.max_p = model_data['max_p']
            self.min_q = model_data['min_q']
            self.max_q = model_data['max_q']
            self.min_r = model_data['min_r']
            self.max_r = model_data['max_r']
            self.min_da = model_data['min_da_radians']
            self.max_da = model_data['max_da_radians']
            self.min_de = model_data['min_de_radians']
            self.max_de = model_data['max_de_radians']
            self.min_dr = model_data['min_dr_radians']
            self.max_dr = model_data['max_dr_radians']
            self.min_dp = model_data['min_dp']
            self.max_dp = model_data['max_dp']
            self.min_load_factor = model_data['min_load_factor']
            self.max_load_factor = model_data['max_load_factor']
            
    def set_derived_values(self):
        self.AR = self.b**2 / self.S
        self.tau = self.calc_tau(self.I_xx, self.I_zz, self.I_xz)
    
    # Calculated like this so that it can also be used by Dymos, where the inertia properties may change (be the subject of optimisation) and tau may need recalculating.
    @staticmethod
    def calc_tau(I_xx, I_zz, I_xz):
        return I_xx*I_zz - jnp.square(I_xz)
    
    # CL based on alpha only - so may be slightly wrong
    def calc_stall_speed(self):
        CLmax = self.CL(self.max_alpha, 0, 0, 0)
        return np.sqrt((2*self.m*self.g)/(self.rho*self.S*CLmax))
    
    # ===============
    # Coefficient functions

    # TODO This is from Ana's Matlab code, but it doesn't agree with her paper
    # Coefficient of lift
    @partial(jax.jit, static_argnums=(0,))
    def CL(self, alpha, q, de, dp):
        # TODO This Cl_dt*dp term isn't present in the paper
        return self.Cl_alpha*(alpha + self.alpha_0) + self.Cl_q*self.q_const*q + \
            self.Cl_de*de + self.Cl_dt*dp
    
    # Coefficient of lateral forces (in body frame? TODO)
    @partial(jax.jit, static_argnums=(0,))
    def CY(self, beta, p, r, da, dr):
        return self.CY_0 + self.CY_beta*beta + self.CY_p*self.b_2V*p + \
            self.CY_r*self.b_2V*r +self.CY_da*da + self.CY_dr*dr
    
    # Coefficient of drag
    # New drag model - fitted by me from Sergio's data - see e-mail to Shane, "Modelling notes" 10/12/24 1:30am.
    # Doesn't use de or dp
    @partial(jax.jit, static_argnums=(0,))
    def CD_freddie(self, alpha, q, de=0, dp=0):
        # Apparently jnp.square() is slower but more accurate than **: https://stackoverflow.com/questions/29361856/python-numpy-square-vs
        return self.Cd_0 + self.Cd_alpha*alpha + self.Cd_alpha2*jnp.square(alpha) + self.Cd_q*q + self.Cd_q2*jnp.square(q)
    
    # New drag model - CD based on value of CL
    @partial(jax.jit, static_argnums=(0,))
    def CD_cl_derived(self, alpha, q, de, dp):
        # Apparently jnp.square() is slower but more accurate than **: https://stackoverflow.com/questions/29361856/python-numpy-square-vs
        return self.Cd_0 + (jnp.square(self.CL(alpha, q, de, dp)) / (jnp.pi*self.AR*self.e_oswald))
    
    # Old (deprecated) coefficient of drag - not valid for negative values of alpha
    @partial(jax.jit, static_argnums=(0,))
    def CD_orig(self, alpha):
        # Apparently jnp.square() is slower but more accurate than **: https://stackoverflow.com/questions/29361856/python-numpy-square-vs
        return self.Cd_0 + self.Cd_alpha*alpha + self.Cd_alpha2*jnp.square(alpha)
    
    # Thrust coefficient
    @partial(jax.jit, static_argnums=(0,))
    def Ct(self, alpha, q, dp):
        # Apparently jnp.square() is slower but more accurate than **: https://stackoverflow.com/questions/29361856/python-numpy-square-vs
        return self.Ct_alpha*alpha + self.Ct_q*self.q_const*q + \
            self.Ct_dt*dp + self.Ct_dt2*jnp.square(dp)

    # Roll moment coefficient
    # TODO Using Ana's names - don't understand why it is called this.
    @partial(jax.jit, static_argnums=(0,))
    def Cl_lat(self, beta, p, r, da, dr):
        return self.Cl_0_lat + self.Cl_beta*beta + self.Cl_p*self.b_2V*p + \
            self.Cl_r*self.b_2V*r + self.Cl_da*da + self.Cl_dr*dr
    
    # Pitch moment coefficient
    @partial(jax.jit, static_argnums=(0,))
    def Cm(self, alpha, q, de, dp):
        return self.Cm_0 + self.Cm_alpha*alpha + self.Cm_q*self.q_const*q + \
            self.Cm_de*de + self.Cm_dt*dp
    
    # Yaw moment coefficient
    @partial(jax.jit, static_argnums=(0,))
    def Cn(self, beta, p, r, da, dr):
        return self.Cn_0 + self.Cn_beta*beta + self.Cn_p*self.b_2V*p + \
            self.Cn_r*self.b_2V*r + self.Cn_da*da + self.Cn_dr*dr
    

    # ===============
    # Force and moment functions
    
    # These calculations are lifted from Ana's code: 'WOT4_DynModel_6DOF.m' (last modified 22/06/2021 14:27)
    # Apparently jnp.square() is slower but more accurate than **: https://stackoverflow.com/questions/29361856/python-numpy-square-vs
    # Force and moment function arguments are written in this way (with default values) to facilitate model plotting in simulator project from which this code was copied.
    
    # Lift
    @partial(jax.jit, static_argnums=(0,))
    def L(self, va=0, alpha=0, q=0, de=0, dp=0):
        return 0.5*self.rho*jnp.square(va)*self.S*self.CL(alpha, q, de, dp)
    
    # Sideforce
    @partial(jax.jit, static_argnums=(0,))
    def C(self, va=0, beta=0, p=0, r=0, da=0, dr=0):
        return 0.5*self.rho*jnp.square(va)*self.S*self.CY(beta, p, r, da, dr)
    
    # Drag
    # def D(self, va=0, alpha=0):
    @partial(jax.jit, static_argnums=(0,))
    def D(self, va=0, alpha=0, q=0, de=0, dp=0):   # New drag model, which uses CL internally.
        return 0.5*self.rho*jnp.square(va)*self.S*self.CD(alpha, q, de, dp)
    
    # Thrust
    @partial(jax.jit, static_argnums=(0,))
    def T(self, alpha=0, q=0, dp=0):
        return 0.5*self.rho*self.S*self.Ct_k*self.Ct(alpha, q, dp)
    
    # Roll moment
    @partial(jax.jit, static_argnums=(0,))
    def L_lat(self, va=0, beta=0, p=0, r=0, da=0, dr=0):
        return 0.5*self.rho*jnp.square(va)*self.S*self.b*self.Cl_lat(beta, p, r, da, dr)
    
    # Pitch moment
    @partial(jax.jit, static_argnums=(0,))
    def M(self, va=0, alpha=0, q=0, de=0, dp=0):
        return 0.5*self.rho*jnp.square(va)*self.S*self.c*self.Cm(alpha, q, de, dp)
    
    # Yaw moment
    @partial(jax.jit, static_argnums=(0,))
    def N(self, va=0, beta=0, p=0, r=0, da=0, dr=0):
        return 0.5*self.rho*jnp.square(va)*self.S*self.b*self.Cn(beta, p, r, da, dr)
    
    @partial(jax.jit, static_argnums=(0,))
    def calculate_forces_and_moments(self, va, alpha, beta, p, q, r, da, de, dr, dp):
        # The optimiser can use a very high throttle value. Saturating it here.
        dp = self.max_dp*jnp.tanh(dp)
        
        # Ana's condition - need to understand the implications of this
        return jax.lax.cond(va > 0,
                            lambda: jnp.array([
                                self.L(va, alpha, q, de, dp),       # lift
                                self.C(va, beta, p, r, da, dr),     # sideforce
                                # AircraftDynamics.D(aircraft_model, va, alpha) # drag
                                self.D(va, alpha, q, de, dp),       # new drag model, based on CL
                                self.T(alpha, q, dp),               # thrust
                                self.L_lat(va, beta, p, r, da, dr), # roll_mom
                                self.M(va, alpha, q, de, dp),       # pitch_mom
                                self.N(va, beta, p, r, da, dr)      # yaw_mom
                                ]),
                            lambda: jnp.zeros(7))