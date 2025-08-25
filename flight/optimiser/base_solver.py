# FT 28/7/24
# This file is to check that everything is working correctly

import openmdao.api as om
import dymos as dm
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from datetime import datetime
import copy
# For debugging
import time

# Add the flight directory to the Python path (when running from FlightSwordLite)
import sys, os
sys.path.append(os.getcwd())

# from flight.optimiser import results_plotting
from flight.analysis import results_plotting

from flight.optimiser import collocation_plotting
from flight.optimiser import config
from flight.optimiser.optimisation_models import BeardDynamicsJax # , BeardDynamicsNoGrads

from flight.simulator.aircraft_model import AircraftModel
from flight.simulator.wind_models import WindModel
from flight.simulator.utils import DataLogger
# from cfd_wind_field_parsers.quic_parser import QuicWindParser

"""
# Second phase, to calculate integrated control effort (to add this as a regularisation term).
class AdditionalValues(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    
    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('da', val=np.zeros(nn), desc='aileron deflection angle', units='rad')
        self.add_input('de', val=np.zeros(nn), desc='elevator deflection angle', units='rad')
        self.add_input('dr', val=np.zeros(nn), desc='rudder deflection angle', units='rad')
        self.add_input('dp', val=np.zeros(nn), desc='throttle')

        # Rate to compute sum of squares of control angles (squaring for positive values and smoothness)
        # Using 'effort' here but really this is just about deflection angles.
        self.add_output('squared_control_effort_rate', val=np.zeros(nn), desc='rate of change of sum of squares of control surface deflections', units='rad/s')
        self.add_output('squared_throttle_effort_rate', val=np.zeros(nn), desc='rate of change of squared throttle')

        # Setup partials
        arange = np.arange(nn)
        self.declare_partials(of='squared_control_effort_rate', wrt='da', rows=arange, cols=arange)
        self.declare_partials(of='squared_control_effort_rate', wrt='de', rows=arange, cols=arange)
        self.declare_partials(of='squared_control_effort_rate', wrt='dr', rows=arange, cols=arange)
        self.declare_partials(of='squared_throttle_effort_rate', wrt='dp', rows=arange, cols=arange)
    
    def compute(self, inputs, outputs):
        da = inputs['da']
        de = inputs['de']
        dr = inputs['dr']
        dp = inputs['dp']

        outputs['squared_control_effort_rate'] = da**2 + de**2 + dr**2
        outputs['squared_throttle_effort_rate'] = dp**2
    
    def compute_partials(self, inputs, partials):
        da = inputs['da']
        de = inputs['de']
        dr = inputs['dr']
        dp = inputs['dp']

        partials['squared_control_effort_rate', 'da'] = 2*da
        partials['squared_control_effort_rate', 'de'] = 2*de
        partials['squared_control_effort_rate', 'dr'] = 2*dr
        partials['squared_throttle_effort_rate', 'dp'] = 2*dp
"""


class ControlODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        
        self.add_input('da', shape=(nn,), desc='aileron deflection angle', units='rad')
        self.add_input('de', shape=(nn,), desc='elevator deflection angle', units='rad')
        self.add_input('dr', shape=(nn,), desc='rudder deflection angle', units='rad')
        # Used to have units=None, but this breaks it.
        self.add_input('dp', shape=(nn,), desc='throttle setting')

        self.add_output('control_surface_int_rate', val=np.zeros(nn), desc='rate of change of integral of summed squared control surface deflections', units='rad')
        # Used to have units=None, but this breaks it.
        self.add_output('throttle_int_rate', val=np.zeros(nn), desc='rate of change of integral of squared throttle values')

        # TODO These are defined to be complex step, but they should be analytic. Check if making them analytic makes it faster.
        self.declare_partials('control_surface_int_rate', ['da', 'de', 'dr'], method='cs')
        self.declare_partials('throttle_int_rate', 'dp', method='cs')
    
    def compute(self, inputs, outputs):
        da = inputs['da']
        de = inputs['de']
        dr = inputs['dr']
        dp = inputs['dp']
        
        # Sum squared control rates
        outputs['control_surface_int_rate'] = np.square(da) + np.square(de) + np.square(dr)
        outputs['throttle_int_rate'] = np.square(dp)

class ControlRateIntegrator(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('time', shape=(nn,), units='s')
        self.add_input('da', shape=(nn,), desc='aileron deflection angle', units='rad')
        self.add_input('de', shape=(nn,), desc='elevator deflection angle', units='rad')

        self.add_output('abs_control_rate_int', val=0.0)

        self.declare_partials(of='abs_control_rate_int', wrt='*', method='cs')

    def compute(self, inputs, outputs):
        times = inputs['time']
        das = inputs['da']
        des = inputs['de']

        # print(f"Times: {times}")
        # print(f"Das: {das}")
        # print(f"Des: {des}")

        da_grads = np.gradient(das)
        de_grads = np.gradient(des)
        abs_da_grads = np.abs(da_grads)
        abs_de_grads = np.abs(de_grads)

        abs_da_rate_int = np.trapz(times, abs_da_grads)
        abs_de_rate_int = np.trapz(times, abs_de_grads)

        outputs['abs_control_rate_int'] = abs_da_rate_int + abs_de_rate_int

class TrajectorySolver:
    def __init__(self, solver_params, foldername=None, results_folder_path=None):
        self.probs = []     # List of Problem instances
        self.num_solves = 0     # Stores the number of times run_traj_solver is called.

        aircraft_model_path = solver_params['aircraft_model_path']
        self.wm = solver_params['wind_manager']
        self.record_sim = solver_params['record_sim']
        self.datalogger_dt = solver_params['datalogger_dt']
        self.refine_control_tx = solver_params.get('refine_control_tx', True)    # Defaults to True if ommitted, to maintain
                                                                                 # the behaviour of the code before this option.
        self.control_rate_int_tx_mult = solver_params.get('control_rate_int_tx_mult', 10)

        # Get aircraft model
        self.aircraft_model = AircraftModel(aircraft_model_path)

        # Set up folders for solution

        if foldername is None:
            foldername = str(datetime.now())
        self.foldername = foldername

        # If a path is provided for the results folder, use it, else use the defaults in the config file.
        if results_folder_path is None:
            datalogger_path = config.default_datalogger_save_path
            dymos_logs_folder_path = config.default_dymos_logs_folder_path
            figure_path = config.default_figure_folder_path
            solver_output_folder_path = config.default_solver_output_folder_path
        else:
            datalogger_path = results_folder_path / 'dataloggers'
            dymos_logs_folder_path = results_folder_path / 'dymos_logs'
            figure_path = results_folder_path / 'figs'
            solver_output_folder_path = results_folder_path / 'solver_output'
        
        # Datalogger files
        self.datalogger_path = datalogger_path
        
        # Log files
        self.dymos_output_path = dymos_logs_folder_path / foldername
        self.dymos_output_path.mkdir(parents=True, exist_ok=True)
        
        # Figures
        if config.save_figs:
            self.figure_path = figure_path / foldername
            self.figure_path.mkdir(parents=True, exist_ok=True)
        
        # IPOPT output
        # Construct ipopt.opt options file to create a log file for this solution.
        # If this is not done for each separate trajectory optimisation, the results of different ones
        # will be overwritten.
        # The ipopt file has to be created because the log file settings can't be
        # configured using the API.
        self.solver_output_folder = solver_output_folder_path / foldername
        self.solver_output_folder.mkdir(parents=True, exist_ok=True)

        ipopt_file_str = f"output_file \"{(self.solver_output_folder / 'IPOPT.out').resolve()}\"\n"
        # TODO TEST THIS - with multiple different calls to run_traj_solver.
        # So that multiple runs and grid refinements don't overwrite the previous solve iterations.
        ipopt_file_str += f"file_append yes\n"
        # ipopt_file_str += f"hessian_approximation exact\n"

        # Save log file, overwriting it if it exists.
        with open(config.solver_settings_filepath, 'w') as f:
            f.write(ipopt_file_str)
            f.flush()

    @staticmethod
    def constrain_position(phase, loc, n, e, d):
        phase.add_boundary_constraint('n', loc=loc, equals=n)
        phase.add_boundary_constraint('e', loc=loc, equals=e)
        phase.add_boundary_constraint('d', loc=loc, equals=d)
    
    # Setup Dymos/OpenMDAO problem
    # 'solve=False': simulate by default. Set to True to optimise trajectory.
    # foldername: if left as None, this will be the date and time. Can be passed in in case the calling script wants to save additional data
    #  so that it can all be tied together by name.
    # 'normalised_segment_ends': list, values in range [-1, 1] (TODO do -1 and 1 need to be included?). If provided, specifies the locations of the
    # transcription points.
    def run_traj_solver(self, params, set_objective, set_additional_constraints, set_initial_values, set_driver_options=(lambda driver: None), solve=False, check_partials=False):

        # Update solve counter
        self.num_solves += 1
        solve_num = self.num_solves

        # =================================
        # Options

        # Transcription settings
        num_dynamics_segments = params['num_dynamics_segments']
        dynamics_seg_order = params['dynamics_seg_order']
        num_control_segments = params['num_control_segments']
        control_seg_order = params['control_seg_order']
        normalised_dynamics_segment_ends = params.get('normalised_dynamics_segment_ends', None)   # This argument is optional

        # Scaling values
        # Unit reference
        refs = params.get('refs', {
            'n': None, 
            'e': None,
            'd': None,
            'u': None,
            'v': None,
            'w': None,
            'phi': None,
            'theta': None,
            'psi': None,
            'p': None,
            'q': None,
            'r': None,
            'da': None,
            'de': None,
            'dr': None,
            'dp': None
        })
        # Zero reference
        ref0s = params.get('ref0s', {
            'n': None, 
            'e': None,
            'd': None,
            'u': None,
            'v': None,
            'w': None,
            'phi': None,
            'theta': None,
            'psi': None,
            'p': None,
            'q': None,
            'r': None,
            'da': None,
            'de': None,
            'dr': None,
            'dp': None
        })
        # Defect unit reference at collocation nodes
        drefs = params.get('drefs', {
            'n': None, 
            'e': None,
            'd': None,
            'u': None,
            'v': None,
            'w': None,
            'phi': None,
            'theta': None,
            'psi': None,
            'p': None,
            'q': None,
            'r': None,
            'da': None,
            'de': None,
            'dr': None,
            'dp': None
        })

        # Solve settings
        num_refinements = params['num_refinements']
        max_iter = params['max_iter']
        # No 'from_sim' - warm starting not implemented at the moment
        time_duration_bounds = params['time_duration_bounds']

        enable_throttle_optimisation = params['enable_throttle_optimisation']

        # =================================
        # OpenMDAO Problem

        prob = om.Problem()

        traj = dm.Trajectory()
        # Trajectory gets added to the problem
        prob.model.add_subsystem('traj', traj)

        # Create a transcription for the control phase
        control_tx = dm.GaussLobatto(num_segments=num_control_segments, order=control_seg_order, compressed=False)

        # This is a bit of a hacky way of doing it
        # Create a transcription for the control integration
        control_rate_int_tx = dm.GaussLobatto(num_segments=self.control_rate_int_tx_mult*num_control_segments, order=3, compressed=False)

        # Create a transcription for the dynamics
        if normalised_dynamics_segment_ends is not None:
            num_dynamics_segments_from_ends = len(normalised_dynamics_segment_ends) - 1
            # TODO Test this
            if num_dynamics_segments_from_ends != num_dynamics_segments:
                raise Exception("Number of dynamics segments must be 'len(normalised_dynamics_segment_ends) - 1'")
            dynamics_tx = dm.GaussLobatto(num_segments=num_dynamics_segments_from_ends, segment_ends=normalised_dynamics_segment_ends, order=dynamics_seg_order, compressed=False)
        else:
            dynamics_tx = dm.GaussLobatto(num_segments=num_dynamics_segments, order=dynamics_seg_order, compressed=False)

        # === Set up control phase ===
        control_phase = dm.Phase(ode_class=ControlODE, transcription=control_tx)
        traj.add_phase('control_phase', control_phase)

        control_phase.set_time_options(fix_initial=True, duration_bounds=time_duration_bounds)
        control_phase.set_refine_options(refine=self.refine_control_tx)

        control_phase.add_control('da', targets=['da'], continuity=True, rate_continuity=True, units='rad', opt=True, ref=refs['da'], ref0=ref0s['da']) # , defect_ref=drefs['da'])
        control_phase.add_control('de', targets=['de'], continuity=True, rate_continuity=True, units='rad', opt=True, ref=refs['de'], ref0=ref0s['de']) # , defect_ref=drefs['de'])
        control_phase.add_control('dr', targets=['dr'], continuity=True, rate_continuity=True, units='rad', opt=False, ref=refs['dr'], ref0=ref0s['dr']) # , defect_ref=drefs['dr'])
        # Used to have units=None, but this broke it.
        control_phase.add_control('dp', targets=['dp'], continuity=True, rate_continuity=True, opt=enable_throttle_optimisation, ref=refs['dp'], ref0=ref0s['dp']) # , defect_ref=drefs['dp'])

        control_phase.add_state('control_surface_int', fix_initial=True, rate_source='control_surface_int_rate', units='rad*s')
        # Used to have units=None, but this broke it.
        control_phase.add_state('throttle_int', fix_initial=True, rate_source='throttle_int_rate')

        # Add alternative timeseries output to provide control inputs for the dynamics phase
        control_phase.add_timeseries('control_to_dyn_ts', transcription=dynamics_tx, subset='control_input')
        # Add alternative timeseries output to provide control inputs for the control rate integration phase
        control_phase.add_timeseries('control_to_rate_int_ts', transcription=control_rate_int_tx, subset='control_input')

        # Feed in times
        #control_phase.add_parameter('input_time', targets=['input_time'])
        #prob.model.connect('traj.control_phase.timeseries.time', 'traj.control_phase.parameters:input_time')
        
        # === Set up dynamics phase ===

        # =================================
        # Add states, controls and parameters

        dynamics_phase = dm.Phase(ode_class=BeardDynamicsJax, transcription=dynamics_tx, ode_init_kwargs={'wind_model': self.wm, 'aircraft_model': self.aircraft_model})
        traj.add_phase('dynamics_phase', dynamics_phase)

        dynamics_phase.set_time_options(fix_initial=True, input_duration=True)

        # States
        dynamics_phase.add_state('n', rate_source='n_dot', targets=['n'], units='m', ref=refs['n'], ref0=ref0s['n'], defect_ref=drefs['n'])
        dynamics_phase.add_state('e', rate_source='e_dot', targets=['e'], units='m', ref=refs['e'], ref0=ref0s['e'], defect_ref=drefs['e'])
        dynamics_phase.add_state('d', rate_source='d_dot', targets=['d'], units='m', ref=refs['d'], ref0=ref0s['d'], defect_ref=drefs['d'])
        dynamics_phase.add_state('u', rate_source='u_dot', targets=['u'], units='m/s', ref=refs['u'], ref0=ref0s['u'], defect_ref=drefs['u'])
        dynamics_phase.add_state('v', rate_source='v_dot', targets=['v'], units='m/s', ref=refs['v'], ref0=ref0s['v'], defect_ref=drefs['v'])
        dynamics_phase.add_state('w', rate_source='w_dot', targets=['w'], units='m/s', ref=refs['w'], ref0=ref0s['w'], defect_ref=drefs['w'])
        dynamics_phase.add_state('phi', rate_source='phi_dot', targets=['phi'], units='rad', ref=refs['phi'], ref0=ref0s['phi'], defect_ref=drefs['phi'])
        dynamics_phase.add_state('theta', rate_source='theta_dot', targets=['theta'], units='rad', ref=refs['theta'], ref0=ref0s['theta'], defect_ref=drefs['theta'])
        dynamics_phase.add_state('psi', rate_source='psi_dot', targets=['psi'], units='rad', ref=refs['psi'], ref0=ref0s['psi'], defect_ref=drefs['psi'])
        dynamics_phase.add_state('p', rate_source='p_dot', targets=['p'], units='rad/s', ref=refs['p'], ref0=ref0s['p'], defect_ref=drefs['p'])
        dynamics_phase.add_state('q', rate_source='q_dot', targets=['q'], units='rad/s', ref=refs['q'], ref0=ref0s['q'], defect_ref=drefs['q'])
        dynamics_phase.add_state('r', rate_source='r_dot', targets=['r'], units='rad/s', ref=refs['r'], ref0=ref0s['r'], defect_ref=drefs['r'])

        dynamics_phase.add_control('da', shape=(1,), opt=False, units='rad')
        dynamics_phase.add_control('de', shape=(1,), opt=False, units='rad')
        dynamics_phase.add_control('dr', shape=(1,), opt=False, units='rad')
        # Used to have units=None, but this broke it.
        dynamics_phase.add_control('dp', shape=(1,), opt=False)

        # Add connections - connect the two phases
        prob.model.connect('traj.control_phase.t_duration_val', 'traj.dynamics_phase.t_duration')
        prob.model.connect('traj.control_phase.control_to_dyn_ts.da', 'traj.dynamics_phase.controls:da')
        prob.model.connect('traj.control_phase.control_to_dyn_ts.de', 'traj.dynamics_phase.controls:de')
        prob.model.connect('traj.control_phase.control_to_dyn_ts.dr', 'traj.dynamics_phase.controls:dr')
        prob.model.connect('traj.control_phase.control_to_dyn_ts.dp', 'traj.dynamics_phase.controls:dp')

        # Parameters
        # No Dymos parameters in this model
        # Example: phase.add_parameter('m', units='kg', targets=['m'])

        # Add states which are for output only
        # Forces
        dynamics_phase.add_timeseries_output('L')
        dynamics_phase.add_timeseries_output('C')
        dynamics_phase.add_timeseries_output('D')
        dynamics_phase.add_timeseries_output('T')
        dynamics_phase.add_timeseries_output('W')
        # Moments
        dynamics_phase.add_timeseries_output('L_lat')
        dynamics_phase.add_timeseries_output('M')
        dynamics_phase.add_timeseries_output('N')

        # Air-relative states
        dynamics_phase.add_timeseries_output('va')
        dynamics_phase.add_timeseries_output('alpha')
        dynamics_phase.add_timeseries_output('beta')

        # Wind
        dynamics_phase.add_timeseries_output('w_n')
        dynamics_phase.add_timeseries_output('w_e')
        dynamics_phase.add_timeseries_output('w_d')

        # For calculating energy change
        dynamics_phase.add_timeseries_output('n_dot')
        dynamics_phase.add_timeseries_output('e_dot')
        dynamics_phase.add_timeseries_output('d_dot')

        # Fictitious forces
        dynamics_phase.add_timeseries_output('Fwn')
        dynamics_phase.add_timeseries_output('Fwe')
        dynamics_phase.add_timeseries_output('Fwd')

        # For constraining based on load factor
        dynamics_phase.add_timeseries_output('load_factor')

        # === Set up control rate integration component ===

        prob.model.add_subsystem('control_deriv_integrator', subsys=ControlRateIntegrator(num_nodes=control_rate_int_tx.grid_data.num_nodes))

        # Connect to the control rate integral calculating component
        prob.model.connect('traj.control_phase.control_to_rate_int_ts.time', 'control_deriv_integrator.time')
        prob.model.connect('traj.control_phase.control_to_rate_int_ts.da', 'control_deriv_integrator.da')
        prob.model.connect('traj.control_phase.control_to_rate_int_ts.de', 'control_deriv_integrator.de')

        # =================================
        # Set objective and constraints

        # Set objective: run callback
        set_objective(prob, control_phase, dynamics_phase, aircraft_model=self.aircraft_model)

        # Set up constraints
        # These constraints should apply to every trajectory optimisation

        # Constraints from aircraft model yaml file
        # Lower limit to prevent singularities
        dynamics_phase.add_path_constraint('va', lower=self.aircraft_model.min_va, upper=self.aircraft_model.max_va)
        dynamics_phase.add_path_constraint('alpha', lower=self.aircraft_model.min_alpha, upper=self.aircraft_model.max_alpha)
        dynamics_phase.add_path_constraint('beta', lower=self.aircraft_model.min_beta, upper=self.aircraft_model.max_beta)

        # Without constraints, numeric (e.g. overflow) errors can be encountered.
        dynamics_phase.add_path_constraint('u', lower=0, upper=100)
        dynamics_phase.add_path_constraint('v', lower=-50, upper=50)
        dynamics_phase.add_path_constraint('w', lower=-50, upper=50)
            
        dynamics_phase.add_path_constraint('p', lower=self.aircraft_model.min_p, upper=self.aircraft_model.max_p)
        dynamics_phase.add_path_constraint('q', lower=self.aircraft_model.min_q, upper=self.aircraft_model.max_q)
        dynamics_phase.add_path_constraint('r', lower=self.aircraft_model.min_r, upper=self.aircraft_model.max_r)

        dynamics_phase.add_path_constraint('da', lower=self.aircraft_model.min_da, upper=self.aircraft_model.max_da)
        dynamics_phase.add_path_constraint('de', lower=self.aircraft_model.min_de, upper=self.aircraft_model.max_de)
        dynamics_phase.add_path_constraint('dr', lower=self.aircraft_model.min_dr, upper=self.aircraft_model.max_dr)
        dynamics_phase.add_path_constraint('dp', lower=self.aircraft_model.min_dp, upper=self.aircraft_model.max_dp)

        # Structural constraint
        dynamics_phase.add_path_constraint('load_factor', lower=self.aircraft_model.min_load_factor, upper=self.aircraft_model.max_load_factor)

        # Constrain out gimbal lock
        dynamics_phase.add_path_constraint('theta', lower=np.radians(-80), upper=np.radians(80))

        # Set problem-specific constraints: run callback
        set_additional_constraints(prob, control_phase, dynamics_phase)

        # =================================
        # Configure

        # Set IPOPT as optimiser
        prob.driver = om.pyOptSparseDriver(optimizer='IPOPT')
        # prob.driver = om.ScipyOptimizeDriver()
        prob.driver.opt_settings['print_level'] = 5
        if max_iter is not None:
            prob.driver.opt_settings['max_iter'] = max_iter
        # For debugging
        # prob.driver.opt_settings['tol'] = 1e-1 # 30

        prob.driver.declare_coloring()

        # Create recorder and attach to problem
        # Have to attach a separate recorder for each solve
        recorder = om.SqliteRecorder(self.dymos_output_path / f'plane_{solve_num}.sql')
        prob.driver.add_recorder(recorder)

        # Driver configuration: run callback
        set_driver_options(prob.driver)

        prob.setup() # force_alloc_complex=True)

        # =================================
        # Set initial state, control and parameter values

        # Applicable to all problems
        # TODO What's the difference between the 'prob.set_val' and 'prob[...]' formulations?
        prob.set_val('traj.control_phase.t_initial', 0.0)
        prob['traj.control_phase.states:control_surface_int'] = 0.0
        prob['traj.control_phase.states:throttle_int'] = 0.0

        # Others: run callback
        set_initial_values(prob, control_phase, dynamics_phase)

        # =================================
        # Run simulation

        prob.run_model()

        if check_partials:
            with open(self.solver_output_folder / 'check_partials_out.txt', 'w') as f:
                cp_out = prob.check_partials(method='fd', compact_print=True, out_stream=f)
                # cp_out = prob.check_partials(method='cs', compact_print=True, out_stream=f)
            self.log_errors(cp_out, self.solver_output_folder)
        
        if solve:
            # NOTE Not implementing warm starting at the moment
            dm.run_problem(prob, run_driver=True, refine_iteration_limit=num_refinements, refine_method='hp', solution_record_file=self.dymos_output_path / 'dymos_solution.db') # , restart=str(prev_sim_log_file_path) if from_sim else None)
        
        # =================================

        # Move grid refinement file to solver_output_folder, if it exists.
        try:
            grid_refinement_file = Path('.') / 'grid_refinement.out'
            # TODO Test this numbering system.
            # grid_refinement_file.rename(self.solver_output_folder / f'grid_refinement_{solve_num}.out')
            shutil.move(grid_refinement_file, self.solver_output_folder / f'grid_refinement_{solve_num}.out')
        except FileNotFoundError:
            print("Grid refinement file doesn't exist - not moving.")
        
        # TODO Is this still a valid comment?
        # Want to copy this for safety but it doesn't seem to copy all of the information, e.g. information
        # within the traj.options dictionary seems to be lost (TODO).
        # copy.deepcopy(prob))
        self.probs.append(prob)
        return prob
    
    # Simulate, generate datalogger and create plots.
    # Note: the simulation could raise a RatesCompException exception, which will propagate up through this
    # method.
    def post_process(self, prob=None, plot=False):
        # If a Problem object is not provided, just use the most recent one.
        if prob is None:
            prob = self.probs[-1]
        
        traj = prob.model.traj
        # This might raise a RatesCompException
        sim_out = traj.simulate(times_per_seg=50, max_step=0.05, record_file=self.dymos_output_path / 'sim.db' if self.record_sim else None)
        # sim_out = traj.simulate(times_per_seg=200, max_step=0.01, record_file=dymos_output_path / 'sim.db' if record_sim else None)

        opt_success = not prob.driver.fail
        print(f"Success: {opt_success}")

        # Generate and save DataLogger, based on *simulation* output.
        dl = self.generate_datalogger(sim_out, self.datalogger_dt, self.foldername, opt_success)

        if plot:
            # TODO Check that the systems which use plane.sql still work.
            results_plotting.generate_plots(self.figure_path, dl, self.aircraft_model, prob, save=config.save_figs)
        
        return dl
    
    # Generate DataLogger which can be passed to simulator for re-simulation
    # and comparison.
    # Interpolate optimisation simulation output at the required dt
    # Create DataLogger instance with values
    # Save DataLogger
    # opt_success == True if the optimisation was successful
    def generate_datalogger(self, sim_out, dt, save_name, opt_success):
        times = sim_out.get_val(f'traj.dynamics_phase.timeseries.time').flatten()
        log_times = np.arange(0, times[-1], dt)
        log_vars = ['n', 'e', 'd', 'u', 'v', 'w', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'va', 'alpha', 'beta', 'da', 'de', 'dr', 'dp', 'L', 'C', 'D', 'T', 'W', 'Fwn', 'Fwe', 'Fwd', 'L_lat', 'M', 'N', 'w_n', 'w_e', 'w_d', 'load_factor'] # , 'da_dot', 'de_dot', 'dr_dot', 'dp_dot']

        log_arr = np.empty(shape=(len(log_vars) + 1, len(log_times)))
        log_arr[0, :] = log_times
        
        # TODO This needs checking
        for log_var_ind in range(len(log_vars)):
            values = sim_out.get_val(f'traj.dynamics_phase.timeseries.{log_vars[log_var_ind]}').flatten()
            # Interpolate at the required dt
            # log_var_ind + 1 because the 0th entry is for the times
            log_arr[log_var_ind + 1, :] = np.interp(log_times, times, values)
        
        # Create DataLogger
        num_steps = len(log_times)
        dl = DataLogger(dt, num_steps, log_arr, num_steps-1, True, opt_success)

        # Save DataLogger - automatically saves it to the same folder as the others.
        dl.save_to_path(self.datalogger_path / f"{save_name}_optimiser")

        # Return DataLogger
        return dl
    # =================================
    # For checking partial derivatives

    # I can't seem to get the built-in check_partials method to print only the required rate errors, so writing a custom one.
    # Note: this only captures the forwards errors.
    def log_errors(self, cp_out, base_folder = '.'):
        folder = Path(base_folder)      # This will still work even if base_folder is already a Path.

        # Main: discretisation nodes, collocation nodes.
        self.write_errors(folder / 'dynamics_errors.md', cp_out['traj.phases.dynamics_phase.rhs_disc'], cp_out['traj.phases.dynamics_phase.rhs_col'], ['n_dot', 'e_dot', 'd_dot', 'u_dot', 'v_dot', 'w_dot', 'phi_dot', 'theta_dot', 'psi_dot', 'p_dot', 'q_dot', 'r_dot'])
        # For control regularisation.
        # NOTE This is from when there was an additional phase for computing the total control etc - phase1.
        # self.write_errors(folder / 'regularisation_errors.md', cp_out['phase1.rhs_disc'], cp_out['phase1.rhs_col'])


    # file should be a path to a Markdown file
    # Helper method for log_errors
    @staticmethod
    def write_errors(filepath, disc, col, key_filter=None):
        if disc.keys() != col.keys():
            raise Exception("disc and col should have matching keys!")
        
        err_dict = {}

        # Want to also find keys (partials) for which the derivative is zero and report them (they could be zero by chance, but this could always be investigated
        # manually).
        zero_combs = []
        abs_discs = []
        rel_discs = []
        abs_cols = []
        rel_cols = []
        for k in disc.keys():
            if (key_filter is None) or (k[0] in key_filter):
                err_dict[k] = {}
                calc_mag = disc[k]['magnitude'][0]
                check_mag = disc[k]['magnitude'][2]
                if (calc_mag == 0) and (check_mag != 0):
                    zero_combs.append(k)
                err_dict[k]['calc_mag'] = calc_mag
                err_dict[k]['check_mag'] = check_mag

                abs_disc = disc[k]['abs error'][0]
                rel_disc = disc[k]['rel error'][0]
                abs_col = col[k]['abs error'][0]
                rel_col = col[k]['rel error'][0]

                err_dict[k]['abs_disc'] = abs_disc
                err_dict[k]['rel_disc'] = rel_disc
                err_dict[k]['abs_col'] = abs_col
                err_dict[k]['rel_col'] = rel_col

                # To get largest errors:
                abs_discs.append(abs_disc)
                rel_discs.append(rel_disc)
                abs_cols.append(abs_col)
                rel_cols.append(rel_col)

        with open(filepath, 'w') as f:
            f.write(f"| var | deriv | calc mag. | check mag. | disc: abs | disc: rel | col: abs | col: rel |\n")
            f.write(f"| - | - | - | - | - | - | - | - |\n")
            for k in err_dict:
                vals = err_dict[k]
                f.write(f"| {k[0]} | {k[1]} | {vals['calc_mag']} | {vals['check_mag']} | {vals['abs_disc']} | {vals['rel_disc']} | {vals['abs_col']} | {vals['rel_col']} |\n")
            f.write("\n\n")
            f.write("Zero combinations\n")
            f.write("=================\n")
            for k in zero_combs:
                f.write(f" - d({k[0]})/d({k[1]})\n")
            f.write("\n\n")
            f.write(f"Max absolute discretisation error: {TrajectorySolver.get_largest_error(abs_discs)}\n")
            f.write("\n")
            f.write(f"Max relative discretisation error: {TrajectorySolver.get_largest_error(rel_discs)}\n")
            f.write("\n")
            f.write(f"Max absolute collocation error: {TrajectorySolver.get_largest_error(abs_cols)}\n")
            f.write("\n")
            f.write(f"Max relative collocation error: {TrajectorySolver.get_largest_error(rel_cols)}\n")

    # Helper method for write_errors
    @staticmethod
    def get_largest_error(l):
        a = np.array(l)
        # Remove NaNs
        a = a[~np.isnan(a)]
        a.sort()
        return a[-1]




# For now we're not doing warm starting. Warm starting from the previous is accomplished
# by interpolating from the previously found states and controls.
"""
# Get path to previous simulation file, for warm starting.
log_folders = [f for f in config.dymos_logs_folder_path.iterdir() if f.is_dir()]
log_folders.sort(key=lambda x: os.path.getmtime(x))
if len(log_folders) >= 1:
    prev_sim_log_file_path = log_folders[-1] / 'sim.db'
else:
    from_sim = False
"""

"""
# Place constraints on the controls (the control rates)
# TODO What should these values be?
control_phase.add_path_constraint('da_dot', lower=-1., upper=1.)
control_phase.add_path_constraint('de_dot', lower=-1., upper=1.)
control_phase.add_path_constraint('dr_dot', lower=-1., upper=1.)
control_phase.add_path_constraint('dp_dot', lower=-1., upper=1.)
"""