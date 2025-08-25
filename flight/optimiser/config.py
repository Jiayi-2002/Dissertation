from pathlib import Path

animation_interval_ms = 20
# For building plotting
building_alpha = 0.3
save_figs = True
default_figure_folder_path                 = Path('.') / 'data' / 'figs'
default_dymos_logs_folder_path      = Path('.') / 'data' / 'dymos_logs'
default_datalogger_save_path        = Path('.') / 'data' / 'dataloggers'
default_solver_output_folder_path   = Path('.') / 'data' / 'solver_output'
# This file has to be called 'ipopt.opt', and must be in the folder that IPOPT is run from.
# solver_settings_filepath    = Path('/home/kinglouis/miniconda3/pkgs/ipopt-3.14.16-h3a0b567_5') / 'ipopt.opt'
solver_settings_filepath    = Path('.') / 'ipopt.opt'