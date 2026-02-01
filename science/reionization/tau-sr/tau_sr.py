import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import argparse
import os
from sklearn.feature_selection import mutual_info_regression

# Package imports
import physo
import physo.learn.monitoring as monitoring

# Local imports
import config as custom_config

# Parallel config :
# Parallel mode may cause issues due to the number of samples, non-parallel mode is recommended
# Single core with so many samples will actually use up to 10 cores via pytorch parallelization along sample dim
PARALLEL_MODE_DEFAULT = True
N_CPUS_DEFAULT        = 8

# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------
parser = argparse.ArgumentParser (description     = "Runs a tau SR job.", # TAU SPECIFIC
                                  formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-sim", "--simulation", default = "amber",
                    help = "Whether to use data from 21cmFast (21f) sim or Amber (amber).")
parser.add_argument("-xe", "--x_essential", default = False,
                    help = "Whether to only use essential x variables.")
parser.add_argument("-fp", "--parameters", default = 5,
                    help = "Number of free parameters set to use.")
parser.add_argument("-ll", "--length_loc", default = custom_config.LENGTH_LOC,
                    help = "Soft length prior location.")
parser.add_argument("-ls", "--length_scale", default = custom_config.LENGTH_SCALE,
                    help = "Soft length prior scale.")
parser.add_argument("-s", "--seed", default = 0,
                    help = "Seed to use.")
parser.add_argument("-p", "--parallel_mode", default = PARALLEL_MODE_DEFAULT,
                    help = "Should parallel mode be used.")
parser.add_argument("-ncpus", "--ncpus", default = N_CPUS_DEFAULT,
                    help = "Nb. of CPUs to use")
config = vars(parser.parse_args())

# Simulation name
SIMULATION = str(config["simulation"])
# Seed
SEED = int(config["seed"])
# Number of free parameters
N_FREE_PARAMS = int(config["parameters"])
# Soft length prior params
LENGTH_LOC_ARG   = int(config["length_loc"])
LENGTH_SCALE_ARG = int(config["length_scale"])
# Whether to only use essential x variables
X_ESSENTIAL_ONLY = bool(int(config["x_essential"]))
# Parallel config
PARALLEL_MODE = bool(config["parallel_mode"])
N_CPUS        = int(config["ncpus"])
# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------

# Avoid race conditions with multiprocessing:
import matplotlib
matplotlib.rcParams['text.usetex'] = False  # avoid TeX rendering

if __name__ == '__main__':

    # region  ## Fixing seed
    seed = SEED
    np.random.seed(seed)
    torch.manual_seed(seed)
    # endregion

    # region ## Loading dataset
    df = pd.read_csv(f'../data/{SIMULATION}/tau_training_data.csv') # TAU SPECIFIC
    # endregion

    # region ## Run name and paths
    # Paths
    RUN_NAME = ("TAU_SR_%s_xe%i_fp%d_lloc%d_lscale%d_s%d" % (SIMULATION, int(X_ESSENTIAL_ONLY), N_FREE_PARAMS, LENGTH_LOC_ARG, LENGTH_SCALE_ARG, SEED)) # TAU SPECIFIC
    PATH_DATA      = "%s_data.csv" % (RUN_NAME)
    PATH_DATA_CORR = "%s_data_features.csv" % (RUN_NAME)
    PATH_DATA_PLOT = "%s_data.png" % (RUN_NAME)

    # Making a directory for this run and run in it
    if not os.path.exists(RUN_NAME):
        os.makedirs(RUN_NAME)
    os.chdir(os.path.join(os.path.dirname(__file__), RUN_NAME, ))

    # endregion

    # region ## Dataset

    # TAU SPECIFIC
    X_names_dict = {
        "21f": {
            "X_names"           : ["OMm","OMb","h","sigma_8", "n_s", "F_STAR10","F_ESC10","ALPHA_STAR","ALPHA_ESC","M_TURN","R_BUBBLE_MAX"],
            "X_names_essential" : ["F_ESC10","M_TURN","F_STAR10","ALPHA_ESC"],},
        "amber": {
            "X_names"           : ['OMm', 'OMb', 'h', 'sigma_8', 'n_s', 'z_mid', 'z_dur', 'z_asy'],
            "X_names_essential" : ['OMb', 'h', 'z_mid', 'z_dur', 'z_asy'], },
            }

    X_names           = X_names_dict[SIMULATION]["X_names"]
    X_names_essential = X_names_dict[SIMULATION]["X_names_essential"]

    if X_ESSENTIAL_ONLY:
        X_names = X_names_essential

    n_dim = len(X_names)
    X = df[X_names].to_numpy().T                # (n_dim, n_samples)
    y_name = 'tau' # TAU SPECIFIC
    y = df[y_name].to_numpy()                   # (n_samples,)
    # Save data
    df = pd.DataFrame(data=np.concatenate((y[np.newaxis, :], X), axis=0).transpose(),
                      columns=[y_name] + X_names)
    df.to_csv(PATH_DATA, sep=";")

    # Dataset plot
    n_dim = X.shape[0]
    fig, ax = plt.subplots(n_dim, 1, figsize=(10,int(n_dim*5)))
    for i in range (n_dim):
        curr_ax = ax if n_dim==1 else ax[i]
        curr_ax.plot(X[i], y, 'k.',)
        curr_ax.set_xlabel("X[%i]"%(i))
        curr_ax.set_ylabel("y")
    plt.show()
    # Save plot
    fig.savefig(PATH_DATA_PLOT)

    # region ## Dataset feature importance

    # Compute Pearson correlation coefficients
    pearsons = np.array([np.corrcoef(X.T[:, i], y)[0, 1] for i in range(X.T.shape[1])])
    # Compute Mutual Information (nonlinear dependence)
    mi = mutual_info_regression(X.T, y, random_state=0)

    # Build combined DataFrame
    df_features = pd.DataFrame({
        'Variable': X_names,
        'Pearson_r': pearsons,
        'Abs_r': np.abs(pearsons),
        'Mutual_Info': mi
    }).sort_values('Mutual_Info', ascending=False)
    df_features.to_csv(PATH_DATA_CORR, sep=";")

    # endregion

    # region ## SR config

    CONFIG = custom_config.custom_config
    # Tuning length prior according to args
    for prior in CONFIG['priors_config']:
        if prior[0] == 'SoftLengthPrior':
            prior[1]['length_loc'] = LENGTH_LOC_ARG
            prior[1]['scale']      = LENGTH_SCALE_ARG

    physo.config.utils.soft_length_plot(CONFIG, save_path="soft_length_prior.png", do_show=False)

    FREE_CONSTS_NAMES = ["c%d"%(i) for i in range(N_FREE_PARAMS)]
    FIXED_CONSTS = [1.]

    OP_NAMES = ["mul", "add", "sub", "div", "inv", "neg", "n2", "sqrt", "exp", "log",] # TAU SPECIFIC
    # pow, sin, tanh
    MAX_N_EVALUATIONS = int(1e99) # Will stop the run regardless of N_EPOCHS
    N_EPOCHS = int(1e99)

    # endregion

    # region ## Logging config

    save_path_training_curves = 'sr_curves.png'
    save_path_log             = 'sr.log'

    run_logger     = lambda : monitoring.RunLogger(save_path = save_path_log,
                                                    do_save = True)

    run_visualiser = lambda : monitoring.RunVisualiser (epoch_refresh_rate = 1,
                                               save_path = save_path_training_curves,
                                               do_show   = False,
                                               do_prints = True,
                                               do_save   = True, )

    # endregion

    # region ## SR Run

    # Printing start
    print("%s : Starting SR task"%(RUN_NAME))

    # Running SR task
    expression, logs = physo.SR(X, y,
                                X_names = X_names,
                                y_name  = y_name,
                                # Fixed constants
                                fixed_consts       = FIXED_CONSTS,
                                # Free constants
                                free_consts_names = FREE_CONSTS_NAMES,
                                # Symbolic operations that can be used to make f
                                op_names = OP_NAMES,
                                get_run_logger     = run_logger,
                                get_run_visualiser = run_visualiser,
                                # Run config
                                run_config = CONFIG,
                                max_n_evaluations = MAX_N_EVALUATIONS,
                                epochs            = N_EPOCHS,
                                # Parallel mode
                                parallel_mode = PARALLEL_MODE,
                                n_cpus        = N_CPUS,
    )

    # Printing end
    print("%s : SR task finished" % (RUN_NAME))

    # endregion



