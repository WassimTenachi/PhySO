import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import argparse

# Internal imports
import physo.benchmark.FeynmanDataset.FeynmanProblem as Feyn
import physo

# Local imports
import feynman_config as fconfig

# Parallel config :
# Parallel mode may cause issues due to the number of samples, non-parallel mode is recommended
# Single core with so many samples will actually use up to 10 cores via pytorch parallelization along sample dim
PARALLEL_MODE_DEFAULT = False
N_CPUS_DEFAULT        = 1

# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------
parser = argparse.ArgumentParser (description     = "Runs a Feynman problem job.",
                                  formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--equation", default = 0,
                    help = "Equation number in the set (e.g. 0 to 99 for bulk eqs and 100 to 119 for bonus eqs).")
parser.add_argument("-t", "--trial", default = 0,
                    help = "Trial number (sets seed accordingly).")
parser.add_argument("-n", "--noise", default = 0.,
                    help = "Noise level fraction.")
parser.add_argument("-p", "--parallel_mode", default = PARALLEL_MODE_DEFAULT,
                    help = "Should parallel mode be used.")
parser.add_argument("-ncpus", "--ncpus", default = N_CPUS_DEFAULT,
                    help = "Nb. of CPUs to use")
config = vars(parser.parse_args())

# Feynman problem number
I_FEYN  = int(config["equation"])
# Trial number
N_TRIAL = int(config["trial"])
# Noise level
NOISE_LEVEL = float(config["noise"])
# Parallel config
PARALLEL_MODE = bool(config["parallel_mode"])
N_CPUS        = int(config["ncpus"])
# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------

if __name__ == '__main__':

    # ----- HYPERPARAMS -----
    FIXED_CONSTS       = fconfig.FIXED_CONSTS
    FIXED_CONSTS_UNITS = fconfig.FIXED_CONSTS_UNITS
    FREE_CONSTS_NAMES  = fconfig.FREE_CONSTS_NAMES
    FREE_CONSTS_UNITS  = fconfig.FREE_CONSTS_UNITS
    OP_NAMES           = fconfig.OP_NAMES
    N_SAMPLES          = fconfig.N_SAMPLES
    CONFIG             = fconfig.CONFIG
    MAX_N_EVALUATIONS  = fconfig.MAX_N_EVALUATIONS
    N_EPOCHS           = fconfig.N_EPOCHS
    ORIGINAL_VAR_NAMES = fconfig.ORIGINAL_VAR_NAMES

    # Fixing seed accordingly with attempt number
    seed = N_TRIAL
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Paths
    RUN_NAME       = "FR_%i_%i_%f"%(I_FEYN, N_TRIAL, NOISE_LEVEL)
    PATH_DATA      = "%s_data.csv"%(RUN_NAME)
    PATH_DATA_PLOT = "%s_data.png"%(RUN_NAME)

    # Making a directory for this run and run in it
    if not os.path.exists(RUN_NAME):
        os.makedirs(RUN_NAME)
    os.chdir(os.path.join(os.path.dirname(__file__), RUN_NAME,))

    # Copying .py this script to the directory
    # shutil.copy2(src = __file__, dst = os.path.join(os.path.dirname(__file__), RUN_NAME))

    # MONITORING CONFIG TO USE
    get_run_logger     = lambda : physo.learn.monitoring.RunLogger(
                                          save_path = 'SR.log',
                                          do_save   = True)
    get_run_visualiser = lambda : physo.learn.monitoring.RunVisualiser (
                                               epoch_refresh_rate = 1,
                                               save_path = 'SR_curves.png',
                                               do_show   = False,
                                               do_prints = True,
                                               do_save   = True, )

    # Loading Feynman problem
    pb = Feyn.FeynmanProblem(I_FEYN, original_var_names=ORIGINAL_VAR_NAMES)

    # Generate data
    X, y = pb.generate_data_points (n_samples = N_SAMPLES)

    # Noise
    y_rms = ((y ** 2).mean()) ** 0.5
    epsilon = NOISE_LEVEL * np.random.normal(0, y_rms, len(y))
    y = y + epsilon

    # Save data
    df = pd.DataFrame(data    = np.concatenate((y[np.newaxis,:], X), axis=0).transpose(),
                      columns = [pb.y_name] + pb.X_names.tolist())
    df.to_csv(PATH_DATA, sep=";")

    # Plot data
    mpl.rcParams.update(mpl.rcParamsDefault)
    n_dim = X.shape[0]
    fig, ax = plt.subplots(n_dim, 1, figsize=(10, n_dim * 4))
    fig.suptitle(pb.formula_original)
    for i in range(n_dim):
        curr_ax = ax if n_dim == 1 else ax[i]
        curr_ax.plot(X[i], y, 'k.', markersize=0.1)
        curr_ax.set_xlabel("%s : %s" % (pb.X_names[i], pb.X_units[i]))
        curr_ax.set_ylabel("%s : %s" % (pb.y_name    , pb.y_units))
    # Save plot
    fig.savefig(PATH_DATA_PLOT)

    # Printing start
    print("%s : Starting SR task"%(RUN_NAME))

    # Running SR task
    expression, logs = physo.SR(X, y,
                # Giving names of variables (for display purposes)
                X_names = pb.X_names,
                # Giving units of input variables
                X_units = pb.X_units,
                # Giving name of root variable (for display purposes)
                y_name  = pb.y_name,
                # Giving units of the root variable
                y_units = pb.y_units,
                # Fixed constants
                fixed_consts       = FIXED_CONSTS,
                # Units of fixed constants
                fixed_consts_units = FIXED_CONSTS_UNITS,
                # Free constants names (for display purposes)
                free_consts_names = FREE_CONSTS_NAMES,
                # Operations allowed
                op_names = OP_NAMES,
                # Units of free constants
                free_consts_units = FREE_CONSTS_UNITS,
                # Run config
                run_config = CONFIG,
                # Run monitoring
                get_run_logger     = get_run_logger,
                get_run_visualiser = get_run_visualiser,
                # Stopping condition
                stop_reward = 1.1,  # not stopping even if perfect 1.0 reward is reached
                max_n_evaluations = MAX_N_EVALUATIONS,
                epochs            = N_EPOCHS,
                # Parallel mode
                parallel_mode = PARALLEL_MODE,
                n_cpus        = N_CPUS,
        )

    # Printing end
    print("%s : SR task finished"%(RUN_NAME))
