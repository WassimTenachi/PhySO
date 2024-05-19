import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import argparse

# Internal imports
import physo.benchmark.ClassDataset.ClassProblem as ClPb
import physo

# Local imports
import classbench_config as fconfig

# Parallel config :
# Parallel mode may cause issues due to the number of samples, non-parallel mode is recommended
# Single core with so many samples will actually use up to 10 cores via pytorch parallelization along sample dim
PARALLEL_MODE_DEFAULT = False
N_CPUS_DEFAULT        = 1


# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------
parser = argparse.ArgumentParser (description     = "Runs a Class Benchmark problem job.",
                                  formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--equation", default = 0,
                    help = "Equation number in the set..")
parser.add_argument("-t", "--trial", default = 0,
                    help = "Trial number (sets seed accordingly).")
parser.add_argument("-n", "--noise", default = 0.,
                    help = "Noise level fraction.")
parser.add_argument("-r", "--n_reals", default = 1,
                    help = "Number of realizations to use.")
parser.add_argument("-p", "--parallel_mode", default = PARALLEL_MODE_DEFAULT,
                    help = "Should parallel mode be used.")
parser.add_argument("-ncpus", "--ncpus", default = N_CPUS_DEFAULT,
                    help = "Nb. of CPUs to use")
config = vars(parser.parse_args())

# Class benchmark problem number
I_PB = int(config["equation"])
# Trial number
N_TRIAL = int(config["trial"])
# Noise level
NOISE_LEVEL = float(config["noise"])
# Nb of realizations
N_REALS = int(config["n_reals"])
# Parallel config
PARALLEL_MODE = bool(config["parallel_mode"])
N_CPUS        = int(config["ncpus"])
# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------

if __name__ == '__main__':

    # ----- HYPERPARAMS -----
    DIMENSIONLESS_RUN  = fconfig.DIMENSIONLESS_RUN
    FIXED_CONSTS       = fconfig.FIXED_CONSTS
    FIXED_CONSTS_UNITS = fconfig.FIXED_CONSTS_UNITS
    CLASS_FREE_CONSTS_NAMES = fconfig.CLASS_FREE_CONSTS_NAMES
    CLASS_FREE_CONSTS_UNITS = fconfig.CLASS_FREE_CONSTS_UNITS
    SPE_FREE_CONSTS_NAMES   = fconfig.SPE_FREE_CONSTS_NAMES
    SPE_FREE_CONSTS_UNITS   = fconfig.SPE_FREE_CONSTS_UNITS
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
    RUN_NAME       = "CR_%i_%i_%i_%f"%(I_PB, N_TRIAL, N_REALS, NOISE_LEVEL)
    PATH_DATA      = "%s_data.csv"    %(RUN_NAME)
    PATH_DATA_PLOT = "%s_data.png"    %(RUN_NAME)
    PATH_DATAGEN_K = "%s_datagen.csv" %(RUN_NAME)

    # Making a directory for this run and run in it
    if not os.path.exists(RUN_NAME):
        os.makedirs(RUN_NAME)
    os.chdir(os.path.join(os.path.dirname(__file__), RUN_NAME,))

    # Copying .py this script to the directory
    # shutil.copy2(src = __file__, dst = os.path.join(os.path.dirname(__file__), RUN_NAME))

    # MONITORING CONFIG TO USE
    get_run_logger     = lambda : physo.learn.monitoring.RunLogger(
                                          save_path = 'SR.log',
                                          do_save   = False)
    get_run_visualiser = lambda : physo.learn.monitoring.RunVisualiser (
                                               epoch_refresh_rate = 1,
                                               save_path = 'SR_curves.png',
                                               do_show   = False,
                                               do_prints = True,
                                               do_save   = True, )

    # Loading Class problem
    pb = ClPb.ClassProblem(i_eq=I_PB, original_var_names=ORIGINAL_VAR_NAMES)

    # Units
    if DIMENSIONLESS_RUN:
        X_units = np.array([fconfig.dimensionless_units, ] * len(pb.X_names))
        y_units = fconfig.dimensionless_units
    else:
        X_units = pb.X_units
        y_units = pb.y_units

    # Generate data
    multi_X, multi_y, multi_K = pb.generate_data_points (n_samples = N_SAMPLES, n_realizations = N_REALS, return_K=True)

    # Noise
    for i_real in range(N_REALS):
        y = multi_y[i_real]
        y_rms = ((y ** 2).mean()) ** 0.5
        epsilon = NOISE_LEVEL * np.random.normal(0, y_rms, len(y))
        multi_y[i_real] = y + epsilon

    # Save data
    # Saving data with columns like i_real | y | x0 | x1 | ...
    i_reals_col = np.arange(N_REALS).repeat(N_SAMPLES)[np.newaxis,:] # (1,     N_REALS * N_SAMPLES)
    X_cols      = np.concatenate(multi_X, axis=1)                    # (N_DIM, N_REALS * N_SAMPLES)
    y_col       = np.concatenate(multi_y, axis=0)[np.newaxis,:]      # (1,     N_REALS * N_SAMPLES)
    df = pd.DataFrame(data    = np.concatenate((i_reals_col, y_col, X_cols), axis=0).transpose(),
                      columns = ["i_real"] + [pb.y_name] + pb.X_names.tolist())
    df = df.astype({"i_real": int})
    df.to_csv(PATH_DATA, sep=";", index=False)

    # Save data generation info
    # Saving data with columns like i_real | k1 | k2 | ...
    df_gen = pd.DataFrame(data    = np.concatenate((np.arange(N_REALS)[np.newaxis,:], multi_K.transpose()), axis=0).transpose(),
                          columns = ["i_real"] + pb.K_names.tolist())
    df_gen = df_gen.astype({"i_real": int})
    df_gen.to_csv(PATH_DATAGEN_K, sep=";", index=False)

    # Plot data
    mpl.rcParams.update(mpl.rcParamsDefault)
    n_dim = multi_X.shape[1]
    fig, ax = plt.subplots(n_dim, 1, figsize=(10, n_dim * 4))
    fig.suptitle(pb.formula_original)
    for i in range(n_dim):
        curr_ax = ax if n_dim == 1 else ax[i]
        curr_ax.set_xlabel("%s : %s" % (pb.X_names[i], pb.X_units[i]))
        curr_ax.set_ylabel("%s : %s" % (pb.y_name, pb.y_units))
        for i_real in range(N_REALS):
            curr_ax.plot(multi_X[i_real, i], multi_y[i_real], '.', markersize=1.)
    # Save plot
    fig.savefig(PATH_DATA_PLOT)

    # Printing start
    print("%s : Starting SR task"%(RUN_NAME))

    # Running SR task
    expression, logs = physo.ClassSR(multi_X, multi_y,
                # Giving names of variables (for display purposes)
                X_names = pb.X_names,
                # Giving units of input variables
                X_units = X_units,
                # Giving name of root variable (for display purposes)
                y_name  = pb.y_name,
                # Giving units of the root variable
                y_units = y_units,
                # Fixed constants
                fixed_consts       = FIXED_CONSTS,
                # Units of fixed constants
                fixed_consts_units = FIXED_CONSTS_UNITS,
                # Free constants names (for display purposes)
                class_free_consts_names = CLASS_FREE_CONSTS_NAMES,
                # Units of free constants
                class_free_consts_units = CLASS_FREE_CONSTS_UNITS,
                # Spe free constants names (for display purposes)
                spe_free_consts_names = SPE_FREE_CONSTS_NAMES,
                # Units of spe free constants
                spe_free_consts_units = SPE_FREE_CONSTS_UNITS,
                # Operations allowed
                op_names = OP_NAMES,
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