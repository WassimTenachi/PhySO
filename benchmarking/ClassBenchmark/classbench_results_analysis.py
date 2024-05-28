import warnings
import numpy as np
import sympy
import pandas as pd
import argparse
import scipy.stats as st
import os
import time
import platform

# Internal imports
import physo.benchmark.ClassDataset.ClassProblem as ClPb
import physo.benchmark.utils.symbolic_utils as su
import physo.benchmark.utils.metrics_utils as metrics_utils
import physo.benchmark.utils.timeout_unix  as timeout_unix
import physo.benchmark.utils.read_logs     as read_logs

# Local imports
import classbench_config as fconfig
from benchmarking import utils as bu


# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------
parser = argparse.ArgumentParser (description     = "Analyzes Class benchmark run results folder (works on ongoing "
                                                    "benchmarks) and produces .csv files containing results and a "
                                                    "summary.",
                                  formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--path", default = ".",
                    help = "Paths to results folder.")
parser.add_argument("-u", "--list_unfinished", default = 1,
                    help = "Save a list of unfinished runs.")
config = vars(parser.parse_args())

RESULTS_PATH    = str(config["path"])
SAVE_UNFINISHED = bool(int(config["list_unfinished"]))

# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------

ORIGINAL_VAR_NAMES = fconfig.ORIGINAL_VAR_NAMES
EXCLUDED_EQS = fconfig.EXCLUDED_EQS

N_TRIALS       = fconfig.N_TRIALS
NOISE_LEVELS   = fconfig.NOISE_LEVELS
N_REALIZATIONS = fconfig.N_REALIZATIONS

# Batch size
BATCH_SIZE = fconfig.CONFIG["learning_config"]["batch_size"]

# Path where to save jobfile to relaunch unfinished jobs
PATH_UNFINISHED_JOBFILE          = os.path.join(RESULTS_PATH, "jobfile_unfinished")
PATH_UNFINISHED_BUSINESS_JOBFILE = os.path.join(RESULTS_PATH, "jobfile_unfinished_business")


# Unfinished jobs list
unfinished_jobs          = []
# Unfinished + target not recovered job list
unfinished_business_jobs = []

# Iterating through Class problems
for i_eq in range (ClPb.N_EQS):
    print("\nProblem #%i"%(i_eq))
    # Loading a problem
    pb = ClPb.ClassProblem(i_eq, original_var_names=ORIGINAL_VAR_NAMES)
    # Making run file only if it is not in excluded problems
    if pb.eq_name not in EXCLUDED_EQS:
        print(pb)
        # Iterating through trials
        for i_trial in range (N_TRIALS):
            for noise_lvl in NOISE_LEVELS:
                for n_reals in N_REALIZATIONS:

                    run_result = {}

                    run_name = "CR_%i_%i_%i_%f"%(i_eq, i_trial, n_reals, noise_lvl)
                    path_run = os.path.join(RESULTS_PATH, run_name)

                    # ----- Loading run data -----

                    path_curves = os.path.join(path_run, "SR_curves_data.csv")

                    try:
                        curves_df = pd.read_csv(path_curves)
                    except:
                        warnings.warn("Unable to load curves data .csv: %s" % (run_name))
                        curves_df = None


                    # ----- Logging run details -----

                    try:
                        n_evals     = curves_df["n_rewarded"].sum()
                        n_epochs    = curves_df["epoch"].iloc[-1]
                        is_started  = curves_df["epoch"].iloc[-1] >= 0
                        is_finished = n_evals >= (fconfig.MAX_N_EVALUATIONS - BATCH_SIZE- 1)
                    except:
                        # If curves were not loaded -> run was not started
                        n_evals     = 0
                        n_epochs    = 0
                        is_started  = False
                        is_finished = False

                    run_details = {
                        '# EVALUATIONS' : n_evals,
                        'STARTED'       : is_started,
                        'FINISHED'      : is_finished,
                        }
                    run_result.update(run_details)


                    # ----- Symbolic equivalence related -----
                    equivalence_report = {
                        "symbolic_solution": False,
                        }

                    # ----- Listing unfinished jobs -----
                    command = "python classbench_run.py -i %i -t %i -n %f -r %i" % (i_eq, i_trial, noise_lvl, n_reals)

                    # If job was not finished let's put it in the joblist of runs to be re-started.

                    if SAVE_UNFINISHED and (not is_finished):
                        unfinished_jobs.append(command)
                        bu.make_jobfile_from_command_list(PATH_UNFINISHED_JOBFILE, unfinished_jobs)

                    if SAVE_UNFINISHED and (not is_finished) and (not equivalence_report["symbolic_solution"]):
                        unfinished_business_jobs.append(command)
                        bu.make_jobfile_from_command_list(PATH_UNFINISHED_BUSINESS_JOBFILE, unfinished_business_jobs)

    else:
        print("Problem excluded.")