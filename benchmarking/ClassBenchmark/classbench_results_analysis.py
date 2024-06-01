import warnings
import numpy as np
import sympy
import pandas as pd
import argparse
import scipy.stats as st
import os
import time
import platform
import torch
import matplotlib.pyplot as plt

# Internal imports
import physo.benchmark.ClassDataset.ClassProblem as ClPb
import physo.benchmark.utils.symbolic_utils as su
import physo.benchmark.utils.metrics_utils as metrics_utils
import physo.benchmark.utils.timeout_unix  as timeout_unix
import physo.benchmark.utils.read_logs     as read_logs
import physo

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
parser.add_argument("-c", "--continue", default = 1,
                    help = "Continues previous analysis run if it exists.")
config = vars(parser.parse_args())

RESULTS_PATH    = str(config["path"])
SAVE_UNFINISHED = bool(int(config["list_unfinished"]))
CONTINUE        = bool(int(config["continue"]))

# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------

ORIGINAL_VAR_NAMES = fconfig.ORIGINAL_VAR_NAMES
EXCLUDED_EQS = fconfig.EXCLUDED_EQS

N_TRIALS       = fconfig.N_TRIALS
NOISE_LEVELS   = fconfig.NOISE_LEVELS
N_REALIZATIONS = fconfig.N_REALIZATIONS

# Batch size
BATCH_SIZE = fconfig.CONFIG["learning_config"]["batch_size"]

# R2 threshold above which an expression is deemed an accuracy solution
R2_ACCURACY_SOLUTION_THRESHOLD = 0.999

# ------------------------------- PATHS -------------------------------
# Where to save raw results of all runs
PATH_RESULTS_SAVE         = os.path.join(RESULTS_PATH, "results_detailed.csv")

# Path where to save jobfile to relaunch unfinished jobs
PATH_UNFINISHED_JOBFILE          = os.path.join(RESULTS_PATH, "jobfile_unfinished")
PATH_UNFINISHED_BUSINESS_JOBFILE = os.path.join(RESULTS_PATH, "jobfile_unfinished_business")


# ------------------------------- TIMEOUT WRAPPER -------------------------------
@timeout_unix.timeout(2) # Max 2s wrapper (works on unix only)
def timed_compare_expr( trial_expr,
                        target_expr,
                        handle_trigo,
                        prevent_zero_frac,
                        prevent_inf_equivalence,
                        round_decimal,
                        verbose,):
    return su.compare_expression( trial_expr              = trial_expr,
                                  target_expr             = target_expr,
                                  handle_trigo            = handle_trigo,
                                  prevent_zero_frac       = prevent_zero_frac,
                                  prevent_inf_equivalence = prevent_inf_equivalence,
                                  round_decimal           = round_decimal,
                                  verbose                 = verbose,)
def untimed_compare_expr( trial_expr,
                        target_expr,
                        handle_trigo,
                        prevent_zero_frac,
                        prevent_inf_equivalence,
                        round_decimal,
                        verbose,):
    return su.compare_expression( trial_expr              = trial_expr,
                                  target_expr             = target_expr,
                                  handle_trigo            = handle_trigo,
                                  prevent_zero_frac       = prevent_zero_frac,
                                  prevent_inf_equivalence = prevent_inf_equivalence,
                                  round_decimal           = round_decimal,
                                  verbose                 = verbose,)
def compare_expr( trial_expr,
                  target_expr,
                  handle_trigo,
                  prevent_zero_frac,
                  prevent_inf_equivalence,
                  round_decimal,
                  verbose,):
    if platform.system() == "Windows":
        return untimed_compare_expr(trial_expr              = trial_expr,
                                    target_expr             = target_expr,
                                    handle_trigo            = handle_trigo,
                                    prevent_zero_frac       = prevent_zero_frac,
                                    prevent_inf_equivalence = prevent_inf_equivalence,
                                    round_decimal           = round_decimal,
                                    verbose                 = verbose,)
    else:
        return timed_compare_expr(trial_expr              = trial_expr,
                                  target_expr             = target_expr,
                                  handle_trigo            = handle_trigo,
                                  prevent_zero_frac       = prevent_zero_frac,
                                  prevent_inf_equivalence = prevent_inf_equivalence,
                                  round_decimal           = round_decimal,
                                  verbose                 = verbose,)

# ------------------------------- MAIN SCRIPT -------------------------------
t00 = time.time()

# List of all lines that will be converted to pd.DataFrame (as df.append is deprecated).
all_results = []

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
        # Iterating through trials
        for i_trial in range (N_TRIALS):
            for noise_lvl in NOISE_LEVELS:
                for n_reals in N_REALIZATIONS:

                    t0 = time.time()

                    print("\n--------------------")
                    print("Analyzing run :")
                    print("i_eq    : %d" % (i_eq))
                    print("i_trial : %d" % (i_trial))
                    print("noise   : %f" % (noise_lvl))
                    print("n_reals : %f" % (n_reals))
                    print("\n")

                    run_name = "CR_%i_%i_%i_%f"%(i_eq, i_trial, n_reals, noise_lvl)
                    path_run = os.path.join(RESULTS_PATH, run_name)

                    do_analysis = True

                    # Only doing analysis if not already done
                    path_result = os.path.join(RESULTS_PATH, run_name, "run_result.csv")
                    if os.path.exists(path_result) and CONTINUE:
                        try:
                            run_result = pd.read_csv(path_result).iloc[0].to_dict()
                            print("Results already analyzed.")
                            do_analysis = False
                        except:
                            do_analysis = True

                    if do_analysis:

                        run_result = {}

                        # Adding problem details
                        run_result.update(
                            {
                                "i_eq"    : i_eq,
                                "i_trial" : i_trial,
                                "noise"   : noise_lvl,
                                "n_reals" : n_reals,
                            }
                        )

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

                        # ----- Most accurate expression found -----

                        # Path to pareto front expressions
                        path_pareto_pkl = os.path.join(path_run, "SR_curves_pareto.pkl")

                        try:
                            pareto_expressions = physo.read_pareto_pkl(path_pareto_pkl)  # (n_reals,)
                            best_expr = pareto_expressions[-1].get_infix_sympy(evaluate_consts=True)[0]
                            best_expr = su.clean_sympy_expr(best_expr)
                        except:
                            best_expr = ""

                        run_result.update(
                            {
                                'last_pareto_expression' : best_expr,
                            }
                        )


                        # ----- Symbolic equivalence related -----

                        # Path to spe free consts values used in the run
                        path_K_vals     = os.path.join(path_run, run_name + "_datagen.csv")
                        # Path to pareto front expressions
                        path_pareto_pkl = os.path.join(path_run, "SR_curves_pareto.pkl")

                        try:
                            #raise Exception("Not implemented.")
                            pareto_expressions = physo.read_pareto_pkl(path_pareto_pkl)                           # (n_reals,)

                            # Iterating through pareto expressions in reverse order
                            for i_pareto in range(len(pareto_expressions)-1, -1, -1):
                                # Getting trial expression
                                # (n_reals,) size as there is one free const value set per realization
                                trial_expr = pareto_expressions[i_pareto].get_infix_sympy(evaluate_consts=True)       # (n_reals,)
                                # Injecting assumptions about input variables
                                trial_expr = [texpr.subs(pb.sympy_X_symbols_dict).simplify() for texpr in trial_expr] # (n_reals,)

                                # Getting target spe free consts values
                                K_vals_df = pd.read_csv(path_K_vals, sep=";")
                                K_vals_df = K_vals_df.drop(columns=["i_real"])
                                K_vals    = K_vals_df.to_numpy().astype(float)    # (n_reals,)
                                # Getting target expression
                                target_expr = pb.get_sympy(K_vals=K_vals)

                                # Comparing on realization 0 only
                                target_expr = target_expr[0]
                                trial_expr  = trial_expr [0]

                                # Getting symbolic equivalence report
                                target_expr = su.clean_sympy_expr(target_expr,)
                                trial_expr  = su.clean_sympy_expr(trial_expr ,)
                                is_equivalent, equivalence_report = compare_expr (
                                                                               trial_expr  = trial_expr,
                                                                               target_expr = target_expr,
                                                                               handle_trigo            = True,
                                                                               prevent_zero_frac       = True,
                                                                               prevent_inf_equivalence = True,
                                                                               verbose                 = True,
                                                                               round_decimal = 2,
                                                                               )
                                equivalent_details = {
                                    'equivalent_expr' : trial_expr,
                                    'i_pareto       ' : i_pareto,
                                                     }
                                if is_equivalent:
                                    print("Found equivalent expression, breaking.")
                                    break
                                else:
                                    equivalent_details.update({'equivalent_expr' : ""})

                        except:
                            # Negative report
                            is_equivalent = False
                            equivalent_details = {
                                'equivalent_expr' : "",
                                'i_pareto       ' : -1,
                                                 }
                            equivalence_report = {
                                'symbolic_error'                : '',
                                'symbolic_fraction'             : '',
                                'symbolic_error_is_zero'        : None,
                                'symbolic_error_is_constant'    : None,
                                'symbolic_fraction_is_constant' : None,
                                'sympy_exception'               : "Unknown error",
                                'symbolic_solution'             : False,
                            }

                        run_result.update(equivalence_report)
                        run_result.update(equivalent_details)

                        # ----- Fit quality related -----

                        # Path to spe free consts values used in the run
                        path_K_vals     = os.path.join(path_run, run_name + "_datagen.csv")
                        path_pareto_pkl = os.path.join(path_run, "SR_curves_pareto.pkl")

                        try:
                            pareto_expressions = physo.read_pareto_pkl(path_pareto_pkl)                           # (n_reals,)

                            # Getting X data (generate_data_points will return y as well from random Ks so we need
                            # to drop it)
                            multi_X, _ = pb.generate_data_points(n_samples = 10_000, n_realizations=n_reals)

                            # Getting predictions
                            # Last expression in pareto front
                            i_pareto = -1
                            trial_expr = pareto_expressions[i_pareto]
                            multi_y_pred = []
                            for i_real in range(n_reals):
                                X = torch.tensor(multi_X[i_real])
                                y_pred = trial_expr.execute(X=X, i_realization=i_real)
                                y_pred = y_pred.detach().numpy()
                                multi_y_pred.append(y_pred)
                            multi_y_pred = np.array(multi_y_pred)

                            # Getting target
                            K_vals_df = pd.read_csv(path_K_vals, sep=";")
                            K_vals_df = K_vals_df.drop(columns=["i_real"])
                            K_vals    = K_vals_df.to_numpy().astype(float)    # (n_reals,)
                            multi_y_target = []
                            for i_real in range(n_reals):
                                y_target = pb.target_function(X=multi_X[i_real], K=K_vals[i_real])
                                multi_y_target.append(y_target)
                            multi_y_target = np.array(multi_y_target)

                            # R2
                            multi_y_target_flatten = np.concatenate(multi_y_target)
                            multi_y_pred_flatten   = np.concatenate(multi_y_pred)
                            test_r2 = metrics_utils.r2(y_target=multi_y_target_flatten, y_pred=multi_y_pred_flatten)
                            is_accuracy_solution = test_r2 > R2_ACCURACY_SOLUTION_THRESHOLD

                            # fig, ax = plt.subplots(1,1, figsize=(10,8))
                            # for i_real in range(n_reals):
                            #     ax.plot(multi_X[i_real][0], multi_y_pred[i_real], 'b.')
                            #     ax.plot(multi_X[i_real][0], multi_y_target[i_real], 'r.')
                            # plt.show()
                            # print(None)

                        except:
                            test_r2 = 0.
                            is_accuracy_solution = False

                        fit_quality = {
                                'test_r2': test_r2,
                                'accuracy_solution': is_accuracy_solution,
                                      }
                        run_result.update(fit_quality)

                        # ----- Listing unfinished jobs -----
                        command = "python classbench_run.py -i %i -t %i -n %f -r %i" % (i_eq, i_trial, noise_lvl, n_reals)

                        run_command = {
                            'RUN COMMAND' : command,
                            'RUN NAME'    : run_name,
                            }
                        run_result.update(run_command)

                        # ----- Timing -----
                        t1 = time.time()
                        run_result.update(
                            {
                                "result_analysis_time": t1 - t0,
                            }
                        )

                    # If job was not finished let's put it in the joblist of runs to be re-started.
                    is_finished   = run_result["FINISHED"]
                    command       = run_result["RUN COMMAND"]
                    is_equivalent = run_result["symbolic_solution"]

                    if SAVE_UNFINISHED and (not is_finished):
                        unfinished_jobs.append(command)
                        bu.make_jobfile_from_command_list(PATH_UNFINISHED_JOBFILE, unfinished_jobs)

                    if SAVE_UNFINISHED and (not is_finished) and (not is_equivalent):
                        unfinished_business_jobs.append(command)
                        bu.make_jobfile_from_command_list(PATH_UNFINISHED_BUSINESS_JOBFILE, unfinished_business_jobs)

                    # ----- Saving single run result -----
                    run_res_df = pd.DataFrame([run_result, ])
                    run_res_df.to_csv(path_result, index=False)

                    # ----- Results .csv -----
                    all_results.append(run_result)
                    df = pd.DataFrame(all_results)
                    df.to_csv(PATH_RESULTS_SAVE, index=False)

    else:
        print("Problem excluded.")


print("--------------------")
print("Total evals:", df["# EVALUATIONS"].sum())

t01 = time.time()
print("Total time : %f s"%(t01 - t00))