import os
import pandas as pd
import time
import argparse
import numpy as np
import sympy
import platform
import torch

# Internal imports
from physo.benchmark.utils import symbolic_utils as su
import physo.benchmark.utils.metrics_utils as metrics_utils
import physo.benchmark.utils.timeout_unix as timeout_unix
from benchmarking import utils as bu
import physo


# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------
parser = argparse.ArgumentParser (description     = "Analyzes MW streams results folder (works on ongoing benchmarks) "
                                                    "and produces .csv files containing results and a summary.",
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


# ------------------------------- PATHS -------------------------------
# Path where to save the results
PATH_RESULTS_SAVE         = os.path.join(RESULTS_PATH, "results_detailed.csv")
PATH_RESULTS_SUMMARY_SAVE = os.path.join(RESULTS_PATH, "results_summary.csv")
# Path where to save jobfile to relaunch unfinished jobs
PATH_UNFINISHED_JOBFILE          = os.path.join(RESULTS_PATH, "jobfile_unfinished")
PATH_UNFINISHED_BUSINESS_JOBFILE = os.path.join(RESULTS_PATH, "jobfile_unfinished_business")
# Path where data is located
PATH_SOURCE_DATA = "streams.csv"

# ------------------------------- TIMEOUT WRAPPER -------------------------------
@timeout_unix.timeout(2) # Max 2s wrapper (works on unix only)
def timed_compare_expr(trial_expr, target_expr):
    return su.compare_expression(
                        trial_expr              = trial_expr,
                        target_expr             = target_expr,
                        handle_trigo            = True,
                        prevent_zero_frac       = True,
                        prevent_inf_equivalence = True,
                        round_decimal           = 2,
                        verbose                 = True, )

def untimed_compare_expr(trial_expr, target_expr):
    return su.compare_expression(
                        trial_expr              = trial_expr,
                        target_expr             = target_expr,
                        handle_trigo            = True,
                        prevent_zero_frac       = True,
                        prevent_inf_equivalence = True,
                        round_decimal           = 1,
                        verbose                 = True, )
def compare_expr(trial_expr, target_expr):
    if platform.system() == "Windows":
        return untimed_compare_expr(trial_expr, target_expr)
    else:
        return timed_compare_expr(trial_expr, target_expr)


# ------------------------------- TARGET EXPRESSION -------------------------------

# Target expression
target_expr_str = np.array([
        # Target functional form : : cste + cste * log (1+r)
        " -2.25433197869809  + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -2.41436719312905  + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -2.2447343815812   + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -1.5859020918779   + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -2.5645735964909   + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -1.62503655452927  + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -2.15162507988352  + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -1.35017082006533  + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -1.34999371983704  + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -0.886739103968145 + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -1.4852048039817   + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -1.93110067461651  + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -2.27868502863944  + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -1.90140528126145  + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -1.79236813095344  + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -2.0266623875962   + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -1.76631299530485  + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -1.62558963183134  + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -2.31576548893306  + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -1.41578671710615  + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -1.17092711945236  + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -1.48716147657053  + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -1.51377591474388  + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -1.9125453150739   + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -1.07822521728563  + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -1.29561794758918  + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -1.56992178190473  + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -2.05315490058432  + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        " -1.68884448806975  + 3.77705922384934 * log( 1.0000075063141 *r + 1.0 ) / r ",
        # Allowing for negative in log as run is done with protected log : cste + cste * log (-(1+r))
        " -2.25433197869809  + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -2.41436719312905  + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -2.2447343815812   + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.5859020918779   + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -2.5645735964909   + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.62503655452927  + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -2.15162507988352  + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.35017082006533  + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.34999371983704  + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -0.886739103968145 + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.4852048039817   + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.93110067461651  + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -2.27868502863944  + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.90140528126145  + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.79236813095344  + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -2.0266623875962   + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.76631299530485  + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.62558963183134  + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -2.31576548893306  + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.41578671710615  + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.17092711945236  + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.48716147657053  + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.51377591474388  + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.9125453150739   + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.07822521728563  + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.29561794758918  + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.56992178190473  + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -2.05315490058432  + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.68884448806975  + 3.77705922384934 * log( - (1.0000075063141 *r + 1.0 ) ) / r ",
        # Equivalent form that sympy is unable to deal with : cste - cste * log (-1/(1+r))
        " -2.25433197869809  - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -2.41436719312905  - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -2.2447343815812   - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.5859020918779   - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -2.5645735964909   - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.62503655452927  - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -2.15162507988352  - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.35017082006533  - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.34999371983704  - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -0.886739103968145 - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.4852048039817   - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.93110067461651  - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -2.27868502863944  - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.90140528126145  - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.79236813095344  - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -2.0266623875962   - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.76631299530485  - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.62558963183134  - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -2.31576548893306  - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.41578671710615  - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.17092711945236  - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.48716147657053  - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.51377591474388  - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.9125453150739   - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.07822521728563  - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.29561794758918  - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.56992178190473  - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -2.05315490058432  - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
        " -1.68884448806975  - 3.77705922384934 * log( - 1 / (1.0000075063141 *r + 1.0 ) ) / r ",
])

r = sympy.Symbol('r', real = True, positive = True)
sympy_X_symbols_dict = {"r": r}

target_expr = np.array([sympy.parse_expr(expr,
                                         local_dict = sympy_X_symbols_dict,
                                         evaluate   = True,
                                         )#.simplify()
# no simplification here as asymmetric treatment of target expr and trial expr can lead to sympy being ineffective
                        for expr in target_expr_str])

# ------------------------------- TARGET DATA -------------------------------

df = pd.read_csv(PATH_SOURCE_DATA)
df["r"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
df["v"] = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2 + df["vz"] ** 2)

stream_ids = np.unique(df["sID"].to_numpy())  # (n_streams,)
stream_dfs = [df[df["sID"] == sID] for sID in stream_ids]  # (n_streams,)
n_streams = len(stream_dfs)

# Dataset
multi_X = []
multi_y = []
for i, df in enumerate(stream_dfs):
    r = (df["r"]/20.0 ).to_numpy()
    v = (df["v"]/200.0).to_numpy()
    E_kin = 0.5 * v**2
    X = np.stack((r, ), axis=0)
    y = E_kin
    multi_X.append(X)
    multi_y.append(y)

n_samples_per_dataset = [X.shape[1] for X in multi_X]

# ------------------------------- RUN FOLDER DETAILS -------------------------------
# Run folders
run_folder_prefix = "StreamsSR_0_"
folders = np.sort(os.listdir(RESULTS_PATH)) # sorting to have the same order on all machines (for reproducibility)

# ------------------------------- ANALYSIS -------------------------------
t00 = time.time()

# Results lines of dict list
run_results = []
# Unfinished jobs list
unfinished_jobs          = []
# Unfinished + target not recovered job list
unfinished_business_jobs = []

for folder in folders:
    # If folder is a run folder
    if folder.startswith(run_folder_prefix):

        t0 = time.time()

        # try:
        run_name = folder[len(run_folder_prefix):]
        i_trial   = int   (run_name.split("_")[0])
        noise     = float (run_name.split("_")[1])
        frac_real = float (run_name.split("_")[2])

        print("--------------------")
        print("Analyzing run :")
        print("i_trial   : %d"%(i_trial))
        print("noise     : %f"%(noise))
        print("frac_real : %f"%(frac_real))

        do_analysis = True

        # Only doing analysis if not already done
        path_result = os.path.join(RESULTS_PATH, folder, "run_result.csv")
        if os.path.exists(path_result) and CONTINUE:
            try:
                run_result = pd.read_csv(path_result).iloc[0].to_dict()
                is_finished   = run_result["is_finished"]
                is_equivalent = run_result["symbolic_solution"]
                print("Results already analyzed.")
                do_analysis = False
            except:
                do_analysis = True

        ## Quick hack to debug
        #if i_trial == 6 and noise == 0.001 and frac_real == 0.25:
        #    do_analysis = True

        if do_analysis:

            run_result = {}

            run_result.update(
                {
                    "i_trial"   : i_trial,
                    "noise"     : noise,
                    "frac_real" : frac_real,
                }
            )

            # --------------- Run log ---------------
            try:
                path_run_log = os.path.join(RESULTS_PATH, folder, "run_curves_data.csv")
                run_log_df = pd.read_csv(path_run_log)

                n_evals     = run_log_df["n_rewarded"].sum()
                is_finished = n_evals >= 240_000 # 250k - batch size

            except:
                n_evals     = 0
                is_finished = False

            run_result.update(
                {
                "n_evals"     : n_evals,
                "is_finished" : is_finished,
                }
            )

            # --------------- Fit quality from logs ---------------

            try:
                path_pareto_csv = os.path.join(RESULTS_PATH, folder, "run_curves_pareto.csv")
                pareto_expressions_df = pd.read_csv(path_pareto_csv)

                r2     = pareto_expressions_df.iloc[-1]["r2"]
                reward = pareto_expressions_df.iloc[-1]["reward"]

            except:
                r2     = 0.
                reward = 0.

            run_result.update(
                {
                    "r2"     : r2,
                    "reward" : reward,
                }
            )

            # --------- Assessing symbolic equivalence ---------
            try:
                # raise(Exception("Not implemented"))
                # Pareto expressions pkl
                path_pareto_pkl = os.path.join(RESULTS_PATH, folder, "run_curves_pareto.pkl")
                pareto_expressions = physo.read_pareto_pkl(path_pareto_pkl)

                # Last expression in pareto front
                # (n_realizations,) size as there is one free const value set per realization
                trial_expr = pareto_expressions[-1].get_infix_sympy(evaluate_consts=True)     # (n_realizations,)

                # Injecting assumptions about r
                trial_expr = [texpr.subs(sympy_X_symbols_dict).simplify() for texpr in trial_expr]

                # todo: whole pareto front ?

                # Comparing any expression found to target expression (with any constants)
                expr = trial_expr[0]
                for it, texpr in enumerate(target_expr):
                    try:
                        expr  = su.clean_sympy_expr(expr,  round_decimal=1)
                        texpr = su.clean_sympy_expr(texpr, round_decimal=1)
                        is_equivalent, report = compare_expr(trial_expr=expr, target_expr=texpr)
                    except:
                        is_equivalent = False
                    if is_equivalent:
                        print("Found equivalent expression, breaking.")
                        break

                save_expr = su.clean_sympy_expr(trial_expr[0], round_decimal=3)
            except:
                is_equivalent = False
                save_expr     = None

            run_result.update(
                {
                    "symbolic_solution": is_equivalent,
                    "expression"       : save_expr

                }
            )

            # --------------- Test fit quality ---------------

            try:
                # Pareto expressions pkl
                path_pareto_pkl = os.path.join(RESULTS_PATH, folder, "run_curves_pareto.pkl")
                pareto_expressions = physo.read_pareto_pkl(path_pareto_pkl)

                # Expression on which to test the fit
                test_expr = pareto_expressions[-1]
                n_spe_free_consts_appearing = np.array([tok.is_spe_free_const for tok in test_expr.tokens]).sum()
                n_free_consts_appearing     = np.array([tok.is_spe_free_const or tok.is_class_free_const for tok in test_expr.tokens]).sum()
                # We have to re-fit the free constants as each run uses a potentially different set of realizations
                # Also depending on the number of realizations used, there might not be enough free dataset specific
                # free constants to fit the expression on all realizations at the same time.
                # Let's do them one by one always using free constants from the 0th realization.

                multi_y_pred = []
                for i_real in range (len(multi_X)):
                    X = torch.tensor(multi_X[i_real])
                    y = torch.tensor(multi_y[i_real])
                    # Fine-tuning spe free constants only
                    # And if they appear in the expression only
                    if n_spe_free_consts_appearing > 0 and n_free_consts_appearing > 0:
                        test_expr.optimize_constants(X, y, i_realization=0, freeze_class_free_consts=True)
                    # If the run was conducted with only 1 realization then we can allow for the class free constants
                    # to be optimized as well as the algo had no knowledge of the disctinction between class and spe
                    # free consts.
                    if frac_real < 1e3 and n_free_consts_appearing > 0:
                        test_expr.optimize_constants(X, y, i_realization=0, freeze_class_free_consts=False)
                    y_pred = test_expr.execute(X, i_realization=0)
                    multi_y_pred.append(y_pred.cpu().detach().numpy())

                # Concatenating all predictions
                multi_y_pred_flatten = np.concatenate(multi_y_pred)
                multi_y_flatten      = np.concatenate(multi_y)
                test_r2 = metrics_utils.r2(y_target=multi_y_flatten, y_pred=multi_y_pred_flatten)

                test_expression_save = su.clean_sympy_expr(test_expr.detach().get_infix_sympy(evaluate_consts=True)[0], round_decimal=3)
                test_expression_save_pre = test_expr.__str__().replace("\n", "")

            except:
                test_r2 = 0.
                test_expression_save     = ""
                test_expression_save_pre = ""

            if np.isnan(test_r2):
                test_r2 = 0.

            run_result.update(
                {
                    "test_r2"             : test_r2,
                    "test_expression"     : test_expression_save,
                    "test_expression_pre" : test_expression_save_pre,
                }
            )

            t1 = time.time()
            run_result.update(
                {
                    "result_analysis_time" : t1 - t0,
                }
            )

        # ----- Saving single run result -----
        run_res_df = pd.DataFrame([run_result,])
        run_res_df.to_csv(path_result, index=False)

        # ----- Results .csv -----
        run_results.append(run_result)
        df = pd.DataFrame(run_results)
        df.to_csv(PATH_RESULTS_SAVE, index=False)

        # ----- Listing unfinished jobs -----

        # If job was not finished let's put it in the joblist of runs to be re-started.

        if SAVE_UNFINISHED and (not is_finished):
            command = "python MW_streams_run.py --trial %i --noise %f --frac_real %f"%(i_trial, noise, frac_real)
            unfinished_jobs.append(command)
            bu.make_jobfile_from_command_list(PATH_UNFINISHED_JOBFILE, unfinished_jobs)

        if SAVE_UNFINISHED and (not is_finished) and (not is_equivalent):
            command = "python MW_streams_run.py --trial %i --noise %f --frac_real %f"%(i_trial, noise, frac_real)
            unfinished_business_jobs.append(command)
            bu.make_jobfile_from_command_list(PATH_UNFINISHED_BUSINESS_JOBFILE, unfinished_business_jobs)


# Saving results one last time with sorted lines
df.sort_values(by=["noise", "frac_real", "i_trial",], inplace=True)
df.to_csv(PATH_RESULTS_SAVE, index=False)

df_grouped = df.groupby(['noise', 'frac_real']).agg({'symbolic_solution' : 'mean',
                                                     'test_r2'           : 'median',
                                                     'n_evals'           : 'mean',
                                                     'is_finished'       : 'all',
                                                        }).reset_index()
# Adding target col
df_grouped["target_formula"] = "E_t +A*R*log((1+(r/R)))/r"
# Re-ordering columns
df_grouped = df_grouped[['noise', 'frac_real', 'target_formula', 'symbolic_solution', 'test_r2', 'n_evals', 'is_finished']]
df_grouped.to_csv(PATH_RESULTS_SUMMARY_SAVE, index=False)

print("--------------------")
print("Total evals:", df["n_evals"].sum())

t01 = time.time()
print("Total time : %f s"%(t01 - t00))

print(None)