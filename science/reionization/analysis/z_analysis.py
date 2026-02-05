import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import torch
import sklearn.metrics as metrics
import os
import sympy
import time

import physo
import physo.benchmark.utils.metrics_utils as metrics_utils
import physo.benchmark.utils as benchmark_utils
import physo.benchmark.utils.symbolic_utils as su

RUNS_PATH = "/Users/wtenachi/Documents/ASTRO_research/projects/reionization-sr/season5/run-results/Z_SR-RUNS-APLHA/"
PATH_DATA = "/Users/wtenachi/Documents/ASTRO_research/projects/reionization-sr/season4/data/"
ZTASK = 'z_asy' #z_asy, z_dur or z_mid

print('Analysis of all SR runs for task:', ZTASK)

# region # ------- COLLECTING ALL EXPRESSIONS ACROSS RUNS ------- #
# list folders starting with task name
run_folders = [f for f in os.listdir(RUNS_PATH) if f.startswith("ZREION_SR_%s"%(ZTASK))] # z_reion specific

# Collecting all expressions and vars used in a df
all_exprs_df = pd.DataFrame(columns=["expression", "vars_used",  "run_name"])

for run_folder in run_folders:
    run_path = os.path.join(RUNS_PATH, run_folder) # my_path/run_folder
    # Paths to results
    result_pkl     = os.path.join(run_path, "sr_curves_pareto.pkl")
    data_used_csv  = os.path.join(run_path, f"{run_folder}_data.csv")
    run_curves_csv = os.path.join(run_path, "sr_curves_data.csv")
    # Resulting expressions and vars used
    try:
        exprs         = physo.read_pareto_pkl(result_pkl)
        run_curves_df = pd.read_csv(run_curves_csv)
    except Exception as e:
        print(f"Could not read expressions from {result_pkl} due to: {str(e)}")
        continue
    vars_used  = pd.read_csv(data_used_csv, sep=';').columns.tolist()[2:]  # skip index and y cols
    # Append to df
    for expr in exprs:
        all_exprs_df = all_exprs_df._append({
            "expression": expr,
            "vars_used" : vars_used,
            "run_name"  : run_folder,
            "n_evals"   : run_curves_df['n_rewarded'].sum(),
        }, ignore_index=True)


print(None)

# endregion

# region # ------- EVALUATING ALL EXPRESSIONS ------- #

# Load training and test data
path_data_train = os.path.join(PATH_DATA, "params_training_data.csv") # z_reion specific
path_data_test  = os.path.join(PATH_DATA, "params_test_data.csv")     # z_reion specific
df_train = pd.read_csv(path_data_train)
df_test  = pd.read_csv(path_data_test)

t00 = time.perf_counter()

path_all_exprs_pkl = os.path.join(RUNS_PATH, "all_exprs_eval_%s.pkl"%(ZTASK)) # z_reion specific
path_all_exprs_csv = os.path.join(RUNS_PATH, "all_exprs_eval_%s.csv"%(ZTASK)) # z_reion specific

# If .pkl exists load it
found_all_exprs = False
if os.path.exists(path_all_exprs_pkl):
    all_exprs_df = pd.read_pickle(path_all_exprs_pkl)
    found_all_exprs = True


if not found_all_exprs:
    for index, row in all_exprs_df.iterrows():
        print('Evaluating expression %d / %d' % (index+1, len(all_exprs_df)))
        expr      = row["expression"]
        vars_used = row["vars_used"]

        df_dict = {"train": df_train, "test": df_test}

        # -- Logging complexity metrics --

        # Length
        try: # Sympy way
            @benchmark_utils.timeout_generic.timeout(1)
            def get_length(expr):
                sympy_expr = sympy.sympify(expr.get_infix_sympy(evaluate_consts=True)[0])
                return benchmark_utils.symbolic_utils.expression_size(sympy_expr)
            length = get_length(expr)
        except Exception as e: # Fallback way
            print('Could not compute length via sympy for expr index %d due to: %s' % (index, str(e)))
            length = expr.size

        all_exprs_df.at[index, "length"] = length

        # Number of free parameters
        try: # Sympy way
            @benchmark_utils.timeout_generic.timeout(1)
            def get_n_free_params(expr):
                """Count number of free parameters in expression."""
                sympy_expr = sympy.sympify(expr.get_infix_sympy(evaluate_consts=True)[0])
                return  benchmark_utils.symbolic_utils.expression_n_floats(sympy_expr)
            n_free_params = get_n_free_params(expr)
        except Exception as e: # Fallback way
            print('Could not compute n_free_params via sympy for expr index %d due to: %s' % (index, str(e)))
            n_free_params = int((expr.free_consts.class_values != 1.).sum()) # counting free params optimized away from default 1.0

        all_exprs_df.at[index, "n_free_params"] = n_free_params

        # -- Logging accuracy metrics --
        for mode in ["train", "test"]:
            df = df_dict[mode]

            X = df[vars_used].to_numpy()                                  # (n_samples, n_dim)
            y_target = df[ZTASK].to_numpy()                               # (n_samples,)       # z_reion specific
            y_pred   = expr(torch.tensor(X.T)).cpu().detach().numpy()     # (n_samples,)

            # R2
            r2 = metrics_utils.r2(y_target=y_target, y_pred=y_pred)
            all_exprs_df.at[index, f"R2_{mode}"] = r2

            # MAE
            MAE = np.mean(np.abs(y_pred - y_target))
            all_exprs_df.at[index, f"MAE_{mode}"] = MAE
else:
    print("Loaded pre-evaluated expressions from pickle.")

# Change dtype to int for length and n_free_params
all_exprs_df["length"]        = all_exprs_df["length"]       .astype(int)
all_exprs_df["n_free_params"] = all_exprs_df["n_free_params"].astype(int)
all_exprs_df["n_evals"]       = all_exprs_df["n_evals"]      .astype(int)

# Save evaluated expressions to .pkl and csv
all_exprs_df.to_pickle(path_all_exprs_pkl)
all_exprs_df.to_csv   (path_all_exprs_csv, index=False)
print('Saved evaluated expressions to:', path_all_exprs_csv)

t11 = time.perf_counter()
print(f"Evaluated all expressions in {t11 - t00:0.4f} seconds")



# endregion

# region # ------- BEST EXPRESSION PRINT ------- #

print("Best MAE test expression:")
print(all_exprs_df.loc[all_exprs_df["MAE_test"].idxmin()])

# Show residuals plot for best expression
best_expr_row = all_exprs_df.loc[all_exprs_df["MAE_test"].idxmin()]
best_expr      = best_expr_row["expression"]
best_vars_used = best_expr_row["vars_used"]
df_test = pd.read_csv(path_data_test)
X_test      = df_test[best_vars_used].to_numpy()                           # (n_samples, n_dim)
y_target    = df_test[ZTASK].to_numpy()                                    # (n_samples,)       # z_reion specific
y_pred_test = best_expr(torch.tensor(X_test.T)).cpu().detach().numpy()     # (n_samples,)
residuals = y_target - y_pred_test

plt.figure(figsize=(7,5))
plt.scatter(y_target, residuals, alpha=0.6, edgecolor="none")
plt.axhline(0, color="r", linestyle="--", linewidth=1)
plt.xlabel("True")
plt.ylabel("Residual (Pred − True)")
plt.tight_layout()
plt.show()

# endregion

# region # ------- PARETO FRONT ------- #

# Based on test mae vs. length
objective1 = "length"
objective2 = "MAE_test"
pareto_df_mae_length = all_exprs_df.iloc[metrics_utils.get_pareto_front(all_exprs_df[objective1].to_numpy(),
                                                                        all_exprs_df[objective2].to_numpy())
                                        ].sort_values(by=objective2, ascending=False).reset_index(drop=True)

# Based on test mae vs. n_free_params
objective1 = "n_free_params"
objective2 = "MAE_test"
pareto_df_mae_fp = all_exprs_df.iloc[metrics_utils.get_pareto_front(all_exprs_df[objective1].to_numpy(),
                                                                    all_exprs_df[objective2].to_numpy())
                                        ].sort_values(by=objective2, ascending=False).reset_index(drop=True)
# Only keep expr with >= 2 free params
pareto_df_mae_fp = pareto_df_mae_fp[pareto_df_mae_fp["n_free_params"] >= 2].reset_index(drop=True)

# endregion

# region # ------- PARETO FRONT PLOT ------- #

# fig, ax = plt.subplots(figsize=(7,5))
# ax.plot(pareto_df_mae_length["length"], pareto_df_mae_length["MAE_test"], 'r-', alpha=1.)
# plt.xlabel("Length")
# plt.ylabel("MAE (test)")
# plt.show()
#
# fig, ax = plt.subplots(figsize=(7,5))
# ax.plot(pareto_df_mae_fp["n_free_params"], pareto_df_mae_fp["MAE_test"], 'r-', alpha=1.)
# plt.xlabel("Number of free parameters")
# plt.ylabel("MAE (test)")
# plt.show()

# endregion

# region # ------- COMPARISON WITH BASELINE ------- #

astro_cols = ["OMm","OMb","h","sigma_8","n_s", "F_STAR10","F_ESC10","ALPHA_STAR","ALPHA_ESC", "M_TURN","L_X","t_STAR","R_BUBBLE_MAX"]

# Functions
def make_callable_equation(eq_str: str, all_var_names: list[str]):
    """
    Returns a callable that evaluates eq_str, auto-detects used vars,
    and attaches metadata so it works when passed ALL_VARS.
    """
    expr = eq_str.replace('^', '**').replace('atan', 'arctan')
    # Detect used variables via regex word boundaries
    used_vars = [v for v in all_var_names
                 if re.search(rf"\b{re.escape(v)}\b", expr)]
    # Safe namespace
    safe_dict = {
        'exp': np.exp,
        'log': np.log,
        'sin': np.sin,
        'tanh': np.tanh,
        'arctan': np.arctan,
        'abs': np.abs,
        'sinh': np.sinh,
        'cosh': np.cosh,
        'safe_sin': np.sin,
        'safe_log': np.log,
        'sqrt':np.sqrt
    }

    def f(args):
        # Expect args as full-length array matching ALL_VARS
        vals = np.asarray(args)
        if vals.ndim != 1:
            raise ValueError('Input must be 1D array of values')
        # Map var->value for used_vars only
        local_vars = {v: vals[all_var_names.index(v)] for v in used_vars}
        return eval(expr, safe_dict, local_vars)
    # Attach metadata
    f.all_var_names = all_var_names
    f.used_vars = used_vars
    return f

baseline_str_dict = {
    "z_asy":"(2.4992032 + ((((exp((F_STAR10 / (((t_STAR * exp(tanh(ALPHA_ESC) / ((F_ESC10 * tanh(sigma_8 + -1.4415569)) / safe_sin(M_TURN)))) ^ -0.40509987) + ALPHA_STAR)) - (((safe_sin(L_X / safe_log(R_BUBBLE_MAX)) - (tanh((safe_sin((R_BUBBLE_MAX - OMb) / -0.37114137) - (((safe_sin(R_BUBBLE_MAX / -0.37114137) / 1.8254017) - safe_sin(R_BUBBLE_MAX + R_BUBBLE_MAX)) * (F_ESC10 - safe_sin(R_BUBBLE_MAX / -0.8155561)))) * (F_STAR10 - safe_sin(R_BUBBLE_MAX * 2.7630098))) * safe_sin(L_X / (R_BUBBLE_MAX * -0.3563518)))) / L_X) * R_BUBBLE_MAX)) + (((((L_X * 0.041705806) - ALPHA_STAR) - tanh(tanh(ALPHA_ESC))) * (safe_sin(OMm + ((ALPHA_ESC * ((n_s * F_STAR10) + (F_ESC10 - -2.155896))) / (ALPHA_STAR / OMb))) * R_BUBBLE_MAX)) - F_STAR10)) / (M_TURN + -5.360221)) - h) + ((F_ESC10 * F_STAR10) * safe_sin(exp((h * (safe_sin(L_X / 1.7154312) - safe_log(ALPHA_STAR))) - tanh(ALPHA_ESC / (0.36466765 / safe_sin(M_TURN)))))))) * ((n_s + -0.2545741) * h)",
    "z_dur":"((((((OMm + h) * safe_sin(((safe_sin(ALPHA_ESC * (((h * (h + OMm)) + OMm) * F_STAR10)) * 0.2653357) / (tanh(1.253315 - safe_sin(M_TURN)) * exp(F_ESC10))) - ALPHA_STAR)) + (exp(safe_sin(M_TURN - (((F_STAR10 * 0.53047323) - ALPHA_STAR) * (ALPHA_ESC - (-0.66976476 * ((OMb ^ -1.2197223) ^ F_ESC10)))))) * tanh(((0.21202238 - OMb) - ((h * (((L_X + (safe_log(R_BUBBLE_MAX) * tanh(0.9052894 - (sigma_8 / (0.9052894 + safe_sin(-0.66976476 - R_BUBBLE_MAX)))))) ^ -0.338148) * F_STAR10)) * (ALPHA_STAR ^ F_ESC10))) * sigma_8))) + ((1.6497698 / (1.3236942 - ((((safe_sin(n_s - (M_TURN * tanh(safe_sin((L_X ^ F_ESC10) + (ALPHA_ESC - ((ALPHA_STAR * ((n_s + -0.27127248) ^ F_STAR10)) * F_ESC10)))))) / (L_X - R_BUBBLE_MAX)) / (1.1171885 - safe_sin(M_TURN))) + ((R_BUBBLE_MAX - (safe_sin(ALPHA_ESC * ((F_STAR10 + F_STAR10) * F_STAR10)) - h)) ^ 0.47583225)) ^ -0.48693913))) ^ 1.759602)) ^ n_s) + (-0.43240064 - OMb)) * (sigma_8 + safe_sin(OMm))",
    "z_mid":"(((safe_sin(exp((safe_sin(ALPHA_STAR + ALPHA_ESC) * (((R_BUBBLE_MAX - ((((OMb ^ ALPHA_ESC) / safe_sin(F_ESC10)) - (n_s / F_STAR10)) / (ALPHA_STAR * ALPHA_STAR))) * 0.040217478) ^ 0.28525034)) / (-1.085362 / (F_ESC10 - (F_STAR10 / -1.289086))))) * ((((((L_X - safe_sin((M_TURN / (F_ESC10 - safe_sin(F_STAR10 / -1.1875426))) * (safe_sin(n_s) / OMm))) * 0.14962274) - M_TURN) / (R_BUBBLE_MAX - (ALPHA_ESC / (ALPHA_ESC - tanh(F_STAR10))))) - ((-0.3020872 + OMb) / exp(exp(safe_sin(F_ESC10 + (1.7674565 - (((0.28954217 * ALPHA_STAR) ^ ALPHA_ESC) / F_ESC10))) + (safe_sin(M_TURN / -0.39442402) - (-0.8451099 / h)))))) - OMb)) + (sigma_8 * (((F_STAR10 + ((h - (((M_TURN - ((((h + sigma_8) - -1.6948011) + ((((0.0685727 ^ tanh(ALPHA_ESC)) * safe_log(ALPHA_STAR)) * safe_sin(M_TURN - (-0.2677172 * (F_ESC10 * F_STAR10)))) / F_STAR10)) * OMm)) + -2.9401636) - F_ESC10)) + M_TURN)) * 2.0168314) ^ n_s))) / (OMb - -0.4522517)) + (F_ESC10 * (0.927484 ^ L_X))"
}
baseline_str = baseline_str_dict[ZTASK] # z_reion specific

# ---- Metrics : accuracy ----
baseline_func = make_callable_equation(baseline_str, astro_cols)

X = df_test[astro_cols].to_numpy()                           # (n_samples, n_dim)
y_target    = df_test[ZTASK].to_numpy()                      # (n_samples,)       # z_reion specific
y_pred      = np.array([baseline_func(row) for row in X])    # (n_samples,)

baseline_mae_test = np.mean(np.abs(y_pred - y_target))
print("Baseline MAE:", baseline_mae_test)
baseline_r2_test  = metrics_utils.r2(y_target=y_target, y_pred=y_pred)

# ---- Metrics : complexity ----

# Replace ^ with ** for sympy compatibility and remove safe_ prefixes
baseline_str_sympy = baseline_str.replace('^', '**').replace('safe_log', 'log').replace('safe_sin', 'sin')
baseline_sympy = sympy.parse_expr(baseline_str_sympy)

# Number of nodes
length        = physo.benchmark.utils.symbolic_utils.expression_size     (baseline_sympy)
n_free_params = physo.benchmark.utils.symbolic_utils.expression_n_floats (baseline_sympy)

# ---- Baseline in dfs ----
baseline_row = {
    "expression": baseline_sympy,
    "vars_used": astro_cols,
    "run_name": "baseline",
    "length": length,
    "n_free_params": n_free_params,
    "MAE_test": baseline_mae_test,
    "R2_test": baseline_r2_test
}

pareto_df_mae_length_w_baseline = pareto_df_mae_length._append(baseline_row, ignore_index=True
                                ).sort_values(by="MAE_test", ascending=False).reset_index(drop=True)
pareto_df_mae_length_w_baseline.to_csv(RUNS_PATH+"pareto_mae_vs_length_%s.csv"%(ZTASK), index=False)

pareto_df_mae_fp_w_baseline = pareto_df_mae_fp._append(baseline_row, ignore_index=True
                                ).sort_values(by="MAE_test", ascending=False).reset_index(drop=True)
pareto_df_mae_fp_w_baseline.to_csv(RUNS_PATH+"pareto_mae_vs_n_free_params_%s.csv"%(ZTASK), index=False)

fig, ax = plt.subplots(figsize=(7,5))
ax.set_title(ZTASK)
ax.plot(pareto_df_mae_length_w_baseline["length"], pareto_df_mae_length_w_baseline["MAE_test"], 'b--', alpha=1.)
ax.plot(pareto_df_mae_length["length"], pareto_df_mae_length["MAE_test"], 'r-', alpha=1.)
plt.xlabel("Length")
plt.ylabel("MAE (test)")
plt.savefig(RUNS_PATH+"pareto_mae_vs_length_%s.png"%(ZTASK))
plt.show()

fig, ax = plt.subplots(figsize=(7,5))
ax.set_title(ZTASK)
ax.plot(pareto_df_mae_fp_w_baseline["n_free_params"], pareto_df_mae_fp_w_baseline["MAE_test"], 'b--', alpha=1.)
ax.plot(pareto_df_mae_fp["n_free_params"], pareto_df_mae_fp["MAE_test"], 'r-', alpha=1.)
plt.xlabel("Number of free parameters")
plt.ylabel("MAE (test)")
plt.savefig(RUNS_PATH+"pareto_mae_vs_n_free_params_%s.png"%(ZTASK))
plt.show()

# IN S5 results:
for i_expr in range (len(pareto_df_mae_length)):
    df_line = pareto_df_mae_length.iloc[i_expr]
    nparams = df_line["n_free_params"]
    acc     = df_line["MAE_test"]
    sympy_expr = df_line['expression'].get_infix_sympy(evaluate_consts=True)[0].simplify()
    sympy_expr = su.clean_sympy_expr(sympy_expr, round_decimal = 3)
    print("\n--------------------------")
    print(f"Expression index {i_expr} | n_free_params = {nparams}")
    print(sympy.pretty(sympy_expr))
    print(f"MAE (test) = {acc:0.6f}")


# endregion
print(None)
