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

RUNS_PATH = "/Users/wtenachi/Documents/ASTRO_research/projects/reionization-sr/season4/run-results/TAU_SR-RUNS-APLHA/"
PATH_DATA = "/Users/wtenachi/Documents/ASTRO_research/projects/reionization-sr/season4/data/"


# region # ------- COLLECTING ALL EXPRESSIONS ACROSS RUNS ------- #

# list folders starting with "TAU_SR" # Tau specific
run_folders = [f for f in os.listdir(RUNS_PATH) if f.startswith("TAU_SR")]

# Collecting all expressions and vars used in a df
all_exprs_df = pd.DataFrame(columns=["expression", "vars_used",  "run_name"])

for run_folder in run_folders:
    run_path = os.path.join(RUNS_PATH, run_folder) # my_path/run_folder
    # Paths to results
    result_pkl    = os.path.join(run_path, "sr_curves_pareto.pkl")
    data_used_csv = os.path.join(run_path, f"{run_folder}_data.csv")
    run_curves_csv = os.path.join(run_path, "sr_curves_data.csv")
    # Resulting expressions and vars used
    exprs      = physo.read_pareto_pkl(result_pkl)
    run_curves_df = pd.read_csv(run_curves_csv)
    vars_used  = pd.read_csv(data_used_csv, sep=';').columns.tolist()[2:]  # skip index and y cols
    # Append to df
    for expr in exprs:
        all_exprs_df = all_exprs_df._append({
            "expression": expr,
            "vars_used": vars_used,
            "run_name": run_folder,
            "n_evals": run_curves_df['n_rewarded'].sum(),
        }, ignore_index=True)

# endregion

# region # ------- EVALUATING ALL EXPRESSIONS ------- #

# Load training and test data
path_data_train = os.path.join(PATH_DATA, "tau_training_data.csv") # Tau specific
path_data_test  = os.path.join(PATH_DATA, "tau_test_data.csv")     # Tau specific
df_train = pd.read_csv(path_data_train)
df_test  = pd.read_csv(path_data_test)

t00 = time.perf_counter()

path_all_exprs_pkl = os.path.join(RUNS_PATH, "all_exprs_eval.pkl")
path_all_exprs_csv = os.path.join(RUNS_PATH, "all_exprs_eval.csv")

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
            y_target = df["tau"].to_numpy()                               # (n_samples,)
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
y_target    = df_test["tau"].to_numpy()                                    # (n_samples,)
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

# region # ------- NN BASELINE ------- #
# Train a simple NN as baseline

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Preparing data
astro_cols = ["OMm", "OMb", "h", "sigma_8", "n_s", "F_STAR10","F_ESC10","ALPHA_STAR","ALPHA_ESC","M_TURN","L_X","t_STAR","R_BUBBLE_MAX"]
X_train = df_train[astro_cols].to_numpy()   # (n_samples, n_dim)
y_train = df_train["tau"].to_numpy()        # (n_samples,)
X_test  = df_test[astro_cols].to_numpy()    # (n_samples, n_dim)
y_test  = df_test["tau"].to_numpy()         # (n_samples,)

# Scaling
scalerX, scalery = StandardScaler(), StandardScaler()
X_train_scaled = scalerX.fit_transform(X_train)
X_test_scaled = scalerX.transform(X_test)
y_train_scaled = scalery.fit_transform(y_train.reshape(-1,1)).ravel()

# NN model
nn_model = MLPRegressor(hidden_layer_sizes=(4,4),
                        activation='tanh',
                        solver='adam',
                        alpha=1e-3,
                        learning_rate_init=1e-3,
                        max_iter=20000,
                        early_stopping=True,        # <-- enables validation
                        validation_fraction=0.1,    # <-- 10% of training data used as validation
                        n_iter_no_change=50,        # <-- patience
                        #tol=1e-8,                  # <-- stop criterion
                        random_state=0)

# Training
print("Training NN baseline...")
t0 = time.perf_counter()
nn_model.fit(X_train_scaled, y_train_scaled)
t1 = time.perf_counter()
print(f"Trained NN baseline in {t1 - t0:0.4f} seconds")
n_params = sum(coef.size for coef in nn_model.coefs_) + sum(intercept.size for intercept in nn_model.intercepts_)
print(f"NN Baseline number of parameters: {n_params}")

# Plotting training loss curve
plt.figure(figsize=(6,4))
plt.plot(nn_model.loss_curve_, lw=2)
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Training Loss", fontsize=12)
plt.title("MLPRegressor Training Loss Curve", fontsize=14)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

# Predicting
y_pred_nn = scalery.inverse_transform(nn_model.predict(X_test_scaled).reshape(-1,1)).ravel()

# Metrics
nn_mae_test = np.mean(np.abs(y_pred_nn - y_test))
nn_r2_test  = metrics.r2_score(y_test, y_pred_nn)

print(f"NN Baseline MAE (test): {nn_mae_test:0.6f}")
print(f"NN Baseline R2  (test): {nn_r2_test:0.6f}")






# endregion

# region # ------- COMPARISON WITH BASELINE ------- #

astro_cols = ["OMm", "OMb", "h", "sigma_8", "n_s", "F_STAR10","F_ESC10","ALPHA_STAR","ALPHA_ESC","M_TURN","L_X","t_STAR","R_BUBBLE_MAX"]

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

baseline_str = "((((-0.059858155 / ((R_BUBBLE_MAX - safe_log((t_STAR + safe_sin(L_X)) + 0.31493956)) - (safe_sin(M_TURN / exp(-1.2960565 - tanh(F_ESC10))) / exp((((R_BUBBLE_MAX - exp(safe_sin(((0.26600757 / (OMb * F_ESC10)) + (M_TURN ^ 1.5187011)) + M_TURN))) - exp(safe_sin(F_STAR10 + ((F_ESC10 + (M_TURN ^ 0.90847105)) - (R_BUBBLE_MAX * OMm))) - safe_log(h))) ^ 1.2344037) * (F_ESC10 * OMb))))) - -0.018643435) + ((OMb * (((n_s * h) * (sigma_8 * 1.4563627)) + -0.35222226)) * exp(n_s))) * ((exp(F_STAR10) + exp(F_ESC10)) + tanh(exp((F_STAR10 + tanh(((ALPHA_STAR - OMm) + safe_sin(ALPHA_ESC)) * ((((safe_sin(M_TURN) + ((R_BUBBLE_MAX ^ (n_s + -0.91018784)) / (R_BUBBLE_MAX ^ ALPHA_STAR))) * -1.8453525) - ((n_s - 1.9665523) * (F_STAR10 * F_ESC10))) + (((OMm * ALPHA_ESC) / -0.25400874) * (F_STAR10 + (F_ESC10 + 0.7730384)))))) + tanh(safe_log(safe_sin(M_TURN))))))) + -0.0043368875"

# ---- Metrics : accuracy ----
baseline_func = make_callable_equation(baseline_str, astro_cols)

X = df_test[astro_cols].to_numpy()                           # (n_samples, n_dim)
y_target    = df_test["tau"].to_numpy()                      # (n_samples,)
y_pred      = np.array([baseline_func(row) for row in X])    # (n_samples,)

baseline_mae_test = np.mean(np.abs(y_pred - y_target))
baseline_r2_test  = metrics_utils.r2(y_target=y_target, y_pred=y_pred)

print(f"Baseline MAE (test): {baseline_mae_test:0.6f}")
print(f"Baseline R2  (test): {baseline_r2_test:0.6f}")
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
pareto_df_mae_length_w_baseline.to_csv(RUNS_PATH+"pareto_mae_vs_length.csv", index=False)

pareto_df_mae_fp_w_baseline = pareto_df_mae_fp._append(baseline_row, ignore_index=True
                                ).sort_values(by="MAE_test", ascending=False).reset_index(drop=True)
pareto_df_mae_fp_w_baseline.to_csv(RUNS_PATH+"pareto_mae_vs_n_free_params.csv", index=False)

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(pareto_df_mae_length_w_baseline["length"], pareto_df_mae_length_w_baseline["MAE_test"], 'b--', alpha=1.)
ax.plot(pareto_df_mae_length["length"], pareto_df_mae_length["MAE_test"], 'r-', alpha=1.)
# NN baseline
ax.axhline(y=nn_mae_test, color='g', linestyle='dashdot', label='NN Baseline MAE')
# log scale for y and x
# ax.set_xscale('log')
# ax.set_yscale('log')
plt.xlabel("Length")
plt.ylabel("MAE (test)")
plt.savefig(RUNS_PATH+"pareto_mae_vs_length.png")
plt.show()

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(pareto_df_mae_fp_w_baseline["n_free_params"], pareto_df_mae_fp_w_baseline["MAE_test"], 'b--', alpha=1.)
ax.plot(pareto_df_mae_fp["n_free_params"], pareto_df_mae_fp["MAE_test"], 'r-', alpha=1.)
# NN baseline
ax.axhline(y=nn_mae_test, color='g', linestyle='dashdot', label='NN Baseline MAE')
# log scale for y and x
# ax.set_xscale('log')
# ax.set_yscale('log')
plt.xlabel("Number of free parameters")
plt.ylabel("MAE (test)")
plt.savefig(RUNS_PATH+"pareto_mae_vs_n_free_params.png")
plt.show()

# Nice optimum at mae = 0.001796
# print(sympy.pretty(pareto_df_mae_length.iloc[11]['expression'].get_infix_sympy(evaluate_consts=True)[0].simplify()))

#

# endregion
print(None)
