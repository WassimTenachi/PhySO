# External packages
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import physo
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import torch
import argparse

# Internal code import
import physo.physym.free_const as free_const
import physo
import physo.learn.monitoring as monitoring
import physo.benchmark.utils.symbolic_utils as su

# Local imports
from science.galaxy_profile import analytic_properties as ap


# Seed
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)


# META PARAMETERS
BARY = "dmo"
FRAC_REALS = 0.1  # Fraction of realizations to use
N_CLASS_FREE_PARAMS = 1
N_SPE_FREE_PARAMS   = 3  # Number of specialized free parameters (eg. Rs, rho0, etc.)


# -------- DATASET ---------
RUN_NAME = "nihaoap_class_sr_v1"

# PATHS
# Defining source data abs path before changing directory
DATA_PATH     = os.path.join(os.path.abspath(''), "NIHAO_data/%s_profiles/"%(BARY))
METADATA_PATH = os.path.join(DATA_PATH, "MvirRvir.dat")
# Making a directory for this run and running in it
if not os.path.exists(RUN_NAME):
    os.makedirs(RUN_NAME)
os.chdir(os.path.join(os.path.abspath(''), RUN_NAME,))

# Simulations' metadata
metadata = pd.read_csv(METADATA_PATH, sep=' ', header=None, names=['sim', 'rvir', 'mvir', 'rvir_bn', 'mvir_bn'])
metadata = metadata.astype({'sim': str, 'rvir': float, 'mvir': float, 'rvir_bn': float, 'mvir_bn': float})

# Removing profiles with missing rvir_bn or mvir_bn
metadata = metadata[~(metadata["rvir_bn"].isna() | metadata["mvir_bn"].isna())]

# Selecting rows with metadata["sim"] starting with 'g1'
# metadata = metadata[metadata["sim"].str.startswith('g1')]

# Subsampling data
metadata = metadata.sample(frac=FRAC_REALS, random_state=SEED)
n_reals  = len(metadata)

# Saving subsampled metadata
metadata.to_csv('subsample.csv', index=False)

# Histograms of rvir_bn and mvir_bn
fig, ax = plt.subplots(1,2, figsize=(10,5))
fig.suptitle('%i profiles' % n_reals)
ax[0].hist(metadata["rvir_bn"], bins=30, histtype='step', color='k')
ax[0].set_xlabel('$R_{vir}\ [kpc]$')
ax[1].hist(metadata["mvir_bn"], bins=30, histtype='step', color='k')
ax[1].set_xlabel('$M_{vir}\ [M_{\odot}]$')
fig.savefig('subsample_histo.png', dpi=300)
plt.show()

# Selection names
sim_names = metadata["sim"]

# Getting profiles
def get_profile(sim_name):
    df = pd.read_csv(DATA_PATH + '%s_profile.dat'%(sim_name), sep=' ', header=None, names=['r', 'n', 'rho'])
    df = df.astype({'r': float, 'n': int, 'rho': float})
    # Selecting profiles points with more than 1000 particles
    df = df[df["n"] > 1000]
    # Metadata for the profile
    md = metadata[metadata["sim"] == sim_name]
    rvir_bn = md["rvir_bn"].values[0]
    mvir_bn = md["mvir_bn"].values[0]
    # Normalizing profile
    df["r"]   = df["r"]   / rvir_bn
    df["rho"] = df["rho"] / ( mvir_bn / ((4/3)*np.pi*(rvir_bn**3)) )
    return df

multi_X = []
multi_y = []
multi_y_weights = []
for sim_name in sim_names:
    df = get_profile(sim_name)
    r    = df["r"].values
    rho  = df["rho"].values
    n    = df["n"].values
    # Poisson uncertainty from n : y_unc = np.sqrt(n) / n = 1 / np.sqrt(n)
    y_weight = np.sqrt(n)      # (n_samples,)
    X = np.stack([r], axis=0)  # (1, n_samples)
    y = rho                    # (n_samples,)
    # Appending to multi_X, multi_y, multi_y_weights
    multi_X.append(X)
    multi_y.append(y)
    multi_y_weights.append(y_weight)
n_samples_per_dataset = np.array([X.shape[1] for X in multi_X])

# Let's evaluate fit quality on log(y) vs log(f(x))
multi_y = [np.log(y) for y in multi_y]

# ------ CONSTANTS AND FREE CONSTANTS ------

# Constants
FIXED_CONSTS = [1.,]
CLASS_FREE_CONSTS_NAMES = ["c%i"%(i) for i in range (N_CLASS_FREE_PARAMS)]
CLASS_FREE_CONSTS_UNITS = [ [0,0] for i in range (N_CLASS_FREE_PARAMS)]

SPE_FREE_CONSTS_NAMES   = ["rho0",] + ["Rs%i"%(i) for i in range (N_SPE_FREE_PARAMS - 1) ]
SPE_FREE_CONSTS_UNITS   = [[1,-3],] + [ [0,1]     for _ in range (N_SPE_FREE_PARAMS - 1) ]

Y_UNITS = [1,-3]
X_UNITS = [[0,1]]

free_const_opti_args = {
            'loss'   : "MSE",
            'method' : 'LBFGS',
            'method_args': {
                        'n_steps' : 50,
                        'tol'     : 1e-99,
                        'lbfgs_func_args' : {
                            'max_iter'       : 4,
                            'line_search_fn' : "strong_wolfe",
                                             },
                            },
        }

# -------- MONITORING --------

save_path_training_curves = 'demo_curves.png'
save_path_log             = 'demo.log'

run_logger     = lambda : monitoring.RunLogger(save_path = save_path_log,
                                                do_save = True)

run_visualiser = lambda : monitoring.RunVisualiser (epoch_refresh_rate = 1,
                                           save_path = save_path_training_curves,
                                           do_show   = False,
                                           do_prints = True,
                                           do_save   = True, )


# ----------- CUSTOM REWARD -----------
#PROPERTIES_TO_CHECK = ["enclosed mass", "potential",] #"radial velocity dispersion"]
PROPERTIES_TO_CHECK = ["enclosed mass", "potential", "radial velocity dispersion"]

import physo.physym.batch_execute as bexec

# During programs evaluation, should parallel execution be used ?
USE_PARALLEL_EXE        = False  # Only worth it if n_all_samples > 1e6
USE_PARALLEL_OPTI_CONST = False  # Only worth it if batch_size > 1k

def my_RewardsComputer(programs,
                    X,
                    y_target,
                    n_samples_per_dataset,
                    y_weights = 1.,
                    free_const_opti_args = None,
                    reward_function = physo.physym.reward.SquashedNRMSE,
                    zero_out_unphysical = False,
                    zero_out_duplicates = False,
                    keep_lowest_complexity_duplicate = False,
                    parallel_mode = False,
                    n_cpus = None,
                    progress_bar = False,
                    ):
    """
    Computes rewards of programs on X data accordingly with target y_target and reward reward_function using torch
    for acceleration.
    Parameters
    ----------
    programs : Program.VectProgram
        Programs contained in batch to evaluate.
    X : torch.tensor of shape (n_dim, ?,) of float
        Values of the input variables of the problem with n_dim = nb of input variables.
    y_target : torch.tensor of shape (?,) of float
        Values of the target symbolic function on input variables contained in X_target.
    n_samples_per_dataset : array_like of shape (n_realizations,) of int
        We assume that X contains multiple datasets with samples of each ataset following each other and each portion
        of X corresponding to a dataset should be treated with its corresponding dataset specific free constants values.
        n_samples_per_dataset is the number of samples for each dataset. Eg. [90, 100, 110] for 3 datasets, this will
        assume that the first 90 samples of X are for the first dataset, the next 100 for the second and the last 110
        for the third.
    y_weights : torch.tensor of shape (?,) of float, optional
        Weights for each data point.
    free_const_opti_args : dict or None, optional
        Arguments to pass to free_const.optimize_free_const for free constant optimization. By default,
        free_const.DEFAULT_OPTI_ARGS arguments are used.

    reward_function : callable
        Function that taking y_target (torch.tensor of shape (?,) of float), y_pred (torch.tensor of shape (?,)
        of float) and  optionally  y_weights (torch.tensor of shape (?,) of float, optional) as key arguments and
        returning a float reward of an individual program.
    zero_out_unphysical : bool
        Should unphysical programs be zeroed out ?
    zero_out_duplicates : bool
        Should duplicate programs (equal symbolic value when simplified) be zeroed out ?
    keep_lowest_complexity_duplicate : bool
        If True, when eliminating duplicates (via zero_out_duplicates = True), the least complex duplicate is kept, else
        a random duplicate is kept.
    Returns
    -------
    rewards : numpy.array of shape (?,) of float
        Rewards of programs.
    """

    # ----- SETUP -----

    # mask : should program reward NOT be zeroed out ie. is program invalid ?
    # By default all programs are considered valid
    mask_valid = np.full(shape=programs.batch_size, fill_value=True, dtype=bool)

    # ----- PHYSICALITY -----
    if zero_out_unphysical:
        # mask : is program physical
        mask_is_physical = programs.is_physical                                                          # (batch_size,)
        # Update mask to zero out unphysical programs
        mask_valid = (mask_valid & mask_is_physical)                                                     # (batch_size,)

    # ----- ANALYTICAL PROPERTIES -----
    mask_has_props = np.full(shape=programs.batch_size, fill_value=False, dtype=bool)  # (batch_size,)
    for i in range (programs.batch_size):
        # Only check analytical properties if program is valid
        print("Checking analytical properties of program %i/%i (%i to analyze)"%(i+1, programs.batch_size, mask_valid.sum()))
        if mask_valid[i]:
            # mask : does program have analytical properties ?
            prog = programs.get_prog(i)
            #prog_str = prog.get_infix_sympy(evaluate_consts=True)[0].__str__()
            prog_sympy = su.clean_sympy_expr(prog.get_infix_sympy())
            prog_str = prog_sympy.__str__()
            print(prog_str)
            # Compute analytical properties
            results_df = ap.check_analytical_properties(density_profile_str=prog_str,
                                                        free_consts_names=["rho0", "Rs0", "Rs1"],
                                                        free_consts_vals={},
                                                        num_check=False,
                                                        verbose=False,)
            properties = PROPERTIES_TO_CHECK
            filtered_df = results_df[results_df["Property"].str.lower().isin([p.lower() for p in properties])]
            conditions = filtered_df["symb_condition"].values
            print(filtered_df[['Property', "symb_condition"]])
            # If all conditions are satisfied, the program has analytical properties
            is_ok = bool(conditions.all())
        else:
            # If program is not valid, it does not have analytical properties
            is_ok = False
        mask_has_props[i] = is_ok
        #todo: check not nan
        print("Found %i programs satisfying analytical properties so far."%(mask_has_props.sum()))
    mask_valid = (mask_valid & mask_has_props)

    # ----- DUPLICATES -----
    if zero_out_duplicates:
        # Compute rewards (even if programs have non-optimized free consts) to serve as a unique numeric identifier of
        # functional forms (programs having equivalent forms will have the same reward).

        # Only use parallel mode if enabled in function param and in USE_PARALLEL_EXE flag.
        # This way users can use flags to specifically enable or disable parallel exe and/or const opti.
        parallel_mode_exe = parallel_mode and USE_PARALLEL_EXE
        rewards_non_opt = programs.batch_exe_reward (X         = X,
                                                     y_target  = y_target,
                                                     y_weights = y_weights,
                                                     reward_function       = reward_function,
                                                     n_samples_per_dataset = n_samples_per_dataset,
                                                     mask            = mask_valid,
                                                     pad_with        = 0.0,
                                                     # Parallel related
                                                     parallel_mode   = parallel_mode_exe,
                                                     n_cpus          = n_cpus,
                                                    )
        # mask : is program a unique one we should keep ?
        # By default, all programs are eliminated.
        mask_unique_keep = np.full(shape=programs.batch_size, fill_value=False, dtype=bool)              # (batch_size,)
        # Identifying unique programs.
        unique_rewards, unique_idx = np.unique(rewards_non_opt, return_index=True)                       # (n_unique,), (n_unique,)
        if keep_lowest_complexity_duplicate:
            unique_idx_lowest_comp = []
            # Iterating through unique rewards
            for r in unique_rewards:
                # mask: does program have current unique reward ?
                mask_have_r = (rewards_non_opt == r)                                                     # (batch_size,)
                # complexities of programs having current unique reward
                complexities_at_r = programs.n_complexity[mask_have_r]                                   # (n_at_r,)
                # idx in batch of program having current unique reward of the lowest complexity
                idx_lowest_comp = np.arange(programs.batch_size)[mask_have_r][complexities_at_r.argmin()]
                unique_idx_lowest_comp.append(idx_lowest_comp)
            # Idx of unique programs (having the lowest complexity among their duplicates)
            unique_idx_lowest_comp = np.array(unique_idx_lowest_comp)
            # Keeping the lowest complexity duplicate of unique programs
            mask_unique_keep[unique_idx_lowest_comp] = True
        else:
            # Keeping first occurrences of unique programs (random)
            mask_unique_keep[unique_idx] = True                                                          # (n_unique,)
        # Update mask to zero out duplicate programs
        mask_valid = (mask_valid & mask_unique_keep)                                                     # (batch_size,)

    # ----- FREE CONST OPTIMIZATION -----
    # If there are free constants in the library, we have to optimize them
    if programs.library.n_free_const > 0:
        # Only use parallel mode if enabled in function param and in USE_PARALLEL_OPTI_CONST flag.
        # This way users can use flags to specifically enable or disable parallel exe and/or const opti.
        parallel_mode_const_opti = parallel_mode and USE_PARALLEL_OPTI_CONST
        # Opti const
        # batch_optimize_free_const (programs, X, y_target, args_opti = free_const_opti_args, mask_valid = mask_valid)
        programs.batch_optimize_constants(X        = X,
                                          y_target = y_target,
                                          free_const_opti_args  = free_const_opti_args,
                                          y_weights             = y_weights,
                                          mask                  = mask_valid,
                                          n_samples_per_dataset = n_samples_per_dataset,
                                          # Parallel related
                                          parallel_mode         = parallel_mode_const_opti,
                                          n_cpus                = n_cpus)

    # ----- REWARDS -----
    # If rewards were already computed at the duplicate elimination step and there are no free constants in the library
    # No need to recompute rewards.
    if zero_out_duplicates and programs.library.n_free_const == 0:
        rewards = rewards_non_opt
    # Else we need to compute rewards
    else:
        # Only use parallel mode if enabled in function param and in USE_PARALLEL_EXE flag.
        # This way users can use flags to specifically enable or disable parallel exe and/or const opti.
        parallel_mode_exe = parallel_mode and USE_PARALLEL_EXE
        rewards = programs.batch_exe_reward (X         = X,
                                             y_target  = y_target,
                                             y_weights = y_weights,
                                             reward_function       = reward_function,
                                             n_samples_per_dataset = n_samples_per_dataset,
                                             mask            = mask_valid,
                                             pad_with        = 0.0,
                                             # Parallel related
                                             parallel_mode   = parallel_mode_exe,
                                             n_cpus          = n_cpus,
                                            )

    # Applying mask (this is redundant)
    rewards = rewards * mask_valid.astype(float)
    # Safety to avoid nan rewards (messes up gradients)
    rewards = np.nan_to_num(rewards, nan=0.)

    return rewards

def my_make_RewardsComputer(reward_function     = physo.physym.reward.SquashedNRMSE,
                         zero_out_unphysical = False,
                         zero_out_duplicates = False,
                         keep_lowest_complexity_duplicate = False,
                         # Parallel related
                         parallel_mode = True,
                         n_cpus        = None,
                         ):
    """
    Helper function to make custom reward computing function.
    Parameters
    ----------
    reward_function : callable
        Function that taking y_target (torch.tensor of shape (?,) of float), y_pred (torch.tensor of shape (?,)
        of float) and  optionally  y_weights (torch.tensor of shape (?,) of float, optional) as key arguments and
        returning a float reward of an individual program.
    zero_out_unphysical : bool
        Should unphysical programs be zeroed out ?
    zero_out_duplicates : bool
        Should duplicate programs (equal symbolic value when simplified) be zeroed out ?
    keep_lowest_complexity_duplicate : bool
        If True, when eliminating duplicates (via zero_out_duplicates = True), the least complex duplicate is kept, else
        a random duplicate is kept.
    parallel_mode : bool
        Tries to use parallel execution if True (availability will be checked by batch_execute.ParallelExeAvailability),
        execution in a loop else.
    n_cpus : int or None
        Number of CPUs to use when running in parallel mode. By default, uses the maximum number of CPUs available.
    Returns
    -------
    rewards_computer : callable
         Custom reward computing function taking programs (vect_programs.VectPrograms), X (torch.tensor of shape (n_dim,?,)
         of float), y_target (torch.tensor of shape (?,) of float), y_weights (torch.tensor of shape (?,) of float),
         n_samples_per_dataset (array_like of shape (n_realizations,) of int) and free_const_opti_args as key arguments
         and returning reward for each program (array_like of float).
    """
    print("custom rewards_computer:")
    # Check that parallel execution is available on this system
    recommended_config = bexec.ParallelExeAvailability()
    is_parallel_mode_available_on_system = recommended_config["parallel_mode"]
    # If not available and parallel_mode was still instructed warn and disable
    if not is_parallel_mode_available_on_system and parallel_mode:
        bexec.ParallelExeAvailability(verbose=True) # prints explanation
        warnings.warn("Parallel mode is not available on this system, switching to non parallel mode.")
        parallel_mode = False

    # rewards_computer
    def rewards_computer(programs, X, y_target, y_weights, n_samples_per_dataset, free_const_opti_args):
        print("hey")
        R = my_RewardsComputer(programs  = programs,
                            X         = X,
                            y_target  = y_target,
                            y_weights = y_weights,
                            n_samples_per_dataset = n_samples_per_dataset,
                            free_const_opti_args  = free_const_opti_args,
                            # Frozen args
                            reward_function     = reward_function,
                            zero_out_unphysical = zero_out_unphysical,
                            zero_out_duplicates = zero_out_duplicates,
                            keep_lowest_complexity_duplicate = keep_lowest_complexity_duplicate,
                            # Parallel related
                            parallel_mode = parallel_mode,
                            n_cpus        = n_cpus,
                            )
        return R

    return rewards_computer


# ------- CONFIGURATION -------
my_config = physo.config.config2b.config2b

# Putting new reward computer in config
reward_config = {
                 "reward_function"     : physo.physym.reward.SquashedNRMSE,
                 "zero_out_unphysical" : True,
                 "zero_out_duplicates" : False,
                 "keep_lowest_complexity_duplicate" : False,
                }
my_config["learning_config"]["rewards_computer"] = my_make_RewardsComputer (**reward_config)
my_config["learning_config"]["custom_rewards_computer"] = True

# Operations allowed
OP_NAMES = ["add", "sub", "mul", "div", "inv", "n2", "sqrt", "neg", "log", "exp"]

# ENFORCING NEW FREE CONSTS PARAMS
my_config["free_const_opti_args"] = free_const_opti_args

# pISO "rho0 / (1 + (r / Rs)**2)"
# ["div", "rho0", "add", "1.0", "n2", "div", "r", "Rs0",]
# NFW : "rho0 / ((r / Rs) * (1 + r / Rs)**2)"
# ["div", "rho0", "mul", "div", "r", "Rs0", "n2", "add", "1.0", "div", "r", "Rs0",]

# tmp
target_prog_str = ["div", "rho0", "add", "1.0", "n2", "div", "r", "Rs0",]
cheater_prior_config = ('SymbolicPrior', {'expression': target_prog_str})
my_config["priors_config"].append(cheater_prior_config)

# ENFORCING EQUATION TO START WITH LOG
def candidate_wrapper(func, X):
    return torch.log(torch.abs(func(X)))


# Hack here
my_config["learning_config"]["batch_size"] = 20 # tmp

MAX_N_EVALUATIONS = int(1e99)
# Allowed to search in an infinitely large search space, research will be stopped by MAX_N_EVALUATIONS
N_EPOCHS          = 1 # int(1e99)  #int(1e99) # tmp 1 #

# ------- RUN -------

# Running SR task
expression, logs = physo.ClassSR(multi_X, multi_y,
            multi_y_weights = multi_y_weights,
            X_names = ["r",],
            X_units = X_UNITS,
            y_name  = "rho",
            y_units = Y_UNITS,
            # Fixed constants
            fixed_consts       = FIXED_CONSTS,
            # Free constants names (for display purposes)
            class_free_consts_names = CLASS_FREE_CONSTS_NAMES,
            class_free_consts_units = CLASS_FREE_CONSTS_UNITS,
            # Spe free constants names (for display purposes)
            spe_free_consts_names = SPE_FREE_CONSTS_NAMES,
            spe_free_consts_units = SPE_FREE_CONSTS_UNITS,
            # Operations allowed
            op_names = OP_NAMES,
            # Wrapper
            candidate_wrapper = candidate_wrapper,
            # Run config
            run_config = my_config,
            # Run monitoring
            get_run_logger     = run_logger,
            get_run_visualiser = run_visualiser,
            # Stopping condition
            stop_reward = 1.1,  # not stopping even if perfect 1.0 reward is reached
            max_n_evaluations = MAX_N_EVALUATIONS,
            epochs            = N_EPOCHS,
            # Parallel mode
            parallel_mode = False,
    )
