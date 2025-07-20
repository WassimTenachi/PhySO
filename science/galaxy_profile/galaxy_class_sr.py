#!/usr/bin/env python
# coding: utf-8

# # $\Phi$-SO demo : Class SR quick start

# ## Class SR definition

# Class Symbolic Regression:
# Automatically finding a single analytical functional form that accurately fits multiple datasets - each governed by its own (possibly) unique set of fitting parameters.
# This hierarchical framework leverages the common constraint that all the members of a single class of physical phenomena follow a common governing law.

# ![class_sr_framework.png](attachment:3b74f0d7-8a06-4663-bf0d-e442c0f40aff.png)

# ## Package import

# In[1]:


# External packages
import numpy as np
import matplotlib.pyplot as plt
import torch

# In[2]:


# Internal code import
import physo
import physo.learn.monitoring as monitoring

# Local imports
from science.galaxy_profile import analytic_properties as ap



# ## Fixing seed

# In[3]:


# Seed
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)


# ## Dataset

# In[4]:


# Making toy synthetic data
multi_X = []
multi_y = []

# Realization 0
x0 = np.random.uniform(-10, 10, 256)
X = np.stack((x0,), axis=0)
y = 1.123*x0 + 10.123
multi_X.append(X)
multi_y.append(y)

# Realization 1
x0 = np.random.uniform(-11, 11, 500)
X = np.stack((x0,), axis=0)
y = 2*1.123*x0 + 10.123
multi_X.append(X)
multi_y.append(y)


n_reals = len(multi_X)
for i in range(n_reals):
    fig, ax = plt.subplots(1,1, figsize=(10,5))
    fig.suptitle("Realization {}".format(i))
    ax.plot(multi_X[i][0], multi_y[i], 'k.')
    ax.set_xlabel("x0")
    ax.set_ylabel("y")
    plt.show()



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
import physo.physym.batch_execute as bexec

# During programs evaluation, should parallel execution be used ?
USE_PARALLEL_EXE        = False  # Only worth it if n_all_samples > 1e6
USE_PARALLEL_OPTI_CONST = True   # Only worth it if batch_size > 1k

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
    print("hola")

    # mask : should program reward NOT be zeroed out ie. is program invalid ?
    # By default all programs are considered valid
    mask_valid = np.full(shape=programs.batch_size, fill_value=True, dtype=bool)

    # ----- ANALYTICAL PROPERTIES -----
    mask_has_props = np.full(shape=programs.batch_size, fill_value=False, dtype=bool)  # (batch_size,)
    for i in range (programs.batch_size):
        print("Checking analytical properties of program %i/%i"%(i+1, programs.batch_size))
        # mask : does program have analytical properties ?
        prog = programs.get_prog(i)
        prog_str = prog.get_infix_sympy(evaluate_consts=True)[0].__str__()
        print(prog_str)
        # Compute analytical properties
        results_df = ap.check_analytical_properties(density_profile_str=prog_str,
                                                    free_consts_names=[],
                                                    free_consts_vals={},
                                                    num_check=False,
                                                    verbose=False,)
        properties = ["enclosed mass", "potential",] #["radial velocity dispersion"]
        filtered_df = results_df[results_df["Property"].str.lower().isin([p.lower() for p in properties])]
        conditions = filtered_df["symb_condition"].values
        # If all conditions are satisfied, the program has analytical properties
        mask_has_props[i] = bool(conditions.all())
        #todo: check not nan
        print("Found %i programs satisfying analytical properties so far."%(mask_has_props.sum()))
    mask_valid = (mask_valid & mask_has_props)

    # ----- PHYSICALITY -----
    if zero_out_unphysical:
        # mask : is program physical
        mask_is_physical = programs.is_physical                                                          # (batch_size,)
        # Update mask to zero out unphysical programs
        mask_valid = (mask_valid & mask_is_physical)                                                     # (batch_size,)

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
my_config = physo.config.config0b.config0b
reward_config = {
                 "reward_function"     : physo.physym.reward.SquashedNRMSE,
                 "zero_out_unphysical" : True,
                 "zero_out_duplicates" : False,
                 "keep_lowest_complexity_duplicate" : False,
                }
my_config["learning_config"]["rewards_computer"] = my_make_RewardsComputer (**reward_config)
my_config["learning_config"]["custom_rewards_computer"] = True

# Running SR task
expression, logs = physo.ClassSR(multi_X, multi_y,
                            # Giving names of variables (for display purposes)
                            X_names = [ "r"       , ],
                            # Associated physical units (ignore or pass zeroes if irrelevant)
                            X_units = [ [0, 0, 0] , ],
                            # Giving name of root variable (for display purposes)
                            y_name  = "y",
                            y_units = [0, 0, 0],
                            # Fixed constants
                            fixed_consts       = [ 1.      ],
                            fixed_consts_units = [ [0, 0, 0] ],
                            # Whole class free constants
                            class_free_consts_names = [ "c0"      ,],
                            class_free_consts_units = [ [0, 0, 0] ,],
                            # Realization specific free constants
                            spe_free_consts_names = [ "rho0"      , "Rs"        ],
                            spe_free_consts_units = [ [0, 0, 0]   , [0, 0, 0]   ],
                            # Run config
                            run_config = my_config,
                            # Symbolic operations that can be used to make f
                            op_names = ["add", "sub", "mul", "div", "inv", "exp", "log"],
                            get_run_logger     = run_logger,
                            get_run_visualiser = run_visualiser,
                            # Parallel mode (only available when running from python scripts, not notebooks)
                            parallel_mode = False,
                            # Number of iterations
                            epochs = 10,
)

