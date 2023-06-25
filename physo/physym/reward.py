import warnings

import numpy as np
import torch as torch
import physo.physym.execute as exec

# During programs evaluation, should parallel execution be used ?
USE_PARALLEL_EXE        = False  # Only worth it if n_samples > 1e6
USE_PARALLEL_OPTI_CONST = True   # Only worth it if batch_size > 1k

def SquashedNRMSE (y_target, y_pred,):
    """
    Squashed NRMSE reward.
    Parameters
    ----------
    y_target : torch.tensor of shape (?,) of float
        Target output data.
    y_pred   : torch.tensor of shape (?,) of float
        Predicted data.
    Returns
    -------
    reward : torch.tensor float
        Reward encoding prediction vs target discrepancy in [0,1].
    """
    sigma_targ = y_target.std()
    RMSE = torch.sqrt(torch.mean((y_pred-y_target)**2))
    NRMSE = (1/sigma_targ)*RMSE
    reward = 1/(1 + NRMSE)
    return reward

def RewardsComputer(programs,
                    X,
                    y_target,
                    free_const_opti_args = None,
                    reward_function = SquashedNRMSE,
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
    free_const_opti_args : dict or None, optional
        Arguments to pass to free_const.optimize_free_const for free constants optimization. By default,
        free_const.DEFAULT_OPTI_ARGS arguments are used.

    reward_function : callable
        Function that taking y_target (torch.tensor of shape (?,) of float) and y_pred (torch.tensor of shape (?,)
        of float) as key arguments and returning a float reward of an individual program.
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
    mask_valid = np.full(shape=programs.batch_size, fill_value=True, dtype=bool)                         # (batch_size,)

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
        rewards_non_opt = programs.batch_exe_reward (X        = X,
                                                     y_target = y_target,
                                                     reward_function = reward_function,
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
    # If there are free constants in the library, we have to optimize.py them
    if programs.library.n_free_const > 0:
        # Only use parallel mode if enabled in function param and in USE_PARALLEL_OPTI_CONST flag.
        # This way users can use flags to specifically enable or disable parallel exe and/or const opti.
        parallel_mode_const_opti = parallel_mode and USE_PARALLEL_OPTI_CONST
        # Opti const
        # batch_optimize_free_const (programs, X, y_target, args_opti = free_const_opti_args, mask_valid = mask_valid)
        programs.batch_optimize_constants(X        = X,
                                          y_target = y_target,
                                          free_const_opti_args = free_const_opti_args,
                                          mask                 = mask_valid,
                                          # Parallel related
                                          parallel_mode        = parallel_mode_const_opti,
                                          n_cpus               = n_cpus)

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
        rewards = programs.batch_exe_reward (X        = X,
                                             y_target = y_target,
                                             reward_function = reward_function,
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


def make_RewardsComputer(reward_function     = SquashedNRMSE,
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
        Reward function to use that takes y_target (torch.tensor of shape (?,) of float) and y_pred (torch.tensor of
        shape (?,) of float) as key arguments and returns a float reward of an individual program.
    zero_out_unphysical : bool
        Should unphysical programs be zeroed out ?
    zero_out_duplicates : bool
        Should duplicate programs (equal symbolic value when simplified) be zeroed out ?
    keep_lowest_complexity_duplicate : bool
        If True, when eliminating duplicates (via zero_out_duplicates = True), the least complex duplicate is kept, else
        a random duplicate is kept.
    parallel_mode : bool
        Tries to use parallel execution if True (availability will be checked by execute.ParallelExeAvailability),
        execution in a loop else.
    n_cpus : int or None
        Number of CPUs to use when running in parallel mode. By default, uses the maximum number of CPUs available.
    Returns
    -------
    rewards_computer : callable
         Custom reward computing function taking programs (program.VectPrograms), X (torch.tensor of shape (n_dim,?,)
         of float), y_target (torch.tensor of shape (?,) of float), free_const_opti_args as key arguments and returning reward for each
         program (array_like of float).
    """
    # Check that parallel execution is available on this system
    recommended_config = exec.ParallelExeAvailability()
    is_parallel_mode_available_on_system = recommended_config["parallel_mode"]
    # If not available and parallel_mode was still instructed warn and disable
    if not is_parallel_mode_available_on_system and parallel_mode:
        exec.ParallelExeAvailability(verbose=True) # prints explanation
        warnings.warn("Parallel mode is not available on this system, switching to non parallel mode.")
        parallel_mode = False

    # rewards_computer
    def rewards_computer(programs, X, y_target, free_const_opti_args):
        R = RewardsComputer(programs = programs,
                            X        = X,
                            y_target = y_target,
                            free_const_opti_args = free_const_opti_args,
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