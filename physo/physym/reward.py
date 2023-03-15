import warnings

import numpy as np
import torch as torch

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

    def batch_compute_rewards (programs, X, y_target, reward_function, mask_valid):
        """
        Helper function to compute reward of a batch of program where mask_valid is True.
        """
        rewards = []
        # Iterating through batch
        for i in range(programs.batch_size):
            # print("rewarding %i/%i" % (i, programs.batch_size))
            # If valid prog, reward should be computed
            if mask_valid[i]:
                try:
                    # Getting prog
                    prog = programs.get_prog(i)
                    # Making prediction
                    y_pred = prog(X)
                    # Computing reward
                    r = reward_function(y_pred=y_pred, y_target=y_target,)
                except:
                    # Safety
                    warnings.warn("Unable to compute reward of prog %i -> r = 0"%(i))
                    r = torch.tensor(0.)
            # If this is not a valid prog, we should not bother computing reward
            else:
                r = torch.tensor(0.)
            # Only using torch for gpu acceleration, no backpropagation happening here -> converting to numpy
            rewards.append(r.detach().cpu().numpy())
        rewards = np.array(rewards)                                                                      # (batch_size,)
        return rewards

    def batch_optimize_free_const (programs, X, y_target, args_opti, mask_valid,):
        """
        Helper function to optimize.py constants of a batch of program where mask_valid is True.
        """
        # Iterating through batch
        for i in range(programs.batch_size):
            #print("%i/%i"%(i, programs.batch_size))
            # print("optimizing free const %i/%i"%(i, programs.batch_size))
            # If this is a valid prog AND it contains free constants then we try to optimize.py its free constants.
            # (Else we should not bother optimizing its free constants)
            if mask_valid[i] and programs.n_free_const_occurrences[i]:
                try:
                    # Getting prog
                    prog = programs.get_prog(i)
                    # Optimizing free constants
                    history = prog.optimize_constants(X, y_target, args_opti=args_opti)
                    # Logging free constant optimization process
                    programs.free_consts.is_opti    [i] = True
                    programs.free_consts.opti_steps [i] = len(history)
                except:
                    # Safety
                    warnings.warn("Unable to optimize.py free constants of prog %i -> r = 0"%(i))
        return None

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
        rewards_non_opt = batch_compute_rewards (programs, X, y_target,
                                                 reward_function = reward_function,
                                                 mask_valid = mask_valid)
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
        batch_optimize_free_const (programs, X, y_target, args_opti = free_const_opti_args, mask_valid = mask_valid)

    # ----- REWARDS -----
    # If rewards were already computed at the duplicate elimination step and there are no free constants in the library
    # No need to recompute rewards.
    if zero_out_duplicates and programs.library.n_free_const == 0:
        rewards = rewards_non_opt
    # Else we need to compute rewards
    else:
        rewards = batch_compute_rewards (programs, X, y_target, reward_function = reward_function, mask_valid = mask_valid)

    # Applying mask (this is redundant)
    rewards = rewards * mask_valid.astype(float)
    # Safety to avoid nan rewards (messes up gradients)
    rewards = np.nan_to_num(rewards, nan=0.)

    return rewards


def make_RewardsComputer(reward_function     = SquashedNRMSE,
                         zero_out_unphysical = False,
                         zero_out_duplicates = False,
                         keep_lowest_complexity_duplicate = False):
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
    Returns
    -------
    rewards_computer : callable
         Custom reward computing function taking programs (program.VectPrograms), X (torch.tensor of shape (n_dim,?,)
         of float), y_target (torch.tensor of shape (?,) of float) as key arguments and returning reward for each
         program (array_like of float).
    """
    rewards_computer = lambda programs, X, y_target, free_const_opti_args : RewardsComputer(programs = programs,
                                                                                            X        = X,
                                                                                            y_target = y_target,
                                                                                            free_const_opti_args = free_const_opti_args,
                                                                                            # Frozen args
                                                                                            reward_function     = reward_function,
                                                                                            zero_out_unphysical = zero_out_unphysical,
                                                                                            zero_out_duplicates = zero_out_duplicates,
                                                                                            keep_lowest_complexity_duplicate = keep_lowest_complexity_duplicate,)
    return rewards_computer