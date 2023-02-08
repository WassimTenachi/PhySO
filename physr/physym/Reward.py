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
        Arguments to pass to FreeConstUtils.optimize_free_const for free constants optimization. By default,
        FreeConstUtils.DEFAULT_OPTI_ARGS arguments are used.

    reward_function : callable
        Function that taking y_target (torch.tensor of shape (?,) of float) and y_pred (torch.tensor of shape (?,)
        of float) as key arguments and returning a float reward of an individual program.
    zero_out_unphysical : bool
        Should unphysical programs be zeroed out ?
    zero_out_duplicates : bool
        Should duplicate programs (equal symbolic value when simplified) be zeroed out ?
    Returns
    -------
    rewards : numpy.array of shape (?,) of float
        Rewards of programs.
    """

    def batch_compute_rewards (programs, X, y_target, mask_valid):
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
                    warnings.warn("Unable to compute reward of prog %i -> r = 0"%(i))
                    r = torch.tensor(0.)
            # If not valid prog, we should not bother computing reward
            else:
                r = torch.tensor(0.)
            rewards.append(r)
        # Only using torch for acceleration, no backpropagation happening here -> converting to numpy
        rewards = torch.stack(rewards).cpu().detach().numpy()
        return rewards

    def batch_optimize_free_const (programs, X, y_target, args_opti, mask_valid,):
        """
        Helper function to optimize constants of a batch of program where mask_valid is True.
        """
        # Iterating through batch
        for i in range(programs.batch_size):
            # print("optimizing free const %i/%i"%(i, programs.batch_size))
            # If this is a valid prog AND it contains free constants then try to optimize consts
            # (Else we should not bother optimizing free constants)
            if mask_valid[i] and programs.n_free_const_occurrences[i]:
                try:
                    # Getting prog
                    prog = programs.get_prog(i)
                    # Optimizing free constants
                    history = prog.optimize_constants(X, y_target, args_opti=args_opti)
                    # Logging free constant optimization
                    programs.free_consts.is_opti    [i] = True
                    programs.free_consts.opti_steps [i] = len(history)
                except:
                    warnings.warn("Unable to optimize free constants of prog %i -> r = 0"%(i))
        return None

    # mask : should program reward NOT be zeroed out ? # By default all programs are considered valid
    mask_valid = np.full(shape=programs.batch_size, fill_value=True, dtype=bool)

    # ----- PHYSICALITY -----
    # mask : is program physical
    mask_is_physical = programs.is_physical
    # Update mask to zero out unphysical programs if necessary
    if zero_out_unphysical:
        mask_valid = (mask_valid & mask_is_physical)

    # ----- DUPLICATES -----
    # todo: compute all rewards (with initial constants values) to use rewards as hash for unique-ness of candidates
    #  (-> no need for simplification for identifying duplicates)
    # todo: put reward of duplicates to 0 + mask encoding which were eliminated

    # ----- FREE CONST OPTI -----
    batch_optimize_free_const (programs, X, y_target, args_opti = free_const_opti_args, mask_valid = mask_valid)

    # ----- REWARDS -----
    rewards = batch_compute_rewards (programs, X, y_target, mask_valid = mask_valid)

    return rewards


def make_RewardsComputer(reward_function     = SquashedNRMSE,
                         zero_out_unphysical = False,
                         zero_out_duplicates = False,):
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
    Returns
    -------
    rewards_computer : callable
         Custom reward computing function taking programs (Program.VectPrograms), X (torch.tensor of shape (n_dim,?,)
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
                                                                                            )
    return rewards_computer