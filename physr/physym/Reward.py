import warnings

import numpy as np
import torch as torch

def Reward_SquashedNRMSE (y_target, y_pred, program=None):
    """
    Squashed NRMSE reward.
    Parameters
    ----------
    y_target : torch.tensor of shape (?,) of float
        Target output data.
    y_pred   : torch.tensor of shape (?,) of float
        Predicted data.
    program : Program.Program or None (optional)
        Program evaluated here (useful if reward should also depend on symbolic information).
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


def Reward_Physical_SquashedNRMSE (y_target, y_pred, program):
    """
    Squashed NRMSE reward or 0 if the program is not physical.
    Parameters
    ----------
    y_target : torch.tensor of shape (?,) of float
        Target output data.
    y_pred   : torch.tensor of shape (?,) of float
        Predicted data.
    program : Program.Program or None (optional)
        Program evaluated here (useful if reward should also depend on symbolic information).
    Returns
    -------
    reward : torch.tensor float
        Reward encoding prediction vs target discrepancy in [0,1].
    """
    reward = Reward_SquashedNRMSE(y_target=y_target, y_pred=y_pred, program=program)
    reward = reward * float(program.is_physical)
    return reward

def ComputeRewards(reward_function, programs, X, y_target):
    """
    Computes rewards of programs on X data accordingly with target y_target and reward reward_function using torch
    for acceleration.
    Parameters
    ----------
    reward_function : callable
        Function that taking y_target (torch.tensor of shape (?,) of float) and y_pred (torch.tensor of shape (?,)
        of float) as key arguments and returning a float reward of an individual program.
    programs : Program.VectProgram
        Programs contained in batch to evaluate.
    X : torch.tensor of shape (n_dim, ?,) of float
        Values of the input variables of the problem with n_dim = nb of input variables.
    y_target : torch.tensor of shape (?,) of float
        Values of the target symbolic function on input variables contained in X_target.
    Returns
    -------
    rewards : numpy.array of shape (?,) of float
        Rewards of programs.
    """
    rewards = []
    for i in range(programs.batch_size):
        try:
            prog = programs.get_prog(i)
            y_pred = prog(X)
            r = reward_function(y_pred=y_pred, y_target=y_target, program=prog)
        except:
            warnings.warn("Unable to compute reward of prog %i -> r = 0"%(i))
            r = torch.tensor(0.)
        rewards.append(r)
    # Only using torch for acceleration, no backpropagation happening here -> converting to numpy
    rewards = torch.stack(rewards).cpu().detach().numpy()
    return rewards