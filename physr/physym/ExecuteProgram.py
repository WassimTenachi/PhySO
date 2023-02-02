import warnings

import numpy as np
import torch as torch

def ExecuteProgram (input_var_data, program_tokens):
    """
    Executes a symbolic function program.
    Parameters
    ----------
    X : torch.tensor of shape (n_dim, ?,) of float
        Values of the input variables of the problem with n_dim = nb of input variables.
    program_tokens : list of Token.Token
        Symbolic function program in reverse Polish notation order.
    Returns
    -------
    y : torch.tensor of shape (?,) of float
        Result of computation.
    """

    # Number of tokens in the program
    n_tokens = len(program_tokens)

    # Current stack of computed results
    curr_stack = []

    # De-stacking program (iterating from last token to first)
    start = n_tokens - 1
    for i in range (start, -1, -1):
        token = program_tokens[i]
        # Terminal token
        if token.arity == 0:
            if token.is_input_var:
                curr_stack.append(input_var_data[token.input_var_id])
            else:
                curr_stack.append(token.function())
        # Non-terminal token
        elif token.arity > 0:
            # Last pending elements are those needed for next computation (in reverse order)
            args = curr_stack[-token.arity:][::-1]
            res = token.function(*args)
            # Removing those pending elements as they were used
            curr_stack = curr_stack[:-token.arity]
            # Appending last result to stack
            curr_stack.append(res)
    y = curr_stack[0]
    return y


def ComputeInfixNotation (program_tokens):
    """
    Computes infix str representation of a program.
    (which is the usual way to note symbolic function: +34 (in polish notation) = 3+4 (in infix notation))
    Parameters
    ----------
    program_tokens : list of Token.Token
        List of tokens making up the program.
    Returns
    -------
    program_str : str
    """
    # Number of tokens in the program
    n_tokens = len(program_tokens)

    # Current stack of computed results
    curr_stack = []

    # De-stacking program (iterating from last token to first)
    start = n_tokens - 1
    for i in range (start, -1, -1):
        token = program_tokens[i]
        # Last pending elements are those needed for next computation (in reverse order)
        args = curr_stack[-token.arity:][::-1]
        if token.arity == 0:
            res = token.sympy_repr
        elif token.arity == 1:
            if token.is_power is True:
                pow = '{:g}'.format(token.power)  # without trailing zeros
                res = "((%s)**(%s))" % (args[0], pow)
            else:
                res = "%s(%s)" % (token.sympy_repr, args[0])
        elif token.arity == 2:
            res = "(%s%s%s)" % (args[0], token.sympy_repr, args[1])
        elif token.arity > 2 :
            args_str = ""
            for arg in args: args_str+="%s,"%arg
            args_str = args_str[:-1] # deleting last ","
            res = "%s(%s)" % (token.sympy_repr, args_str)
        if token.arity > 0:
            # Removing those pending elements as they were used
            curr_stack = curr_stack[:-token.arity]
        # Appending last result to stack
        curr_stack.append(res)
    return curr_stack[0]


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