import torch
import numpy as np

# ------------------------------------------------------------------------------------------------------
# ---------------------------------------- FREE CONSTANTS TABLE ----------------------------------------
# ------------------------------------------------------------------------------------------------------

class FreeConstantsTable:
    """
    Contains free constants values.
    """
    def __init__(self, batch_size, library):
        # Library
        self.library = library

        # Shape
        self.batch_size = batch_size
        self.n_free_const = self.library.n_free_const # Number of free constants
        self.shape = (self.batch_size, self.n_free_const,)

        # Initial values
        self.init_val = self.library.free_constants_init_val                           # (n_free_const,) of float
        # Free constants values for each program # as torch tensor for fast computation (sent to device in Batch)
        values_array = np.tile(self.init_val, reps=(self.batch_size, 1))               # (batch_size, n_free_const,) of float
        # If free_constants_init_val already contains torch tensors, they are converted by np.tile (if on same device)
        self.values = torch.tensor(values_array)                                       # (batch_size, n_free_const,) of float
        self.values = self.values
        # mask : is free constant optimized
        self.is_opti    = np.full(shape=self.batch_size, fill_value=False, dtype=bool)  # (batch_size,) of bool
        # Number of epochs necessary to optimize.py free constant
        self.opti_steps = np.full(shape=self.batch_size, fill_value=False, dtype=int )  # (batch_size,) of int

    def __repr__(self):
        s = "FreeConstantsTable for %s : %s"%(self.library.free_constants_tokens, self.shape,)
        return s

# ------------------------------------------------------------------------------------------------------
# ------------------------------------ FREE CONSTANTS OPTIMIZATION -------------------------------------
# ------------------------------------------------------------------------------------------------------

# ------------ Loss to use for free constant optimization ------------


def MSE_loss (func, params, y_target):
    """
    Loss for free constant optimization.
    Parameters
    ----------
    func : callable
        Function which's constants should be optimized taking params as argument.
    params : torch.tensor of shape (n_free_const,)
        Free constants to optimize.py.
    y_target : torch.tensor of shape (?,)
        Target output of function.
    Returns
    -------
    loss : float
        Value of error to be minimized.
    """
    loss = torch.mean((func(params) - y_target)**2)
    return loss


LOSSES = {
    "MSE": MSE_loss
}

# ------------ Optimizer for free constant optimization ------------

# --- LBFGS ---

DEFAULT_LBFGS_OPTI_ARGS = {
    'n_steps' : 30,
    'tol'     : 1e-6,
    'lbfgs_func_args' : {
        'max_iter'       : 4,
        'line_search_fn' : "strong_wolfe",
                         },
}

def LBFGS_optimizer (params, f, n_steps=10, tol=1e-6, lbfgs_func_args={}):
    """
    Params optimizer (wrapper around torch.optim.LBFGS).
    See: https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html
    Parameters
    ----------
    params : torch.tensor of shape (n_free_const,)
        Parameters to optimize.py.
    f : callable
        Function to minimize, taking params as argument.
    n_steps : int
        Number of optimization steps.
    tol : float
        Error tolerance, early stops if error < tol.
    lbfgs_func_args : dict
        Arguments to pass to torch.optim.LBFGS
    Returns
    -------
    history : numpy.array of shape (?,)
        Loss history (? <= n_steps).
    """
    params.requires_grad = True

    lbfgs = torch.optim.LBFGS([params], **lbfgs_func_args)

    def closure():
        lbfgs.zero_grad()
        objective = f(params)
        objective.backward()
        return objective

    history = []
    for i in range(n_steps):
        history.append(f(params).item())
        lbfgs.step(closure)
        if history[-1] < tol:
            break

    history = np.array(history)

    return history

# --- DICTS ---

OPTIMIZERS = {
    "LBFGS" : LBFGS_optimizer
}

OPTIMIZERS_DEFAULT_ARGS = {
    "LBFGS" : DEFAULT_LBFGS_OPTI_ARGS
}

# ------------ WRAPPER ------------

DEFAULT_OPTI_ARGS = {
    'loss'   : "MSE",
    'method' : 'LBFGS',
    'method_args': OPTIMIZERS_DEFAULT_ARGS['LBFGS'],
}

def optimize_free_const (func,
                         params,
                         y_target,
                         loss        = "MSE",
                         method      = "LBFGS",
                         method_args = None):
    """
    Optimizes free constants params so that func output matches y_target.
    Parameters
    ----------
    func : callable
        Function which's constants should be optimized taking params as argument.
    params : torch.tensor of shape (n_free_const,)
        Free constants to optimize.py.
    y_target : torch.tensor of shape (?,)
        Target output of function.
    """

    # Getting loss
    err_msg = "Loss should be a string contained in the dict of available const optimization losses, see " \
              "free_const.LOSSES : %s"%(LOSSES)
    assert isinstance(loss, str), err_msg
    assert loss in LOSSES, err_msg
    loss = LOSSES[loss]

    # Getting optimizer
    err_msg = "Optimizer should be a string contained in the dict of available const optimizers, see " \
              "free_const.OPTIMIZERS: %s"%(OPTIMIZERS)
    assert isinstance(method, str), err_msg
    assert method in OPTIMIZERS, err_msg
    optimizer = OPTIMIZERS[method]

    # Getting optimizer_args
    if method_args is None:
        err_msg = "Optimizer args should be given or defined in free_const.OPTIMIZERS_DEFAULT_ARGS: %s" % (OPTIMIZERS_DEFAULT_ARGS)
        assert method in OPTIMIZERS_DEFAULT_ARGS, err_msg
        optimizer_args = OPTIMIZERS_DEFAULT_ARGS[method]
    else:
        optimizer_args = method_args

    # Loss wrapper : loss_params
    loss_params = lambda params : loss(func = func, params = params, y_target = y_target)

    # Running optimizer
    history = optimizer (params = params, f = loss_params, **optimizer_args)

    # # Making free const positive values only #abs_free_const
    # params = torch.abs(params)

    return history
