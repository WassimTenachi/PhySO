import torch
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------------------
# ---------------------------------------- FREE CONSTANTS TABLE ----------------------------------------
# ------------------------------------------------------------------------------------------------------

class FreeConstantsTable:
    """
    Contains free constants values.

    Attributes
    ----------
    batch_size : int
        Number of programs in batch.
    library : library.Library
        Library of tokens that can appear in programs.
    n_realizations : int
        Number of realizations for each program, ie. number of datasets each program has to fit.
        Dataset specific free constants will have different values different for each realization.

    is_opti : torch.tensor of shape (batch_size,) of bool
        Is set of free constants optimized.
    opti_steps : torch.tensor of shape (batch_size,) of int
        Number of iterations necessary to optimize free constant.
    class_values : torch.tensor of shape (batch_size, n_class_free_const,)
        Free constants values for each program.
    spe_values : torch.tensor of shape (batch_size, n_spe_free_const, n_realizations,)
        Realization specific free constants values for each program.
    """
    def __init__(self, batch_size, library, n_realizations=1):
        """"
        Parameters
        ----------
        batch_size : int
            Number of programs in batch.
        library : library.Library
            Library of tokens that can appear in programs.
        n_realizations : int
            Number of realizations for each program, ie. number of datasets each program has to fit.
            Dataset specific free constants will have different values different for each realization.
        """

        self.library = library

        # Shape
        self.batch_size   = batch_size

        # Optimization logs
        # Using torch tensors for is_opti and opti_steps as it is the only way to have FreeConstantsTable be able to
        # modify its own is_opti and opti_steps attributes in multiprocessing.
        # mask : Is set of free constants optimized
        self.is_opti    = torch.full(size=(self.batch_size,), fill_value=False, dtype=bool)    # (batch_size,) of bool
        # Number of iterations necessary to optimize free constant
        self.opti_steps = torch.full(size=(self.batch_size,), fill_value=False, dtype=int )    # (batch_size,) of int

        # Class free constants
        self.n_class_free_const = self.library.n_class_free_const  # Number of class free constants
        self.reset_class_values()

        # Spe free constants
        self.n_spe_free_const = self.library.n_spe_free_const  # Number of realization specific free constants
        self.n_realizations = n_realizations
        self.reset_spe_values()

        # Shape
        self.shape = (self.batch_size, self.n_class_free_const,), (self.batch_size, self.n_spe_free_const, self.n_realizations,)
        self.n_free_const_tokens = library.n_free_const  # Total number of free constant tokens
        self.n_free_const_values = self.n_class_free_const + self.n_spe_free_const*self.n_realizations # Total number of adjustable values (for each program in batch)

        return None

    def reset_class_values (self):
        """
        Reset class free constants values to initial values.
        """
        init_val = self.library.class_free_constants_init_val                         # (n_class_free_const,) of float
        # Free constants values for each program as torch tensor for fast computation (sent to device in batch.py)
        # If init_val already contains torch tensors, they are converted by np.tile (if on same device)
        values_array = np.tile(init_val, reps=(self.batch_size, 1))                   # (batch_size, n_class_free_const,) of float
        self.class_values = torch.tensor(values_array)                                # (batch_size, n_class_free_const,) of float
        self.class_values = self.class_values                                         # (batch_size, n_class_free_const,) of float
        return None

    def reset_spe_values (self):
        """
        Reset spe free constants values to initial values.
        """
        # Check and pad init_val if necessary to match n_realizations (for single float init val)
        self.library.check_and_pad_spe_free_const_init_val (self.n_realizations)
        init_val = self.library.spe_free_constants_init_val                           # (n_spe_free_const, n_realizations,) of float
        # Free constants values for each program as torch tensor for fast computation (sent to device in batch.py)
        # If init_val already contains torch tensors, they are converted by np.tile (if on same device)
        values_array = np.tile(init_val, reps=(self.batch_size, 1, 1))                # (batch_size, n_spe_free_const, n_realizations,) of float
        self.spe_values = torch.tensor(values_array)                                  # (batch_size, n_spe_free_const, n_realizations,) of float
        return None

    def __repr__(self):
        s = "FreeConstantsTable"
        s_class = " -> Class consts (%s) : %s"  %(self.library.class_free_constants_names, (self.batch_size, self.n_class_free_const,))
        s_spe   = " -> Spe consts   (%s) : %s"  %(self.library.spe_free_constants_names  , (self.batch_size, self.n_spe_free_const, self.n_realizations,))
        s += "\n" + s_class + "\n" + s_spe
        return s

    def detach (self):
        """
        Detach values from computation graph and copies is_opti and opti_steps to detach them from higher level table
        if there is one.
        """
        self.class_values = self.class_values.clone().detach()
        self.spe_values   = self.spe_values  .clone().detach()
        self.is_opti      = self.is_opti     .clone().detach()
        self.opti_steps   = self.opti_steps  .clone().detach()
        return self

    def to (self, device):
        """
        Send all values to device.
        """
        self.class_values = self.class_values.to(device)
        self.spe_values   = self.spe_values  .to(device)
        self.is_opti      = self.is_opti     .to(device)
        self.opti_steps   = self.opti_steps  .to(device)
        return self

    def cpu (self):
        """
        Send all values to cpu.
        """
        self.to("cpu")
        return self

    def get_const_of_prog (self, prog_idx):
        """
        Return a FreeConstantsTable object with values for a single program (batch_size=1).
        """
        res = FreeConstantsTable (batch_size=1, library=self.library, n_realizations=self.n_realizations)
        # Returning arrays of (1,...) to have a reference to the original arrays
        res.class_values = self.class_values[prog_idx:prog_idx+1,:]             # (1, n_class_free_const,)
        res.spe_values   = self.spe_values  [prog_idx:prog_idx+1,:,:]           # (1, n_spe_free_const, n_realizations,)
        res.is_opti      = self.is_opti     [prog_idx:prog_idx+1]               # (1,) of bool
        res.opti_steps   = self.opti_steps  [prog_idx:prog_idx+1]               # (1,) of int
        return res

    def flatten_like_data (self, n_samples_per_dataset):
        """
        Flattens free constants values to match flattened datasets.
        This is useful for computing together class free consts and spe free consts and all datasets at the same time
        during a single program execution.
        Parameters
        ----------
        n_samples_per_dataset : array_like of shape (n_realizations,) of int
            Number of samples for each dataset (eg. [90, 100, 110] for 3 datasets).
        Returns
        -------
        class_const_flatten, spe_const_flatten : torch.tensor of shape (batch_size, n_class_free_const, n_all_samples), torch.tensor of shape (batch_size, n_spe_free_const, n_all_samples)
            Flattened free constants values.
        """
        n_all_samples = n_samples_per_dataset.sum()  # Total number of samples

        # ---- Handling spe free constants ----
        # (Spe free const are different for each dataset/realization)
        detected_device = self.spe_values.device
        n_samples_per_dataset_torch = torch.tensor(n_samples_per_dataset).to(detected_device)
        spe_const_flatten = self.spe_values.repeat_interleave(n_samples_per_dataset_torch, dim=-1)                 # (batch_size, n_spe_free_const, n_all_samples)
        #Alternative way to do the same thing (new way is now 30% faster).
        #flattened = []                                                                                            # (n_realizations,) of (batch_size, n_spe_free_const, [n_samples depends on dataset],)
        #for i_real in range(self.n_realizations):
        #    flattened.append(
        #        torch.tile(self.spe_values[:, :, i_real], (n_samples_per_dataset[i_real], 1, 1)).permute(1, 2, 0) # (batch_size, n_spe_free_const, [n_samples depends on dataset],)
        #    )
        #spe_const_flatten = torch.cat(flattened, axis=-1)                                                         # (batch_size, n_spe_free_const, n_all_samples)


        # ---- Handling class free constants ----
        # (Class free const are the same for all datasets/realizations)
        class_const_flatten = self.class_values[:,:,np.newaxis].repeat((1,1,n_all_samples))                        # (batch_size, n_class_free_const, n_all_samples)
        # Alternative ways to do the same thing (new way is not faster)
        # class_const_flatten = torch.tile(self.class_values, (n_all_samples, 1,1)).permute(1, 2, 0)
        # class_const_flatten = self.class_values.repeat((n_all_samples, 1, 1)).permute(1, 2, 0)
        # class_const_flatten = self.class_values.repeat((n_all_samples, 1, 1)).transpose(0, 1).transpose(1, 2)
        # class_const_flatten = self.class_values[:,:,np.newaxis].repeat((1,1,n_all_samples))
        # class_const_flatten = self.class_values.unsqueeze(-1).repeat((1,1,n_all_samples))
        return class_const_flatten, spe_const_flatten

    def df (self):
        """
        Return a pandas dataframe with free constants values.
        Returns
        -------
        df : pandas.DataFrame
            Dataframe with free constants values of shape (batch_size, n_class_free_const + n_spe_free_const*n_realizations).
        """

        # Class free const df
        values = self.class_values.cpu().detach().numpy()                  # (batch_size, n_class_free_const)
        names  = self.library.class_free_constants_names                   # (n_class_free_const,)
        df_class = pd.DataFrame(values, columns=names)                     # (batch_size, n_class_free_const)

        # Spe free const df
        values = self.spe_values.cpu().detach().numpy()
        values = np.reshape(values, (self.batch_size, self.n_spe_free_const*self.n_realizations))            # (batch_size, n_spe_free_const*n_realizations)
        names  = self.library.spe_free_constants_names                                                       # (n_spe_free_const,)
        # Adding realization number to names (eg. "k0_0", "k0_1", "k0_2", ..., "k1_0", "k1_1", "k1_2", ...)
        names  = [name + "_%s" % (i) for name in names for i in range(self.n_realizations)]                  # (n_spe_free_const*n_realizations,)
        df_spe = pd.DataFrame(values, columns=names)                                                         # (batch_size, n_spe_free_const*n_realizations)

        # Concatenating
        df = pd.concat([df_class, df_spe], axis=1)                                                           # (batch_size, n_class_free_const + n_spe_free_const*n_realizations)

        return df

# ------------------------------------------------------------------------------------------------------
# ------------------------------------ FREE CONSTANTS OPTIMIZATION -------------------------------------
# ------------------------------------------------------------------------------------------------------

# ------------ Loss to use for free constant optimization ------------


def MSE_loss (func, params, y_target, y_weights = 1.):
    """
    Loss for free constant optimization.
    Parameters
    ----------
    func : callable
        Function which's constants should be optimized taking params as argument.
    params : list of torch.tensor
        Free constants to optimize.
    y_target : torch.tensor of shape (?,)
        Target output of function.
    y_weights : torch.tensor of shape (?,) of float, optional
        Weights for each data point. By default, no weights are used.
    Returns
    -------
    loss : float
        Value of error to be minimized.
    """
    err = y_weights * (func(params) - y_target)**2
    loss = torch.mean(err)
    return loss


LOSSES = {
    "MSE": MSE_loss
}

# ------------ Optimizer for free constant optimization ------------

# --- LBFGS ---

DEFAULT_LBFGS_OPTI_ARGS = {
    'n_steps' : 30,
    'tol'     : 1e-10, # Most of SR search time will be spent on incorrect programs, so we can afford a very low tolerance
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
    params : list of torch.tensor
        Free constants to optimize.
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

    if type(params) == list:
        for p in params:
            p.requires_grad = True
        params_topass = params
    else:
        params.requires_grad = True
        params_topass = [params,]

    lbfgs = torch.optim.LBFGS(params_topass, **lbfgs_func_args)

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
                         y_weights   = 1.,
                         loss        = "MSE",
                         method      = "LBFGS",
                         method_args = None):
    """
    Optimizes free constants params so that func output matches y_target.
    Parameters
    ----------
    func : callable
        Function which's constants should be optimized taking params as argument.
    params : list of torch.tensor
        Free constants to optimize.
    y_target : torch.tensor of shape (?,)
        Target output of function.
    y_weights : torch.tensor of shape (?,) of float, optional
        Weights for each data point. By default, no weights are used.
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
    loss_params = lambda params : loss(func = func, params = params, y_target = y_target, y_weights = y_weights)

    # Running optimizer
    history = optimizer (params = params, f = loss_params, **optimizer_args)

    return history
