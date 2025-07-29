import numpy as np

# Internal imports
import physo.task.args_handler as args_handler
import physo.task.sr as sr
from physo.physym import library as Lib
from physo.physym import prior as Prior
import physo.physym.free_const as free_const
from physo.physym import vect_programs as VProg

def get_library(
                # X
                X_names = ["x1", "x2"],
                X_units = None,
                # y
                y_name = "y",
                y_units = None,
                # Fixed constants
                fixed_consts       = [1.],
                fixed_consts_units = None,
                # Class free constants
                class_free_consts_names    = ["c0", "c1"],
                class_free_consts_units    = None,
                class_free_consts_init_val = None,
                # Spe Free constants
                spe_free_consts_names    = None,
                spe_free_consts_units    = None,
                spe_free_consts_init_val = None,
                # Operations to use
                op_names          = args_handler.default_op_names,
                use_protected_ops = True,
                # Number of realizations
                n_realizations = 1,
                # Device to use
                device = "cpu",
        ):
    """
    X_names : array_like of shape (n_dim,) of str or None (optional)
        Names of input variables (for display purposes).
    X_units : array_like of shape (n_dim, n_units) of float or None (optional)
        Units vector for each input variables (n_units <= 7). By default, assumes dimensionless.

    y_name : str or None (optional)
        Name of the root variable (for display purposes).
    y_units : array_like of shape (n_units) of float or None (optional)
        Units vector for the root variable (n_units <= 7). By default, assumes dimensionless.

    fixed_consts : array_like of shape (?,) of float or None (optional)
        Values of choosable fixed constants. By default, no fixed constants are used.
    fixed_consts_units : array_like of shape (?, n_units) of float or None (optional)
        Units vector for each fixed constant (n_units <= 7). By default, assumes dimensionless.

    class_free_consts_names : array_like of shape (?,) of str or None (optional)
        Names of free constants (for display purposes).
    class_free_consts_units : array_like of shape (?, n_units) of float or None (optional)
        Units vector for each free constant (n_units <= 7). By default, assumes dimensionless.
    class_free_consts_init_val : dict of { str : float } or None (optional)
        Dictionary containing free constants names as keys (eg. 'c0', 'c1', 'c2') and corresponding float initial
        values to use during optimization process (eg. 1., 1., 1.). None will result in the usage of
        token.DEFAULT_FREE_CONST_INIT_VAL as initial values. None by default.

    spe_free_consts_names : array_like of shape (?,) of str or None (optional)
        Names of free constants (for display purposes).
    spe_free_consts_units : array_like of shape (?, n_units) of float or None (optional)
        Units vector for each free constant (n_units <= 7). By default, assumes dimensionless.
    spe_free_consts_init_val : dict of { str : float } or dict of { str : array_like of shape (n_realizations,) of floats } or None, optional
        Dictionary containing realization specific free constants names as keys (eg. 'k0', 'k1', 'k2') and
        corresponding float initial values to use during optimization process (eg. 1., 1., 1.). Realization
        specific initial values can be used by providing a vector of shape (n_realizations,) for each constant
        in lieu of a single float per constant. None will result in the usage of token.DEFAULT_FREE_CONST_INIT_VAL
        as initial values. None by default.

    op_names : array_like of shape (?) of str or None (optional)
        Names of choosable symbolic operations (see physo.physym.functions for a list of available operations).
        By default, uses operations listed in physo.task.args_handler.default_op_names.
    use_protected_ops : bool (optional)
        If True, uses protected operations (e.g. division by zero is avoided). True by default. (see
        physo.physym.functions for a list of available protected operations).

    n_realizations : int or None, optional
        Number of realizations for each program, ie. number of datasets each program has to fit.
        Dataset specific free constants will have different values different for each realization.
        Uses 1 by default (if None).
    device : str, optional
        Device on which free constants will be stored and computations will be performed.

    Returns
    -------
    library : physym.library.Library
            Library of choosable tokens.

    """
    # Build the library configuration and checks arguments
    library_config = args_handler.check_library_args(

                # X
                X_names = X_names,
                X_units = X_units,
                # y
                y_name  = y_name,
                y_units = y_units,
                # Fixed constants
                fixed_consts       = fixed_consts,
                fixed_consts_units = fixed_consts_units,
                # Class free constants
                class_free_consts_names    = class_free_consts_names,
                class_free_consts_units    = class_free_consts_units,
                class_free_consts_init_val = class_free_consts_init_val,
                # Spe Free constants
                spe_free_consts_names    = spe_free_consts_names,
                spe_free_consts_units    = spe_free_consts_units,
                spe_free_consts_init_val = spe_free_consts_init_val,
                # Operations to use
                op_names = op_names,
                use_protected_ops = use_protected_ops,
                # Number of realizations
                n_realizations = n_realizations,
                # Device to use
                device = device,
                )

    library = Lib.Library(**library_config)

    return library