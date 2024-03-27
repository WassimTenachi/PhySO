import numpy as np
import torch
import warnings

# Internal imports
from physo.config.config0 import config0
import physo.learn.monitoring as monitoring
import physo

# DEFAULT RUN CONFIG TO USE
default_config = config0

# DEFAULT MONITORING CONFIG TO USE
get_default_run_logger = lambda : monitoring.RunLogger(
                                      save_path = 'SR.log',
                                      do_save   = True)
get_default_run_visualiser = lambda : monitoring.RunVisualiser (
                                           epoch_refresh_rate = 1,
                                           save_path = 'SR_curves.png',
                                           do_show   = False,
                                           do_prints = True,
                                           do_save   = True, )

# DEFAULT ALLOWED OPERATIONS
default_op_names = ["mul", "add", "sub", "div", "inv", "n2", "sqrt", "neg", "exp", "log", "sin", "cos"]
default_stop_after_n_epochs = 5

def SR(X, y, y_weights=1.,
            # X
            X_names = None,
            X_units = None,
            # y
            y_name  = None,
            y_units = None,
            # Fixed constants
            fixed_consts = None,
            fixed_consts_units = None,
            # Class free constants
            free_consts_names    = None,
            free_consts_units    = None,
            free_consts_init_val = None,
            # Operations to use
            op_names = None,
            use_protected_ops = True,
            # Stopping
            stop_reward = 1.,
            max_n_evaluations = None,
            epochs = None,
            # Default run config to use
            run_config = default_config,
            # Default run monitoring
            get_run_logger     = get_default_run_logger,
            get_run_visualiser = get_default_run_visualiser,
            # Parallel mode
            parallel_mode = True,
            n_cpus        = None,
            ):
    """
    Runs a symbolic regression task.
    (Wrapper around physo.task.fit)

    Parameters
    ----------

    X : numpy.array of shape (n_dim, ?,) of float
        Values of the input variables of the problem with n_dim = nb of input variables.
    y : numpy.array of shape (?,) of float
        Values of the target symbolic function to recover when applied on input variables contained in X.
    y_weights : np.array of shape (?,) of float
                or float, optional
        Weight values to apply to y data.
        Or single float to apply to all (for default value = 1.).

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

    free_consts_names : array_like of shape (?,) of str or None (optional)
        Names of free constants (for display purposes).
    free_consts_units : array_like of shape (?, n_units) of float or None (optional)
        Units vector for each free constant (n_units <= 7). By default, assumes dimensionless.
    free_consts_init_val : dict of { str : float } or None (optional)
        Dictionary containing free constants names as keys (eg. 'c0', 'c1', 'c2') and corresponding float initial
        values to use during optimization process (eg. 1., 1., 1.). None will result in the usage of
        token.DEFAULT_FREE_CONST_INIT_VAL as initial values. None by default.

    op_names : array_like of shape (?) of str or None (optional)
        Names of choosable symbolic operations (see physo.physym.functions for a list of available operations).
        By default, uses operations listed in physo.task.sr.default_op_names.
    use_protected_ops : bool (optional)
        If True, uses protected operations (e.g. division by zero is avoided). True by default.
         (see physo.physym.functions for a list of available protected operations).

    stop_reward : float (optional)
        Early stops if stop_reward is reached by a program (= 1 by default), use stop_reward = (1-1e-5) when using free
        constants.
    max_n_evaluations : int or None (optional)
        Maximum number of unique expression evaluations allowed (for benchmarking purposes). Immediately terminates
        the symbolic regression task if the limit is about to be reached. The parameter max_n_evaluations is distinct
        from batch_size * n_epochs because batch_size * n_epochs sets the number of expressions generated but a lot of
        these are not evaluated because they have inconsistent units.

    epochs : int or None (optional)
        Number of epochs to perform. By default, uses the number in the default config file.
    run_config : dict (optional)
        Run configuration (by default uses physo.task.sr.default_config)
    get_run_logger : callable returning physo.learn.monitoring.RunLogger (optional)
        Run logger (by default uses physo.task.sr.get_default_run_logger)
    get_run_visualiser : callable returning physo.learn.monitoring.RunVisualiser (optional)
        Run visualiser (by default uses physo.task.sr.get_default_run_visualiser)

    parallel_mode : bool (optional)
        Parallel execution if True, execution in a loop else. True by default. Overides parameter in run_config.
    n_cpus : int or None (optional)
        Number of CPUs to use when running in parallel mode. Uses max nb. of CPUs by default.
        Overrides parameter in run_config.

    Returns
    -------
    best_expression, run_logger : physo.physym.program.Program, physo.learn.monitoring.RunLogger
        Best analytical expression found and run logger.
    """
    # Transmitting arguments to ClassSR as SR is just a sub-case of ClassSR where there is only one realization
    # and no dataset spe free constants.
    best_expression, run_logger = physo.ClassSR(
                # Wrapping to make a 1 realization dataset
                multi_X = [X,],
                multi_y = [y,],
                multi_y_weights = [y_weights,],
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
                class_free_consts_names    = free_consts_names,
                class_free_consts_units    = free_consts_units ,
                class_free_consts_init_val = free_consts_init_val,
                # Spe Free constants (not used in SR)
                spe_free_consts_names    = None,
                spe_free_consts_units    = None,
                spe_free_consts_init_val = None,
                # Operations to use
                op_names          = op_names,
                use_protected_ops = use_protected_ops,
                # Stopping
                stop_reward       = stop_reward,
                max_n_evaluations = max_n_evaluations,
                epochs            = epochs,
                # Default run config to use
                run_config = run_config,
                # Default run monitoring
                get_run_logger     = get_run_logger,
                get_run_visualiser = get_run_visualiser,
                # Parallel mode
                parallel_mode = parallel_mode,
                n_cpus        = n_cpus,
    )

    return best_expression, run_logger