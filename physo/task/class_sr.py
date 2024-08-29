# Internal imports
from physo.config.config0b import config0b
from physo.task.fit import fit
import physo.task.args_handler as args_handler
import physo.physym.execute as exec

# DEFAULT RUN CONFIG TO USE (Using one with more steps for free constants optimization)
default_config = config0b

def ClassSR(multi_X, multi_y, multi_y_weights=1.,
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
            class_free_consts_names    = None,
            class_free_consts_units    = None,
            class_free_consts_init_val = None,
            # Spe Free constants
            spe_free_consts_names    = None,
            spe_free_consts_units    = None,
            spe_free_consts_init_val = None,
            # Operations to use
            op_names = None,
            use_protected_ops = True,
            # Stopping
            stop_reward = 1.,
            max_n_evaluations = None,
            stop_after_n_epochs = args_handler.default_stop_after_n_epochs,
            epochs = None,
            # Candidate wrapper
            candidate_wrapper = None,
            # Default run config to use
            run_config = None,
            # Default run monitoring
            get_run_logger     = None,
            get_run_visualiser = None,
            # Parallel mode
            parallel_mode = True,
            n_cpus        = None,
            device        = 'cpu',
            ):
    """
    Runs a class symbolic regression task ie. searching for a single functional form fitting multiple datasets
    allowing each dataset to have its own realization specific free constant values (spe_free_consts) but using
    the same class free constants (class_free_consts) for all datasets.
    (Wrapper around physo.task.fit)

    Parameters
    ----------

    multi_X : list of len (n_realizations,) of np.array of shape (n_dim, ?,) of float
        List of X (one per realization). With X being values of the input variables of the problem with n_dim = nb
        of input variables.
    multi_y :  list of len (n_realizations,) of np.array of shape (?,) of float
        List of y (one per realization). With y being values of the target symbolic function on input variables
        contained in X.
    multi_y_weights : list of len (n_realizations,) of np.array of shape (?,) of float or array_like of (n_realizations,) of float or float, optional
        List of y_weights (one per realization). With y_weights being weights to apply to y data.
        Or list of weights one per entire realization.
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

    stop_reward : float (optional)
        Early stops if stop_reward is reached by a program (= 1 by default), use stop_reward = (1-1e-5) when using free
        constants.
    max_n_evaluations : int or None (optional)
        Maximum number of unique expression evaluations allowed (for benchmarking purposes). Immediately terminates
        the symbolic regression task if the limit is about to be reached. The parameter max_n_evaluations is distinct
        from batch_size * n_epochs because batch_size * n_epochs sets the number of expressions generated but a lot of
        these are not evaluated because they have inconsistent units.
    stop_after_n_epochs : int or None (optional)
        Number of additional epochs to do after early stop condition is reached.
        Uses args_handler.default_stop_after_n_epochs by default.
    epochs : int or None (optional)
        Number of epochs to perform. By default, uses the number in the default config file.

    candidate_wrapper : callable or None (optional)
        Wrapper to apply to candidate program's output, candidate_wrapper taking func, X as arguments where func is
        a candidate program callable (taking X as arg). By default = None, no wrapper is applied (identity).
    run_config : dict or None (optional)
        Run configuration (by default uses physo.task.class_sr.default_config)
        See physo/config/ for examples of run configurations.
    get_run_logger : callable returning physo.learn.monitoring.RunLogger or None (optional)
        Run logger (by default uses physo.task.args_handler.get_default_run_logger)
    get_run_visualiser : callable returning physo.learn.monitoring.RunVisualiser or None (optional)
        Run visualiser (by default uses physo.task.args_handler.get_default_run_visualiser)

    parallel_mode : bool (optional)
        Parallel execution if True, execution in a loop else. True by default. Overrides parameter in run_config.
    n_cpus : int or None (optional)
        Number of CPUs to use when running in parallel mode. Uses max nb. of CPUs by default.
        Overrides parameter in run_config.
    device : str (optional)
        Device to use for computations (eg. 'cpu', 'cuda'). 'cpu' by default.

    Returns
    -------
    best_expression, run_logger : physo.physym.program.Program, physo.learn.monitoring.RunLogger
        Best analytical expression found and run logger.
    """

    # Default run config to use
    if run_config is None:
        run_config = default_config
    if get_run_logger is None:
        get_run_logger = args_handler.get_default_run_logger
    if get_run_visualiser is None:
        get_run_visualiser = args_handler.get_default_run_visualiser

    # ------------------------------- HANDLING ARGUMENTS -------------------------------
    # Transmitting all arguments to be handled
    # Ie. all except:
    #   stop_reward
    #   max_n_evaluations
    #   stop_after_n_epochs

    handled_args = args_handler.check_args_and_build_run_config(
                                    multi_X         = multi_X,
                                    multi_y         = multi_y,
                                    multi_y_weights = multi_y_weights,
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
                                    op_names          = op_names,
                                    use_protected_ops = use_protected_ops,
                                    # Stopping
                                    epochs = epochs,
                                    # Candidate wrapper
                                    candidate_wrapper = candidate_wrapper,
                                    # Default run config to use
                                    run_config = run_config,
                                    # Default run monitoring
                                    get_run_logger     = get_run_logger,
                                    get_run_visualiser = get_run_visualiser,
                                    # Parallel mode
                                    parallel_mode = parallel_mode,
                                    n_cpus        = n_cpus,
                                    device        = device,
                            )

    multi_X         = handled_args['multi_X']
    multi_y         = handled_args['multi_y']
    multi_y_weights = handled_args['multi_y_weights']
    run_config      = handled_args['run_config']

    # ------------------------------- MISC -------------------------------

    # Show progress bar
    exec.SHOW_PROGRESS_BAR = True

    # ------------------------------- RUN -------------------------------

    print("SR task started...")
    rewards, candidates = fit (multi_X         = multi_X,
                               multi_y         = multi_y,
                               multi_y_weights = multi_y_weights,
                               candidate_wrapper = candidate_wrapper,
                               run_config        = run_config,
                               stop_reward         = stop_reward,
                               stop_after_n_epochs = stop_after_n_epochs,
                               max_n_evaluations   = max_n_evaluations,
                               )

    # ------------------------------- RESULTS -------------------------------

    run_logger = run_config["run_logger"]

    pareto_front_complexities, pareto_front_programs, pareto_front_r, pareto_front_rmse = run_logger.get_pareto_front()
    best_expression = pareto_front_programs[-1]

    return best_expression, run_logger