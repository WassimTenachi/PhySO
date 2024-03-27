import numpy as np
import torch
import warnings

# Internal imports
from physo.config.config0b import config0b
import physo.learn.monitoring as monitoring
from physo.task.fit import fit
import physo.physym.execute as exec
import physo.physym.dataset as Dataset
import physo

# DEFAULT RUN CONFIG TO USE (Using one with more steps for free constants optimization)
default_config = config0b

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
    multi_y_weights : list of len (n_realizations,) of np.array of shape (?,) of float
                       or array_like of (n_realizations,) of float
                       or float, optional
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
    spe_free_consts_init_val : dict of { str : float }
                               or dict of { str : array_like of shape (n_realizations,) of floats }
                               or None, optional
        Dictionary containing realization specific free constants names as keys (eg. 'k0', 'k1', 'k2') and
        corresponding float initial values to use during optimization process (eg. 1., 1., 1.). Realization
        specific initial values can be used by providing a vector of shape (n_realizations,) for each constant
        in lieu of a single float per constant. None will result in the usage of token.DEFAULT_FREE_CONST_INIT_VAL
        as initial values. None by default.

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
    # --- DEVICE ---
    DEVICE = 'cpu'

    # ------------------------------- HANDLING ARGUMENTS -------------------------------

    # --- DATA ---
    # Data checking and conversion to torch if necessary is now handled by Dataset class which is called by Batch class.
    # We use it here to infer n_dim (this will also run most other assertions unrelated to the library which is unknown
    # here and extra time) and sending data to device.
    dataset = Dataset.Dataset(multi_X=multi_X, multi_y=multi_y, multi_y_weights=multi_y_weights)
    # Getting number of input variables
    n_dim   = dataset.n_dim
    # Sending data to device and using sent data
    dataset.to(DEVICE)
    multi_X         = dataset.multi_X
    multi_y         = dataset.multi_y
    multi_y_weights = dataset.multi_y_weights

    # -- X_names --
    # Handling input variables names
    if X_names is None:
        # If None use x00, x01... names
        X_names = ["x%s"%(str(i).zfill(2)) for i in range(n_dim)]
    X_names = np.array(X_names)
    assert X_names.dtype.char == "U", "Input variables names should be strings."
    assert X_names.shape == (n_dim,), "There should be one input variable name per dimension in X."

    # -- X_units --
    # Handling input variables units
    if X_units is None:
        warnings.warn("No units given for input variables, assuming dimensionless units.")
        X_units = [[0,0,0] for _ in range(n_dim)]
    X_units = np.array(X_units).astype(float)
    assert X_units.shape[0] == n_dim, "There should be one input variable units per dimension in X."

    # --- y_name ---
    if y_name is None:
        y_name = "y"
    y_name = str(y_name)

    # --- y_units ---
    if y_units is None:
        warnings.warn("No units given for root variable, assuming dimensionless units.")
        y_units = [0,0,0]
    y_units = np.array(y_units).astype(float)
    assert len(y_units.shape) == 1, "y_units must be a 1D units vector"

    # --- n_fixed_consts ---
    if fixed_consts is not None:
        n_fixed_consts = len(fixed_consts)
    else:
        n_fixed_consts = 0
        fixed_consts = []
        warnings.warn("No information about fixed constants, not using any.")

    # --- fixed_consts_names ---
    # Rounding name to avoid using too long names (eg. for np.pi)
    fixed_consts_names = np.array([str(round(c, 4)) for c in fixed_consts])
    fixed_consts       = np.array(fixed_consts).astype(float)

    # --- fixed_consts_units ---
    if fixed_consts_units is None:
        warnings.warn("No units given for fixed constants, assuming dimensionless units.")
        fixed_consts_units = [[0,0,0] for _ in range(n_fixed_consts)]
    fixed_consts_units = np.array(fixed_consts_units).astype(float)
    assert fixed_consts_units.shape[0] == n_fixed_consts, "There should be one fixed constant units vector per fixed constant in fixed_consts_names"

    # --- n_class_free_consts ---
    if class_free_consts_names is not None:
        n_class_free_consts = len(class_free_consts_names)
    elif class_free_consts_units is not None:
        n_class_free_consts = len(class_free_consts_units)
    else:
        n_class_free_consts = 0
        warnings.warn("No information about free constants, not using any.")

    # --- class_free_consts_names ---
    if class_free_consts_names is None:
        # If None use c00, c01... names
        class_free_consts_names = ["c%s"%(str(i).zfill(2)) for i in range(n_class_free_consts)]
    # Convert to strings (this helps pass str assert in case array is empty)
    class_free_consts_names = np.array(class_free_consts_names).astype(str)
    assert class_free_consts_names.dtype.char == "U", "class_free_consts_names should be strings."
    assert class_free_consts_names.shape == (n_class_free_consts,), \
        "There should be one class free constant name per units in class_free_consts_units"

    # --- class_free_consts_units ---
    if class_free_consts_units is None:
        warnings.warn("No units given for free constants, assuming dimensionless units.")
        class_free_consts_units = [[0,0,0] for _ in range(n_class_free_consts)]
    class_free_consts_units = np.array(class_free_consts_units).astype(float)
    assert class_free_consts_units.shape[0] == n_class_free_consts, \
        "There should be one class free constant units vector per free constant in class_free_consts_names"

    # --- class_free_consts_init_val ---
    if class_free_consts_init_val is None:
        class_free_consts_init_val = np.ones(n_class_free_consts)
    class_free_consts_init_val = np.array(class_free_consts_init_val).astype(float)
    assert class_free_consts_init_val.shape[0] == n_class_free_consts, \
        "There should be one class free constant initial value per free constant in class_free_consts_names"

    # --- n_spe_free_consts ---
    if spe_free_consts_names is not None:
        n_spe_free_consts = len(spe_free_consts_names)
    elif spe_free_consts_units is not None:
        n_spe_free_consts = len(spe_free_consts_units)
    else:
        n_spe_free_consts = 0
        warnings.warn("No information about free constants, not using any.")

    # --- spe_free_consts_names ---
    if spe_free_consts_names is None:
        # If None use c00, c01... names
        spe_free_consts_names = ["k%s"%(str(i).zfill(2)) for i in range(n_spe_free_consts)]
    # Convert to strings (this helps pass str assert in case array is empty)
    spe_free_consts_names = np.array(spe_free_consts_names).astype(str)
    assert spe_free_consts_names.dtype.char == "U", "spe_free_consts_names should be strings."
    assert spe_free_consts_names.shape == (n_spe_free_consts,), \
        "There should be one spe free constant name per units in spe_free_consts_units"

    # --- spe_free_consts_units ---
    if spe_free_consts_units is None:
        warnings.warn("No units given for free constants, assuming dimensionless units.")
        spe_free_consts_units = [[0,0,0] for _ in range(n_spe_free_consts)]
    spe_free_consts_units = np.array(spe_free_consts_units).astype(float)
    assert spe_free_consts_units.shape[0] == n_spe_free_consts, \
        "There should be one spe free constant units vector per free constant in spe_free_consts_names"

    # --- spe_free_consts_init_val ---
    if spe_free_consts_init_val is None:
        spe_free_consts_init_val = np.ones(n_spe_free_consts)
    # Do not convert to array as user may use a mix of single floats and (n_realizations,) arrays
    assert len(spe_free_consts_init_val) == n_spe_free_consts, \
        "There should be one spe free constant initial value per free constant in spe_free_consts_names"

    # --- op_names ---
    if op_names is None:
        op_names = default_op_names

    # ------------------------------- WRAPPING -------------------------------

    # Converting fixed constants to torch and sending to device
    fixed_consts = torch.tensor(fixed_consts).to(DEVICE)

    # Embedding wrapping
    args_make_tokens = {
                    # operations
                    "op_names"             : op_names,
                    "use_protected_ops"    : use_protected_ops,
                    # input variables
                    "input_var_ids"        : {X_names[i]: i          for i in range(n_dim)},
                    "input_var_units"      : {X_names[i]: X_units[i] for i in range(n_dim)},
                    # constants
                    "constants"            : {fixed_consts_names[i] : fixed_consts[i]       for i in range(n_fixed_consts)},
                    "constants_units"      : {fixed_consts_names[i] : fixed_consts_units[i] for i in range(n_fixed_consts)},
                    # class_free_constants
                    "class_free_constants"          : {class_free_consts_names[i]                                 for i in range(n_class_free_consts)},
                    "class_free_constants_units"    : {class_free_consts_names[i] : class_free_consts_units   [i] for i in range(n_class_free_consts)},
                    "class_free_constants_init_val" : {class_free_consts_names[i] : class_free_consts_init_val[i] for i in range(n_class_free_consts)},
                    # spe_free_constants
                    "spe_free_constants"          : {spe_free_consts_names[i]                               for i in range(n_spe_free_consts)},
                    "spe_free_constants_units"    : {spe_free_consts_names[i] : spe_free_consts_units   [i] for i in range(n_spe_free_consts)},
                    "spe_free_constants_init_val" : {spe_free_consts_names[i] : spe_free_consts_init_val[i] for i in range(n_spe_free_consts)},
                        }

    library_config = {"args_make_tokens"  : args_make_tokens,
                      "superparent_units" : y_units,
                      "superparent_name"  : y_name,
                    }

    # Monitoring
    run_logger     = get_run_logger()
    run_visualiser = get_run_visualiser()
    # Updating config
    run_config.update({
        "library_config"       : library_config,
        "run_logger"           : run_logger,
        "run_visualiser"       : run_visualiser,
    })
    # Update reward_config
    run_config["reward_config"].update({
        # with parallel config
        "parallel_mode" : parallel_mode,
        "n_cpus"        : n_cpus,
        })
    #  Updating reward config for parallel mode
    reward_config = run_config["reward_config"]
    run_config["learning_config"]["rewards_computer"] = physo.physym.reward.make_RewardsComputer(**reward_config)

    # Number of epochs
    if epochs is not None:
        run_config["learning_config"]["n_epochs"] = epochs

    # Show progress bar
    exec.SHOW_PROGRESS_BAR = True

    # ------------------------------- RUN -------------------------------

    print("SR task started...")
    rewards, candidates = fit (multi_X = multi_X,
                               multi_y = multi_y,
                               multi_y_weights = multi_y_weights,
                               run_config      = run_config,
                               stop_reward         = stop_reward,
                               stop_after_n_epochs = default_stop_after_n_epochs,
                               max_n_evaluations   = max_n_evaluations,
                               )

    # ------------------------------- RESULTS -------------------------------

    pareto_front_complexities, pareto_front_programs, pareto_front_r, pareto_front_rmse = run_logger.get_pareto_front()
    best_expression = pareto_front_programs[-1]

    return best_expression, run_logger