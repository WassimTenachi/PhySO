import numpy as np
import torch
import warnings

# Internal imports
import physo.learn.monitoring as monitoring
import physo.physym.dataset as Dataset
import physo

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
default_stop_after_n_epochs = 10


def check_priors_config(priors_config, max_time_step):
    """
    Checks that the prior configurations are valid with respect to the max_time_step.
    Parameters
    ----------
    priors_config : list of tuples (str : dict)
        List of priors. List containing tuples with prior name as first item in couple (see prior.PRIORS_DICT for list
        of available priors) and additional arguments (besides library and programs) to be passed to priors as second
        item of couple, leave None for priors that do not require arguments.
    max_time_step : int
        Max number of tokens programs can contain.
    """
    # Asserting that max_time_step is >= HardLengthPrior's max_length
    for prior_config in priors_config:
        if prior_config[0] == "HardLengthPrior":
            assert max_time_step >= prior_config[1]["max_length"], \
                "max_time_step should be greater than or equal to HardLengthPrior's max_length."

def check_library_args(
        # X
        X_names,
        X_units,
        # y
        y_name,
        y_units,
        # Fixed constants
        fixed_consts,
        fixed_consts_units,
        # Class free constants
        class_free_consts_names,
        class_free_consts_units,
        class_free_consts_init_val,
        # Spe Free constants
        spe_free_consts_names,
        spe_free_consts_units,
        spe_free_consts_init_val,
        # Operations to use
        op_names,
        use_protected_ops,
        # Number of dimensions
        n_dim = None,
        # Number of realizations
        n_realizations = 1,
        # Device to use
        device = "cpu",
    ):

    # Number of dimensions
    if n_dim is None:
        if X_names is not None:
            n_dim = len(X_names)
        elif X_units is not None:
            n_dim = len(X_units)
        else:
            raise ValueError("n_dim should be given or X_names or X_units should be provided to infer it.")

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
        warnings.warn("No information about class free constants, not using any.")

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
        if n_class_free_consts > 0:
            warnings.warn("No units given for class free constants, assuming dimensionless units.")
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
        # Only warning if there are multiple realizations
        if n_realizations > 1:
            warnings.warn("No information about spe free constants, not using any.")

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
        if n_spe_free_consts > 0:
            warnings.warn("No units given for spe free constants, assuming dimensionless units.")
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

    # ------------------------------- WRAPPING LIBRARY -------------------------------

    # Converting fixed constants to torch and sending to device
    fixed_consts = torch.tensor(fixed_consts).to(device)

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
    return library_config

def check_args_and_build_run_config(multi_X, multi_y, multi_y_weights,
            # X
            X_names,
            X_units,
            # y
            y_name ,
            y_units,
            # Fixed constants
            fixed_consts,
            fixed_consts_units,
            # Class free constants
            class_free_consts_names   ,
            class_free_consts_units   ,
            class_free_consts_init_val,
            # Spe Free constants
            spe_free_consts_names   ,
            spe_free_consts_units   ,
            spe_free_consts_init_val,
            # Operations to use
            op_names,
            use_protected_ops,
            # Stopping
            epochs,
            # Candidate wrapper
            candidate_wrapper,
            # Default run config to use
            run_config,
            # Default run monitoring
            get_run_logger,
            get_run_visualiser,
            # Parallel mode
            parallel_mode,
            n_cpus,
            device,
    ):
    """
    Checks arguments of SR and ClassSR functions and builds run_config for physo.task.fit.
    """

    # ------------------------------- DATASETS -------------------------------

    # Data checking and conversion to torch if necessary is now handled by Dataset class which is called by Batch class.
    # We use it here to infer n_dim (this will also run most other assertions unrelated to the library which is unknown
    # here and extra time) and sending data to device.
    dataset = Dataset.Dataset(multi_X=multi_X, multi_y=multi_y, multi_y_weights=multi_y_weights)
    # Getting number of input variables
    n_dim   = dataset.n_dim
    # Getting number of realizations
    n_realizations = dataset.n_realizations
    # Sending data to device and using sent data
    dataset.to(device)
    multi_X         = dataset.multi_X
    multi_y         = dataset.multi_y
    multi_y_weights = dataset.multi_y_weights

    # ------------------------------- LIBRARY ARGS -------------------------------

    library_config = check_library_args(
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
            # Number of dimensions
            n_dim = n_dim,
            # Number of realizations
            n_realizations = n_realizations,
            # Device to use
            device = device,
            )

    # Updating config
    run_config.update({
        "library_config" : library_config,
    })

    # ------------------------------- MONITORING -------------------------------
    run_logger     = get_run_logger()
    run_visualiser = get_run_visualiser()

    # Updating config
    run_config.update({
        "run_logger"           : run_logger,
        "run_visualiser"       : run_visualiser,
    })

    # ------------------------------- PARALLEL CONFIG AND BUILDING RewardsComputer -------------------------------

    # Update reward_config
    run_config["reward_config"].update({
        # with parallel config
        "parallel_mode" : parallel_mode,
        "n_cpus"        : n_cpus,
        })
    #  Updating reward config for parallel mode
    reward_config = run_config["reward_config"]
    run_config["learning_config"]["rewards_computer"] = physo.physym.reward.make_RewardsComputer(**reward_config)

    # ------------------------------- EPOCHS -------------------------------

    # Number of epochs (using epochs args in run_config if it was given).
    if epochs is not None:
        run_config["learning_config"]["n_epochs"] = epochs

    # ------------------------------- MAX_TIME_STEP ASSERTIONS -------------------------------


    check_priors_config(priors_config = run_config["priors_config"],
                        max_time_step = run_config["learning_config"]["max_time_step"])

    # ------------------------------- LEARNING HYPERPARAMS ASSERTIONS -------------------------------
    # risk_factor should be a float >= 0 and <= 1
    risk_factor = run_config["learning_config"]["risk_factor"]
    try:
        risk_factor = float(risk_factor)
    except:
        raise ValueError("risk_factor should be castable to a float.")
    assert isinstance(risk_factor, float), "risk_factor should be a float."
    assert 0 <= risk_factor <= 1, "risk_factor should be >= 0 and <= 1."

    # gamma_decay should be a float
    gamma_decay = run_config["learning_config"]["gamma_decay"]
    try:
        gamma_decay = float(gamma_decay)
    except:
        raise ValueError("gamma_decay should be castable to a float.")
    assert isinstance(gamma_decay, float), "gamma_decay should be a float."

    # entropy_weight should be a float
    entropy_weight = run_config["learning_config"]["entropy_weight"]
    try:
        entropy_weight = float(entropy_weight)
    except:
        raise ValueError("entropy_weight should be castable to a float.")
    assert isinstance(entropy_weight, float), "entropy_weight should be a float."

    # ------------------------------- CANDIDATE_WRAPPER -------------------------------
    # candidate_wrapper should be callable or None
    assert candidate_wrapper is None or callable(candidate_wrapper), "candidate_wrapper should be callable or None."

    # ------------------------------- RETURN -------------------------------
    # Returning
    handled_args = {
        "multi_X"         : multi_X,
        "multi_y"         : multi_y,
        "multi_y_weights" : multi_y_weights,
        "run_config"      : run_config,
    }
    return handled_args
