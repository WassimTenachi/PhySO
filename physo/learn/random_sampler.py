import numpy as np

# Internal imports
import physo.task.args_handler as args_handler
import physo.task.sr as sr
from physo.physym import library as Lib
from physo.physym import prior as Prior
import physo.physym.free_const as free_const
from physo.physym import vect_programs as VProg

def generate_expressions(
            # Batch size
            batch_size=1000,
            # Max time step
            max_time_step=50,

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

            # Priors configuration
            priors_config = None,

            # Number of realizations
            n_realizations = 1,
            # Device to use
            device="cpu",

    ):
    """
    batch_size : int
        Number of programs in batch.
    max_time_step : int
        Max number of tokens programs can contain.

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

    priors_config : list of tuples (str : dict), optional
        List of priors. List containing tuples with prior name as first item in couple (see prior.PRIORS_DICT for list
        of available priors) and additional arguments (besides library and programs) to be passed to priors as second
        item of couple, leave None for priors that do not require arguments. By default, uses priors from config
        physo.task.sr.default_config.

    n_realizations : int or None, optional
        Number of realizations for each program, ie. number of datasets each program has to fit.
        Dataset specific free constants will have different values different for each realization.
        Uses 1 by default (if None).
    device : str, optional
        Device on which free constants will be stored and computations will be performed.
    """

    # ----- Library ------

    # Build the library configuration
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

    lib = Lib.Library(**library_config)

    # ----- Programs -----

    progs = VProg.VectPrograms(batch_size=batch_size, max_time_step=max_time_step, library=lib, n_realizations=1)

    # ----- Priors -----

    if priors_config is None:
        priors_config = sr.default_config["priors_config"]
    # Check priors configuration
    args_handler.check_priors_config(priors_config=priors_config, max_time_step=max_time_step)

    # Prior
    prior = Prior.make_PriorCollection( programs=progs,
                                        library=lib,
                                        priors_config=priors_config, )

    # ----- Generate programs -----
    # Nb of choosable tokens
    n_choices = progs.n_choices

    for i in range(max_time_step):
        print("Token number %i/%i" % (i+1, max_time_step))

        # -- Sample next token --
        # Random probs for next tokens
        next_tokens_probs = np.random.random((batch_size, n_choices))                          # (batch_size, n_choices)
        next_tokens_probs /= next_tokens_probs.sum(axis=-1, keepdims=True)                     # (batch_size, n_choices)

        # -- Prior --
        # (embedding output)
        prior_array = prior().astype(np.float32)                                               # (batch_size, n_choices)

        # 0 protection so there is always something to sample
        epsilon = 0  # 1e-14 #1e0*np.finfo(np.float32).eps
        prior_array[prior_array == 0] = epsilon
        is_able_to_sample = (prior_array.sum(axis=-1) > 0.)  # (batch_size,)
        assert is_able_to_sample.all(), "Prior(s) make it impossible to successfully sample expression(s) as all " \
                                        "choosable tokens have 0 prob for %i/%i programs." % (is_able_to_sample.sum(),
                                                                                              batch_size)

        # -- Actions --
        # Apply the prior to the next tokens
        next_tokens_probs *= prior_array                                                       # (batch_size, n_choices)
        next_tokens_probs /= next_tokens_probs.sum(axis=-1, keepdims=True)                     # (batch_size, n_choices)

        # Sample next tokens
        # next_tokens_idx = np.array([np.random.choice(n_choices, p=probs) for probs in next_tokens_probs])  # (batch_size,)
        # Vectorized version of the above line:
        rand_vals = np.random.rand(batch_size)                              # (batch_size,)
        cum_probs = next_tokens_probs.cumsum(axis=1)                        # (batch_size, n_choices)
        next_tokens_idx = (cum_probs > rand_vals[:, None]).argmax(axis=1)   # (batch_size,)

        progs.append(next_tokens_idx)

    return progs