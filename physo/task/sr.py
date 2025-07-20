import numpy as np
import torch
import warnings
import os
import pandas as pd

# Internal imports
from physo.config.config0 import config0
from physo.config.config3_expA import config3

import physo.task.args_handler as args_handler
import physo

# DEFAULT RUN CONFIG TO USE
default_config          = config0
default_config_w_struct = config3

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
            stop_after_n_epochs = args_handler.default_stop_after_n_epochs,
            epochs = None,
            # Structure analysis
            structure_analysis = None,
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
    Runs a symbolic regression task.
    (Wrapper around physo.task.fit)

    Parameters
    ----------

    X : numpy.array of shape (n_dim, ?,) of float
        Values of the input variables of the problem with n_dim = nb of input variables.
    y : numpy.array of shape (?,) of float
        Values of the target symbolic function to recover when applied on input variables contained in X.
    y_weights : np.array of shape (?,) of float or float, optional
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

    structure_analysis : bool (optional)
        If True, a structure analysis of the problem is performed before in order to detect separabilities that can help
        the symbolic regression task, eg. detecting that problem y = f(x0,x1,x2,x3) can be simplified to
        y = f(x0) * f(x1) + f(x2,x3). False by default.

    candidate_wrapper : callable or None (optional)
        Wrapper to apply to candidate program's output, candidate_wrapper taking func, X as arguments where func is
        a candidate program callable (taking X as arg). By default = None, no wrapper is applied (identity).
    run_config : dict or None (optional)
        Run configuration (by default uses physo.task.sr.default_config)
        See physo/config/ for examples of run configurations.
        If structure_analysis is True, uses physo.task.sr.default_config_w_struct by default.
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

    # ------- Default run config to use -------
    # Default run config to use
    using_default_config = False
    if run_config is None:
        using_default_config = True
        run_config = default_config
    if get_run_logger is None:
        get_run_logger = args_handler.get_default_run_logger
    if get_run_visualiser is None:
        get_run_visualiser = args_handler.get_default_run_visualiser

    # ------- Structure analysis -------
    # By default, structure analysis is done according to the run_config
    # - Custom run_config  -> custom structure_analysis
    # - Default run_config -> no structure_analysis
    if structure_analysis is None:
        do_structure_analysis = run_config['struct_analysis']['structure_analysis'] # Consistent with run_config
    else:
        do_structure_analysis = bool(structure_analysis) # We will have to check that this is consistent with run_config
    # If using default config, use default config for structure analysis
    if do_structure_analysis and using_default_config:
        run_config = default_config_w_struct # Consistent with run_config
    # Check if structure_analysis is consistent with run_config, we can not run structure analysis if its hyperparams
    # are not in run_config
    assert do_structure_analysis == run_config['struct_analysis']['structure_analysis'], "structure_analysis (%s) is inconsistent with run_config (%s)" %(structure_analysis, run_config['struct_analysis']['structure_analysis'])

    if do_structure_analysis:

        # --- Save path ---
        if get_run_visualiser().do_save:
            structure_do_save   = True
            structure_save_path = ''.join(get_run_visualiser().save_path.split('.')[:-1]) + "_structure_analysis.csv"    # save_path with extension replaced by '_structure_analysis.csv',
        else:
            structure_do_save   = False
            structure_save_path = None

        # --- Check if analysis was already done ---
        # Check if file structure_save_path does exist, if it does no need to run the analysis
        do_run = True
        if structure_do_save:
            if os.path.exists(structure_save_path):
                try:
                    df = pd.read_csv(structure_save_path)
                    structure        = eval(df["structure_prefix"].iloc[0]) # Loading the structure
                    structure_pretty = df["structure"].iloc[0]
                    do_run = False
                except:
                    pass

        # --- Run ---
        if do_run:
            structure,structure_pretty = physo.StructureAnalysis(
                        X = X,
                        y = y,
                        y_weights = y_weights,
                        # X
                        X_names = X_names,
                        # Candidate wrapper
                        candidate_wrapper = candidate_wrapper,
                        # Default run config to use
                        run_config = run_config,
                        # Save path
                        do_save   = structure_do_save,
                        save_path = structure_save_path,
                        # Parallel mode
                        device  = device,
            )
        else:
            print("Structure analysis was already performed, loading it from : %s" %structure_save_path)

        # --- Encoding structure into prior ---
        print("-> Encoding prior that expressions should be like : %s" %structure_pretty)
        struct_prior_args = {"structure": structure}
        struct_prior_args.update(run_config["struct_analysis"]["prior_config"])
        struct_prior_config = ("StructurePrior", struct_prior_args)
        run_config["priors_config"].append(struct_prior_config)
        # todo: adjust global length prior accordingly with the structure ?

    # ------- To Class SR for run -------
    # Transmitting arguments to ClassSR as SR is just a sub-case of ClassSR where there is only one realization
    # and no dataset spe free constants.
    best_expression, run_logger = physo.ClassSR(
                # Wrapping to make a 1 realization dataset (single real SR specific)
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
                # Spe Free constants are not used in normal SR (single real SR specific)
                spe_free_consts_names    = None,
                spe_free_consts_units    = None,
                spe_free_consts_init_val = None,
                # Operations to use
                op_names          = op_names,
                use_protected_ops = use_protected_ops,
                # Stopping
                stop_reward         = stop_reward,
                max_n_evaluations   = max_n_evaluations,
                stop_after_n_epochs = stop_after_n_epochs,
                epochs              = epochs,
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

    return best_expression, run_logger