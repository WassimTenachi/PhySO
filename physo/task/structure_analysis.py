import numpy as np
import torch
import warnings
import pandas as pd

import physo.task.sr
# Internal imports
import physo.task.args_handler as args_handler
from physo.physym import library as Lib
from physo.physym import prior as Prior
from physo.physym import vect_programs as VProg
from physo.task import sr as SR

# DEFAULT RUN CONFIG TO USE
default_config = SR.default_config_w_struct

def StructureAnalysis(X, y, y_weights=1.,
                    # X
                    X_names = None,
                    # Candidate wrapper
                    candidate_wrapper = None,
                    # Default run config to use
                    run_config = None,
                    # Save path
                    do_save   = True,
                    save_path = None,
                    # Default run monitoring
                    get_run_logger     = None,
                    get_run_visualiser = None,
                    # Device
                    device        = 'cpu',
        ):
    """
    Performs a structure analysis of the dataset in order to detect separabilities, eg. detecting that data like
    y = f(x0,x1,x2,x3) can be modeled by y = f1(x0) * f2(x1) + f3(x2,x3).

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

    candidate_wrapper : callable or None (optional)
        Wrapper to apply to candidate program's output, candidate_wrapper taking func, X as arguments where func is
        a candidate program callable (taking X as arg). By default = None, no wrapper is applied (identity).
    run_config : dict or None (optional)
        Run configuration (by default uses physo.task.sr.default_config_w_struct)
        See physo/config/ for examples of run configurations.

    do_save : bool (optional)
        If True, saves the results of the task. True by default.
    save_path : str or None (optional)
        Path to save the results of the task.

    get_run_logger : callable returning physo.learn.monitoring.RunLogger or None (optional)
        Run logger (by default uses physo.task.args_handler.get_default_run_logger)
    get_run_visualiser : callable returning physo.learn.monitoring.RunVisualiser or None (optional)
        Run visualiser (by default uses physo.task.args_handler.get_default_run_visualiser)

    device : str (optional)
        Device to use for computations (eg. 'cpu', 'cuda'). 'cpu' by default.

    Returns
    -------
    structure : list of (str or list of str)
        Detected structure underlying the dataset in prefix notation, eg. ["add", "mul", ["x0"], ["x1",], ["x2", "x3"]].
        if data is separable as y = f(x0,x1,x2,x3) = f1(x0) * f2(x1) + f3(x2,x3).

    """

    # ------- Default run config to use -------
    # Default run config to use
    if run_config is None:
        run_config = default_config
    if get_run_logger is None:
        get_run_logger = args_handler.get_default_run_logger
    if get_run_visualiser is None:
        get_run_visualiser = args_handler.get_default_run_visualiser

    # ------- Check and build run config -------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Raises some warnings due to some missing infos (this is ok as we do not need them here)
        handled_args = args_handler.check_args_and_build_run_config(
                                        multi_X = [X,],
                                        multi_y = [y,],
                                        multi_y_weights = [y_weights,],
                                        # X
                                        X_names = X_names,
                                        X_units = None,
                                        # y
                                        y_name  = None,
                                        y_units = None,
                                        # Fixed constants
                                        fixed_consts       = None,
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
                                        op_names          = None,
                                        use_protected_ops = None,
                                        # Stopping
                                        epochs = None,
                                        # Candidate wrapper
                                        candidate_wrapper = candidate_wrapper,
                                        # Default run config to use
                                        run_config = run_config,
                                        # Default run monitoring
                                        get_run_logger     = get_run_logger,
                                        get_run_visualiser = get_run_visualiser,
                                        # Parallel mode
                                        parallel_mode = None,
                                        n_cpus        = None,
                                        device        = device,
                                )

    # ------- Run -------
    print("Structure analysis task started...")
    print("-> Not implemented yet.")
    # todo : figure out the device cpu gpu situation, we might want to run structure analysis on gpu but the rest on cpu
    # todo : figure out candidate_wrapper
    # todo : figure out y_weights

    raise NotImplementedError("Structure analysis is not implemented yet.")
    # todo: for now let's pretended the process returned this:
    result = [torch.add, [0,], [1,]]

    # ------- Cleaning -------
    # Making a nice representation of the structure
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Raises some warnings due to some units provided (this is ok)
        lib = Lib.Library(args_make_tokens = handled_args["run_config"]["library_config"]["args_make_tokens"],
                          superparent_name = handled_args["run_config"]["library_config"]["superparent_name"])

    structure       = []
    structure_w_ids = []
    for node in result:
        # Input variable
        if isinstance(node, list):
            nid = node
            n   = [lib.input_var_name_from_id[id] for id in node]
        # Multiplicative separability
        elif node is torch.multiply:
            nid = "mul"
            n   = "mul"
        # Additive separability
        elif node is torch.add:
            nid = "add"
            n   = "add"
        else:
            raise ValueError("Unknown node type %s", node)
        structure       .append(n)
        structure_w_ids .append(nid)

    # ------- Return -------
    # Using Prior to make fancy representation of the structure
    progs = VProg.VectPrograms(batch_size=1, max_time_step=10, library=lib, n_realizations=1) # dummy progs for prior
    prior = Prior.StructurePrior(library=lib, programs=progs, structure=structure)

    # Saving struct as .csv
    df = pd.DataFrame()
    df["structure"]              = [prior.structure_repr(),]
    df["structure_prefix"]       = [str(structure),]
    df["structure_prefix_w_ids"] = [str(structure_w_ids),]

    if do_save:
        if save_path is None:
            save_path = "structure_analysis.csv"
        df.to_csv(save_path, index=False)

    return structure


