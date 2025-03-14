import warnings

import numpy as np
import torch as torch
import torch.multiprocessing as mp

from tqdm import tqdm
SHOW_PROGRESS_BAR = False

def EnforceStartMethod():
    # Only enforce the use of spawn start method if not already spawn
    if mp.get_start_method() != "spawn":
        print("Enforcing spawn multiprocessing start method.")
        mp.set_start_method("spawn", force=True)

# Enforcing the use of spawn start method as soon as this file is imported
EnforceStartMethod()

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ PARALLEL EXECUTION DIAGNOSIS ------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

def IsNotebook():
    try:
        if 'google.colab' in str(get_ipython()):
            return True   # Google Colab
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def ParallelExeAvailability(verbose=False):
    """
    Checks if parallel run is available on this system and produces a recommended config.
    Parameters
    ----------
    verbose : bool
        Prints log.
    Returns
    -------
    recommended_config : dict
        bool recommended_config[parallel_mode] : will parallel mode  work on this system ?
        int recommended_config[n_cpus] : Nb. of CPUs to use.

    """

    # Gathering info
    is_notebook       = IsNotebook()
    is_cuda_available = torch.cuda.is_available()
    max_ncpus         = mp.cpu_count() # Nb. of CPUs available

    # Typically MACs / Windows systems return spawn and LINUX systems return fork. Empirical results:
    # Linux / Intel -> fork
    # Linux / AMD   -> fork
    # MAC / ARM     -> spawn
    # MAC / Intel   -> spawn
    # Windows / Intel -> spawn

    # Is parallel mode available or not
    parallel_mode = True

    # Enforcing the use of spawn start method
    # This is necessary because:
    # 1. fork + physo installed in env     -> parallel mode is always inefficient (for both SR and class SR)
    # 2. fork + physo not installed in env -> parallel mode is efficient for SR but does NOT RUN for class SR
    # EnforceStartMethod() # mp.set_start_method("spawn", force=True)
    # Done at file import

    # spawn + notebook causes issues
    if mp.get_start_method() == "spawn" and is_notebook:
        parallel_mode = False
        msg = "Parallel mode is not available because physo is being ran from a notebook using 'spawn' " \
              "multiprocessing start method (multiprocessing.get_start_method() = 'spawn'). Run physo from a " \
              "python script to use parallel mode."
        print(msg)
        warnings.warn(msg)

    # CUDA available causes issues on some systems even when sending to proper device
    if is_cuda_available:
        parallel_mode = False
        msg = "Parallel mode is not available because having a CUDA-able version of pytorch was found to cause issues " \
              "on some systems (even if the dataset is sent to the proper device). Please install the vanilla non " \
              "CUDA-able version of pytorch via conda install pytorch (returning torch.cuda.is_available() = False) " \
              "to use parallel mode."
        print(msg)
        warnings.warn(msg)

    # recommended config
    recommended_config = {
        "parallel_mode" : parallel_mode,
        "n_cpus" : max_ncpus,
    }

    # Report
    if verbose:
        print("\nMultiprocessing start method :", mp.get_start_method())
        print("Running from notebook :", is_notebook)
        print("Is CUDA available :", is_cuda_available)  # OK if dataset on CPU
        print("Total nb. of CPUs : ", max_ncpus)
        print("Recommended config", recommended_config)

    # Too many issues with cuda available + parallel mode on linux (even when sending to proper device).
    #if is_cuda_available and parallel_mode == True:
    #    warnings.warn("Both CUDA and CPU parallel mode are available. If you plan on using CPU parallel mode (which is "
    #                  "typically faster), please ensure that the dataset is manually transferred to the CPU device as "
    #                  "it will automatically be sent to the CUDA device otherwise which will cause a conflict.")
    #

    return recommended_config

# ------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- PARALLEL EXECUTION -----------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# Utils pickable function (non nested definition) executing a program (for parallelization purposes)
def task_exe(prog, X, i_realization, n_samples_per_dataset):
    try:
        res = prog(X=X, i_realization=i_realization, n_samples_per_dataset=n_samples_per_dataset)
    except:
        res = 0.
    return res

def BatchExecution (progs, X,
                    # Realization related
                    i_realization         = 0,
                    n_samples_per_dataset = None,
                    # Mask
                    mask     = None,
                    pad_with = np.nan,
                    # Parallel mode related
                    n_cpus        = 1,
                    parallel_mode = False):
    """
    Executes prog(X) for each prog in progs and returns the results.
    NB: Parallel execution is typically slower because of communication time (parallel_mode = False is recommended).
    Parallel mode causes inter-process communication errors on some systems (probably due to the large number of
    information to pass which would not work in fork mode but would work in spawn mode ?).
    Parameters
    ----------
    progs : vect_programs.VectPrograms
        Programs in the batch.
    X : torch.tensor of shape (n_dim, n_samples,) of float
        Values of the input variables of the problem with n_dim = nb of input variables.
    i_realization : int, optional
        Index of realization to use for dataset specific free constants (0 by default).
    n_samples_per_dataset : array_like of shape (n_realizations,) of int or None, optional
        Overrides i_realization if given. If given assumes that X contains multiple datasets with samples of each
        dataset following each other and each portion of X corresponding to a dataset should be treated with its
        corresponding dataset specific free constants values. n_samples_per_dataset is the number of samples for
        each dataset. Eg. [90, 100, 110] for 3 datasets, this will assume that the first 90 samples of X are for
        the first dataset, the next 100 for the second and the last 110 for the third.
    mask : array_like of shape (progs.batch_size) of bool, optional
        Only programs where mask is True are executed. By default, all programs are executed.
    pad_with : float, optional
        Value to pad with where mask is False. (Default = nan).
    n_cpus : int, optional
        Number of CPUs to use when running in parallel mode.
    parallel_mode : bool, optional
        Parallel execution if True, execution in a loop else.
    Returns
    -------
    y_batch : torch.tensor of shape (progs.batch_size, n_samples,) of float
        Returns result of execution for each program in progs. Returns NaNs for programs that are not executed
        (where mask is False).
    """
    # mask : should program be executed ?
    # By default, all programs of batch are executed
    # ? = mask.sum() # Number of programs to execute
    if mask is None:
        mask = np.full(shape=(progs.batch_size), fill_value=True)                           # (batch_size)

    # Number of data point per dimension
    n_samples = X.shape[1]

    # ----- Parallel mode -----
    if parallel_mode:
        # Opening a pull of processes
        # pool = mp.get_context("fork").Pool(processes=n_cpus)
        # mp.set_start_method("spawn", force=True)
        pool = mp.Pool(processes=n_cpus)
        results = []
        for i in range(progs.batch_size):
            # Computing y = prog(X) where mask is True
            if mask[i]:
                # Getting minimum executable skeleton pickable program
                prog = progs.get_prog(prog_idx=i, skeleton=True)
                result = pool.apply_async(task_exe, args=(prog, X, i_realization, n_samples_per_dataset))
                results.append(result)

        # Waiting for all tasks to complete and collecting the results
        results = [result.get() for result in results]

        # Closing the pool of processes
        pool.close()
        pool.join()

    # ----- Non parallel mode -----
    else:
        results = []
        for i in range (progs.batch_size):
            # Computing y = prog(X) where mask is True
            if mask[i]:
                prog = progs.get_prog(prog_idx=i, skeleton=True)
                result = task_exe(prog, X, i_realization, n_samples_per_dataset)           # (n_samples,)
                results.append(result)

    # ----- Results -----
    # Stacking results
    results = torch.stack(results)                                                         # (?, n_samples)
    # Batch of evaluation results
    y_batch = torch.full((progs.batch_size, n_samples), pad_with, dtype=results.dtype)     # (batch_size, n_samples)
    # Updating y_batch with results
    y_batch[mask] = results                                                                # (?, n_samples)

    return y_batch

# Utils pickable function (non nested definition) executing a program (for parallelization purposes)
def task_exe_wrapper_reduce(prog, X, reduce_wrapper, i_realization, n_samples_per_dataset):
    try:
        y_pred = prog(X=X, i_realization=i_realization, n_samples_per_dataset=n_samples_per_dataset)
        res = reduce_wrapper(y_pred)
        # Kills gradients ! Necessary to minimize communications so it won't crash on some systems. (BatchExecution doc for
        # details on this issue)
        res = float(res)
    except:
        res = 0.
    return res

def BatchExecutionReduceGather (progs, X, reduce_wrapper,
                                # Realization related
                                i_realization         = 0,
                                n_samples_per_dataset = None,
                                # Mask
                                mask     = None,
                                pad_with = np.nan,
                                # Parallel mode related
                                n_cpus        = 1,
                                parallel_mode = False
                                ):
    """
    Executes prog(X) for each prog in progs and gathers reduce_wrapper(prog(X)) as a result.
    NB: Parallel execution is typically slower because of communication time (even just gathering a float).
    Parameters
    ----------
    progs : vect_programs.VectPrograms
        Programs in the batch.
    X : torch.tensor of shape (n_dim, n_samples,) of float
        Values of the input variables of the problem with n_dim = nb of input variables.
    reduce_wrapper : callable
        Function returning a single float number when applied on prog(X). The function must be pickable
        (defined explicitly at the highest level when using parallel_mode).
    i_realization : int, optional
        Index of realization to use for dataset specific free constants (0 by default).
    n_samples_per_dataset : array_like of shape (n_realizations,) of int or None, optional
        Overrides i_realization if given. If given assumes that X contains multiple datasets with samples of each
        dataset following each other and each portion of X corresponding to a dataset should be treated with its
        corresponding dataset specific free constants values. n_samples_per_dataset is the number of samples for
        each dataset. Eg. [90, 100, 110] for 3 datasets, this will assume that the first 90 samples of X are for
        the first dataset, the next 100 for the second and the last 110 for the third.
    mask : array_like of shape (progs.batch_size) of bool
        Only programs where mask is True are executed. By default, all programs are executed.
    pad_with : float, optional
        Value to pad with where mask is False. (Default = nan).
    n_cpus : int
        Number of CPUs to use when running in parallel mode.
    parallel_mode : bool
        Parallel execution if True, execution in a loop else.
    Returns
    -------
    results : numpy.array of shape (progs.batch_size,) of float
        Returns reduce_wrapper(prog(X)) for each program in progs. Returns NaNs for programs that are not executed
        (where mask is False).
    """
    pb = lambda x: x
    #if SHOW_PROGRESS_BAR:
    #    pb = tqdm

    # mask : should program be executed ?
    # By default, all programs of batch are executed
    # ? = mask.sum() # Number of programs to execute
    if mask is None:
        mask = np.full(shape=(progs.batch_size), fill_value=True)                           # (batch_size)

    # ----- Parallel mode -----
    if parallel_mode:
        # Opening a pull of processes
        # pool = mp.get_context("fork").Pool(processes=n_cpus)
        # mp.set_start_method("spawn", force=True)
        pool = mp.Pool(processes=n_cpus)
        results = []
        for i in range(progs.batch_size):
            # Computing y = prog(X) where mask is True
            if mask[i]:
                # Getting minimum executable skeleton pickable program
                prog = progs.get_prog(i, skeleton=True)
                result = pool.apply_async(task_exe_wrapper_reduce, args=(prog, X, reduce_wrapper, i_realization, n_samples_per_dataset))
                results.append(result)

        # Waiting for all tasks to complete and collecting the results
        results = [result.get() for result in results]

        # Closing the pool of processes
        pool.close()
        pool.join()

    # ----- Non parallel mode -----
    else:
        results = []
        for i in pb(range(progs.batch_size)):
            # Computing y = prog(X) where mask is True
            if mask[i]:
                prog = progs.get_prog(i, skeleton=True)
                result = task_exe_wrapper_reduce(prog, X, reduce_wrapper, i_realization, n_samples_per_dataset) # float
                results.append(result)

    # ----- Results -----
    # Stacking results
    results = np.array(results)                                                            # (?,)
    # Batch of evaluation results
    res = np.full((progs.batch_size,), pad_with, dtype=results.dtype)                      # (batch_size,)
    # Updating res with results
    res[mask] = results                                                                    # (?,)

    return res



# Utils pickable function (non nested definition) executing a program (for parallelization purposes)
def task_exe_reward(prog, X, y_target, reward_function, y_weights, i_realization, n_samples_per_dataset):
    y_pred = prog(X=X, i_realization=i_realization, n_samples_per_dataset=n_samples_per_dataset)
    res = reward_function(y_target=y_target, y_pred=y_pred, y_weights=y_weights)
    # Kills gradients ! Necessary to minimize communications so it won't crash on some systems. (BatchExecution doc for
    # details on this issue)
    res = float(res)
    return res

def BatchExecutionReward (progs, X, y_target, reward_function, y_weights = 1.,
                          # Realization related
                          i_realization         = 0,
                          n_samples_per_dataset = None,
                          # Mask
                          mask     = None,
                          pad_with = np.nan,
                          # Parallel mode related
                          n_cpus        = 1,
                          parallel_mode = False
                          ):
    """
    Executes prog(X) for each prog in progs and gathers reward_function(y_target, prog(X), y_weights) as a result.
    NB: Parallel execution is typically slower because of communication time (even just gathering a float).
    Parameters
    ----------
    progs : vect_programs.VectPrograms
        Programs in the batch.
    X : torch.tensor of shape (n_dim, n_samples,) of float
        Values of the input variables of the problem with n_dim = nb of input variables.
    y_target : torch.tensor of shape (n_samples,) of float
        Values of target output.
    reward_function : callable
        Function that taking y_target (torch.tensor of shape (?,) of float), y_pred (torch.tensor of shape (?,)
        of float) and y_weights (torch.tensor of shape (?,) of float) as key arguments and returning a float reward of
        an individual program. The function must be pickable (defined explicitly at the highest level when using
        parallel_mode).
    y_weights : torch.tensor of shape (?,) of float, optional
        Weights for each data point.
    i_realization : int, optional
        Index of realization to use for dataset specific free constants (0 by default).
    n_samples_per_dataset : array_like of shape (n_realizations,) of int or None, optional
        Overrides i_realization if given. If given assumes that X contains multiple datasets with samples of each
        dataset following each other and each portion of X corresponding to a dataset should be treated with its
        corresponding dataset specific free constants values. n_samples_per_dataset is the number of samples for
        each dataset. Eg. [90, 100, 110] for 3 datasets, this will assume that the first 90 samples of X are for
        the first dataset, the next 100 for the second and the last 110 for the third.
    mask : array_like of shape (progs.batch_size) of bool
        Only programs where mask is True are executed. By default, all programs are executed.
    pad_with : float
        Value to pad with where mask is False. (Default = nan).
    n_cpus : int
        Number of CPUs to use when running in parallel mode.
    parallel_mode : bool
        Parallel execution if True, execution in a loop else.
    Returns
    -------
    results : numpy.array of shape (progs.batch_size,) of float
        Returns reduce_wrapper(prog(X)) for each program in progs. Returns NaNs for programs that are not executed
        (where mask is False).
    """
    pb = lambda x: x
    #if SHOW_PROGRESS_BAR:
    #    pb = tqdm

    # mask : should program be executed ?
    # By default, all programs of batch are executed
    # ? = mask.sum() # Number of programs to execute
    if mask is None:
        mask = np.full(shape=(progs.batch_size), fill_value=True)                           # (batch_size)

    # ----- Parallel mode -----
    if parallel_mode:
        # Opening a pull of processes
        # pool = mp.get_context("fork").Pool(processes=n_cpus)
        # mp.set_start_method("spawn", force=True)
        pool = mp.Pool(processes=n_cpus)
        results = []
        for i in range(progs.batch_size):
            # Computing y = prog(X) where mask is True
            if mask[i]:
                # Getting minimum executable skeleton pickable program
                prog = progs.get_prog(i, skeleton=True)
                result = pool.apply_async(task_exe_reward, args=(prog, X, y_target, reward_function, y_weights, i_realization, n_samples_per_dataset))
                results.append(result)

        # Waiting for all tasks to complete and collecting the results
        results = [result.get() for result in results]

        # Closing the pool of processes
        pool.close()
        pool.join()

    # ----- Non parallel mode -----
    else:
        results = []
        for i in pb(range(progs.batch_size)):
            # Computing y = prog(X) where mask is True
            if mask[i]:
                prog = progs.get_prog(i, skeleton=True)
                result = task_exe_reward(prog, X, y_target, reward_function, y_weights, i_realization, n_samples_per_dataset) # float
                results.append(result)

    # ----- Results -----
    # Stacking results
    results = np.array(results)                                                            # (?,)
    # Batch of evaluation results
    res = np.full((progs.batch_size,), pad_with, dtype=results.dtype)                      # (batch_size,)
    # Updating res with results
    res[mask] = results                                                                    # (?,)

    return res



# Utils pickable function (non nested definition) optimizing the free consts of a program (for parallelization purposes)
def task_free_const_opti(prog, X, y_target, free_const_opti_args, y_weights, i_realization, n_samples_per_dataset):
    try:
        history = prog.optimize_constants(X=X, y_target=y_target, args_opti=free_const_opti_args, y_weights=y_weights, i_realization=i_realization, n_samples_per_dataset=n_samples_per_dataset)
    except:
        # Safety
        warnings.warn("Unable to optimize free constants of prog %s -> r = 0" % (str(prog)))
    return None

def BatchFreeConstOpti (progs, X, y_target, free_const_opti_args=None, y_weights = 1.,
                        # Realization related
                        i_realization         = 0,
                        n_samples_per_dataset = None,
                        # Mask
                        mask     = None,
                        # Parallel mode related
                        n_cpus        = 1,
                        parallel_mode = False
                        ):
    """
    Optimizes the free constants of each program in progs.
    NB: Parallel execution is typically faster.
    Parameters
    ----------
    progs : vect_programs.VectPrograms
        Programs in the batch.
    X : torch.tensor of shape (n_dim, n_samples,) of float
        Values of the input variables of the problem with n_dim = nb of input variables.
    y_target : torch.tensor of shape (n_samples,) of float
        Values of target output.
    free_const_opti_args : dict or None, optional
        Arguments to pass to free_const.optimize_free_const. By default, free_const.DEFAULT_OPTI_ARGS
        arguments are used.
    y_weights : torch.tensor of shape (?,) of float, optional
        Weights for each data point.
    i_realization : int, optional
        Index of realization to use for dataset specific free constants (0 by default).
    n_samples_per_dataset : array_like of shape (n_realizations,) of int or None, optional
        Overrides i_realization if given. If given assumes that X contains multiple datasets with samples of each
        dataset following each other and each portion of X corresponding to a dataset should be treated with its
        corresponding dataset specific free constants values. n_samples_per_dataset is the number of samples for
        each dataset. Eg. [90, 100, 110] for 3 datasets, this will assume that the first 90 samples of X are for
        the first dataset, the next 100 for the second and the last 110 for the third.
    mask : array_like of shape (progs.batch_size) of bool
        Only programs' constants where mask is True are optimized. By default, all programs' constants are opitmized.
    n_cpus : int
        Number of CPUs to use when running in parallel mode.
    parallel_mode : bool
        Parallel execution if True, execution in a loop else.
    """
    pb = lambda x: x
    if SHOW_PROGRESS_BAR:
        pb = tqdm

    # mask : should program be executed ?
    # By default, all programs of batch are executed
    # ? = mask.sum() # Number of programs to execute
    if mask is None:
        mask = np.full(shape=(progs.batch_size), fill_value=True)                           # (batch_size)

    # Parallel mode
    if parallel_mode:
        # Opening a pull of processes
        # pool = mp.get_context("fork").Pool(processes=n_cpus)
        # mp.set_start_method("spawn", force=True)
        pool = mp.Pool(processes=n_cpus)
        for i in range(progs.batch_size):
            # Optimizing free constants of programs where mask is True and only if it actually contains free constants
            # (Else we should not bother optimizing its free constants)
            if mask[i] and progs.n_free_const_occurrences[i]:
                # Getting minimum executable skeleton pickable program
                prog = progs.get_prog(i, skeleton=True)
                pool.apply_async(task_free_const_opti, args=(prog, X, y_target, free_const_opti_args, y_weights, i_realization, n_samples_per_dataset))
        # Closing the pool of processes
        pool.close()
        pool.join()

    # Non parallel mode
    else:
        for i in pb(range(progs.batch_size)):
            # Optimizing free constants of programs where mask is True and only if it actually contains free constants
            # (Else we should not bother optimizing its free constants)
            if mask[i] and progs.n_free_const_occurrences[i]:
                # Getting minimum executable skeleton pickable program
                prog = progs.get_prog(i, skeleton=True)
                task_free_const_opti(prog, X = X, y_target = y_target, free_const_opti_args = free_const_opti_args, y_weights=y_weights, i_realization=i_realization, n_samples_per_dataset=n_samples_per_dataset)

    return None
