import warnings

import numpy as np
import torch as torch
import torch.multiprocessing as mp

from tqdm import tqdm
SHOW_PROGRESS_BAR = False

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ SINGLE EXECUTION ------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

def ExecuteProgram (input_var_data, program_tokens, free_const_values=None):
    """
    Executes a symbolic function program.
    Parameters
    ----------
    input_var_data : torch.tensor of shape (n_dim, ?,) of float
        Values of the input variables of the problem with n_dim = nb of input variables.
    program_tokens : list of token.Token
        Symbolic function program in reverse Polish notation order.
    free_const_values : torch.tensor of shape (n_free_const,) of float or None
        Current values of free constants with for program made of program_tokens n_free_const = nb of choosable free
        constants (library.n_free_constants). free_const_values must be given if program_tokens contains one or more
         free const tokens.
    Returns
    -------
    y : torch.tensor of shape (?,) of float
        Result of computation.
    """

    # Size
    (n_dim, data_size,) = input_var_data.shape

    # Number of tokens in the program
    n_tokens = len(program_tokens)

    # Current stack of computed results
    curr_stack = []

    # De-stacking program (iterating from last token to first)
    start = n_tokens - 1
    for i in range (start, -1, -1):
        token = program_tokens[i]
        # Terminal token
        if token.arity == 0:
            # Function type token
            if token.var_type == 0:
                #curr_stack.append(token.function())
                raise ValueError("Function of arity = 0 encountered. Use var_type = 3 for fixed constants.")
            # Input variable (eg. x0, x1 etc.)
            elif token.var_type == 1:
                curr_stack.append(input_var_data[token.var_id])
            # Free constant variable (eg. c0, c1 etc.)
            elif token.var_type == 2:
                if free_const_values is not None:
                    # curr_stack.append(torch.abs(free_const_values[token.var_id])) # Making free const positive values only #abs_free_const
                    curr_stack.append(free_const_values[token.var_id])
                else:
                    raise ValueError("Free constant encountered in program evaluation but free constant values were "
                                     "not given.")
            # Fixed constant (eg. pi, 1 etc.)
            elif token.var_type == 3:
                curr_stack.append(token.fixed_const)
            else:
                raise NotImplementedError("Token of unknown var_type encountered in ExecuteProgram.")
        # Non-terminal token
        elif token.arity > 0:
            # Last pending elements are those needed for next computation (in reverse order)
            args = curr_stack[-token.arity:][::-1]
            res = token.function(*args)
            # Removing those pending elements as they were used
            curr_stack = curr_stack[:-token.arity]
            # Appending last result to stack
            curr_stack.append(res)
    y = curr_stack[0]
    return y

def ComputeInfixNotation (program_tokens):
    """
    Computes infix str representation of a program.
    (which is the usual way to note symbolic function: +34 (in polish notation) = 3+4 (in infix notation))
    Parameters
    ----------
    program_tokens : list of token.Token
        List of tokens making up the program.
    Returns
    -------
    program_str : str
    """
    # Number of tokens in the program
    n_tokens = len(program_tokens)

    # Current stack of computed results
    curr_stack = []

    # De-stacking program (iterating from last token to first)
    start = n_tokens - 1
    for i in range (start, -1, -1):
        token = program_tokens[i]
        # Last pending elements are those needed for next computation (in reverse order)
        args = curr_stack[-token.arity:][::-1]
        if token.arity == 0:
            res = token.sympy_repr
        elif token.arity == 1:
            if token.is_power is True:
                pow = '{:g}'.format(token.power)  # without trailing zeros
                res = "((%s)**(%s))" % (args[0], pow)
            else:
                res = "%s(%s)" % (token.sympy_repr, args[0])
        elif token.arity == 2:
            res = "(%s%s%s)" % (args[0], token.sympy_repr, args[1])
        elif token.arity > 2 :
            args_str = ""
            for arg in args: args_str+="%s,"%arg
            args_str = args_str[:-1] # deleting last ","
            res = "%s(%s)" % (token.sympy_repr, args_str)
        if token.arity > 0:
            # Removing those pending elements as they were used
            curr_stack = curr_stack[:-token.arity]
        # Appending last result to stack
        curr_stack.append(res)
    return curr_stack[0]

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
    is_notebook = IsNotebook()
    is_cuda_available = torch.cuda.is_available()
    mp_start_method = mp.get_start_method()  # Fork or Spawn ? # mp.get_context("fork").Pool(processes=n_cpus)
    max_ncpus = mp.cpu_count() # Nb. of CPUs available

    # Typically MACs / Windows systems return spawn and LINUX systems return fork. Empirical results:
    # Linux / Intel -> fork
    # Linux / AMD   -> fork
    # MAC / ARM     -> spawn
    # MAC / Intel   -> spawn
    # Windows / Intel -> spawn

    # Is parallel mode available or not
    parallel_mode = True

    # spawn (MAC/Windows) + notebook causes issues
    if mp_start_method == "spawn" and is_notebook:
        parallel_mode = False
        msg = "Parallel mode is not available because physo is being ran from a notebook on a system returning " \
              "multiprocessing.get_start_method() = 'spawn' (typically MACs/Windows). Run physo from the terminal to " \
              "use parallel mode."
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
        print("\ndefault get_start_method :", mp_start_method)
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
def task_exe(prog, X):
    try:
        res = prog(X)
    except:
        res = 0.
    return res

def BatchExecution (progs, X, mask = None, n_cpus = 1, parallel_mode = False):
    """
    Executes prog(X) for each prog in progs and returns the results.
    NB: Parallel execution is typically slower because of communication time (parallel_mode = False is recommended).
    Parallel mode causes inter-process communication errors on some systems (probably due to the large number of
    information to pass which would not work in fork mode but would work in spawn mode ?).
    Parameters
    ----------
    progs : program.VectPrograms
        Programs in the batch.
    X : torch.tensor of shape (n_dim, n_samples,) of float
        Values of the input variables of the problem with n_dim = nb of input variables.
    mask : array_like of shape (progs.batch_size) of bool
        Only programs where mask is True are executed. By default, all programs are executed.
    n_cpus : int
        Number of CPUs to use when running in parallel mode.
    parallel_mode : bool
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
        pool = mp.Pool(processes=n_cpus)
        results = []
        for i in range(progs.batch_size):
            # Computing y = prog(X) where mask is True
            if mask[i]:
                # Getting minimum executable skeleton pickable program
                prog = progs.get_prog(i, skeleton=True)
                result = pool.apply_async(task_exe, args=(prog, X,))
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
                prog = progs.get_prog(i, skeleton=True)
                result = task_exe(prog, X)                                                 # (n_samples,)
                results.append(result)

    # ----- Results -----
    # Stacking results
    results = torch.stack(results)                                                         # (?, n_samples)
    # Batch of evaluation results
    y_batch = torch.full((progs.batch_size, n_samples), torch.nan, dtype=results.dtype)    # (batch_size, n_samples)
    # Updating y_batch with results
    y_batch[mask] = results                                                                # (?, n_samples)

    return y_batch

# Utils pickable function (non nested definition) executing a program (for parallelization purposes)
def task_exe_wrapper_reduce(prog, X, reduce_wrapper):
    try:
        y_pred = prog(X)
        res = reduce_wrapper(y_pred)
        # Kills gradients ! Necessary to minimize communications so it won't crash on some systems. (BatchExecution doc for
        # details on this issue)
        res = float(res)
    except:
        res = 0.
    return res

def BatchExecutionReduceGather (progs, X, reduce_wrapper, mask = None, pad_with = np.NaN, n_cpus = 1, parallel_mode = False):
    """
    Executes prog(X) for each prog in progs and gathers reduce_wrapper(prog(X)) as a result.
    NB: Parallel execution is typically slower because of communication time (even just gathering a float).
    Parameters
    ----------
    progs : program.VectPrograms
        Programs in the batch.
    X : torch.tensor of shape (n_dim, n_samples,) of float
        Values of the input variables of the problem with n_dim = nb of input variables.
    reduce_wrapper : callable
        Function returning a single float number when applied on prog(X). The function must be pickable
        (defined explicitly at the highest level when using parallel_mode).
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
        pool = mp.Pool(processes=n_cpus)
        results = []
        for i in range(progs.batch_size):
            # Computing y = prog(X) where mask is True
            if mask[i]:
                # Getting minimum executable skeleton pickable program
                prog = progs.get_prog(i, skeleton=True)
                result = pool.apply_async(task_exe_wrapper_reduce, args=(prog, X, reduce_wrapper))
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
                result = task_exe_wrapper_reduce(prog, X, reduce_wrapper)                 # float
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
def task_exe_reward(prog, X, y_target, reward_function):
    y_pred = prog(X)
    res = reward_function(y_target, y_pred)
    # Kills gradients ! Necessary to minimize communications so it won't crash on some systems. (BatchExecution doc for
    # details on this issue)
    res = float(res)
    return res

def BatchExecutionReward (progs, X, y_target, reward_function, mask = None, pad_with = np.NaN, n_cpus = 1, parallel_mode = False):
    """
    Executes prog(X) for each prog in progs and gathers reward_function(y_target, prog(X)) as a result.
    NB: Parallel execution is typically slower because of communication time (even just gathering a float).
    Parameters
    ----------
    progs : program.VectPrograms
        Programs in the batch.
    X : torch.tensor of shape (n_dim, n_samples,) of float
        Values of the input variables of the problem with n_dim = nb of input variables.
    y_target : torch.tensor of shape (n_samples,) of float
        Values of target output.
    reward_function : callable
        Function that taking y_target (torch.tensor of shape (?,) of float) and y_pred (torch.tensor of shape (?,)
        of float) as key arguments and returning a float reward of an individual program. The function must be pickable
        (defined explicitly at the highest level when using parallel_mode).
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
        pool = mp.Pool(processes=n_cpus)
        results = []
        for i in range(progs.batch_size):
            # Computing y = prog(X) where mask is True
            if mask[i]:
                # Getting minimum executable skeleton pickable program
                prog = progs.get_prog(i, skeleton=True)
                result = pool.apply_async(task_exe_reward, args=(prog, X, y_target, reward_function))
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
                result = task_exe_reward(prog, X, y_target, reward_function)              # float
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
def task_free_const_opti(prog, X, y_target, free_const_opti_args):
    try:
        history = prog.optimize_constants(X=X, y_target=y_target, args_opti=free_const_opti_args)
    except:
        # Safety
        warnings.warn("Unable to optimize free constants of prog %s -> r = 0" % (str(prog)))
    return None

def BatchFreeConstOpti (progs, X, y_target, free_const_opti_args, mask = None, n_cpus = 1, parallel_mode = False):
    """
    Optimizes the free constants of each program in progs.
    NB: Parallel execution is typically faster.
    Parameters
    ----------
    progs : program.VectPrograms
        Programs in the batch.
    X : torch.tensor of shape (n_dim, n_samples,) of float
        Values of the input variables of the problem with n_dim = nb of input variables.
    y_target : torch.tensor of shape (n_samples,) of float
        Values of target output.
    args_opti : dict or None, optional
        Arguments to pass to free_const.optimize_free_const. By default, free_const.DEFAULT_OPTI_ARGS
        arguments are used.
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
        pool = mp.Pool(processes=n_cpus)
        for i in range(progs.batch_size):
            # Optimizing free constants of programs where mask is True and only if it actually contains free constants
            # (Else we should not bother optimizing its free constants)
            if mask[i] and progs.n_free_const_occurrences[i]:
                # Getting minimum executable skeleton pickable program
                prog = progs.get_prog(i, skeleton=True)
                pool.apply_async(task_free_const_opti, args=(prog, X, y_target, free_const_opti_args))
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
                task_free_const_opti(prog, X = X, y_target = y_target, free_const_opti_args = free_const_opti_args)

    return None
