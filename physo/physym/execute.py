import warnings

import numpy as np
import torch as torch
import torch.multiprocessing as mp


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
            # Fixed constant (eg. pi, 1 etc.)
            if token.var_type == 0:
                curr_stack.append(token.function())
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


def ParallelExeAvailability(verbose=False):
    """
    Checks if parallel run is available on this system.
    Parameters
    ----------
    verbose : bool
        Prints log.
    Returns
    -------
    is_parallel_exe_available : bool
    """
    if verbose:
        print("default get_start_method", mp.get_start_method())
    is_parallel_exe_available = True
    print("Is parallel execution available:", is_parallel_exe_available)
    return is_parallel_exe_available


# Utils pickable function (non nested definition) executing a program (for parallelization purposes)
def task_exe(prog, X):
    res = prog(X)
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
    res = reduce_wrapper(prog(X))
    return res

def BatchExecutionReduceGather (progs, X, reduce_wrapper, mask = None, n_cpus = 1, parallel_mode = False):
    """
    Executes prog(X) for each prog in progs and gathers reduce_wrapper(prog(X)) as a result.
    NB: Parallel execution is typically faster because of communication time is lower when just gathering a float.
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
    n_cpus : int
        Number of CPUs to use when running in parallel mode.
    parallel_mode : bool
        Parallel execution if True, execution in a loop else.
    Returns
    -------
    results : torch.tensor of shape (progs.batch_size,) of float
        Returns reduce_wrapper(prog(X)) for each program in progs. Returns NaNs for programs that are not executed
        (where mask is False).
    """
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
        for i in range (progs.batch_size):
            # Computing y = prog(X) where mask is True
            if mask[i]:
                prog = progs.get_prog(i, skeleton=True)
                result = task_exe_wrapper_reduce(prog, X, reduce_wrapper)                 # float
                results.append(result)

    # ----- Results -----
    # Stacking results
    results = torch.stack(results)                                                         # (?,)
    # Batch of evaluation results
    res = torch.full((progs.batch_size,), torch.nan, dtype=results.dtype)                  # (batch_size,)
    # Updating res with results
    res[mask] = results                                                                    # (?,)

    return res

# Utils pickable function (non nested definition) optimizing the free consts of a program (for parallelization purposes)
def task_free_const_opti(prog, X, y_target, free_const_opti_args):
    history = prog.optimize_constants(X=X, y_target=y_target, args_opti=free_const_opti_args)
    return None

def BatchFreeConstOpti (progs, X, y_target, free_const_opti_args, mask = None, n_cpus = 1, parallel_mode = True):
    """
    Executes prog(X) for each prog in progs and returns the results.
    NB: Parallel execution is typically faster.
    Parameters
    ----------
    progs : program.VectPrograms
        Programs in the batch.
    X : torch.tensor of shape (n_dim, ?,) of float
        Values of the input variables of the problem with n_dim = nb of input variables.
    y_target : torch.tensor of shape (?,) of float
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
            # Optimizing free constants of programs where mask is True
            if mask[i]:
                # Getting minimum executable skeleton pickable program
                prog = progs.get_prog(i, skeleton=True)
                pool.apply_async(task_free_const_opti, args=(prog, X, y_target, free_const_opti_args))
        # Closing the pool of processes
        pool.close()
        pool.join()

    # Non parallel mode
    else:
        for i in range (progs.batch_size):
            # Optimizing free constants of programs where mask is True
            if mask[i]:
                # Getting minimum executable skeleton pickable program
                prog = progs.get_prog(i, skeleton=True)
                task_free_const_opti(prog, X = X, y_target = y_target, free_const_opti_args = free_const_opti_args)

    return None
