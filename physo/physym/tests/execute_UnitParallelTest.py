import unittest
import numpy as np
import time as time
import torch as torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

# Internal imports
import physo.physym.reward
from physo.physym import execute as Exec
from physo.physym import library as Lib
from physo.physym import program as Prog

# List of number of CPUs to test based on the total nb of CPUs (if 8 cores returns [8, 1, 2 4,])
def get_ncpus(max_ncpus):
    result = 0
    exponent = 0
    results = []
    while result < max_ncpus:
        result = 2 ** exponent
        exponent = exponent + 1
        if result < max_ncpus:
            results.append(result)
    # start with the highest then lowest to immediately see difference
    results = [max_ncpus,] + results
    return np.array(results)

# Pickable test function for BatchExecutionReduceGather
def TEST_REDUCE_WRAPPER(y):
    return y.mean()

class ExecuteProgramTest(unittest.TestCase):

    # Test parallelized execution (due to large data communication causes error on some linux systems -> commented).
    # def test_ParallelizedExe (self):
        #
        # DEVICE = 'cpu'
        # if torch.cuda.is_available():
        #     DEVICE = 'cuda'
        #
        # # LIBRARY CONFIG
        # args_make_tokens = {
        #                 # operations
        #                 "op_names"             : "all",  # or ["mul", "neg", "inv", "sin"]
        #                 "use_protected_ops"    : True,
        #                 # input variables
        #                 "input_var_ids"        : {"x" : 0         },
        #                 "input_var_units"      : {"x" : [0, 0, 0] },
        #                 "input_var_complexity" : {"x" : 0.        },
        #                 # constants
        #                 "constants"            : {"pi" : np.pi     , "1" : 1         },
        #                 "constants_units"      : {"pi" : [0, 0, 0] , "1" : [0, 0, 0] },
        #                 "constants_complexity" : {"pi" : 0.        , "1" : 1.        },
        #                 # free constants
        #                 "free_constants"            : {"a"             , "b"              },
        #                 "free_constants_init_val"   : {"a" : 1.        , "b"  : 1.        },
        #                 "free_constants_units"      : {"a" : [0, 0, 0] , "b"  : [0, 0, 0] },
        #                 "free_constants_complexity" : {"a" : 0.        , "b"  : 0.        },
        #                    }
        # my_lib = Lib.Library(args_make_tokens = args_make_tokens,
        #                      superparent_units = [0, 0, 0], superparent_name = "y")
        #
        # # TEST PROGRAM
        # batch_size = 10000
        # test_program_str = ["mul", "a", "sin", "mul", "x", "b"]
        # test_program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str])
        # test_program_length = len(test_program_str)
        # test_program_idx = np.tile(test_program_idx, reps=(batch_size,1))
        #
        # # BATCH
        # my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=test_program_length, library=my_lib)
        # my_programs.set_programs(test_program_idx)
        #
        # # TEST DATA
        # ideal_params = [1.14, 0.936] # Mock target free constants
        # n_params = len(ideal_params)
        # x = torch.tensor(np.linspace(-10, 10, 1000))
        # # Sending dataset to device te simulate a real environment
        # X = torch.stack((x,), axis=0).to(DEVICE)
        # y_target  = ideal_params[0]*torch.sin(ideal_params[1]*x).to(DEVICE)
        #
        # # MASK: WHICH PROGRAM SHOULD BE EXECUTED
        # # Simulating an evenly spread 90% load task
        # mask = np.random.rand(batch_size) < 0.9
        #
        # # Function to run the hyper-task once with a given config
        # def run (parallel=True, n_cpus=1):
        #     # Run tasks
        #     t0 = time.perf_counter()
        #     results = Exec.BatchExecution(progs=my_programs,
        #                                   X = X,
        #                                   mask = mask,
        #                                   parallel_mode = parallel,
        #                                   n_cpus = n_cpus)
        #     t1 = time.perf_counter()
        #     task_time = (t1 - t0) * 1e3 / mask.sum()
        #     #torch.set_printoptions(threshold=100)
        #     #print(results)
        #     return task_time
        #
        # is_parallel_exe_available = Exec.ParallelExeAvailability(verbose=True)
        #
        # # EFFICIENCY CURVE (NUMBER OF CPUS VS TASK TIME)
        #
        # print("\nParallelized execution test:")
        #
        # max_ncpus = mp.cpu_count()
        # print("Total nb. of CPUs: ", max_ncpus)
        #
        # # Getting computation times as a function of the number of CPUs
        # times = []
        # ncpus_list = get_ncpus(max_ncpus)
        # print("Testing nb. of CPUs = ", ncpus_list)
        # for ncpus in ncpus_list:
        #     print("n cpu =", ncpus)
        #     task_time = run(parallel=True, n_cpus=ncpus)
        #     # task_time = np.exp(-ncpus) # fast mock plot
        #     print("-> %f ms per task"%(task_time))
        #     times.append(task_time)
        # times = np.array(times)
        #
        # # Getting computation times when running in a non-parallelized loop
        # print("Not parallelized")
        # not_parallelized_time = run(parallel=False)
        # print("-> %f ms per task" % (not_parallelized_time))
        #
        # # Plot
        # fig,ax = plt.subplots(1,1)
        # fig.suptitle("Efficiency curve: execution")
        # ax.plot(ncpus_list, times, 'ko')
        # ax.plot(1, not_parallelized_time, 'ro', label="not parallelized")
        # ax.set_xlabel("Nb. of CPUs")
        # ax.set_ylabel("time [ms]")
        # ax.legend()
        # plt.show()

        # return None

    # Test parallelized execution + gathering of reduced floats
    def test_A_ParallelizedExeReduceGather (self):

        # Testing everything on CPU. If user has CUDA and wants to use CPU parallel mode, it is their responsability to
        # send the dataset to the proper device.
        DEVICE = 'cpu'
        #if torch.cuda.is_available():
        #    DEVICE = 'cuda'

        # LIBRARY CONFIG
        args_make_tokens = {
                        # operations
                        "op_names"             : "all",  # or ["mul", "neg", "inv", "sin"]
                        "use_protected_ops"    : True,
                        # input variables
                        "input_var_ids"        : {"x" : 0         },
                        "input_var_units"      : {"x" : [0, 0, 0] },
                        "input_var_complexity" : {"x" : 0.        },
                        # constants
                        "constants"            : {"pi" : np.pi     , "1" : 1         },
                        "constants_units"      : {"pi" : [0, 0, 0] , "1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 0.        , "1" : 1.        },
                        # free constants
                        "free_constants"            : {"a"             , "b"              },
                        "free_constants_init_val"   : {"a" : 1.        , "b"  : 1.        },
                        "free_constants_units"      : {"a" : [0, 0, 0] , "b"  : [0, 0, 0] },
                        "free_constants_complexity" : {"a" : 0.        , "b"  : 0.        },
                           }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [0, 0, 0], superparent_name = "y")

        # TEST PROGRAM
        batch_size = 10000
        # Using protected functions (eg. log) and (fixed constants eg. 1) to test their pickability
        # test_program_str = ["mul", "a", "sin", "mul", "x", "b",]
        # test_program_str = ["add", "mul", "a", "sin", "mul", "x", "b", "exp", "log", "x",]
        test_program_str = ["add", "mul", "a", "sin", "mul", "x", "b", "exp", "log", "add", "x", "sub", "1", "1"]
        test_program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str])
        test_program_length = len(test_program_str)
        test_program_idx = np.tile(test_program_idx, reps=(batch_size,1))

        # BATCH
        my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=test_program_length, library=my_lib)
        my_programs.set_programs(test_program_idx)

        # TEST DATA
        ideal_params = [1.14, 0.936] # Mock target free constants
        n_params = len(ideal_params)
        x = torch.tensor(np.linspace(-10, 10, 1000)) # parallel exe is worth it N > int(1e6)
        # Sending dataset to device te simulate a real environment
        X = torch.stack((x,), axis=0).to(DEVICE)
        y_target  = ideal_params[0]*torch.sin(ideal_params[1]*x).to(DEVICE)

        # MASK: WHICH PROGRAM SHOULD BE EXECUTED
        # Simulating an evenly spread 90% load task
        mask = np.random.rand(batch_size) < 0.9

        # Function to run the hyper-task once with a given config
        def run (parallel=True, n_cpus=1):
            # Run tasks
            t0 = time.perf_counter()
            results = Exec.BatchExecutionReduceGather(progs=my_programs,
                                                      X = X,
                                                      reduce_wrapper= TEST_REDUCE_WRAPPER,
                                                      mask = mask,
                                                      parallel_mode = parallel,
                                                      n_cpus = n_cpus)
            # Through VectPrograms method
            # results = my_programs.batch_exe_reduce_gather(
            #                                           X = X,
            #                                           reduce_wrapper= TEST_REDUCE_WRAPPER,
            #                                           mask = mask,
            #                                           parallel_mode = parallel,
            #                                           n_cpus = n_cpus)
            t1 = time.perf_counter()
            task_time = (t1 - t0) * 1e3 / mask.sum()
            #torch.set_printoptions(threshold=100)
            #print(results)
            return task_time

        is_parallel_exe_available = Exec.ParallelExeAvailability(verbose=True)

        # EFFICIENCY CURVE (NUMBER OF CPUS VS TASK TIME)

        print("\nParallelized execution test:")

        max_ncpus = mp.cpu_count()
        print("Total nb. of CPUs: ", max_ncpus)

        # Getting computation times as a function of the number of CPUs
        times = []
        ncpus_list = get_ncpus(max_ncpus)
        print("Testing nb. of CPUs = ", ncpus_list)
        for ncpus in ncpus_list:
            print("n cpu =", ncpus)
            task_time = run(parallel=True, n_cpus=ncpus)
            # task_time = np.exp(-ncpus) # fast mock plot
            print("-> %f ms per task"%(task_time))
            times.append(task_time)
        times = np.array(times)
        # Ordering results
        ncpus_list = np.array(ncpus_list[1:].tolist() + [ncpus_list[0]])
        times = np.array(times[1:].tolist() + [times[0]])

        # Getting computation times when running in a non-parallelized loop
        print("Not parallelized")
        not_parallelized_time = run(parallel=False)
        print("-> %f ms per task" % (not_parallelized_time))

        # Plot
        fig,ax = plt.subplots(1,1)
        # Is feature used ?
        enabled = physo.physym.reward.USE_PARALLEL_EXE
        fig.suptitle("Efficiency curve: execution and reduced gathering\n "
                     "Using parallelization in physo run : %s"%(str(enabled)))
        ax.plot(ncpus_list, times, 'k--')
        ax.plot(ncpus_list, times, 'ko')
        ax.plot(1, not_parallelized_time, 'ro', label="not parallelized")
        ax.set_xlabel("Nb. of CPUs")
        ax.set_ylabel("time [ms]")
        ax.legend()
        plt.show()

        return None
    # Test parallelized execution + gathering of reduced reward floats
    def test_B_ParallelizedExeReward (self):

        # Testing everything on CPU. If user has CUDA and wants to use CPU parallel mode, it is their responsability to
        # send the dataset to the proper device.
        DEVICE = 'cpu'
        #if torch.cuda.is_available():
        #    DEVICE = 'cuda'

        # LIBRARY CONFIG
        args_make_tokens = {
                        # operations
                        "op_names"             : "all",  # or ["mul", "neg", "inv", "sin"]
                        "use_protected_ops"    : True,
                        # input variables
                        "input_var_ids"        : {"x" : 0         },
                        "input_var_units"      : {"x" : [0, 0, 0] },
                        "input_var_complexity" : {"x" : 0.        },
                        # constants
                        "constants"            : {"pi" : np.pi     , "1" : 1         },
                        "constants_units"      : {"pi" : [0, 0, 0] , "1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 0.        , "1" : 1.        },
                        # free constants
                        "free_constants"            : {"a"             , "b"              },
                        "free_constants_init_val"   : {"a" : 1.        , "b"  : 1.        },
                        "free_constants_units"      : {"a" : [0, 0, 0] , "b"  : [0, 0, 0] },
                        "free_constants_complexity" : {"a" : 0.        , "b"  : 0.        },
                           }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [0, 0, 0], superparent_name = "y")

        # TEST PROGRAM
        batch_size = 10000
        # Using protected functions (eg. log) and (fixed constants eg. 1) to test their pickability
        # test_program_str = ["mul", "a", "sin", "mul", "x", "b",]
        # test_program_str = ["add", "mul", "a", "sin", "mul", "x", "b", "exp", "log", "x",]
        test_program_str = ["add", "mul", "a", "sin", "mul", "x", "b", "exp", "log", "add", "x", "sub", "1", "1"]
        test_program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str])
        test_program_length = len(test_program_str)
        test_program_idx = np.tile(test_program_idx, reps=(batch_size,1))

        # BATCH
        my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=test_program_length, library=my_lib)
        my_programs.set_programs(test_program_idx)

        # TEST DATA
        ideal_params = [1.14, 0.936] # Mock target free constants
        n_params = len(ideal_params)
        x = torch.tensor(np.linspace(-10, 10, 1000)) # parallel exe is worth it N > int(1e6)
        # Sending dataset to device te simulate a real environment
        X = torch.stack((x,), axis=0).to(DEVICE)
        y_target  = ideal_params[0]*torch.sin(ideal_params[1]*x).to(DEVICE)

        # MASK: WHICH PROGRAM SHOULD BE EXECUTED
        # Simulating an evenly spread 90% load task
        mask = np.random.rand(batch_size) < 0.9

        # Function to run the hyper-task once with a given config
        def run (parallel=True, n_cpus=1):
            # Run tasks
            t0 = time.perf_counter()
            results = Exec.BatchExecutionReward (progs=my_programs,
                                                 X = X,
                                                 y_target = y_target,
                                                 reward_function = physo.physym.reward.SquashedNRMSE,
                                                 mask = mask,
                                                 parallel_mode = parallel,
                                                 n_cpus = n_cpus)
            # Through VectPrograms method
            # results = my_programs.batch_exe_reward (
            #                                      X = X,
            #                                      y_target = y_target,
            #                                      reward_function = physo.physym.reward.SquashedNRMSE,
            #                                      mask = mask,
            #                                      parallel_mode = parallel,
            #                                      n_cpus = n_cpus)
            t1 = time.perf_counter()
            task_time = (t1 - t0) * 1e3 / mask.sum()
            #torch.set_printoptions(threshold=100)
            #print(results)
            return task_time

        is_parallel_exe_available = Exec.ParallelExeAvailability(verbose=True)

        # EFFICIENCY CURVE (NUMBER OF CPUS VS TASK TIME)

        print("\nParallelized execution / reward gather test:")

        max_ncpus = mp.cpu_count()
        print("Total nb. of CPUs: ", max_ncpus)

        # Getting computation times as a function of the number of CPUs
        times = []
        ncpus_list = get_ncpus(max_ncpus)
        print("Testing nb. of CPUs = ", ncpus_list)
        for ncpus in ncpus_list:
            print("n cpu =", ncpus)
            task_time = run(parallel=True, n_cpus=ncpus)
            # task_time = np.exp(-ncpus) # fast mock plot
            print("-> %f ms per task"%(task_time))
            times.append(task_time)
        times = np.array(times)
        # Ordering results
        ncpus_list = np.array(ncpus_list[1:].tolist() + [ncpus_list[0]])
        times = np.array(times[1:].tolist() + [times[0]])

        # Getting computation times when running in a non-parallelized loop
        print("Not parallelized")
        not_parallelized_time = run(parallel=False)
        print("-> %f ms per task" % (not_parallelized_time))

        # Plot
        fig,ax = plt.subplots(1,1)
        # Is feature used ?
        enabled = physo.physym.reward.USE_PARALLEL_EXE
        fig.suptitle("Efficiency curve: execution and reduced gathering of rewards\n "
                     "Using parallelization in physo run : %s"%(str(enabled)))
        ax.plot(ncpus_list, times, 'k--')
        ax.plot(ncpus_list, times, 'ko')
        ax.plot(1, not_parallelized_time, 'ro', label="not parallelized")
        ax.set_xlabel("Nb. of CPUs")
        ax.set_ylabel("time [ms]")
        ax.legend()
        plt.show()

        return None

    # Test parallelized execution of free constant optimization
    def test_C_ParallelizedExeFreeConstants (self):

        # Testing everything on CPU. If user has CUDA and wants to use CPU parallel mode, it is their responsability to
        # send the dataset to the proper device.
        DEVICE = 'cpu'
        #if torch.cuda.is_available():
        #    DEVICE = 'cuda'

        # LIBRARY CONFIG
        args_make_tokens = {
                        # operations
                        "op_names"             : "all",  # or ["mul", "neg", "inv", "sin"]
                        "use_protected_ops"    : True,
                        # input variables
                        "input_var_ids"        : {"x" : 0         },
                        "input_var_units"      : {"x" : [0, 0, 0] },
                        "input_var_complexity" : {"x" : 0.        },
                        # constants
                        "constants"            : {"pi" : np.pi     , "1" : 1         },
                        "constants_units"      : {"pi" : [0, 0, 0] , "1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 0.        , "1" : 1.        },
                        # free constants
                        "free_constants"            : {"a"             , "b"              },
                        "free_constants_init_val"   : {"a" : 1.        , "b"  : 1.        },
                        "free_constants_units"      : {"a" : [0, 0, 0] , "b"  : [0, 0, 0] },
                        "free_constants_complexity" : {"a" : 0.        , "b"  : 0.        },
                           }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [0, 0, 0], superparent_name = "y")

        # TEST PROGRAM
        batch_size = 10000
        # Testing with protected ops
        # Using protected functions (eg. log) and (fixed constants eg. 1) to test their pickability
        # test_program_str = ["mul", "a", "sin", "mul", "x", "b",]
        # test_program_str = ["add", "mul", "a", "sin", "mul", "x", "b", "exp", "log", "x",]
        test_program_str = ["add", "mul", "a", "sin", "mul", "x", "b", "exp", "log", "add", "x", "sub", "1", "1"]
        test_program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str])
        test_program_length = len(test_program_str)
        test_program_idx = np.tile(test_program_idx, reps=(batch_size,1))

        # BATCH
        my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=test_program_length, library=my_lib)
        my_programs.set_programs(test_program_idx)

        # TEST DATA
        ideal_params = [1.14, 0.936] # Mock target free constants
        n_params = len(ideal_params)
        x = torch.tensor(np.linspace(-10, 10, 1000))
        # Sending dataset to device te simulate a real environment
        X = torch.stack((x,), axis=0).to(DEVICE)
        y_target  = ideal_params[0]*torch.sin(ideal_params[1]*x).to(DEVICE)

        # FREE CONST OPTI CONFIG
        free_const_opti_args = {
            'loss'   : "MSE",
            'method' : 'LBFGS',
            'method_args': {
                        'n_steps' : 20,
                        'tol'     : 1e-8,
                        'lbfgs_func_args' : {
                            'max_iter'       : 4,
                            'line_search_fn' : "strong_wolfe",
                                             },
                            },
        }

        # MASK: WHICH PROGRAM SHOULD BE EXECUTED
        # Simulating an evenly spread 90% load task
        mask = np.random.rand(batch_size) < 0.9

        # Function to run the hyper-task once with a given config
        def run (parallel=True, n_cpus=1):
            # reset before each run, so it is not easier (early stop) to optimize free const next time
            my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=test_program_length, library=my_lib)
            my_programs.set_programs(test_program_idx)
            # Run tasks
            t0 = time.perf_counter()
            Exec.BatchFreeConstOpti(progs = my_programs,
                                    X = X,
                                    y_target = y_target,
                                    free_const_opti_args = free_const_opti_args,
                                    mask = mask,
                                    parallel_mode= parallel,
                                    n_cpus = n_cpus, )
            # Through VectPrograms method
            # my_programs.batch_optimize_constants(
            #                                     X = X,
            #                                     y_target = y_target,
            #                                     free_const_opti_args = free_const_opti_args,
            #                                     mask = mask,
            #                                     parallel_mode= parallel,
            #                                     n_cpus = n_cpus, )
            t1 = time.perf_counter()
            task_time = (t1 - t0) * 1e3 / mask.sum()
            return task_time

        is_parallel_exe_available = Exec.ParallelExeAvailability(verbose=True)

        # EFFICIENCY CURVE (NUMBER OF CPUS VS TASK TIME)

        print("\nParallelized free constant optimization test:")

        max_ncpus = mp.cpu_count()
        print("Total nb. of CPUs: ", max_ncpus)
        # Getting computation times as a function of the number of CPUs
        times = []
        ncpus_list = get_ncpus(max_ncpus)
        print("Testing nb. of CPUs = ", ncpus_list)
        for ncpus in ncpus_list:
            print("n cpu =", ncpus)
            task_time = run(parallel=True, n_cpus=ncpus)
            # task_time = np.exp(-ncpus) # fast mock plot
            print("-> %f ms per task"%(task_time))
            times.append(task_time)
        times = np.array(times)
        # Ordering results
        ncpus_list = np.array(ncpus_list[1:].tolist() + [ncpus_list[0]])
        times = np.array(times[1:].tolist() + [times[0]])

        # Getting computation times when running in a non-parallelized loop
        print("Not parallelized")
        not_parallelized_time = run(parallel=False)
        print("-> %f ms per task" % (not_parallelized_time))

        # Plot
        fig,ax = plt.subplots(1,1)
        enabled = physo.physym.reward.USE_PARALLEL_OPTI_CONST
        fig.suptitle("Efficiency curve: free const. opti.\n Using parallelization in physo run : %s"%(str(enabled)))
        ax.plot(ncpus_list, times, 'k--')
        ax.plot(ncpus_list, times, 'ko')
        ax.plot(1, not_parallelized_time, 'ro', label="not parallelized")
        ax.set_xlabel("Nb. of CPUs")
        ax.set_ylabel("time [ms]")
        ax.legend()
        plt.show()

        return None

if __name__ == '__main__':
    unittest.main(verbosity=2)
