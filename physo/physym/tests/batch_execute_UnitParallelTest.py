import unittest
import platform
import numpy as np
import time as time
import pandas as pd
import torch as torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import os
import importlib.util


# Internal imports
import physo.physym.reward
from physo.physym import batch_execute as BExec
from physo.physym import library as Lib
from physo.physym import vect_programs as VProg

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

# Should the tests be done via VectPrograms method which in turns uses BatchExecution ?
BOOL_DO_TEST_VIA_VECTPROGRAMS = False
# Should the testing figures be saved ?
DO_SAVE_FIGS = True

# System info for logging (using a function to get up-to-date get_start_method)
def get_system_info():
    system_info = {
        "mp_start_method" : mp.get_start_method(),
        "physo_installed" : False if importlib.util.find_spec("physo") is None else True,
        "torch_version"   : torch.__version__,
        "python_version"  : platform.python_version(),
        "os_type"         : platform.system(),
        "cpu_model"       : platform.processor(),
    }
    return system_info

class ExecuteProgramTest(unittest.TestCase):

    # Test parallelized execution (due to large data communication causes error on some linux systems -> commented).
    # def test_0_ParallelizedExe (self):
    #
    #     DEVICE = 'cpu'
    #     if torch.cuda.is_available():
    #         DEVICE = 'cuda'
    #
    #     # LIBRARY CONFIG
    #     args_make_tokens = {
    #                     # operations
    #                     "op_names"             : "all",  # or ["mul", "neg", "inv", "sin"]
    #                     "use_protected_ops"    : True,
    #                     # input variables
    #                     "input_var_ids"        : {"x" : 0         },
    #                     "input_var_units"      : {"x" : [0, 0, 0] },
    #                     "input_var_complexity" : {"x" : 0.        },
    #                     # constants
    #                     "constants"            : {"pi" : np.pi     , "1" : 1         },
    #                     "constants_units"      : {"pi" : [0, 0, 0] , "1" : [0, 0, 0] },
    #                     "constants_complexity" : {"pi" : 0.        , "1" : 1.        },
    #                     # free constants
    #                     "free_constants"            : {"a"             , "b"              },
    #                     "free_constants_init_val"   : {"a" : 1.        , "b"  : 1.        },
    #                     "free_constants_units"      : {"a" : [0, 0, 0] , "b"  : [0, 0, 0] },
    #                     "free_constants_complexity" : {"a" : 0.        , "b"  : 0.        },
    #                        }
    #     my_lib = Lib.Library(args_make_tokens = args_make_tokens,
    #                          superparent_units = [0, 0, 0], superparent_name = "y")
    #
    #     # TEST PROGRAM
    #     batch_size = 10000
    #     test_program_str = ["mul", "a", "sin", "mul", "x", "b"]
    #     test_program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str])
    #     test_program_length = len(test_program_str)
    #     test_program_idx = np.tile(test_program_idx, reps=(batch_size,1))
    #
    #     # BATCH
    #     my_programs = VProg.VectPrograms(batch_size=batch_size, max_time_step=test_program_length, library=my_lib, n_realizations=1)
    #     my_programs.set_programs(test_program_idx)
    #
    #     # TEST DATA
    #     ideal_params = [1.14, 0.936] # Mock target free constants
    #     n_params = len(ideal_params)
    #     x = torch.tensor(np.linspace(-10, 10, 100))
    #     # Sending dataset to device te simulate a real environment
    #     X = torch.stack((x,), axis=0).to(DEVICE)
    #     y_target  = ideal_params[0]*torch.sin(ideal_params[1]*x).to(DEVICE)
    #
    #     my_programs.free_consts.class_values[:, 0] = ideal_params[0]
    #     my_programs.free_consts.class_values[:, 1] = ideal_params[1]
    #
    #     # MASK: WHICH PROGRAM SHOULD BE EXECUTED
    #     # Simulating an evenly spread 90% load task
    #     mask = np.random.rand(batch_size) < 0.9
    #
    #     # Function to run the hyper-task once with a given config
    #     def run (parallel=True, n_cpus=1):
    #         # Run tasks
    #         t0 = time.perf_counter()
    #         results = BExec.BatchExecution(progs         = my_programs,
    #                                       X             = X,
    #                                       mask          = mask,
    #                                       parallel_mode = parallel,
    #                                       n_cpus        = n_cpus)
    #         t1 = time.perf_counter()
    #         task_time = (t1 - t0) * 1e3 / mask.sum()
    #         #torch.set_printoptions(threshold=100)
    #         #print(results)
    #         n_correct_computations = float((results == y_target.repeat((batch_size, 1))).all(axis=1).sum())
    #         perc_correct_computations = 100*n_correct_computations/mask.sum()
    #         print(" -> Correct computations: %f %%"%(perc_correct_computations))
    #         assert perc_correct_computations == 100, "Not all computations were correct"
    #
    #         return task_time
    #
    #     is_parallel_exe_available = BExec.ParallelExeAvailability(verbose=True)
    #
    #     # EFFICIENCY CURVE (NUMBER OF CPUS VS TASK TIME)
    #
    #     print("\nParallelized execution test:")
    #
    #     max_ncpus = mp.cpu_count()
    #     print("Total nb. of CPUs: ", max_ncpus)
    #
    #     # Getting computation times as a function of the number of CPUs
    #     times = []
    #     ncpus_list = get_ncpus(max_ncpus)
    #     print("Testing nb. of CPUs = ", ncpus_list)
    #     for ncpus in ncpus_list:
    #         print("n cpu =", ncpus)
    #         task_time = run(parallel=True, n_cpus=ncpus)
    #         # task_time = np.exp(-ncpus) # fast mock plot
    #         print("-> %f ms per task"%(task_time))
    #         times.append(task_time)
    #     times = np.array(times)
    #
    #     # Getting computation times when running in a non-parallelized loop
    #     print("Not parallelized")
    #     not_parallelized_time = run(parallel=False)
    #     print("-> %f ms per task" % (not_parallelized_time))
    #
    #     # Plot
    #     fig,ax = plt.subplots(1,1)
    #     fig.suptitle("Efficiency curve: execution")
    #     ax.plot(ncpus_list, times, 'ko')
    #     ax.plot(1, not_parallelized_time, 'ro', label="not parallelized")
    #     ax.set_xlabel("Nb. of CPUs")
    #     ax.set_ylabel("time [ms]")
    #     ax.legend()
    #     plt.show()
    #
    #     return None

    # Test parallelized execution (due to large data communication causes error on some linux systems -> commented).
    # def test_0_ParallelizedExe_with_spe_consts (self):
    #
    #     DEVICE = 'cpu'
    #     if torch.cuda.is_available():
    #         DEVICE = 'cuda'
    #
    #     # -------------------------------------- Making fake datasets --------------------------------------
    #
    #     multi_X = []
    #     for n_samples in [90, 100, 110]:
    #         x1 = np.linspace(0, 10, n_samples)
    #         x2 = np.linspace(0, 1 , n_samples)
    #         X = np.stack((x1,x2),axis=0)
    #         X = torch.tensor(X).to(DEVICE)
    #         multi_X.append(X)
    #     multi_X = multi_X*10                         # (n_realizations,) of (n_dim, [n_samples depends on dataset],)
    #
    #     n_samples_per_dataset = np.array([X.shape[1] for X in multi_X])
    #     n_all_samples = n_samples_per_dataset.sum()
    #     n_realizations = len(multi_X)
    #     def flatten_multi_data (multi_data,):
    #         """
    #         Flattens multiple datasets into a single one for vectorized evaluation.
    #         Parameters
    #         ----------
    #         multi_data : list of length (n_realizations,) of torch.tensor of shape (..., [n_samples depends on dataset],)
    #             List of datasets to be flattened.
    #         Returns
    #         -------
    #         torch.tensor of shape (..., n_all_samples)
    #             Flattened data (n_all_samples = sum([n_samples depends on dataset])).
    #         """
    #         flattened_data = torch.cat(multi_data, axis=-1) # (..., n_all_samples)
    #         return flattened_data
    #
    #     def unflatten_multi_data (flattened_data):
    #         """
    #         Unflattens a single data into multiple ones.
    #         Parameters
    #         ----------
    #         flattened_data : torch.tensor of shape (..., n_all_samples)
    #             Flattened data (n_all_samples = sum([n_samples depends on dataset])).
    #         Returns
    #         -------
    #         list of len (n_realizations,) of torch.tensor of shape (..., [n_samples depends on dataset],)
    #             Unflattened data.
    #         """
    #         return list(torch.split(flattened_data, n_samples_per_dataset.tolist(), dim=-1)) # (n_realizations,) of (..., [n_samples depends on dataset],)
    #
    #     # y_weights_per_dataset = np.array([0, 0.001, 1.0]*10) # Shows weights work
    #     y_weights_per_dataset = torch.tensor(np.array([1., 1., 1.]*10))
    #     multi_y_weights = [torch.full(size=(n_samples_per_dataset[i],), fill_value=y_weights_per_dataset[i]) for i in range (n_realizations)]
    #     y_weights_flatten = flatten_multi_data(multi_y_weights)
    #
    #     multi_X_flatten = flatten_multi_data(multi_X)  # (n_dim, n_all_samples)
    #
    #     # Making fake ideal parameters
    #     # n_spe_params   = 3
    #     # n_class_params = 2
    #     random_shift       = (np.random.rand(n_realizations,3)-0.5)*0.8
    #     ideal_spe_params   = torch.tensor(np.array([1.123, 0.345, 0.116]) + random_shift) # (n_realizations, n_spe_params,)
    #     ideal_spe_params   = ideal_spe_params.transpose(0,1)                              # (n_spe_params, n_realizations)
    #     ideal_class_params = torch.tensor(np.array([1.389, 1.005]))                       # (n_class_params, )
    #
    #     ideal_spe_params_flatten = torch.cat(
    #         [torch.tile(ideal_spe_params[:,i], (n_samples_per_dataset[i],1)).transpose(0,1) for i in range (n_realizations)], # (n_realizations,) of (n_spe_params, [n_samples depends on dataset],)
    #         axis = 1
    #     ) # (n_spe_params, n_all_samples)
    #
    #     ideal_class_params_flatten = torch.tile(ideal_class_params, (n_all_samples,1)).transpose(0,1) # (n_class_params, n_all_samples)
    #
    #     def trial_func (X, params, class_params):
    #         y = params[0]*torch.exp(-params[1]*X[0])*torch.cos(class_params[0]*X[0]+params[2]) + class_params[1]*X[1]
    #         return y
    #
    #     y_ideals_flatten = trial_func (multi_X_flatten, ideal_spe_params_flatten, ideal_class_params_flatten) # (n_all_samples,)
    #     multi_y_ideals   = unflatten_multi_data(y_ideals_flatten)                                         # (n_realizations,) of (n_samples depends on dataset,)
    #
    #
    #     # params[0]*torch.exp(-params[1]*X[0])*torch.cos(class_params[0]*X[0]+params[2]) + class_params[1]*X[1]
    #     # k0 * exp(-k1 * t) * cos(c0 * t + k2) + c1 * l
    #     # "add", "mul", "mul", "k0", "exp", "mul", "neg", "k1", "t", "cos", "add", "mul", "c0", "t", "k2", "mul", "c1", "l"
    #
    #     k0_init = [1.,1.,1.]*10 # np.full(n_realizations, 1.)
    #     # consts
    #     pi     = torch.tensor (np.pi) .to(DEVICE)
    #     const1 = torch.tensor (1.)    .to(DEVICE)
    #
    #     # LIBRARY CONFIG
    #     args_make_tokens = {
    #                     # operations
    #                     "op_names"             : "all",
    #                     "use_protected_ops"    : True,
    #                     # input variables
    #                     "input_var_ids"        : {"t" : 0         , "l" : 1          },
    #                     "input_var_units"      : {"t" : [1, 0, 0] , "l" : [0, 1, 0]  },
    #                     "input_var_complexity" : {"t" : 0.        , "l" : 1.         },
    #                     # constants
    #                     "constants"            : {"pi" : pi        , "const1" : const1    },
    #                     "constants_units"      : {"pi" : [0, 0, 0] , "const1" : [0, 0, 0] },
    #                     "constants_complexity" : {"pi" : 1.        , "const1" : 1.        },
    #                     # free constants
    #                     "class_free_constants"            : {"c0"              , "c1"               },
    #                     "class_free_constants_init_val"   : {"c0" : 1.         , "c1"  : 1.         },
    #                     "class_free_constants_units"      : {"c0" : [-1, 0, 0] , "c1"  : [0, -1, 0] },
    #                     "class_free_constants_complexity" : {"c0" : 1.         , "c1"  : 1.         },
    #                     # free constants
    #                     "spe_free_constants"            : {"k0"              , "k1"               , "k2"               },
    #                     "spe_free_constants_init_val"   : {"k0" : k0_init    , "k1"  : 1.         , "k2"  : 1.         },
    #                     "spe_free_constants_units"      : {"k0" : [0, 0, 0]  , "k1"  : [-1, 0, 0] , "k2"  : [0, 0, 0]  },
    #                     "spe_free_constants_complexity" : {"k0" : 1.         , "k1"  : 1.         , "k2"  : 1.         },
    #                        }
    #     my_lib = Lib.Library(args_make_tokens = args_make_tokens,
    #                          superparent_units = [0, 0, 0], superparent_name = "y")
    #
    #     # TEST PROGRAM
    #     batch_size = 10000
    #     test_program_str = ["add", "mul", "mul", "k0"  , "exp", "mul", "neg", "k1", "t", "cos", "add", "mul", "c0", "t", "k2", "mul", "c1", "l", ]
    #     test_program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str])
    #     test_program_length = len(test_program_str)
    #     test_program_idx = np.tile(test_program_idx, reps=(batch_size,1))
    #
    #     # BATCH
    #     my_programs = VProg.VectPrograms(batch_size=batch_size, max_time_step=test_program_length, library=my_lib, n_realizations=n_realizations)
    #     my_programs.set_programs(test_program_idx)
    #
    #     # ENCODING CORRECT FREE CONSTANTS
    #     my_programs.free_consts.class_values = ideal_class_params.repeat((batch_size,1))   # (batch_size, n_class_params)
    #     my_programs.free_consts.spe_values   = ideal_spe_params.repeat((batch_size, 1, 1)) # (batch_size, n_spe_params, n_realizations)
    #
    #
    #     # MASK: WHICH PROGRAM SHOULD BE EXECUTED
    #     # Simulating an evenly spread 90% load task
    #     mask = np.random.rand(batch_size) < 0.9
    #
    #     # Function to run the hyper-task once with a given config
    #     def run (parallel=True, n_cpus=1):
    #         # Run tasks
    #         t0 = time.perf_counter()
    #         results = BExec.BatchExecution(progs                 = my_programs,
    #                                       X                     = multi_X_flatten,
    #                                       n_samples_per_dataset = n_samples_per_dataset, # Realization related
    #                                       mask                  = mask,
    #                                       parallel_mode         = parallel,
    #                                       n_cpus                = n_cpus)
    #         t1 = time.perf_counter()
    #         task_time = (t1 - t0) * 1e3 / mask.sum()
    #         #torch.set_printoptions(threshold=100)
    #         #print(results)
    #         n_correct_computations = float((results == y_ideals_flatten.repeat((batch_size, 1))).all(axis=1).sum())
    #         perc_correct_computations = 100*n_correct_computations/mask.sum()
    #         print(" -> Correct computations: %f %%"%(perc_correct_computations))
    #         assert perc_correct_computations == 100, "Not all computations were correct"
    #
    #         #
    #         # # Sanity plot
    #         # y_computed_flatten = results[0]
    #         # multi_y_computed = unflatten_multi_data(y_computed_flatten)
    #         # fig, ax = plt.subplots(1,1,figsize=(10,5))
    #         # for i in range(n_realizations):
    #         #     ax.plot(multi_X[i][0], multi_y_ideals   [i].cpu().detach().numpy(), 'o', )
    #         #     ax.plot(multi_X[i][0], multi_y_computed [i].cpu().detach().numpy(), 'r-',)
    #         # ax.legend()
    #         # plt.show()
    #         # for i in range(n_realizations):
    #         #     mse = torch.mean((multi_y_computed[i] - multi_y_ideals[i])**2)
    #         #     print("%i, mse = %f"%(i, mse))
    #
    #         return task_time
    #
    #     is_parallel_exe_available = BExec.ParallelExeAvailability(verbose=True)
    #
    #     # EFFICIENCY CURVE (NUMBER OF CPUS VS TASK TIME)
    #
    #     print("\nParallelized execution test:")
    #
    #     max_ncpus = mp.cpu_count()
    #     print("Total nb. of CPUs: ", max_ncpus)
    #
    #     # Getting computation times as a function of the number of CPUs
    #     times = []
    #     ncpus_list = get_ncpus(max_ncpus)
    #     print("Testing nb. of CPUs = ", ncpus_list)
    #     for ncpus in ncpus_list:
    #         print("n cpu =", ncpus)
    #         task_time = run(parallel=True, n_cpus=ncpus)
    #         # task_time = np.exp(-ncpus) # fast mock plot
    #         print("-> %f ms per task"%(task_time))
    #         times.append(task_time)
    #     times = np.array(times)
    #
    #     # Getting computation times when running in a non-parallelized loop
    #     print("Not parallelized")
    #     not_parallelized_time = run(parallel=False)
    #     print("-> %f ms per task" % (not_parallelized_time))
    #
    #     # Plot
    #     fig,ax = plt.subplots(1,1)
    #     fig.suptitle("Efficiency curve: execution")
    #     ax.plot(ncpus_list, times, 'ko')
    #     ax.plot(1, not_parallelized_time, 'ro', label="not parallelized")
    #     ax.set_xlabel("Nb. of CPUs")
    #     ax.set_ylabel("time [ms]")
    #     ax.legend()
    #     plt.show()
    #
    #     return None

    # Test parallelized execution + gathering of reduced floats
    def test_03_A_ParallelizedExeReduceGather (self):

        # Testing everything on CPU. If user has CUDA and wants to use CPU parallel mode, it is their responsibility to
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
        test_program_str = ["mul", "a", "sin", "mul", "x", "b",]
        # test_program_str = ["add", "mul", "a", "sin", "mul", "x", "b", "exp", "log", "x",]
        # test_program_str = ["add", "mul", "a", "sin", "mul", "x", "b", "exp", "log", "add", "x", "sub", "1", "1"]
        # -> using a*sin(x*b) + exp(log(x+1-1)) may produce NaN resulting in error
        test_program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str])
        test_program_length = len(test_program_str)
        test_program_idx = np.tile(test_program_idx, reps=(batch_size,1))

        # BATCH
        my_programs = VProg.VectPrograms(batch_size=batch_size, max_time_step=test_program_length, library=my_lib, n_realizations=1)
        my_programs.set_programs(test_program_idx)

        # TEST DATA
        ideal_params = [1.14, 0.936] # Mock target free constants
        n_params = len(ideal_params)
        x = torch.tensor(np.linspace(-10, 10, 1000)) # parallel exe is worth it N > int(1e6)
        # Sending dataset to device te simulate a real environment
        X = torch.stack((x,), axis=0).to(DEVICE)
        y_target  = ideal_params[0]*torch.sin(ideal_params[1]*x).to(DEVICE)

        my_programs.free_consts.class_values[:, 0] = ideal_params[0]
        my_programs.free_consts.class_values[:, 1] = ideal_params[1]

        expected_results = TEST_REDUCE_WRAPPER(y_target).repeat((batch_size)) # (batch_size,)
        expected_results = expected_results.cpu().detach().numpy()

        # MASK: WHICH PROGRAM SHOULD BE EXECUTED
        # Simulating an evenly spread 90% load task
        mask = np.random.rand(batch_size) < 0.9

        # Function to run the hyper-task once with a given config
        def run (parallel=True, n_cpus=1):
            # Run tasks
            t0 = time.perf_counter()
            if not BOOL_DO_TEST_VIA_VECTPROGRAMS:
                results = BExec.BatchExecutionReduceGather(progs=my_programs,
                                                          X = X,
                                                          reduce_wrapper= TEST_REDUCE_WRAPPER,
                                                          mask = mask,
                                                          parallel_mode = parallel,
                                                          n_cpus = n_cpus)
            else:
                # Through VectPrograms method
                results = my_programs.batch_exe_reduce_gather(
                                                           X = X,
                                                           reduce_wrapper= TEST_REDUCE_WRAPPER,
                                                           mask = mask,
                                                           parallel_mode = parallel,
                                                           n_cpus = n_cpus)
            t1 = time.perf_counter()
            task_time = (t1 - t0) * 1e3 / mask.sum()
            #torch.set_printoptions(threshold=100)
            n_correct_computations = float((results == expected_results).sum())
            perc_correct_computations = 100*n_correct_computations/mask.sum()
            print(" -> Correct computations: %f %%"%(perc_correct_computations))
            assert perc_correct_computations == 100, "Not all computations were correct"

            return task_time

        is_parallel_exe_available = BExec.ParallelExeAvailability(verbose=True)

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

        # Logging
        is_parallel_effective = (np.min(times) < not_parallelized_time)
        log_times = times      .tolist() + [not_parallelized_time,]
        log_ncpus = ncpus_list .tolist() + ["np",]
        df = pd.DataFrame({"ncpus": log_ncpus, "time [ms]": log_times})
        df["parallel_effective"] = is_parallel_effective
        # adding system info to each line
        system_info = get_system_info()
        for key in system_info.keys():
            df[key] = system_info[key]

        name = "perf_A_ParallelizedExeReduceGather"

        if DO_SAVE_FIGS:
            fig.savefig(name + ".png")
            # If the file already exists, append to it
            if os.path.isfile(name + ".csv"):
                df.to_csv(name + ".csv", mode='a', header=False)
            else:
                df.to_csv(name + ".csv")

        plt.show()

        return None

    # Test parallelized execution + gathering of reduced reward floats
    def test_04_B_ParallelizedExeReward (self):

        # Testing everything on CPU. If user has CUDA and wants to use CPU parallel mode, it is their responsibility to
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
        my_programs = VProg.VectPrograms(batch_size=batch_size, max_time_step=test_program_length, library=my_lib, n_realizations=1)
        my_programs.set_programs(test_program_idx)

        # TEST DATA
        ideal_params = [1.14, 0.936] # Mock target free constants
        n_params = len(ideal_params)
        x = torch.tensor(np.linspace(-10, 10, 1000)) # parallel exe is worth it N > int(1e6)
        # Sending dataset to device te simulate a real environment
        X = torch.stack((x,), axis=0).to(DEVICE)
        y_target  = ideal_params[0]*torch.sin(ideal_params[1]*x).to(DEVICE)

        # Not spoiling free constants to get different values for y_pred and y_target
        #my_programs.free_consts.class_values[:, 0] = ideal_params[0]
        #my_programs.free_consts.class_values[:, 1] = ideal_params[1]
        y_pred = my_programs.get_prog(0).execute(X)

        expected_results = physo.physym.reward.SquashedNRMSE(y_target=y_target, y_pred=y_pred).repeat((batch_size)) # (batch_size,)
        expected_results = expected_results.cpu().detach().numpy()

        # MASK: WHICH PROGRAM SHOULD BE EXECUTED
        # Simulating an evenly spread 90% load task
        mask = np.random.rand(batch_size) < 0.9

        # Function to run the hyper-task once with a given config
        def run (parallel=True, n_cpus=1):
            # Run tasks
            t0 = time.perf_counter()
            if not BOOL_DO_TEST_VIA_VECTPROGRAMS:
                results = BExec.BatchExecutionReward (progs=my_programs,
                                                     X = X,
                                                     y_target = y_target,
                                                     reward_function = physo.physym.reward.SquashedNRMSE,
                                                     mask = mask,
                                                     parallel_mode = parallel,
                                                     n_cpus = n_cpus)
            else:
                # Through VectPrograms method
                results = my_programs.batch_exe_reward (
                                                     X = X,
                                                     y_target = y_target,
                                                     reward_function = physo.physym.reward.SquashedNRMSE,
                                                     mask = mask,
                                                     parallel_mode = parallel,
                                                     n_cpus = n_cpus)
            t1 = time.perf_counter()
            task_time = (t1 - t0) * 1e3 / mask.sum()
            #torch.set_printoptions(threshold=100)
            n_correct_computations    = float((results == expected_results).sum())
            perc_correct_computations = 100*n_correct_computations/mask.sum()
            print(" -> Correct computations: %f %%"%(perc_correct_computations))
            assert perc_correct_computations == 100, "Not all computations were correct"

            return task_time

        is_parallel_exe_available = BExec.ParallelExeAvailability(verbose=True)

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

        # Logging
        is_parallel_effective = (np.min(times) < not_parallelized_time)
        log_times = times      .tolist() + [not_parallelized_time,]
        log_ncpus = ncpus_list .tolist() + ["np",]
        df = pd.DataFrame({"ncpus": log_ncpus, "time [ms]": log_times})
        df["parallel_effective"] = is_parallel_effective
        # adding system info to each line
        system_info = get_system_info()
        for key in system_info.keys():
            df[key] = system_info[key]

        name = "perf_B_ParallelizedExeReward"

        if DO_SAVE_FIGS:
            fig.savefig(name + ".png")
            # If the file already exists, append to it
            if os.path.isfile(name + ".csv"):
                df.to_csv(name + ".csv", mode='a', header=False)
            else:
                df.to_csv(name + ".csv")

        plt.show()

        return None

    # Test parallelized execution of free constant optimization
    def test_01_C_ParallelizedExeFreeConstants (self):

        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Testing everything on CPU. If user has CUDA and wants to use CPU parallel mode, it is their responsibility to
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
        my_programs = VProg.VectPrograms(batch_size=batch_size, max_time_step=test_program_length, library=my_lib, n_realizations=1)
        my_programs.set_programs(test_program_idx)

        # TEST DATA
        ideal_params = torch.tensor([1.14, 0.936]) # Mock target free constants
        n_params = len(ideal_params)
        x = torch.tensor(np.linspace(-10, 10, 1000))
        # Sending dataset to device te simulate a real environment
        X = torch.stack((x,), axis=0).to(DEVICE)
        y_target  = ideal_params[0]*torch.sin(ideal_params[1]*x).to(DEVICE)

        # Expected results
        tol = 1e-4
        expected_class_vals = ideal_params.repeat((batch_size,1))               # (batch_size, n_class_params)

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
            my_programs = VProg.VectPrograms(batch_size=batch_size, max_time_step=test_program_length, library=my_lib, n_realizations=1)
            my_programs.set_programs(test_program_idx)
            # Run tasks
            t0 = time.perf_counter()
            if not BOOL_DO_TEST_VIA_VECTPROGRAMS:
                BExec.BatchFreeConstOpti(progs = my_programs,
                                        X = X,
                                        y_target = y_target,
                                        free_const_opti_args = free_const_opti_args,
                                        mask = mask,
                                        parallel_mode= parallel,
                                        n_cpus = n_cpus, )
            else:
                # Through VectPrograms method
                my_programs.batch_optimize_constants(
                                                    X = X,
                                                    y_target = y_target,
                                                    free_const_opti_args = free_const_opti_args,
                                                    mask = mask,
                                                    parallel_mode= parallel,
                                                    n_cpus = n_cpus, )
            t1 = time.perf_counter()
            task_time = (t1 - t0) * 1e3 / mask.sum()
            #torch.set_printoptions(threshold=100)
            #print(expected_class_vals)
            n_correct_computations = float((torch.abs((my_programs.free_consts.class_values - expected_class_vals)) < tol).all(axis=-1).sum())
            perc_correct_computations = 100*n_correct_computations/mask.sum()
            # Checking that logging worked
            is_correct_log = torch.logical_and((my_programs.free_consts.is_opti == True), (my_programs.free_consts.opti_steps > 0))
            n_correct_log = float(is_correct_log.sum())
            perc_correct_log = 100*n_correct_log/mask.sum()
            print(" -> Correct computations : %f %%" % (perc_correct_computations))
            print(" -> Correct logging      : %f %%" % (perc_correct_log)         )
            assert perc_correct_computations == 100, "Not all computations were correct"
            assert perc_correct_log          == 100, "Not all logging were correct"

            return task_time

        is_parallel_exe_available = BExec.ParallelExeAvailability(verbose=True)

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

        # Logging
        is_parallel_effective = (np.min(times) < not_parallelized_time)
        log_times = times      .tolist() + [not_parallelized_time,]
        log_ncpus = ncpus_list .tolist() + ["np",]
        df = pd.DataFrame({"ncpus": log_ncpus, "time [ms]": log_times})
        df["parallel_effective"] = is_parallel_effective
        # adding system info to each line
        system_info = get_system_info()
        for key in system_info.keys():
            df[key] = system_info[key]

        name = "perf_C_ParallelizedExeFreeConstants"

        if DO_SAVE_FIGS:
            fig.savefig(name + ".png")
            # If the file already exists, append to it
            if os.path.isfile(name + ".csv"):
                df.to_csv(name + ".csv", mode='a', header=False)
            else:
                df.to_csv(name + ".csv")

        plt.show()

        return None

    # Test parallelized execution + gathering of reduced floats
    def test_05_A_ParallelizedExeReduceGather_with_spe_consts (self):

        # Testing everything on CPU. If user has CUDA and wants to use CPU parallel mode, it is their responsibility to
        # send the dataset to the proper device.
        DEVICE = 'cpu'
        #if torch.cuda.is_available():
        #    DEVICE = 'cuda'


        # -------------------------------------- Making fake datasets --------------------------------------

        multi_X = []
        for n_samples in [90, 100, 110]:
            x1 = np.linspace(0, 10, n_samples)
            x2 = np.linspace(0, 1 , n_samples)
            X = np.stack((x1,x2),axis=0)
            X = torch.tensor(X).to(DEVICE)
            multi_X.append(X)
        multi_X = multi_X*10                         # (n_realizations,) of (n_dim, [n_samples depends on dataset],)

        n_samples_per_dataset = np.array([X.shape[1] for X in multi_X])
        n_all_samples = n_samples_per_dataset.sum()
        n_realizations = len(multi_X)
        def flatten_multi_data (multi_data,):
            """
            Flattens multiple datasets into a single one for vectorized evaluation.
            Parameters
            ----------
            multi_data : list of length (n_realizations,) of torch.tensor of shape (..., [n_samples depends on dataset],)
                List of datasets to be flattened.
            Returns
            -------
            torch.tensor of shape (..., n_all_samples)
                Flattened data (n_all_samples = sum([n_samples depends on dataset])).
            """
            flattened_data = torch.cat(multi_data, axis=-1) # (..., n_all_samples)
            return flattened_data

        def unflatten_multi_data (flattened_data):
            """
            Unflattens a single data into multiple ones.
            Parameters
            ----------
            flattened_data : torch.tensor of shape (..., n_all_samples)
                Flattened data (n_all_samples = sum([n_samples depends on dataset])).
            Returns
            -------
            list of len (n_realizations,) of torch.tensor of shape (..., [n_samples depends on dataset],)
                Unflattened data.
            """
            return list(torch.split(flattened_data, n_samples_per_dataset.tolist(), dim=-1)) # (n_realizations,) of (..., [n_samples depends on dataset],)

        # y_weights_per_dataset = np.array([0, 0.001, 1.0]*10) # Shows weights work
        y_weights_per_dataset = torch.tensor(np.array([1., 1., 1.]*10))
        multi_y_weights = [torch.full(size=(n_samples_per_dataset[i],), fill_value=y_weights_per_dataset[i]) for i in range (n_realizations)]
        y_weights_flatten = flatten_multi_data(multi_y_weights)

        multi_X_flatten = flatten_multi_data(multi_X)  # (n_dim, n_all_samples)

        # Making fake ideal parameters
        # n_spe_params   = 3
        # n_class_params = 2
        random_shift       = (np.random.rand(n_realizations,3)-0.5)*0.8
        ideal_spe_params   = torch.tensor(np.array([1.123, 0.345, 0.116]) + random_shift) # (n_realizations, n_spe_params,)
        ideal_spe_params   = ideal_spe_params.transpose(0,1)                              # (n_spe_params, n_realizations)
        ideal_class_params = torch.tensor(np.array([1.389, 1.005]))                       # (n_class_params, )

        ideal_spe_params_flatten = torch.cat(
            [torch.tile(ideal_spe_params[:,i], (n_samples_per_dataset[i],1)).transpose(0,1) for i in range (n_realizations)], # (n_realizations,) of (n_spe_params, [n_samples depends on dataset],)
            axis = 1
        ) # (n_spe_params, n_all_samples)

        ideal_class_params_flatten = torch.tile(ideal_class_params, (n_all_samples,1)).transpose(0,1) # (n_class_params, n_all_samples)

        def trial_func (X, params, class_params):
            y = params[0]*torch.exp(-params[1]*X[0])*torch.cos(class_params[0]*X[0]+params[2]) + class_params[1]*X[1]
            return y

        y_ideals_flatten = trial_func (multi_X_flatten, ideal_spe_params_flatten, ideal_class_params_flatten) # (n_all_samples,)
        multi_y_ideals   = unflatten_multi_data(y_ideals_flatten)                                         # (n_realizations,) of (n_samples depends on dataset,)


        # params[0]*torch.exp(-params[1]*X[0])*torch.cos(class_params[0]*X[0]+params[2]) + class_params[1]*X[1]
        # k0 * exp(-k1 * t) * cos(c0 * t + k2) + c1 * l
        # "add", "mul", "mul", "k0", "exp", "mul", "neg", "k1", "t", "cos", "add", "mul", "c0", "t", "k2", "mul", "c1", "l"

        k0_init = [1.,1.,1.]*10 # np.full(n_realizations, 1.)
        # consts
        pi     = torch.tensor (np.pi) .to(DEVICE)
        const1 = torch.tensor (1.)    .to(DEVICE)

        # LIBRARY CONFIG
        args_make_tokens = {
                        # operations
                        "op_names"             : "all",
                        "use_protected_ops"    : True,
                        # input variables
                        "input_var_ids"        : {"t" : 0         , "l" : 1          },
                        "input_var_units"      : {"t" : [1, 0, 0] , "l" : [0, 1, 0]  },
                        "input_var_complexity" : {"t" : 0.        , "l" : 1.         },
                        # constants
                        "constants"            : {"pi" : pi        , "const1" : const1    },
                        "constants_units"      : {"pi" : [0, 0, 0] , "const1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 1.        , "const1" : 1.        },
                        # free constants
                        "class_free_constants"            : {"c0"              , "c1"               },
                        "class_free_constants_init_val"   : {"c0" : 1.         , "c1"  : 1.         },
                        "class_free_constants_units"      : {"c0" : [-1, 0, 0] , "c1"  : [0, -1, 0] },
                        "class_free_constants_complexity" : {"c0" : 1.         , "c1"  : 1.         },
                        # free constants
                        "spe_free_constants"            : {"k0"              , "k1"               , "k2"               },
                        "spe_free_constants_init_val"   : {"k0" : k0_init    , "k1"  : 1.         , "k2"  : 1.         },
                        "spe_free_constants_units"      : {"k0" : [0, 0, 0]  , "k1"  : [-1, 0, 0] , "k2"  : [0, 0, 0]  },
                        "spe_free_constants_complexity" : {"k0" : 1.         , "k1"  : 1.         , "k2"  : 1.         },
                           }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [0, 0, 0], superparent_name = "y")

        # TEST PROGRAM
        batch_size = 10000
        test_program_str = ["add", "mul", "mul", "k0"  , "exp", "mul", "neg", "k1", "t", "cos", "add", "mul", "c0", "t", "k2", "mul", "c1", "l", ]
        test_program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str])
        test_program_length = len(test_program_str)
        test_program_idx = np.tile(test_program_idx, reps=(batch_size,1))

        # BATCH
        my_programs = VProg.VectPrograms(batch_size=batch_size, max_time_step=test_program_length, library=my_lib, n_realizations=n_realizations)
        my_programs.set_programs(test_program_idx)

        # ENCODING CORRECT FREE CONSTANTS
        my_programs.free_consts.class_values = ideal_class_params.repeat((batch_size,1))   # (batch_size, n_class_params)
        my_programs.free_consts.spe_values   = ideal_spe_params.repeat((batch_size, 1, 1)) # (batch_size, n_spe_params, n_realizations)

        expected_results = TEST_REDUCE_WRAPPER(y_ideals_flatten).repeat((batch_size)) # (batch_size,)
        expected_results = expected_results.cpu().detach().numpy()

        # MASK: WHICH PROGRAM SHOULD BE EXECUTED
        # Simulating an evenly spread 90% load task
        mask = np.random.rand(batch_size) < 0.9

        # Function to run the hyper-task once with a given config
        def run (parallel=True, n_cpus=1):
            # Run tasks
            t0 = time.perf_counter()
            if not BOOL_DO_TEST_VIA_VECTPROGRAMS:
                results = BExec.BatchExecutionReduceGather(progs=my_programs,
                                                          X = multi_X_flatten,
                                                          reduce_wrapper= TEST_REDUCE_WRAPPER,
                                                          n_samples_per_dataset = n_samples_per_dataset, # Realization related
                                                          mask = mask,
                                                          parallel_mode = parallel,
                                                          n_cpus = n_cpus)
            else:
                # Through VectPrograms method
                results = my_programs.batch_exe_reduce_gather(
                                                          X = multi_X_flatten,
                                                          reduce_wrapper= TEST_REDUCE_WRAPPER,
                                                          n_samples_per_dataset = n_samples_per_dataset, # Realization related
                                                          mask = mask,
                                                          parallel_mode = parallel,
                                                          n_cpus = n_cpus)
            t1 = time.perf_counter()
            task_time = (t1 - t0) * 1e3 / mask.sum()
            #torch.set_printoptions(threshold=100)
            n_correct_computations = float((results == expected_results).sum())
            perc_correct_computations = 100*n_correct_computations/mask.sum()
            print(" -> Correct computations: %f %%"%(perc_correct_computations))
            assert perc_correct_computations == 100, "Not all computations were correct"

            return task_time

        is_parallel_exe_available = BExec.ParallelExeAvailability(verbose=True)

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

        # Logging
        is_parallel_effective = (np.min(times) < not_parallelized_time)
        log_times = times      .tolist() + [not_parallelized_time,]
        log_ncpus = ncpus_list .tolist() + ["np",]
        df = pd.DataFrame({"ncpus": log_ncpus, "time [ms]": log_times})
        df["parallel_effective"] = is_parallel_effective
        # adding system info to each line
        system_info = get_system_info()
        for key in system_info.keys():
            df[key] = system_info[key]

        name = "perf_A_ParallelizedExeReduceGather_with_spe_consts"

        if DO_SAVE_FIGS:
            fig.savefig(name + ".png")
            # If the file already exists, append to it
            if os.path.isfile(name + ".csv"):
                df.to_csv(name + ".csv", mode='a', header=False)
            else:
                df.to_csv(name + ".csv")

        plt.show()

        return None

    # Test parallelized execution + gathering of reduced reward floats
    def test_06_B_ParallelizedExeReward_with_spe_consts (self):

        # Testing everything on CPU. If user has CUDA and wants to use CPU parallel mode, it is their responsibility to
        # send the dataset to the proper device.
        DEVICE = 'cpu'
        #if torch.cuda.is_available():
        #    DEVICE = 'cuda'

        # -------------------------------------- Making fake datasets --------------------------------------

        multi_X = []
        for n_samples in [90, 100, 110]:
            x1 = np.linspace(0, 10, n_samples)
            x2 = np.linspace(0, 1 , n_samples)
            X = np.stack((x1,x2),axis=0)
            X = torch.tensor(X).to(DEVICE)
            multi_X.append(X)
        multi_X = multi_X*10                         # (n_realizations,) of (n_dim, [n_samples depends on dataset],)

        n_samples_per_dataset = np.array([X.shape[1] for X in multi_X])
        n_all_samples = n_samples_per_dataset.sum()
        n_realizations = len(multi_X)
        def flatten_multi_data (multi_data,):
            """
            Flattens multiple datasets into a single one for vectorized evaluation.
            Parameters
            ----------
            multi_data : list of length (n_realizations,) of torch.tensor of shape (..., [n_samples depends on dataset],)
                List of datasets to be flattened.
            Returns
            -------
            torch.tensor of shape (..., n_all_samples)
                Flattened data (n_all_samples = sum([n_samples depends on dataset])).
            """
            flattened_data = torch.cat(multi_data, axis=-1) # (..., n_all_samples)
            return flattened_data

        def unflatten_multi_data (flattened_data):
            """
            Unflattens a single data into multiple ones.
            Parameters
            ----------
            flattened_data : torch.tensor of shape (..., n_all_samples)
                Flattened data (n_all_samples = sum([n_samples depends on dataset])).
            Returns
            -------
            list of len (n_realizations,) of torch.tensor of shape (..., [n_samples depends on dataset],)
                Unflattened data.
            """
            return list(torch.split(flattened_data, n_samples_per_dataset.tolist(), dim=-1)) # (n_realizations,) of (..., [n_samples depends on dataset],)

        y_weights_per_dataset = np.array([0, 0.001, 1.0]*10) # Shows weights work
        # y_weights_per_dataset = torch.tensor(np.array([1., 1., 1.]*10))
        multi_y_weights = [torch.full(size=(n_samples_per_dataset[i],), fill_value=y_weights_per_dataset[i]) for i in range (n_realizations)]
        y_weights_flatten = flatten_multi_data(multi_y_weights)

        multi_X_flatten = flatten_multi_data(multi_X)  # (n_dim, n_all_samples)

        # Making fake ideal parameters
        # n_spe_params   = 3
        # n_class_params = 2
        random_shift       = (np.random.rand(n_realizations,3)-0.5)*0.8
        ideal_spe_params   = torch.tensor(np.array([1.123, 0.345, 0.116]) + random_shift) # (n_realizations, n_spe_params,)
        ideal_spe_params   = ideal_spe_params.transpose(0,1)                              # (n_spe_params, n_realizations)
        ideal_class_params = torch.tensor(np.array([1.389, 1.005]))                       # (n_class_params, )

        ideal_spe_params_flatten = torch.cat(
            [torch.tile(ideal_spe_params[:,i], (n_samples_per_dataset[i],1)).transpose(0,1) for i in range (n_realizations)], # (n_realizations,) of (n_spe_params, [n_samples depends on dataset],)
            axis = 1
        ) # (n_spe_params, n_all_samples)

        ideal_class_params_flatten = torch.tile(ideal_class_params, (n_all_samples,1)).transpose(0,1) # (n_class_params, n_all_samples)

        def trial_func (X, params, class_params):
            y = params[0]*torch.exp(-params[1]*X[0])*torch.cos(class_params[0]*X[0]+params[2]) + class_params[1]*X[1]
            return y

        y_ideals_flatten = trial_func (multi_X_flatten, ideal_spe_params_flatten, ideal_class_params_flatten) # (n_all_samples,)
        multi_y_ideals   = unflatten_multi_data(y_ideals_flatten)                                         # (n_realizations,) of (n_samples depends on dataset,)


        # params[0]*torch.exp(-params[1]*X[0])*torch.cos(class_params[0]*X[0]+params[2]) + class_params[1]*X[1]
        # k0 * exp(-k1 * t) * cos(c0 * t + k2) + c1 * l
        # "add", "mul", "mul", "k0", "exp", "mul", "neg", "k1", "t", "cos", "add", "mul", "c0", "t", "k2", "mul", "c1", "l"

        k0_init = [1.,1.,1.]*10 # np.full(n_realizations, 1.)
        # consts
        pi     = torch.tensor (np.pi) .to(DEVICE)
        const1 = torch.tensor (1.)    .to(DEVICE)

        # LIBRARY CONFIG
        args_make_tokens = {
                        # operations
                        "op_names"             : "all",
                        "use_protected_ops"    : True,
                        # input variables
                        "input_var_ids"        : {"t" : 0         , "l" : 1          },
                        "input_var_units"      : {"t" : [1, 0, 0] , "l" : [0, 1, 0]  },
                        "input_var_complexity" : {"t" : 0.        , "l" : 1.         },
                        # constants
                        "constants"            : {"pi" : pi        , "const1" : const1    },
                        "constants_units"      : {"pi" : [0, 0, 0] , "const1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 1.        , "const1" : 1.        },
                        # free constants
                        "class_free_constants"            : {"c0"              , "c1"               },
                        "class_free_constants_init_val"   : {"c0" : 1.         , "c1"  : 1.         },
                        "class_free_constants_units"      : {"c0" : [-1, 0, 0] , "c1"  : [0, -1, 0] },
                        "class_free_constants_complexity" : {"c0" : 1.         , "c1"  : 1.         },
                        # free constants
                        "spe_free_constants"            : {"k0"              , "k1"               , "k2"               },
                        "spe_free_constants_init_val"   : {"k0" : k0_init    , "k1"  : 1.         , "k2"  : 1.         },
                        "spe_free_constants_units"      : {"k0" : [0, 0, 0]  , "k1"  : [-1, 0, 0] , "k2"  : [0, 0, 0]  },
                        "spe_free_constants_complexity" : {"k0" : 1.         , "k1"  : 1.         , "k2"  : 1.         },
                           }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [0, 0, 0], superparent_name = "y")

        # TEST PROGRAM
        batch_size = 10000
        test_program_str = ["add", "mul", "mul", "k0"  , "exp", "mul", "neg", "k1", "t", "cos", "add", "mul", "c0", "t", "k2", "mul", "c1", "l", ]
        test_program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str])
        test_program_length = len(test_program_str)
        test_program_idx = np.tile(test_program_idx, reps=(batch_size,1))

        # BATCH
        my_programs = VProg.VectPrograms(batch_size=batch_size, max_time_step=test_program_length, library=my_lib, n_realizations=n_realizations)
        my_programs.set_programs(test_program_idx)

        # Not spoiling free constants to get different values for y_pred and y_target
        #my_programs.free_consts.class_values = ideal_class_params.repeat((batch_size,1))   # (batch_size, n_class_params)
        #my_programs.free_consts.spe_values   = ideal_spe_params.repeat((batch_size, 1, 1)) # (batch_size, n_spe_params, n_realizations)

        y_pred_flatten = my_programs.get_prog(0).execute(multi_X_flatten, n_samples_per_dataset=n_samples_per_dataset)

        expected_results = physo.physym.reward.SquashedNRMSE(y_target=y_ideals_flatten, y_pred=y_pred_flatten, y_weights=y_weights_flatten).repeat((batch_size)) # (batch_size,)
        expected_results = expected_results.cpu().detach().numpy()

        # MASK: WHICH PROGRAM SHOULD BE EXECUTED
        # Simulating an evenly spread 90% load task
        mask = np.random.rand(batch_size) < 0.9

        # Function to run the hyper-task once with a given config
        def run (parallel=True, n_cpus=1):
            # Run tasks
            t0 = time.perf_counter()
            if not BOOL_DO_TEST_VIA_VECTPROGRAMS:
                results = BExec.BatchExecutionReward (progs     = my_programs,
                                                     X         = multi_X_flatten,
                                                     y_target  = y_ideals_flatten,
                                                     y_weights = y_weights_flatten,
                                                     reward_function = physo.physym.reward.SquashedNRMSE,
                                                     n_samples_per_dataset = n_samples_per_dataset, # Realization related
                                                     mask = mask,
                                                     parallel_mode = parallel,
                                                     n_cpus = n_cpus)
            else:
                # Through VectPrograms method
                results = my_programs.batch_exe_reward (
                                                     X         = multi_X_flatten,
                                                     y_target  = y_ideals_flatten,
                                                     y_weights = y_weights_flatten,
                                                     reward_function = physo.physym.reward.SquashedNRMSE,
                                                     n_samples_per_dataset = n_samples_per_dataset, # Realization related
                                                     mask = mask,
                                                     parallel_mode = parallel,
                                                     n_cpus = n_cpus)
            t1 = time.perf_counter()
            task_time = (t1 - t0) * 1e3 / mask.sum()
            #torch.set_printoptions(threshold=100)
            n_correct_computations    = float((results == expected_results).sum())
            perc_correct_computations = 100*n_correct_computations/mask.sum()
            print(" -> Correct computations: %f %%"%(perc_correct_computations))
            assert perc_correct_computations == 100, "Not all computations were correct"

            return task_time

        is_parallel_exe_available = BExec.ParallelExeAvailability(verbose=True)

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

        # Logging
        is_parallel_effective = (np.min(times) < not_parallelized_time)
        log_times = times      .tolist() + [not_parallelized_time,]
        log_ncpus = ncpus_list .tolist() + ["np",]
        df = pd.DataFrame({"ncpus": log_ncpus, "time [ms]": log_times})
        df["parallel_effective"] = is_parallel_effective
        # adding system info to each line
        system_info = get_system_info()
        for key in system_info.keys():
            df[key] = system_info[key]

        name = "perf_B_ParallelizedExeReward_with_spe_consts"

        if DO_SAVE_FIGS:
            fig.savefig(name + ".png")
            # If the file already exists, append to it
            if os.path.isfile(name + ".csv"):
                df.to_csv(name + ".csv", mode='a', header=False)
            else:
                df.to_csv(name + ".csv")

        plt.show()

        return None

    # Test parallelized execution of free constant optimization
    def test_02_C_ParallelizedExeFreeConstants_with_spe_consts (self):

        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Testing everything on CPU. If user has CUDA and wants to use CPU parallel mode, it is their responsibility to
        # send the dataset to the proper device.
        DEVICE = 'cpu'
        #if torch.cuda.is_available():
        #    DEVICE = 'cuda'

        # -------------------------------------- Making fake datasets --------------------------------------

        multi_X = []
        for n_samples in [90, 100, 110]:
            x1 = np.linspace(0, 10, n_samples)
            x2 = np.linspace(0, 1 , n_samples)
            X = np.stack((x1,x2),axis=0)
            X = torch.tensor(X).to(DEVICE)
            multi_X.append(X)
        n_real_multi = 10
        multi_X = multi_X*n_real_multi                         # (n_realizations,) of (n_dim, [n_samples depends on dataset],)

        n_samples_per_dataset = np.array([X.shape[1] for X in multi_X])
        n_all_samples = n_samples_per_dataset.sum()
        n_realizations = len(multi_X)
        def flatten_multi_data (multi_data,):
            """
            Flattens multiple datasets into a single one for vectorized evaluation.
            Parameters
            ----------
            multi_data : list of length (n_realizations,) of torch.tensor of shape (..., [n_samples depends on dataset],)
                List of datasets to be flattened.
            Returns
            -------
            torch.tensor of shape (..., n_all_samples)
                Flattened data (n_all_samples = sum([n_samples depends on dataset])).
            """
            flattened_data = torch.cat(multi_data, axis=-1) # (..., n_all_samples)
            return flattened_data

        def unflatten_multi_data (flattened_data):
            """
            Unflattens a single data into multiple ones.
            Parameters
            ----------
            flattened_data : torch.tensor of shape (..., n_all_samples)
                Flattened data (n_all_samples = sum([n_samples depends on dataset])).
            Returns
            -------
            list of len (n_realizations,) of torch.tensor of shape (..., [n_samples depends on dataset],)
                Unflattened data.
            """
            return list(torch.split(flattened_data, n_samples_per_dataset.tolist(), dim=-1)) # (n_realizations,) of (..., [n_samples depends on dataset],)

        #y_weights_per_dataset = np.array([0, 0.001, 1.0]*10) # Shows weights work
        y_weights_per_dataset = torch.tensor(np.array([1., 1., 1.]*n_real_multi))
        multi_y_weights = [torch.full(size=(n_samples_per_dataset[i],), fill_value=y_weights_per_dataset[i]) for i in range (n_realizations)]
        y_weights_flatten = flatten_multi_data(multi_y_weights)

        multi_X_flatten = flatten_multi_data(multi_X)  # (n_dim, n_all_samples)

        # Making fake ideal parameters
        # n_spe_params   = 3
        # n_class_params = 2
        random_shift       = (np.random.rand(n_realizations,3)-0.5)*0.8
        ideal_spe_params   = torch.tensor(np.array([1.123, 0.345, 0.116]) + random_shift) # (n_realizations, n_spe_params,)
        ideal_spe_params   = ideal_spe_params.transpose(0,1)                              # (n_spe_params, n_realizations)
        ideal_class_params = torch.tensor(np.array([1.389, 1.005]))                       # (n_class_params, )

        ideal_spe_params_flatten = torch.cat(
            [torch.tile(ideal_spe_params[:,i], (n_samples_per_dataset[i],1)).transpose(0,1) for i in range (n_realizations)], # (n_realizations,) of (n_spe_params, [n_samples depends on dataset],)
            axis = 1
        ) # (n_spe_params, n_all_samples)

        ideal_class_params_flatten = torch.tile(ideal_class_params, (n_all_samples,1)).transpose(0,1) # (n_class_params, n_all_samples)

        def trial_func (X, params, class_params):
            y = params[0]*torch.exp(-params[1]*X[0])*torch.cos(class_params[0]*X[0]+params[2]) + class_params[1]*X[1]
            return y

        y_ideals_flatten = trial_func (multi_X_flatten, ideal_spe_params_flatten, ideal_class_params_flatten) # (n_all_samples,)
        multi_y_ideals   = unflatten_multi_data(y_ideals_flatten)                                         # (n_realizations,) of (n_samples depends on dataset,)


        # params[0]*torch.exp(-params[1]*X[0])*torch.cos(class_params[0]*X[0]+params[2]) + class_params[1]*X[1]
        # k0 * exp(-k1 * t) * cos(c0 * t + k2) + c1 * l
        # "add", "mul", "mul", "k0", "exp", "mul", "neg", "k1", "t", "cos", "add", "mul", "c0", "t", "k2", "mul", "c1", "l"

        k0_init = [1.,1.,1.]*n_real_multi # np.full(n_realizations, 1.)
        # consts
        pi     = torch.tensor (np.pi) .to(DEVICE)
        const1 = torch.tensor (1.)    .to(DEVICE)

        # LIBRARY CONFIG
        args_make_tokens = {
                        # operations
                        "op_names"             : "all",
                        "use_protected_ops"    : True,
                        # input variables
                        "input_var_ids"        : {"t" : 0         , "l" : 1          },
                        "input_var_units"      : {"t" : [1, 0, 0] , "l" : [0, 1, 0]  },
                        "input_var_complexity" : {"t" : 0.        , "l" : 1.         },
                        # constants
                        "constants"            : {"pi" : pi        , "const1" : const1    },
                        "constants_units"      : {"pi" : [0, 0, 0] , "const1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 1.        , "const1" : 1.        },
                        # free constants
                        "class_free_constants"            : {"c0"              , "c1"               },
                        "class_free_constants_init_val"   : {"c0" : 1.         , "c1"  : 1.         },
                        "class_free_constants_units"      : {"c0" : [-1, 0, 0] , "c1"  : [0, -1, 0] },
                        "class_free_constants_complexity" : {"c0" : 1.         , "c1"  : 1.         },
                        # free constants
                        "spe_free_constants"            : {"k0"              , "k1"               , "k2"               },
                        "spe_free_constants_init_val"   : {"k0" : k0_init    , "k1"  : 1.         , "k2"  : 1.         },
                        "spe_free_constants_units"      : {"k0" : [0, 0, 0]  , "k1"  : [-1, 0, 0] , "k2"  : [0, 0, 0]  },
                        "spe_free_constants_complexity" : {"k0" : 1.         , "k1"  : 1.         , "k2"  : 1.         },
                           }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [0, 0, 0], superparent_name = "y")

        # TEST PROGRAM
        batch_size = 256 # 32 tasks per CPU with 8 CPUs
        test_program_str = ["add", "mul", "mul", "k0"  , "exp", "mul", "neg", "k1", "t", "cos", "add", "mul", "c0", "t", "k2", "mul", "c1", "l", ]
        test_program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str])
        test_program_length = len(test_program_str)
        test_program_idx = np.tile(test_program_idx, reps=(batch_size,1))

        # BATCH
        my_programs = VProg.VectPrograms(batch_size=batch_size, max_time_step=test_program_length, library=my_lib, n_realizations=n_realizations)
        my_programs.set_programs(test_program_idx)

        # Expected results
        tol = 5*1e-3
        expected_class_vals = ideal_class_params.repeat((batch_size,1))               # (batch_size, n_class_params)
        expected_spe_vals   = ideal_spe_params.repeat((batch_size, 1, 1))             # (batch_size, n_spe_params, n_realizations)

        # FREE CONST OPTI CONFIG
        free_const_opti_args = {
            'loss'   : "MSE",
            'method' : 'LBFGS',
            'method_args': {
                        'n_steps' : 50,
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
            my_programs = VProg.VectPrograms(batch_size=batch_size, max_time_step=test_program_length, library=my_lib, n_realizations=n_realizations)
            my_programs.set_programs(test_program_idx)
            # Run tasks
            t0 = time.perf_counter()
            if not BOOL_DO_TEST_VIA_VECTPROGRAMS:
                BExec.BatchFreeConstOpti(progs = my_programs,
                                        X = multi_X_flatten,
                                        y_target = y_ideals_flatten,
                                        free_const_opti_args = free_const_opti_args,
                                        y_weights= y_weights_flatten,
                                        n_samples_per_dataset = n_samples_per_dataset,
                                        mask = mask,
                                        parallel_mode= parallel,
                                        n_cpus = n_cpus, )
            else:
                # Through VectPrograms method
                my_programs.batch_optimize_constants(
                                        X = multi_X_flatten,
                                        y_target = y_ideals_flatten,
                                        free_const_opti_args = free_const_opti_args,
                                        y_weights= y_weights_flatten,
                                        n_samples_per_dataset = n_samples_per_dataset,
                                        mask = mask,
                                        parallel_mode= parallel,
                                        n_cpus = n_cpus, )
            t1 = time.perf_counter()
            task_time = (t1 - t0) * 1e3 / mask.sum()
            #torch.set_printoptions(threshold=100)
            #print(expected_class_vals)
            is_correct_class_vals = (torch.abs((my_programs.free_consts.class_values - expected_class_vals)) < tol).all(axis=-1) # (batch_size,)
            is_correct_spe_vals   = (torch.abs((my_programs.free_consts.spe_values   - expected_spe_vals  )) < tol).all(axis=1)  # (batch_size, n_realizations)
            # Not checking where weights are too low
            is_correct_spe_vals   = is_correct_spe_vals [:, y_weights_per_dataset > 0.1]                                         # (batch_size, n_weights>0.1)
            is_correct_spe_vals   = is_correct_spe_vals.all(axis=-1)                                                             # (batch_size,)
            n_correct_computations = float((torch.logical_and(is_correct_class_vals, is_correct_spe_vals)).sum())
            perc_correct_computations = 100*n_correct_computations/mask.sum()
            # Checking that logging worked
            is_correct_log = torch.logical_and((my_programs.free_consts.is_opti == True), (my_programs.free_consts.opti_steps > 0))
            n_correct_log = float(is_correct_log.sum())
            perc_correct_log = 100*n_correct_log/mask.sum()
            print(" -> Correct computations : %f %%" % (perc_correct_computations))
            print(" -> Correct logging      : %f %%" % (perc_correct_log)         )
            assert perc_correct_computations == 100, "Not all computations were correct"
            assert perc_correct_log          == 100, "Not all logging were correct"

            return task_time

        is_parallel_exe_available = BExec.ParallelExeAvailability(verbose=True)

        # EFFICIENCY CURVE (NUMBER OF CPUS VS TASK TIME)

        print("\nParallelized free constant optimization test (mdho2d scenario):")

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
        fig.suptitle("Efficiency curve: free const. opti. (mdho2d scenario)\n Using parallelization in physo run : %s"%(str(enabled)))
        ax.plot(ncpus_list, times, 'k--')
        ax.plot(ncpus_list, times, 'ko')
        ax.plot(1, not_parallelized_time, 'ro', label="not parallelized")
        ax.set_xlabel("Nb. of CPUs")
        ax.set_ylabel("time [ms]")
        ax.legend()

        # Logging
        is_parallel_effective = (np.min(times) < not_parallelized_time)
        log_times = times      .tolist() + [not_parallelized_time,]
        log_ncpus = ncpus_list .tolist() + ["np",]
        df = pd.DataFrame({"ncpus": log_ncpus, "time [ms]": log_times})
        df["parallel_effective"] = is_parallel_effective
        # adding system info to each line
        system_info = get_system_info()
        for key in system_info.keys():
            df[key] = system_info[key]

        name = "perf_C_ParallelizedExeFreeConstants_with_spe_consts"

        if DO_SAVE_FIGS:
            fig.savefig(name + ".png")
            # If the file already exists, append to it
            if os.path.isfile(name + ".csv"):
                df.to_csv(name + ".csv", mode='a', header=False)
            else:
                df.to_csv(name + ".csv")

        plt.show()

        return None

    # Test parallelized execution of free constant optimization
    # This test is the same as the previous one, but with weights (y_weights), this feature is already tested
    # somewhere else, so let's comment this test
    # def test_C_ParallelizedExeFreeConstants_with_weights_and_spe_consts (self):
    #
    #     # Testing everything on CPU. If user has CUDA and wants to use CPU parallel mode, it is their responsibility to
    #     # send the dataset to the proper device.
    #     DEVICE = 'cpu'
    #     #if torch.cuda.is_available():
    #     #    DEVICE = 'cuda'
    #
    #     # -------------------------------------- Making fake datasets --------------------------------------
    #
    #     multi_X = []
    #     for n_samples in [90, 100, 110]:
    #         x1 = np.linspace(0, 10, n_samples)
    #         x2 = np.linspace(0, 1 , n_samples)
    #         X = np.stack((x1,x2),axis=0)
    #         X = torch.tensor(X).to(DEVICE)
    #         multi_X.append(X)
    #     multi_X = multi_X*10                         # (n_realizations,) of (n_dim, [n_samples depends on dataset],)
    #
    #     n_samples_per_dataset = np.array([X.shape[1] for X in multi_X])
    #     n_all_samples = n_samples_per_dataset.sum()
    #     n_realizations = len(multi_X)
    #     def flatten_multi_data (multi_data,):
    #         """
    #         Flattens multiple datasets into a single one for vectorized evaluation.
    #         Parameters
    #         ----------
    #         multi_data : list of length (n_realizations,) of torch.tensor of shape (..., [n_samples depends on dataset],)
    #             List of datasets to be flattened.
    #         Returns
    #         -------
    #         torch.tensor of shape (..., n_all_samples)
    #             Flattened data (n_all_samples = sum([n_samples depends on dataset])).
    #         """
    #         flattened_data = torch.cat(multi_data, axis=-1) # (..., n_all_samples)
    #         return flattened_data
    #
    #     def unflatten_multi_data (flattened_data):
    #         """
    #         Unflattens a single data into multiple ones.
    #         Parameters
    #         ----------
    #         flattened_data : torch.tensor of shape (..., n_all_samples)
    #             Flattened data (n_all_samples = sum([n_samples depends on dataset])).
    #         Returns
    #         -------
    #         list of len (n_realizations,) of torch.tensor of shape (..., [n_samples depends on dataset],)
    #             Unflattened data.
    #         """
    #         return list(torch.split(flattened_data, n_samples_per_dataset.tolist(), dim=-1)) # (n_realizations,) of (..., [n_samples depends on dataset],)
    #
    #     y_weights_per_dataset = np.array([0, 0.001, 1.0]*10) # Shows weights work
    #     #y_weights_per_dataset = torch.tensor(np.array([1., 1., 1.]*10))
    #     multi_y_weights = [torch.full(size=(n_samples_per_dataset[i],), fill_value=y_weights_per_dataset[i]) for i in range (n_realizations)]
    #     y_weights_flatten = flatten_multi_data(multi_y_weights)
    #
    #     multi_X_flatten = flatten_multi_data(multi_X)  # (n_dim, n_all_samples)
    #
    #     # Making fake ideal parameters
    #     # n_spe_params   = 3
    #     # n_class_params = 2
    #     random_shift       = (np.random.rand(n_realizations,3)-0.5)*0.8
    #     ideal_spe_params   = torch.tensor(np.array([1.123, 0.345, 0.116]) + random_shift) # (n_realizations, n_spe_params,)
    #     ideal_spe_params   = ideal_spe_params.transpose(0,1)                              # (n_spe_params, n_realizations)
    #     ideal_class_params = torch.tensor(np.array([1.389, 1.005]))                       # (n_class_params, )
    #
    #     ideal_spe_params_flatten = torch.cat(
    #         [torch.tile(ideal_spe_params[:,i], (n_samples_per_dataset[i],1)).transpose(0,1) for i in range (n_realizations)], # (n_realizations,) of (n_spe_params, [n_samples depends on dataset],)
    #         axis = 1
    #     ) # (n_spe_params, n_all_samples)
    #
    #     ideal_class_params_flatten = torch.tile(ideal_class_params, (n_all_samples,1)).transpose(0,1) # (n_class_params, n_all_samples)
    #
    #     def trial_func (X, params, class_params):
    #         y = params[0]*torch.exp(-params[1]*X[0])*torch.cos(class_params[0]*X[0]+params[2]) + class_params[1]*X[1]
    #         return y
    #
    #     y_ideals_flatten = trial_func (multi_X_flatten, ideal_spe_params_flatten, ideal_class_params_flatten) # (n_all_samples,)
    #     multi_y_ideals   = unflatten_multi_data(y_ideals_flatten)                                         # (n_realizations,) of (n_samples depends on dataset,)
    #
    #
    #     # params[0]*torch.exp(-params[1]*X[0])*torch.cos(class_params[0]*X[0]+params[2]) + class_params[1]*X[1]
    #     # k0 * exp(-k1 * t) * cos(c0 * t + k2) + c1 * l
    #     # "add", "mul", "mul", "k0", "exp", "mul", "neg", "k1", "t", "cos", "add", "mul", "c0", "t", "k2", "mul", "c1", "l"
    #
    #     k0_init = [1.,1.,1.]*10 # np.full(n_realizations, 1.)
    #     # consts
    #     pi     = torch.tensor (np.pi) .to(DEVICE)
    #     const1 = torch.tensor (1.)    .to(DEVICE)
    #
    #     # LIBRARY CONFIG
    #     args_make_tokens = {
    #                     # operations
    #                     "op_names"             : "all",
    #                     "use_protected_ops"    : True,
    #                     # input variables
    #                     "input_var_ids"        : {"t" : 0         , "l" : 1          },
    #                     "input_var_units"      : {"t" : [1, 0, 0] , "l" : [0, 1, 0]  },
    #                     "input_var_complexity" : {"t" : 0.        , "l" : 1.         },
    #                     # constants
    #                     "constants"            : {"pi" : pi        , "const1" : const1    },
    #                     "constants_units"      : {"pi" : [0, 0, 0] , "const1" : [0, 0, 0] },
    #                     "constants_complexity" : {"pi" : 1.        , "const1" : 1.        },
    #                     # free constants
    #                     "class_free_constants"            : {"c0"              , "c1"               },
    #                     "class_free_constants_init_val"   : {"c0" : 1.         , "c1"  : 1.         },
    #                     "class_free_constants_units"      : {"c0" : [-1, 0, 0] , "c1"  : [0, -1, 0] },
    #                     "class_free_constants_complexity" : {"c0" : 1.         , "c1"  : 1.         },
    #                     # free constants
    #                     "spe_free_constants"            : {"k0"              , "k1"               , "k2"               },
    #                     "spe_free_constants_init_val"   : {"k0" : k0_init    , "k1"  : 1.         , "k2"  : 1.         },
    #                     "spe_free_constants_units"      : {"k0" : [0, 0, 0]  , "k1"  : [-1, 0, 0] , "k2"  : [0, 0, 0]  },
    #                     "spe_free_constants_complexity" : {"k0" : 1.         , "k1"  : 1.         , "k2"  : 1.         },
    #                        }
    #     my_lib = Lib.Library(args_make_tokens = args_make_tokens,
    #                          superparent_units = [0, 0, 0], superparent_name = "y")
    #
    #     # TEST PROGRAM
    #     batch_size = 256 # 32 tasks per CPU with 8 CPUs
    #     test_program_str = ["add", "mul", "mul", "k0"  , "exp", "mul", "neg", "k1", "t", "cos", "add", "mul", "c0", "t", "k2", "mul", "c1", "l", ]
    #     test_program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str])
    #     test_program_length = len(test_program_str)
    #     test_program_idx = np.tile(test_program_idx, reps=(batch_size,1))
    #
    #     # BATCH
    #     my_programs = VProg.VectPrograms(batch_size=batch_size, max_time_step=test_program_length, library=my_lib, n_realizations=n_realizations)
    #     my_programs.set_programs(test_program_idx)
    #
    #     # Expected results
    #     tol = 5*1e-3
    #     expected_class_vals = ideal_class_params.repeat((batch_size,1))               # (batch_size, n_class_params)
    #     expected_spe_vals   = ideal_spe_params.repeat((batch_size, 1, 1))             # (batch_size, n_spe_params, n_realizations)
    #
    #     # FREE CONST OPTI CONFIG
    #     free_const_opti_args = {
    #         'loss'   : "MSE",
    #         'method' : 'LBFGS',
    #         'method_args': {
    #                     'n_steps' : 50,
    #                     'tol'     : 1e-8,
    #                     'lbfgs_func_args' : {
    #                         'max_iter'       : 4,
    #                         'line_search_fn' : "strong_wolfe",
    #                                          },
    #                         },
    #     }
    #
    #     # MASK: WHICH PROGRAM SHOULD BE EXECUTED
    #     # Simulating an evenly spread 90% load task
    #     mask = np.random.rand(batch_size) < 0.9
    #
    #     # Function to run the hyper-task once with a given config
    #     def run (parallel=True, n_cpus=1):
    #         # reset before each run, so it is not easier (early stop) to optimize free const next time
    #         my_programs = VProg.VectPrograms(batch_size=batch_size, max_time_step=test_program_length, library=my_lib, n_realizations=n_realizations)
    #         my_programs.set_programs(test_program_idx)
    #         # Run tasks
    #         t0 = time.perf_counter()
    #         if not BOOL_DO_TEST_VIA_VECTPROGRAMS:
    #             BExec.BatchFreeConstOpti(progs = my_programs,
    #                                     X = multi_X_flatten,
    #                                     y_target = y_ideals_flatten,
    #                                     free_const_opti_args = free_const_opti_args,
    #                                     y_weights= y_weights_flatten,
    #                                     n_samples_per_dataset = n_samples_per_dataset,
    #                                     mask = mask,
    #                                     parallel_mode= parallel,
    #                                     n_cpus = n_cpus, )
    #         else:
    #             # Through VectPrograms method
    #             my_programs.batch_optimize_constants(
    #                                     X = multi_X_flatten,
    #                                     y_target = y_ideals_flatten,
    #                                     free_const_opti_args = free_const_opti_args,
    #                                     y_weights= y_weights_flatten,
    #                                     n_samples_per_dataset = n_samples_per_dataset,
    #                                     mask = mask,
    #                                     parallel_mode= parallel,
    #                                     n_cpus = n_cpus, )
    #         t1 = time.perf_counter()
    #         task_time = (t1 - t0) * 1e3 / mask.sum()
    #         #torch.set_printoptions(threshold=100)
    #         #print(expected_class_vals)
    #
    #         # Assertions
    #         is_correct_class_vals = (torch.abs((my_programs.free_consts.class_values - expected_class_vals)) < tol).all(axis=-1) # (batch_size,)
    #         is_correct_spe_vals   = (torch.abs((my_programs.free_consts.spe_values   - expected_spe_vals  )) < tol).all(axis=1)  # (batch_size, n_realizations)
    #         # Not checking where weights are too low
    #         is_correct_spe_vals_where_ok_weights  = is_correct_spe_vals [:, y_weights_per_dataset > 0.5 ]                         # (batch_size, n_weights>0.1)
    #         # Checking that values where not recovered where weights are too low
    #         is_correct_spe_vals_where_low_weights = is_correct_spe_vals [:, y_weights_per_dataset < 1e-8]                         # (batch_size, n_weights<1e-8)
    #         assert is_correct_spe_vals_where_low_weights.sum() == 0, "Values where weights are too low were recovered"
    #         # Checking that values where recovered where weights are not too low
    #         is_correct_spe_vals_where_ok_weights   = is_correct_spe_vals_where_ok_weights.all(axis=-1)                                                             # (batch_size,)
    #         n_correct_computations = float((torch.logical_and(is_correct_class_vals, is_correct_spe_vals_where_ok_weights)).sum())
    #         perc_correct_computations = 100*n_correct_computations/mask.sum()
    #         print(" -> Correct computations: %f %%"%(perc_correct_computations))
    #         assert perc_correct_computations == 100, "Not all computations were correct"
    #
    #
    #         return task_time
    #
    #     is_parallel_exe_available = BExec.ParallelExeAvailability(verbose=True)
    #
    #     # EFFICIENCY CURVE (NUMBER OF CPUS VS TASK TIME)
    #
    #     print("\nParallelized free constant optimization test (wmdho2d scenario):")
    #
    #     max_ncpus = mp.cpu_count()
    #     print("Total nb. of CPUs: ", max_ncpus)
    #     # Getting computation times as a function of the number of CPUs
    #     times = []
    #     ncpus_list = get_ncpus(max_ncpus)
    #     print("Testing nb. of CPUs = ", ncpus_list)
    #     for ncpus in ncpus_list:
    #         print("n cpu =", ncpus)
    #         task_time = run(parallel=True, n_cpus=ncpus)
    #         # task_time = np.exp(-ncpus) # fast mock plot
    #         print("-> %f ms per task"%(task_time))
    #         times.append(task_time)
    #     times = np.array(times)
    #     # Ordering results
    #     ncpus_list = np.array(ncpus_list[1:].tolist() + [ncpus_list[0]])
    #     times = np.array(times[1:].tolist() + [times[0]])
    #
    #     # Getting computation times when running in a non-parallelized loop
    #     print("Not parallelized")
    #     not_parallelized_time = run(parallel=False)
    #     print("-> %f ms per task" % (not_parallelized_time))
    #
    #     # Plot
    #     fig,ax = plt.subplots(1,1)
    #     enabled = physo.physym.reward.USE_PARALLEL_OPTI_CONST
    #     fig.suptitle("Efficiency curve: free const. opti. (wmdho2d scenario)\n Using parallelization in physo run : %s"%(str(enabled)))
    #     ax.plot(ncpus_list, times, 'k--')
    #     ax.plot(ncpus_list, times, 'ko')
    #     ax.plot(1, not_parallelized_time, 'ro', label="not parallelized")
    #     ax.set_xlabel("Nb. of CPUs")
    #     ax.set_ylabel("time [ms]")
    #     ax.legend()
    #     plt.show()
    #
    #     return None

if __name__ == '__main__':
    unittest.main(verbosity=2)
