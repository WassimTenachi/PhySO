import unittest

import matplotlib.pyplot as plt
import numpy as np
import time as time
import warnings
import torch

# Internal imports
from physo.physym import library as Lib
from physo.physym import program as Prog
from physo.physym import dimensional_analysis as phy
from physo.physym.functions import data_conversion, data_conversion_inv
import physo.physym.free_const as free_const


def make_lib():
    # LIBRARY CONFIG
    args_make_tokens = {
                    # operations
                    "op_names"             : "all",  # or ["mul", "neg", "inv", "sin"]
                    "use_protected_ops"    : False,
                    # input variables
                    "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                    "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                    "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                    # constants
                    "constants"            : {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                    "constants_units"      : {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] },
                    "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                    # free constants
                    "free_constants"            : {"c0"             , "c1"               , "c2"             },
                    "free_constants_init_val"   : {"c0" : 1.        , "c1"  : 10.        , "c2" : 1.        },
                    "free_constants_units"      : {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                    "free_constants_complexity" : {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                           }

    my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                         superparent_units = [1, -2, 1], superparent_name = "y")
    return my_lib

class ProgramTest(unittest.TestCase):

    def test_creation(self):

        k0_init = [9,10,11]*10 # np.full(n_realizations, 1.)
        n_realizations = len(k0_init)
        # consts
        pi     = data_conversion (np.pi)
        const1 = data_conversion (1.)

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
                        "class_free_constants_init_val"   : {"c0" : 21.        , "c1"  : 22.         },
                        "class_free_constants_units"      : {"c0" : [-1, 0, 0] , "c1"  : [0, -1, 0] },
                        "class_free_constants_complexity" : {"c0" : 1.         , "c1"  : 1.         },
                        # free constants
                        "spe_free_constants"            : {"k0"              , "k1"               , "k2"               },
                        "spe_free_constants_init_val"   : {"k0" : k0_init    , "k1"  : 2.         , "k2"  : 3.         },
                        "spe_free_constants_units"      : {"k0" : [0, 0, 0]  , "k1"  : [-1, 0, 0] , "k2"  : [0, 0, 0]  },
                        "spe_free_constants_complexity" : {"k0" : 1.         , "k1"  : 1.         , "k2"  : 1.         },
                           }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [0, 0, 0], superparent_name = "y")

        # TEST PROGRAMS
        test_programs_idx = []
        test_prog_str_0 = ["add", "mul", "mul", "k0"  , "exp", "mul", "neg", "k1", "t", "cos", "add", "mul", "c0", "t", "k2", "mul", "c1", "l", ]
        test_prog_str_1 = ["mul", "n2" , "c0" , "cos" , "div", "t"  , "c1" ,]

        # Converting into idx
        test_prog_tokens_0 = np.array([my_lib.lib_name_to_token[tok_str] for tok_str in test_prog_str_0])
        test_prog_tokens_1 = np.array([my_lib.lib_name_to_token[tok_str] for tok_str in test_prog_str_1])

        # Creating programs wo free constants
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Raises some warnings due to free constants table not being provided
                prog0 = Prog.Program(tokens=test_prog_tokens_0, library=my_lib, n_realizations=n_realizations, free_consts=None, is_physical=None, candidate_wrapper=None)
                prog1 = Prog.Program(tokens=test_prog_tokens_1, library=my_lib, n_realizations=n_realizations, free_consts=None, is_physical=None, candidate_wrapper=None)
        except:
            self.fail("Program creation failed.")

        # Creating programs with free constants
        try:
            free_consts_table_0 = free_const.FreeConstantsTable(batch_size=1, library=my_lib, n_realizations=n_realizations)
            free_consts_table_1 = free_const.FreeConstantsTable(batch_size=1, library=my_lib, n_realizations=n_realizations)
            prog0 = Prog.Program(tokens=test_prog_tokens_0, library=my_lib, n_realizations=n_realizations, free_consts=free_consts_table_0, is_physical=None, candidate_wrapper=None)
            prog1 = Prog.Program(tokens=test_prog_tokens_1, library=my_lib, n_realizations=n_realizations, free_consts=free_consts_table_1, is_physical=None, candidate_wrapper=None)
        except:
            self.fail("Program creation failed.")


    # Test program execution on a complicated function (no free consts)
    def test_execution (self):

        DEVICE = 'cpu'
        if torch.cuda.is_available():
            DEVICE = 'cuda'

        # DATA
        N = int(1e6)

        # input var
        x = data_conversion  (np.linspace(0.04, 4, N)  ).to(DEVICE)
        v = data_conversion  (np.linspace(0.10, 10, N) ).to(DEVICE)
        t = data_conversion  (np.linspace(0.06, 6, N)  ).to(DEVICE)
        data = torch.stack((x, v, t), axis=0)

        # consts
        pi = data_conversion (np.pi).to(DEVICE)
        const1 = data_conversion (1.).to(DEVICE)

        # free consts
        c  = data_conversion (3e8).to(DEVICE)
        M  = data_conversion (1e6).to(DEVICE)

        # LIBRARY CONFIG
        args_make_tokens = {
                        # operations
                        "op_names"             : "all",  # or ["mul", "neg", "inv", "sin"]
                        "use_protected_ops"    : False,
                        # input variables
                        "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                        "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                        "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                        # constants
                        "constants"            : {"pi" : pi        , "1" : const1    , "c" : c         , "M" : M         },
                        "constants_units"      : {"pi" : [0, 0, 0] , "1" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] },
                        "constants_complexity" : {"pi" : 0.        , "1" : 1.        , "c" : 0.        , "M" : 1.        },
                           }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [1, -2, 1], superparent_name = "y")

        # PROGRAM
        test_program_str = ["mul", "mul", "M", "n2", "c", "sub", "inv", "sqrt", "sub", "1", "div", "n2", "v", "n2",
                            "c", "cos", "div", "sub", "1", "div", "v", "c", "div", "div", "x", "t", "c"]
        test_program     = [my_lib.lib_name_to_token[name] for name in test_program_str]
        free_const_table = free_const.FreeConstantsTable(batch_size=1, library=my_lib, n_realizations=1)
        prog = Prog.Program(tokens=test_program, library=my_lib, free_consts=free_const_table, n_realizations=1)
        # EXPECTED RES
        expected_res     = M*(c**2)*(1./torch.sqrt(1.-(v**2)/(c**2))-torch.cos((1.-(v/c))/((x/t)/c)))

        N = 100
        # EXECUTION
        t0 = time.perf_counter()
        for _ in range (N):
            res = prog.execute(data)
        t1 = time.perf_counter()
        print("\nprog.execute time = %.3f ms"%((t1-t0)*1e3/N))

        # EXECUTION (wo tokens)
        t0 = time.perf_counter()
        for _ in range (N):
            expected_res     = M*(c**2)*(1./torch.sqrt(1.-(v**2)/(c**2))-torch.cos((1.-(v/c))/((x/t)/c)))
        t1 = time.perf_counter()
        print("\nprog.execute time (wo tokens) = %.3f ms"%((t1-t0)*1e3/N))

        # TEST
        works_bool = np.array_equal(data_conversion_inv(res.cpu()), data_conversion_inv(expected_res.cpu()),)
        self.assertTrue(works_bool)
        return None

    # Test program execution on a complicated function
    def test_execution_with_free_consts (self):

        DEVICE = 'cpu'
        if torch.cuda.is_available():
            DEVICE = 'cuda'

        # DATA
        N = int(1e6)

        # input var
        x = data_conversion  (np.linspace(0.04, 4, N)  ).to(DEVICE)
        v = data_conversion  (np.linspace(0.10, 10, N) ).to(DEVICE)
        t = data_conversion  (np.linspace(0.06, 6, N)  ).to(DEVICE)
        data = torch.stack((x, v, t), axis=0)

        # consts
        pi = data_conversion (np.pi).to(DEVICE)
        const1 = data_conversion (1.).to(DEVICE)

        # free consts
        c  = data_conversion (3e8).to(DEVICE)
        M  = data_conversion (1e6).to(DEVICE)
        free_const_values = torch.stack((M, c), axis=0)
        # (M, c) in alphabetical order as library will give them ids based on that order

        # LIBRARY CONFIG
        args_make_tokens = {
                        # operations
                        "op_names"             : "all",  # or ["mul", "neg", "inv", "sin"]
                        "use_protected_ops"    : False,
                        # input variables
                        "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                        "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                        "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                        # constants
                        "constants"            : {"pi" : pi        , "1" : const1    },
                        "constants_units"      : {"pi" : [0, 0, 0] , "1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 0.        , "1" : 1.        },
                        # free constants
                        "free_constants"            : {"c"              , "M"             },
                        "free_constants_init_val"   : {"c" : 1.         , "M" : 1.        },
                        "free_constants_units"      : {"c" : [1, -1, 0] , "M" : [0, 0, 1] },
                        "free_constants_complexity" : {"c" : 0.         , "M" : 1.        },
                           }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [1, -2, 1], superparent_name = "y")

        # PROGRAM
        test_program_str = ["mul", "mul", "M", "n2", "c", "sub", "inv", "sqrt", "sub", "1", "div", "n2", "v", "n2",
                            "c", "cos", "div", "sub", "1", "div", "v", "c", "div", "div", "x", "t", "c"]
        test_program     = [my_lib.lib_name_to_token[name] for name in test_program_str]
        free_const_table = free_const.FreeConstantsTable(batch_size=1, library=my_lib, n_realizations=1)
        free_const_table.class_values[0] = free_const_values
        prog = Prog.Program(tokens=test_program, library=my_lib, free_consts=free_const_table, n_realizations=1)
        # EXPECTED RES
        expected_res     = M*(c**2)*(1./torch.sqrt(1.-(v**2)/(c**2))-torch.cos((1.-(v/c))/((x/t)/c)))

        N = 100
        # EXECUTION
        t0 = time.perf_counter()
        for _ in range (N):
            res = prog.execute(data)
        t1 = time.perf_counter()
        print("\nprog.execute time = %.3f ms"%((t1-t0)*1e3/N))

        # EXECUTION (wo tokens)
        t0 = time.perf_counter()
        for _ in range (N):
            expected_res     = M*(c**2)*(1./torch.sqrt(1.-(v**2)/(c**2))-torch.cos((1.-(v/c))/((x/t)/c)))
        t1 = time.perf_counter()
        print("\nprog.execute time (wo tokens) = %.3f ms"%((t1-t0)*1e3/N))

        # TEST
        works_bool = np.array_equal(data_conversion_inv(res.cpu()), data_conversion_inv(expected_res.cpu()),)
        self.assertTrue(works_bool)
        return None

    # Test program execution in Class SR scenario
    def test_execution_with_class_and_spe_free_consts (self):

        DEVICE = 'cpu'
        if torch.cuda.is_available():
            DEVICE = 'cuda'

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
            return torch.split(flattened_data, n_samples_per_dataset.tolist(), dim=-1) # (n_realizations,) of (..., [n_samples depends on dataset],)

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
        ideal_class_params = torch.tensor(np.array([1.389, 1.005]))                       # (n_class_params, )

        ideal_spe_params_flatten = torch.cat(
            [torch.tile(ideal_spe_params[i], (n_samples_per_dataset[i],1)).transpose(0,1) for i in range (n_realizations)], # (n_realizations,) of (n_spe_params, [n_samples depends on dataset],)
            axis = 1
        ) # (n_spe_params, n_all_samples)

        ideal_class_params_flatten = torch.tile(ideal_class_params, (n_all_samples,1)).transpose(0,1) # (n_class_params, n_all_samples)

        def trial_func (X, params, class_params):
            y = params[0]*torch.exp(-params[1]*X[0])*torch.cos(class_params[0]*X[0]+params[2]) + class_params[1]*X[1]
            return y

        y_ideals_flatten = trial_func (multi_X_flatten, ideal_spe_params_flatten, ideal_class_params_flatten) # (n_all_samples,)
        multi_y_ideals   = unflatten_multi_data(y_ideals_flatten)                                         # (n_datasets,) of (n_samples depends on dataset,)


        # params[0]*torch.exp(-params[1]*X[0])*torch.cos(class_params[0]*X[0]+params[2]) + class_params[1]*X[1]
        # k0 * exp(-k1 * t) * cos(c0 * t + k2) + c1 * l
        # "add", "mul", "mul", "k0", "exp", "mul", "neg", "k1", "t", "cos", "add", "mul", "c0", "t", "k2", "mul", "c1", "l"

        k0_init = [9,10,11]*10 # np.full(n_realizations, 1.)
        # consts
        pi     = data_conversion (np.pi) .to(DEVICE)
        const1 = data_conversion (1.)    .to(DEVICE)

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
                        "class_free_constants_init_val"   : {"c0" : 21.        , "c1"  : 22.         },
                        "class_free_constants_units"      : {"c0" : [-1, 0, 0] , "c1"  : [0, -1, 0] },
                        "class_free_constants_complexity" : {"c0" : 1.         , "c1"  : 1.         },
                        # free constants
                        "spe_free_constants"            : {"k0"              , "k1"               , "k2"               },
                        "spe_free_constants_init_val"   : {"k0" : k0_init    , "k1"  : 2.         , "k2"  : 3.         },
                        "spe_free_constants_units"      : {"k0" : [0, 0, 0]  , "k1"  : [-1, 0, 0] , "k2"  : [0, 0, 0]  },
                        "spe_free_constants_complexity" : {"k0" : 1.         , "k1"  : 1.         , "k2"  : 1.         },
                           }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [0, 0, 0], superparent_name = "y")

        # TEST PROGRAMS
        test_programs_idx = []
        test_prog_str_0 = ["add", "mul", "mul", "k0"  , "exp", "mul", "neg", "k1", "t", "cos", "add", "mul", "c0", "t", "k2", "mul", "c1", "l", ]
        test_tokens_0 = [my_lib.lib_name_to_token[name] for name in test_prog_str_0]

        free_const_table = free_const.FreeConstantsTable(batch_size=1, library=my_lib, n_realizations=n_realizations)
        free_const_table.class_values[0] = ideal_class_params
        free_const_table.spe_values  [0] = ideal_spe_params.transpose(0,1) # (n_spe_params, n_realizations)

        prog = Prog.Program(tokens=test_tokens_0, library=my_lib, free_consts=free_const_table, n_realizations=n_realizations)


        # Test execution on all datasets in flattened form
        N = 1000
        t0 = time.perf_counter()
        for _ in range (N):
            y_computed_flatten = prog.execute(X                     = multi_X_flatten,
                                              n_samples_per_dataset = n_samples_per_dataset,
                                              )
        t1 = time.perf_counter()
        print("\nprog.execute time (flattened class SR) = %.3f ms"%((t1-t0)*1e3/N))
        #multi_y_computed = unflatten_multi_data(y_computed_flatten)
        works_bool = (y_computed_flatten == y_ideals_flatten).all()
        self.assertTrue(works_bool)

        # Test execution on all datasets but one by one
        t0 = time.perf_counter()
        for _ in range (N):
            multi_y_computed = []
            for i in range(n_realizations):
                y_computed = prog.execute(X              = multi_X[i],
                                          i_realization  = i,
                                          )
                multi_y_computed.append(y_computed)
        t1 = time.perf_counter()
        print("\nprog.execute time (one-by-one class SR) = %.3f ms"%((t1-t0)*1e3/N))

        for i in range(n_realizations):
            works_bool = (multi_y_computed[i] == multi_y_ideals[i]).all()
            self.assertTrue(works_bool)

        # # Sanity plot
        # fig, ax = plt.subplots(1,1,figsize=(10,5))
        # for i in range(n_realizations):
        #     ax.plot(multi_X[i][0], multi_y_ideals   [i].cpu().detach().numpy(), 'o', )
        #     ax.plot(multi_X[i][0], multi_y_computed [i].cpu().detach().numpy(), 'r-',)
        # ax.legend()
        # plt.show()
        # for i in range(n_realizations):
        #     mse = torch.mean((multi_y_computed[i] - multi_y_ideals[i])**2)
        #     print("%i, mse = %f"%(i, mse))


        return None

    # Test program const optimization (normal SR scenario)
    def test_optimize (self):

        ideal_class_params = torch.tensor([1.389, 1.005]) # (n_class_params, )

        # Synthetic data
        x0 = torch.linspace(0, 10, 1000)
        x1 = torch.linspace(-5, 1 , 1000)
        X = torch.stack((x0,x1),axis=0)
        y_ideals = ideal_class_params[0]*X[0] + ideal_class_params[1] + X[1]

        # consts
        pi     = data_conversion (np.pi)
        const1 = data_conversion (1.)

        # LIBRARY CONFIG
        args_make_tokens = {
                        # operations
                        "op_names"             : "all",  # or ["mul", "neg", "inv", "sin"]
                        "use_protected_ops"    : False,
                        # input variables
                        "input_var_ids"        : {"x0" : 0         , "x1" : 1         },
                        "input_var_units"      : {"x0" : [0, 0, 0] , "x1" : [0, 0, 0] },
                        "input_var_complexity" : {"x0" : 0.        , "x1" : 1.        },
                        # constants
                        "constants"            : {"pi" : pi        , "1" : const1    },
                        "constants_units"      : {"pi" : [0, 0, 0] , "1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 0.        , "1" : 1.        },
                        # free constants
                        "free_constants"            : {"a"             , "b"             },
                        "free_constants_init_val"   : {"a" : 1.        , "b" : 1.        },
                        "free_constants_units"      : {"a" : [0, 0, 0] , "b" : [0, 0, 0] },
                        "free_constants_complexity" : {"a" : 0.        , "b" : 1.        },
                           }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [0, 0, 0], superparent_name = "y")

        # PROGRAM
        test_program_str = ["add", "add", "mul", "a", "x0", "b", "x1",]
        test_program     = [my_lib.lib_name_to_token[name] for name in test_program_str]
        free_const_table = free_const.FreeConstantsTable(batch_size=1, library=my_lib, n_realizations=1)
        prog = Prog.Program(tokens=test_program, library=my_lib, free_consts=free_const_table, n_realizations=1)

        # OPTIMIZATION
        history = prog.optimize_constants(X=X, y_target=y_ideals,)

        # Execution for results
        y_pred  = prog.execute(X=X,)
        # Testing that optimization processed was logged
        works_bool = (prog.free_consts.is_opti[0] == True) and (prog.free_consts.opti_steps[0] > 0)
        self.assertTrue(works_bool)
        # Testing that constants were recovered
        tol = 1e-6
        works_bool = (torch.abs(prog.free_consts.class_values[0] - ideal_class_params)<tol).all()
        self.assertTrue(works_bool)
        # Testing that MSEs is low
        mse_tol = 1e-8
        mse = torch.mean((y_pred - y_ideals)**2)
        works_bool = (mse < mse_tol)
        self.assertTrue(works_bool)

        return None

    # Test program const optimization in Class SR scenario
    def test_optimize_with_class_and_spe_free_consts (self):

        DEVICE = 'cpu'
        if torch.cuda.is_available():
            DEVICE = 'cuda'

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
            return torch.split(flattened_data, n_samples_per_dataset.tolist(), dim=-1) # (n_realizations,) of (..., [n_samples depends on dataset],)

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
        multi_y_ideals   = unflatten_multi_data(y_ideals_flatten)                                         # (n_datasets,) of (n_samples depends on dataset,)


        # params[0]*torch.exp(-params[1]*X[0])*torch.cos(class_params[0]*X[0]+params[2]) + class_params[1]*X[1]
        # k0 * exp(-k1 * t) * cos(c0 * t + k2) + c1 * l
        # "add", "mul", "mul", "k0", "exp", "mul", "neg", "k1", "t", "cos", "add", "mul", "c0", "t", "k2", "mul", "c1", "l"

        k0_init = [1.,1.,1.]*10 # np.full(n_realizations, 1.)
        # consts
        pi     = data_conversion (np.pi) .to(DEVICE)
        const1 = data_conversion (1.)    .to(DEVICE)

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
        test_programs_idx = []
        test_prog_str_0 = ["add", "mul", "mul", "k0"  , "exp", "mul", "neg", "k1", "t", "cos", "add", "mul", "c0", "t", "k2", "mul", "c1", "l", ]
        test_tokens_0 = [my_lib.lib_name_to_token[name] for name in test_prog_str_0]
        free_const_table = free_const.FreeConstantsTable(batch_size=1, library=my_lib, n_realizations=n_realizations)
        prog = Prog.Program(tokens=test_tokens_0, library=my_lib, free_consts=free_const_table, n_realizations=n_realizations)

        # ------- Test optimization (all datasets flattened way) -------
        N = 10
        t0 = time.perf_counter()
        for _ in range (N):
            # Resetting free constants to initial values for timing
            free_const_table = free_const.FreeConstantsTable(batch_size=1, library=my_lib, n_realizations=n_realizations)
            prog.free_consts = free_const_table
            # Optimization
            history = prog.optimize_constants(X                     = multi_X_flatten,
                                              y_target              = y_ideals_flatten,
                                              n_samples_per_dataset = n_samples_per_dataset,
                                            )
        t1 = time.perf_counter()
        print("\nprog.optimize_constants time (flattened class SR) = %.3f ms"%((t1-t0)*1e3/N))

        # Execution for results
        y_computed_flatten = prog.execute(X = multi_X_flatten, n_samples_per_dataset = n_samples_per_dataset,)
        multi_y_computed = unflatten_multi_data(y_computed_flatten)
        # Testing that optimization processed was logged
        works_bool = (prog.free_consts.is_opti[0] == True) and (prog.free_consts.opti_steps[0] > 0)
        self.assertTrue(works_bool)
        # Testing that constants were recovered
        tol = 5*1e-3
        works_bool = (torch.abs(prog.free_consts.class_values[0] - ideal_class_params)<tol).all()
        self.assertTrue(works_bool)
        works_bool = (torch.abs(prog.free_consts.spe_values[0] - ideal_spe_params)<tol).all()
        above_tol = torch.abs(prog.free_consts.spe_values[0] - ideal_spe_params)[(torch.abs(prog.free_consts.spe_values[0] - ideal_spe_params)>=tol)]
        self.assertTrue(works_bool, "above_tol = %s"%above_tol)
        # Testing that MSEs are low
        mse_tol = 1e-6
        MSEs = torch.tensor([torch.mean((multi_y_computed[i] - multi_y_ideals[i])**2) for i in range(n_realizations)])
        works_bool = (MSEs < mse_tol).all()
        above_tol = MSEs[MSEs>=mse_tol]
        self.assertTrue(works_bool, "above_tol = %s"%above_tol)

        # ------- Test optimization (one-by-one) -------
        # Actually this makes no sense to test this as this will optimize class free constants one time per realization
        # but they are supposed to be common to all realizations. There is no point in optimizing them one by one even
        # if this was faster. Commenting out for now.
        #
        # t0 = time.perf_counter()
        # for _ in range (N):
        #    # Resetting free constants to initial values for timing
        #    free_const_table = free_const.FreeConstantsTable(batch_size=1, library=my_lib, n_realizations=n_realizations)
        #    prog.free_consts = free_const_table
        #    for i in range(n_realizations):
        #        # Optimization
        #        history = prog.optimize_constants(X             = multi_X[i],
        #                                          y_target      = multi_y_ideals[i],
        #                                          i_realization = i,
        #                                          )
        # t1 = time.perf_counter()
        # print("\nprog.optimize_constants time (one-by-one class SR) = %.3f ms"%((t1-t0)*1e3/N))
        #
        # # Execution for results
        # y_computed_flatten = prog.execute(X = multi_X_flatten, n_samples_per_dataset = n_samples_per_dataset,)
        # multi_y_computed = unflatten_multi_data(y_computed_flatten)
        # # Testing that optimization processed was logged
        # works_bool = (prog.free_consts.is_opti[0] == True) and (prog.free_consts.opti_steps[0] > 0)
        # self.assertTrue(works_bool)
        # # Testing that constants were recovered
        # tol = 5*1e-3
        # works_bool = (torch.abs(prog.free_consts.class_values[0] - ideal_class_params)<tol).all()
        # self.assertTrue(works_bool)
        # works_bool = (torch.abs(prog.free_consts.spe_values[0] - ideal_spe_params)<tol).all()
        # above_tol = torch.abs(prog.free_consts.spe_values[0] - ideal_spe_params)[(torch.abs(prog.free_consts.spe_values[0] - ideal_spe_params)>=tol)]
        # self.assertTrue(works_bool, "above_tol = %s"%above_tol)
        # # Testing that MSEs are low
        # mse_tol = 1e-6
        # MSEs = torch.tensor([torch.mean((multi_y_computed[i] - multi_y_ideals[i])**2) for i in range(n_realizations)])
        # works_bool = (MSEs < mse_tol).all()
        # above_tol = MSEs[MSEs>=mse_tol]
        # self.assertTrue(works_bool, "above_tol = %s"%above_tol)

        # ------- Sanity plot -------
        # fig, ax = plt.subplots(1,1,figsize=(10,5))
        # for i in range(n_realizations):
        #      ax.plot(multi_X[i][0], multi_y_ideals   [i].cpu().detach().numpy(), 'o', )
        #      ax.plot(multi_X[i][0], multi_y_computed [i].cpu().detach().numpy(), 'r-',)
        # ax.legend()
        #
        # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        # ax.plot(history)
        # plt.show()
        #
        # for i in range(n_realizations):
        #      mse = torch.mean((multi_y_computed[i] - multi_y_ideals[i])**2)
        #      print("%i, mse = %f"%(i, mse))
        return None


class VectProgramsTest(unittest.TestCase):

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------ INIT ------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # Test VectPrograms init
    def test_make_VectPrograms(self):
        # BATCH CONFIG
        batch_size = 10000
        max_time_step = 32
        my_lib = make_lib()
        # BATCH
        try:
            my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=max_time_step, library=my_lib)
        except:
            self.fail("VectPrograms creation failed.")

    # Test VectPrograms init
    def test_make_VectPrograms_assertions(self):
        # BATCH CONFIG
        my_lib = make_lib()
        # BATCH
        with self.assertRaises(AssertionError, ):
            my_programs = Prog.VectPrograms(batch_size='1000', max_time_step=32, library=my_lib)
        with self.assertRaises(AssertionError, ):
            my_programs = Prog.VectPrograms(batch_size=1000.0, max_time_step=32, library=my_lib)
        with self.assertRaises(AssertionError, ):
            my_programs = Prog.VectPrograms(batch_size=-1, max_time_step=32, library=my_lib)
        with self.assertRaises(AssertionError, ):
            my_programs = Prog.VectPrograms(batch_size=1000, max_time_step='32', library=my_lib)
        with self.assertRaises(AssertionError, ):
            my_programs = Prog.VectPrograms(batch_size=1000, max_time_step=32.0, library=my_lib)
        with self.assertRaises(AssertionError, ):
            my_programs = Prog.VectPrograms(batch_size=1000, max_time_step=-1, library=my_lib)

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- APPEND ASSERTIONS ------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # Exceed max_time_step
    def test_append_exceed_time_step(self):
        # BATCH CONFIG
        batch_size = 6
        max_time_step = 5
        my_lib = make_lib()
        # BATCH
        my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=max_time_step, library=my_lib)
        # APPEND
        # Exceeding time step by appending terminal tokens to raise this error
        terminal_token_idx = my_lib.n_choices - 1
        next_tokens_idx = np.full(batch_size, terminal_token_idx, int)
        with self.assertRaises(IndexError, ):
            for _ in range (max_time_step+1):
                my_programs.append(next_tokens_idx)
        return None

    # Test append wrong type
    def test_append_wrong_arg_type(self):
        # BATCH CONFIG
        batch_size = 1000
        max_time_step = 32
        my_lib = make_lib()
        # BATCH
        my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=max_time_step, library=my_lib)
        # APPEND
        next_tokens_idx = np.random.randint(low=0, high=my_lib.n_choices, size=batch_size)
        with self.assertRaises(AssertionError, ):
            my_programs.append(next_tokens_idx.astype(float))

    # Test append wrong shape
    def test_append_wrong_arg_shape(self):
        # BATCH CONFIG
        batch_size = 1000
        max_time_step = 32
        my_lib = make_lib()
        # BATCH
        my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=max_time_step, library=my_lib)
        # APPEND
        next_tokens_idx = np.random.randint(low=0, high=my_lib.n_choices, size=batch_size + 99)
        with self.assertRaises(AssertionError, ):
            my_programs.append(next_tokens_idx)

    # Test append wrong min/max
    def test_append_wrong_arg_min_max(self):
        # BATCH CONFIG
        batch_size = 1000
        max_time_step = 32
        my_lib = make_lib()
        # BATCH
        my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=max_time_step, library=my_lib)
        # APPEND
        next_tokens_idx = np.random.randint(low=-1, high=my_lib.n_choices, size=batch_size)
        with self.assertRaises(AssertionError):
            my_programs.append(next_tokens_idx)
        next_tokens_idx = np.random.randint(low=0, high=my_lib.n_choices+1, size=batch_size)
        with self.assertRaises(AssertionError):
            my_programs.append(next_tokens_idx)

    # Test append too many dummies necessary to complete
    def test_append_not_enough_space_for_dummies(self):
        # BATCH CONFIG
        batch_size = int(1e5)
        max_time_step = 8
        my_lib = make_lib()
        np.random.seed(seed=42)
        # Not enough space for dummies with unsafe number of steps
        with self.assertRaises(IndexError):
            # BATCH
            my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=max_time_step, library=my_lib)
            # ADDING NEW TOKENS
            for step in range(1, my_programs.safe_max_time_step*4):
                next_tokens_idx = np.random.randint(low=0, high=my_lib.n_choices, size=batch_size)
                my_programs.append(next_tokens_idx)
        # Enough space for dummies with safe number of steps
        try:
            # BATCH
            my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=max_time_step, library=my_lib)
            # ADDING NEW TOKENS
            for step in range(1, my_programs.safe_max_time_step):
                next_tokens_idx = np.random.randint(low=0, high=my_lib.n_choices, size=batch_size)
                my_programs.append(next_tokens_idx)
        except:
            self.fail("VectPrograms append failed.")
        return None

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------- FAMILY RELATIONSHIPS ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def test_family_relationships(self):
        # LIBRARY CONFIG
        args_make_tokens = {
                        # operations
                        "op_names"             : "all",  # or ["mul", "neg", "inv", "sin"]
                        "use_protected_ops"    : True,
                        # input variables
                        "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                        "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                        "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                        # constants
                        "constants"            : {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       , "const1" : 1         },
                        "constants_units"      : {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] , "const1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        , "const1" : 1.        },
                        # free constants
                        "free_constants"            : {"c0"             , "c1"               , "c2"             },
                        "free_constants_init_val"   : {"c0" : 1.        , "c1"  : 10.        , "c2" : 1.        },
                        "free_constants_units"      : {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                        "free_constants_complexity" : {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                           }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [1, -2, 1], superparent_name = "y")

        # TEST PROGRAM
        test_program_str = ["mul", "mul", "M", "n2", "c", "sub", "inv", "sqrt", "sub", "c1", "div", "n2", "v", "n2",
                            "c", "cos", "div", "sub", "const1", "div", "v", "c", "div", "v", "c"]
        test_program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str])
        test_program_length = len(test_program_str)
        test_program_idx = test_program_idx[np.newaxis, :]

        # BATCH
        my_programs = Prog.VectPrograms(batch_size=1, max_time_step=test_program_length, library=my_lib)
        my_programs.set_programs(test_program_idx)
        #for i in range (test_program_length):
        #    batch.append(test_program_idx[:, i])

        # CURSOR
        cursor = Prog.Cursor(programs = my_programs, prog_idx= 0, pos = 0)
        works_bool = cursor.set_pos(0).child(0).child(0).sibling.parent.sibling.child(0).child().child().child(1).child(
            0).child().parent.parent.sibling.__repr__() == "c1"

        # TEST
        self.assertTrue(works_bool)

        test = my_programs.get_infix_pretty(prog_idx=0)
        return None

    def test_ancestors_relationships(self):
        # LIBRARY CONFIG
        args_make_tokens = {
                        # operations
                        "op_names"             : "all",  # or ["mul", "neg", "inv", "sin"]
                        "use_protected_ops"    : True,
                        # input variables
                        "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                        "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                        "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                        # constants
                        "constants"            : {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       , "const1" : 1         },
                        "constants_units"      : {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] , "const1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        , "const1" : 1.        },
                            }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [1, -2, 1], superparent_name = "y")

        # TEST PROGRAM
        test_program_str = ["mul", "mul", "M", "n2", "c", "sub", "inv", "sqrt", "sub", "const1", "div", "n2", "v", "n2",
                            "c", "cos", "div", "sub", "const1", "div", "v", "c", "div", "v", "c"]
        test_program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str])
        test_program_length = len(test_program_str)
        test_program_idx = test_program_idx[np.newaxis, :]

        # BATCH
        my_programs = Prog.VectPrograms(batch_size=1, max_time_step=test_program_length, library=my_lib)
        my_programs.set_programs(test_program_idx)

        # CURSOR
        cursor = Prog.Cursor(programs = my_programs, prog_idx= 0, pos = 0)

        # TEST has_ancestors_mask
        np.array_equal(my_programs.tokens.has_ancestors_mask, np.full((1, test_program_length), True))

        # TEST n_ancestors and ancestors_pos
        for i in range (len(test_program_str)):
            cursor.set_pos(i)
            # TEST n_ancestors
            expected_n_ancestors = my_programs.tokens.depth[0, i] + 1
            computed_n_ancestors = my_programs.tokens.n_ancestors[0, i]
            # test
            works_bool = (expected_n_ancestors == computed_n_ancestors)
            self.assertTrue(works_bool)
            # TEST ancestors_pos
            # manual ancestors search
            expected_ancestors = [cursor.pos]
            for j in range (expected_n_ancestors-1):
                cursor.set_pos(expected_ancestors[-1])
                expected_ancestors.append(cursor.parent.pos)
            expected_ancestors = np.array(expected_ancestors[::-1])
            # computed ancestors
            computed_ancestors = my_programs.tokens.ancestors_pos[0, i, :expected_n_ancestors]
            # test
            works_bool = np.array_equal(expected_ancestors, computed_ancestors)
            self.assertTrue(works_bool)
        return None

    # Test program management regarding units (units tests are in dimensional_analysis_UnitTest.py)
    def test_units_related(self):
        # LIBRARY CONFIG
        args_make_tokens = {
                        # operations
                        "op_names"             : "all",  # or ["mul", "neg", "inv", "sin"]
                        "use_protected_ops"    : True,
                        # input variables
                        "input_var_ids"        : {"z" : 0         , "v" : 1          , "t" : 2,        },
                        "input_var_units"      : {"z" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                        "input_var_complexity" : {"z" : 0.        , "v" : 1.         , "t" : 0.,       },
                        # constants
                        "constants"            : {"g" : 9.8        , "c" : 3e8       , "m" : 1e6       , "E0" : 1         },
                        "constants_units"      : {"g" : [1, -2, 0] , "c" : [1, -1, 0], "m" : [0, 0, 1] , "E0" : [2, -2, 1] },
                        "constants_complexity" : {"g" : 0.         , "c" : 0.        , "m" : 1.        , "E0" : 1.        },
                            }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [2, -2, 1], superparent_name = "y")

        # TEST PROGRAMS
        test_programs_idx = []
        test_programs_str = [
            ["add", "mul", "mul", "m" , "z", "z" , "E0",],
            ["add", "mul", "mul", "m" , "g", "z" , "E0",],
            ["add", "mul", "m"  , "n2", "z", "E0", "-" ,],
        ]
        # Using terminal token placeholder that will be replaced by '-' void token in append function
        test_programs_str = np.char.replace(test_programs_str, '-', 't')

        # Expected behavior
        expected_is_physical = np.array([False,  True, False])
        # Only expressing expectations for cases
        o = phy.UNITS_ANALYSIS_NOT_PERFORMED_CASE_CODE
        n = np.NAN
        expected_units_analysis_cases = np.array([
            [ n,  n, n, n, n, n, o],
            [ n,  n, n, n, n, n, n],
            [ n,  n, n, n, n, o, o]])
        coords_expected_no_case = np.where(expected_units_analysis_cases == o)

        # Converting into idx
        for test_program_str in test_programs_str :
            test_programs_idx.append(np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str]))
        test_programs_idx = np.array(test_programs_idx)

        # Initializing programs
        my_programs = Prog.VectPrograms(batch_size=test_programs_idx.shape[0], max_time_step=test_programs_idx.shape[1], library=my_lib)

        # Appending tokens
        for i in range (test_programs_idx.shape[1]):
            my_programs.assign_required_units(ignore_unphysical = True)
            my_programs.append(test_programs_idx[:,i])

        coords_observed_no_case = np.where(my_programs.units_analysis_cases == o)

        # Test that unphysical programs were properly detected
        bool_works = np.array_equal(my_programs.is_physical, expected_is_physical)
        self.assertEqual(bool_works, True)

        # Test that units requirements was not performed where it is useless to perform it
        bool_works = np.array_equal(coords_expected_no_case, coords_observed_no_case)
        self.assertEqual(bool_works, True)

        return None


    def test_get_family_relationship_idx_interface(self):
        # LIBRARY CONFIG
        args_make_tokens = {
                        # operations
                        "op_names"             : "all",  # or ["mul", "neg", "inv", "sin"]
                        "use_protected_ops"    : True,
                        # input variables
                        "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                        "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                        "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                        # constants
                        "constants"            : {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       , "const1" : 1         },
                        "constants_units"      : {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] , "const1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        , "const1" : 1.        },
                            }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [1, -2, 1], superparent_name = "y")

        # TEST PROGRAM
        test_program_str = ["mul", "mul", "M", "n2", "c", "sub", "inv", "sqrt", "sub", "const1", "div", "n2", "v", "n2",
                            "c", "cos", "div", "sub", "const1", "div", "v", "c", "div", "v", "c"]
        test_program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str])
        test_program_length = len(test_program_str)
        test_program_idx = test_program_idx[np.newaxis, :]

        # BATCH
        my_programs = Prog.VectPrograms(batch_size=1, max_time_step=test_program_length, library=my_lib)
        my_programs.set_programs(test_program_idx)

        # ----- TEST GET PARENT IDX -----
        no_parent_idx_filler = 8888
        # parent of token at step = last step (c) should be a div
        test_step = len(test_program_str) - 1
        parent_idx   = my_programs.get_parent_idx(my_programs.coords_of_step(test_step))
        parent_token = my_lib.lib_tokens[parent_idx][0]
        works_bool = parent_token.name == "div"
        self.assertTrue(works_bool)
        # parent of token at step = 1 (mul) should be a mul
        test_step = 1
        parent_idx   = my_programs.get_parent_idx(my_programs.coords_of_step(test_step))
        parent_token = my_lib.lib_tokens[parent_idx][0]
        works_bool = parent_token.name == "mul"
        self.assertTrue(works_bool)
        # parent of token at step = 0 (mul) should no exit (superparent)
        test_step = 0
        parent_idx   = my_programs.get_parent_idx(coords = my_programs.coords_of_step(test_step),
                                                  no_parent_idx_filler = no_parent_idx_filler)
        works_bool = (parent_idx == no_parent_idx_filler)
        self.assertTrue(works_bool)

        # ----- TEST GET SIBLING IDX -----
        no_sibling_idx_filler = 8888
        # sibling of token at step = last step (c) should be a v
        test_step = len(test_program_str) - 1
        sibling_idx   = my_programs.get_sibling_idx(my_programs.coords_of_step(test_step))
        sibling_token = my_lib.lib_tokens[sibling_idx][0]
        works_bool = sibling_token.name == "v"
        self.assertTrue(works_bool)
        # sibling of token at step = 0 should not exist
        test_step = 0
        sibling_idx   = my_programs.get_sibling_idx(coords = my_programs.coords_of_step(test_step),
                                                   no_sibling_idx_filler = no_parent_idx_filler)
        works_bool = (sibling_idx == no_sibling_idx_filler)
        self.assertTrue(works_bool)

        # ----- TEST GET ANCESTORS IDX ----
        #is_ancestor = my_programs.get_is_ancestor()

        return None

if __name__ == '__main__':
    unittest.main(verbosity=2)
