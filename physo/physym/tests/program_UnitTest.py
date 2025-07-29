import unittest

import matplotlib.pyplot as plt
import numpy as np
import time as time
import warnings
import torch
import pickle
import os

# Internal imports
from physo.physym import library as Lib
from physo.physym import program as Prog
from physo.physym import dimensional_analysis as phy
from physo.physym.functions import data_conversion, data_conversion_inv
import physo.physym.free_const as free_const
from physo.physym import vect_programs as VProg

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

    def test_pickability(self):

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
        multi_y_ideals   = unflatten_multi_data(y_ideals_flatten)                                         # (n_realizations,) of (n_samples depends on dataset,)


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

        # Test pickability
        try:
            pickle.dumps(prog)
        except:
            self.fail("Program pickability failed.")

        # Test save
        fpath = "test_prog.pkl"
        try:
            prog.save(fpath)
            os.remove(fpath) if os.path.exists(fpath) else None
        except:
            os.remove(fpath) if os.path.exists(fpath) else None
            self.fail("Program save failed.")

        # Test pickle load
        try:
            prog.save(fpath)
            prog_loaded = Prog.load_program_pkl(fpath)
            os.remove(fpath) if os.path.exists(fpath) else None
        except:
            os.remove(fpath) if os.path.exists(fpath) else None
            self.fail("Program pickle load failed.")

        return None

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
        multi_y_ideals   = unflatten_multi_data(y_ideals_flatten)                                         # (n_realizations,) of (n_samples depends on dataset,)


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
        print("\nprog.execute time (flattened mdho2d scenario) = %.3f ms"%((t1-t0)*1e3/N))
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
        print("\nprog.execute time (one-by-one mdho2d scenario) = %.3f ms"%((t1-t0)*1e3/N))

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

        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)

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

    # Test program const optimization when no constants are present
    def test_optimize_no_consts_in_prog (self):

        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)

        ideal_class_params = torch.tensor([1.389, 1.005]) # (n_class_params, )

        # Synthetic data
        x0 = torch.linspace(0, 10, 1000)
        x1 = torch.linspace(-5, 1 , 1000)
        X = torch.stack((x0,x1),axis=0)
        y_ideals = X[0] + X[1]

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
        test_program_str = ["add", "x0", "x1",]
        test_program     = [my_lib.lib_name_to_token[name] for name in test_program_str]
        free_const_table = free_const.FreeConstantsTable(batch_size=1, library=my_lib, n_realizations=1)
        prog = Prog.Program(tokens=test_program, library=my_lib, free_consts=free_const_table, n_realizations=1)

        # OPTIMIZATION
        try:
            history = prog.optimize_constants(X=X, y_target=y_ideals,)
        except:
            self.fail("Program optimization failed when no free constants are present.")

        # Execution for results
        y_pred  = prog.execute(X=X,)
        # Testing that optimization processed was logged
        expected_opti_steps = 0
        works_bool = (prog.free_consts.is_opti[0] == True) and (prog.free_consts.opti_steps[0] == expected_opti_steps)
        self.assertTrue(works_bool)

        return None

    # Test program const optimization in Class SR scenario
    def test_optimize_with_spe_free_consts (self):

        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)

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
        print("\nprog.optimize_constants time (flattened mdho2d scenario) = %.3f ms"%((t1-t0)*1e3/N))

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
        # print("\nprog.optimize_constants time (one-by-one mdho2d scenario) = %.3f ms"%((t1-t0)*1e3/N))
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

    # Test program const optimization in Class SR scenario
    def test_optimize_with_spe_free_consts_with_weights (self):

        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)

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
        free_const_opti_args = {
            'loss': "MSE",
            'method': 'LBFGS',
            'method_args': {
                'n_steps': 120,
                'tol': 1e-16,
                'lbfgs_func_args': {
                    'max_iter': 4,
                    'line_search_fn': "strong_wolfe",
                },
            },
        }

        N = 10
        t0 = time.perf_counter()
        for _ in range (N):
            # Resetting free constants to initial values for timing
            free_const_table = free_const.FreeConstantsTable(batch_size=1, library=my_lib, n_realizations=n_realizations)
            prog.free_consts = free_const_table
            # Optimization
            history = prog.optimize_constants(X                     = multi_X_flatten,
                                              y_weights             = y_weights_flatten,
                                              y_target              = y_ideals_flatten,
                                              n_samples_per_dataset = n_samples_per_dataset,
                                              args_opti             = free_const_opti_args,
                                            )
        t1 = time.perf_counter()
        print("\nprog.optimize_constants time (flattened wmdho2d scenario) = %.3f ms"%((t1-t0)*1e3/N))

        # --------------------------- TESTS ---------------------------

        # Execution for results
        y_computed_flatten = prog.execute(X = multi_X_flatten, n_samples_per_dataset = n_samples_per_dataset,)
        multi_y_computed = unflatten_multi_data(y_computed_flatten)

        # --------- Testing that optimization processed was logged ---------
        works_bool = (prog.free_consts.is_opti[0] == True) and (prog.free_consts.opti_steps[0] > 0)
        self.assertTrue(works_bool)

        # --------- Testing that constants were recovered ---------
        tol = 1*1e-3

        # -> Class free constants
        works_bool = (torch.abs(prog.free_consts.class_values[0] - ideal_class_params)<tol).all()
        self.assertTrue(works_bool)

        # -> Spe free constants
        spe_init = torch.tensor(prog.free_consts.library.spe_free_constants_init_val)           # (n_spe_params, n_realizations)
        # Checking that constants with high weights are recovered
        i_reals  = y_weights_per_dataset > 0.9
        is_recov = torch.abs(prog.free_consts.spe_values[0][:,i_reals] - ideal_spe_params[:,i_reals]) < tol
        works_bool = is_recov.all()
        self.assertTrue(works_bool)
        # Checking that constants with non zero-weights changed
        i_reals  = y_weights_per_dataset > 0.0
        is_recov = torch.abs(prog.free_consts.spe_values[0][:,i_reals] - spe_init[:,i_reals]) > 0.
        works_bool = is_recov.all()
        self.assertTrue(works_bool)
        # Checking that constants with zero-weights did not change
        i_reals  = y_weights_per_dataset == 0.0
        is_recov = torch.abs(prog.free_consts.spe_values[0][:,i_reals] - spe_init[:,i_reals]) == 0
        works_bool = is_recov.all()
        self.assertTrue(works_bool)

        # --------- MSE ---------
        mse_tol = 1e-6
        MSEs = torch.tensor([torch.mean((multi_y_computed[i] - multi_y_ideals[i])**2) for i in range(n_realizations)]) # (n_realizations,)

        # Testing that MSEs are low for high weights
        works_bool = (MSEs[y_weights_per_dataset > 0.5] < mse_tol).all()
        self.assertTrue(works_bool)

        # Testing that MSEs are high for low weights
        works_bool = (MSEs[y_weights_per_dataset == 0.] > mse_tol).all()
        self.assertTrue(works_bool)

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
        # print("\nprog.optimize_constants time (one-by-one wmdho2d scenario) = %.3f ms"%((t1-t0)*1e3/N))
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

if __name__ == '__main__':
    unittest.main(verbosity=2)
