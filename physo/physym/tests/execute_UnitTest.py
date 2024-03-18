import unittest
import numpy as np
import time as time
import torch as torch
import sympy as sympy
import matplotlib.pyplot as plt

# Internal imports
from physo.physym import execute as Exec
from physo.physym import library as Lib
from physo.physym.functions import data_conversion, data_conversion_inv

class ExecuteProgramTest(unittest.TestCase):

    # Test program execution on a complicated function
    def test_ExecuteProgram (self):

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
                        "constants"            : {"pi" : pi        , "c" : c         , "M" : M         , "1" : const1    },
                        "constants_units"      : {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] , "1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        , "1" : 1.        },
                           }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [1, -2, 1], superparent_name = "y")

        # PROGRAM
        test_program_str = ["mul", "mul", "M", "n2", "c", "sub", "inv", "sqrt", "sub", "1", "div", "n2", "v", "n2",
                            "c", "cos", "div", "sub", "1", "div", "v", "c", "div", "div", "x", "t", "c"]
        test_program     = [my_lib.lib_name_to_token[name] for name in test_program_str]
        # EXPECTED RES
        expected_res     = M*(c**2)*(1./torch.sqrt(1.-(v**2)/(c**2))-torch.cos((1.-(v/c))/((x/t)/c)))

        N = 100
        # EXECUTION
        t0 = time.perf_counter()
        for _ in range (N):
            res = Exec.ExecuteProgram(input_var_data = data, program_tokens = test_program, )
        t1 = time.perf_counter()
        print("\nExecuteProgram time = %.3f ms"%((t1-t0)*1e3/N))

        # EXECUTION (wo tokens)
        t0 = time.perf_counter()
        for _ in range (N):
            expected_res     = M*(c**2)*(1./torch.sqrt(1.-(v**2)/(c**2))-torch.cos((1.-(v/c))/((x/t)/c)))
        t1 = time.perf_counter()
        print("\nExecuteProgram time (wo tokens) = %.3f ms"%((t1-t0)*1e3/N))

        # TEST
        works_bool = np.array_equal(data_conversion_inv(res.cpu()), data_conversion_inv(expected_res.cpu()),)
        self.assertTrue(works_bool)
        return None

    # Test program execution on a complicated function
    def test_ExecuteProgram_with_free_consts (self):

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
        # EXPECTED RES
        expected_res     = M*(c**2)*(1./torch.sqrt(1.-(v**2)/(c**2))-torch.cos((1.-(v/c))/((x/t)/c)))

        N = 100
        # EXECUTION
        t0 = time.perf_counter()
        for _ in range (N):
            res = Exec.ExecuteProgram(input_var_data = data, class_free_consts_vals = free_const_values, program_tokens = test_program, )
        t1 = time.perf_counter()
        print("\nExecuteProgram time = %.3f ms"%((t1-t0)*1e3/N))

        # EXECUTION (wo tokens)
        t0 = time.perf_counter()
        for _ in range (N):
            expected_res     = M*(c**2)*(1./torch.sqrt(1.-(v**2)/(c**2))-torch.cos((1.-(v/c))/((x/t)/c)))
        t1 = time.perf_counter()
        print("\nExecuteProgram time (wo tokens) = %.3f ms"%((t1-t0)*1e3/N))

        # TEST
        works_bool = np.array_equal(data_conversion_inv(res.cpu()), data_conversion_inv(expected_res.cpu()),)
        self.assertTrue(works_bool)
        return None

    # Test program execution in Class SR scenario
    def test_ExecuteProgram_with_class_and_spe_free_consts (self):

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
        multi_y_ideals   = unflatten_multi_data(y_ideals_flatten)                                            # (n_realizations,) of (n_samples depends on dataset,)


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

        # Test execution on all datasets in flattened form
        N = 1000
        t0 = time.perf_counter()
        for _ in range (N):
            y_computed_flatten = Exec.ExecuteProgram(input_var_data         = multi_X_flatten,
                                                     class_free_consts_vals = ideal_class_params_flatten,
                                                     spe_free_consts_vals   = ideal_spe_params_flatten,
                                                     program_tokens         = test_tokens_0, )
        t1 = time.perf_counter()
        print("\nExecuteProgram time (flattened class SR) = %.3f ms"%((t1-t0)*1e3/N))
        #multi_y_computed = unflatten_multi_data(y_computed_flatten)
        works_bool = (y_computed_flatten == y_ideals_flatten).all()
        self.assertTrue(works_bool)

        # Test execution on all datasets but one by one
        t0 = time.perf_counter()
        for _ in range (N):
            multi_y_computed = []
            for i in range(n_realizations):
                y_computed = Exec.ExecuteProgram(input_var_data         = multi_X[i],
                                                 class_free_consts_vals = ideal_class_params,
                                                 spe_free_consts_vals   = ideal_spe_params[i],
                                                 program_tokens         = test_tokens_0, )
                multi_y_computed.append(y_computed)
        t1 = time.perf_counter()
        print("\nExecuteProgram time (one-by-one class SR) = %.3f ms"%((t1-t0)*1e3/N))

        for i in range(n_realizations):
            works_bool = (multi_y_computed[i] == multi_y_ideals[i]).all()
            self.assertTrue(works_bool)

        # Sanity plot
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

    # Test program infix notation on a complicated function
    def test_ComputeInfixNotation(self):

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
                        "constants"            : {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       , "1" : 1         },
                        "constants_units"      : {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] , "1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        , "1" : 1.        },
                            }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [1, -2, 1], superparent_name = "y")

        # TEST PROGRAM
        test_program_str = ["mul", "mul", "M", "n2", "c", "sub", "inv", "sqrt", "sub", "1", "div", "n2", "v", "n2",
                            "c", "cos", "div", "sub", "1", "div", "v", "c", "pi"]
        test_program     = np.array([my_lib.lib_name_to_token[tok_str] for tok_str in test_program_str])
        # Infix output
        t0 = time.perf_counter()
        N = 100
        for _ in range (N):
            infix_str = Exec.ComputeInfixNotation(test_program)
        t1 = time.perf_counter()
        print("\nComputeInfixNotation time = %.3f ms"%((t1-t0)*1e3/N))
        infix = sympy.parsing.sympy_parser.parse_expr(infix_str)
        # Expected infix output
        expected_str = "M*(c**2.)*(1./((1.-(v**2)/(c**2))**0.5)-cos((1.-(v/c))/pi))"
        expected = sympy.parsing.sympy_parser.parse_expr(expected_str)
        # difference
        diff = sympy.simplify(infix - expected, rational = True)
        works_bool = diff == 0
        self.assertTrue(works_bool)

if __name__ == '__main__':
    unittest.main(verbosity=2)
