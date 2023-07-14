import unittest
import numpy as np
import time as time
import torch as torch
import sympy as sympy

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
            res = Exec.ExecuteProgram(input_var_data = data, free_const_values = free_const_values, program_tokens = test_program, )
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
