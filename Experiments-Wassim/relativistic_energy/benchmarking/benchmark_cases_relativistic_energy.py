import numpy as np
import torch

L0 = ["mul", "add", "sub", "div", "inv", "n2", "sqrt", "exp", "log", "sin", "cos"]

### ------------------------------ mechanical_energy ------------------------------
def TestCase_mechanical_energy (device):
    # ------ Test case name ------
    test_case_name = "mechanical_energy" # CASE-SPECIFIC

    # ------ Data points ------
    data_size = 50
    low, up = -10, 10

    z  = np.random.uniform(low, up, data_size)
    vz = np.random.uniform(low, up, data_size)
    x0_array = z
    x1_array = vz
    X_array = np.stack((x0_array, x1_array), axis=0)
    m = 1.5
    g = 9.8
    y_array = m*g*z + m*vz**2 #+ 0.5*m*vz**2

    # ------ Vectors ------
    # Stack of all input variables
    X = torch.tensor(X_array).to(device)
    # Output of symbolic function to guess
    y = torch.tensor(y_array).to(device)

    # ------ Constants ------
    const1 = torch.tensor(np.array(1.)).to(device)
    m = torch.tensor(np.array(m)).to(device)
    g = torch.tensor(np.array(g)).to(device)

    # ------ Target Function ------
    target_program_str = ["add", "mul", "mul", "m", "g", "z", "mul", "m", "n2", "v"] # CASE-SPECIFIC

    # ------ Library Config ------

    args_make_tokens = {
                    # operations
                    "op_names"             : L0, #["mul", "add", "sub", "div", "inv", "n2", "sqrt", "exp", "log", "sin", "cos"],
                    "use_protected_ops"    : True,
                    # input variables
                    "input_var_ids"        : {"z" : 0         , "v" : 1         , },
                    "input_var_units"      : {"z" : [1, 0, 0] , "v" : [1, -1, 0], },
                    "input_var_complexity" : {"z" : 1.        , "v" : 1.        , },
                    # constants
                    "constants"            : {"1" : const1    , },
                    "constants_units"      : {"1" : [0, 0, 0] , },
                    "constants_complexity" : {"1" : 1.        , },
                    # free constants
                    "free_constants"            : {"m"              , "g"              ,},
                    "free_constants_init_val"   : {"m" : 1.         , "g" : 1.         ,},
                    "free_constants_units"      : {"m" : [0, 0, 1]  , "g" : [1, -2, 0] ,},
                    "free_constants_complexity" : {"m" : 1.         , "g" : 1.         ,},
                        }

    library_config = {"args_make_tokens"  : args_make_tokens,
                      "superparent_units" : [2, -2, 1],
                      "superparent_name"  : "E",
                    }

    # ------ Ideal reward ------
    expected_ideal_reward = 1.
    candidate_wrapper = None

    # ------ Config ------
    test_case_dict = {"name"                  : test_case_name,
                      "X"                     : X,
                      "y"                     : y,
                      "library_config"        : library_config,
                      "target_program_str"    : target_program_str,
                      "expected_ideal_reward" : expected_ideal_reward,
                      "candidate_wrapper"     : candidate_wrapper,
                       }
    return test_case_dict


### ------------------------------ relativistic_energy ------------------------------
def TestCase_relativistic_energy (device):
    # ------ Test case name ------
    test_case_name = "relativistic_energy" # CASE-SPECIFIC

    # ------ Data points ------
    data_size = int(1e3)

    # Data points
    c = 10.
    data_lowbound, data_upbound = -10, 10
    m = np.random.uniform(data_lowbound, data_upbound, data_size)
    data_lowbound, data_upbound = -0.9 * c, 0.9 * c
    v = np.random.uniform(data_lowbound, data_upbound, data_size)
    x0_array = m
    x1_array = v
    X_array = np.stack((x0_array, x1_array), axis=0)
    y_array = m * (c ** 2) * (1 / (1 - ((v ** 2) / (c ** 2))) ** 0.5)

    # ------ Vectors ------
    # Stack of all input variables
    X = torch.tensor(X_array).to(device)
    # Output of symbolic function to guess
    y = torch.tensor(y_array).to(device)

    # ------ Constants ------
    const1 = torch.tensor(np.array(1.) .astype(float)).to(device)
    c      = torch.tensor(np.array(c)  .astype(float)).to(device)

    # ------ Target Function ------
    target_program_str = ["mul", "mul", "m", "n2", "c", "inv", "sqrt", "sub", "1", "div", "n2", "v", "n2", "c", ]

    # ------ Library Config ------

    args_make_tokens = {
                    # operations
                    "op_names"             : L0, #["mul", "add", "sub", "div", "inv", "n2", "sqrt", "exp", "log", "sin", "cos"],
                    "use_protected_ops"    : True,
                    # input variables
                    "input_var_ids"        : {"m" : 0         , "v" : 1         ,},
                    "input_var_units"      : {"m" : [0, 0, 1] , "v" : [1, -1, 0] ,},
                    "input_var_complexity" : {"m" : 1.        , "v" : 1.        ,},
                    # constants
                    "constants"            : {"1" : const1    , "c" : c          },
                    "constants_units"      : {"1" : [0, 0, 0] , "c" : [1, -1, 0] },
                    "constants_complexity" : {"1" : 1.        , "c" : 1.         },
                    # free constants
                    #"free_constants"            : {"c"              },
                    #"free_constants_init_val"   : {"c" : 1.         },
                    #"free_constants_units"      : {"c" : [1, -1, 0] },
                    #"free_constants_complexity" : {"c" : 1.         },
                        }

    library_config = {"args_make_tokens"  : args_make_tokens,
                    "superparent_units" : [2, -2, 1],
                    "superparent_name"  : "E",
                    }

    # ------ Ideal reward ------
    expected_ideal_reward = 1.
    candidate_wrapper = None

    # ------ Config ------
    test_case_dict = {"name"                  : test_case_name,
                      "X"                     : X,
                      "y"                     : y,
                      "library_config"        : library_config,
                      "target_program_str"    : target_program_str,
                      "expected_ideal_reward" : expected_ideal_reward,
                      "candidate_wrapper"     : candidate_wrapper,
                       }
    return test_case_dict



TEST_CASES = [
    TestCase_mechanical_energy,
	TestCase_relativistic_energy,
]
