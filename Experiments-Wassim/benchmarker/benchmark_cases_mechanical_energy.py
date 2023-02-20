import numpy as np
import torch

L0 = ["mul", "add", "sub", "div", "inv", "n2", "sqrt", "exp", "log", "sin", "cos"]

### ------------------------------ mechanical_energy ------------------------------
def TestCase_mechanical_energy (device):
    # ------ Test case name ------
    test_case_name = "mechanical_energy" # CASE-SPECIFIC
    # ------ Constants ------
    const1 = 1.
    # ------ Data points ------
    n_dim = 1                           # CASE-SPECIFIC
    data_size = 50                      # CASE-SPECIFIC
    low, up = -10, 10                   # CASE-SPECIFIC

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
    target_program_str = ["add", "mul", "mul", "m", "g", "z", "mul", "m", "n2", "v_z"] # CASE-SPECIFIC

    # ------ Library Config ------

    args_make_tokens = {
                    # operations
                    "op_names"             : ["mul", "add", "sub", "div", "inv", "n2", "sqrt", "exp", "log", "sin", "cos"],
                    "use_protected_ops"    : True,
                    # input variables
                    "input_var_ids"        : {"z" : 0         , "v_z" : 1         , },
                    "input_var_units"      : {"z" : [1, 0, 0] , "v_z" : [1, -1, 0], },
                    "input_var_complexity" : {"z" : 1.        , "v_z" : 1.        , },
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




TEST_CASES = [
    TestCase_mechanical_energy,
]
