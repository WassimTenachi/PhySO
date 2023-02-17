import numpy as np
import torch

L0 = ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"]

### ------------------------------ Nguyen-1 ------------------------------
def TestCase_mechanical_energy (device):
    # ------ Test case name ------
    test_case_name = "mechanical_energy" # CASE-SPECIFIC
    # ------ Constants ------
    const1 = 1.
    # ------ Data points ------
    n_dim = 1                           # CASE-SPECIFIC
    data_size = 20                      # CASE-SPECIFIC
    low, up = -1, 1 # CASE-SPECIFIC

    x0_array = np.sort(np.random.uniform(low, up, data_size))
    stack = [x0_array] + [np.random.uniform(low, up, data_size) for i in range (n_dim-1)]
    X_array = np.stack(stack, axis=0)
    y_array = x0_array**3 + x0_array**2 + x0_array**1 # CASE-SPECIFIC

    # Stack of all input variables
    X = torch.tensor(X_array).to(device)
    # Output of symbolic function to guess
    y = torch.tensor(y_array).to(device)

    # ------ Target Function ------
    target_program_str = ["add", "add", "mul", "mul", "x", "x", "x", "mul", "x", "x", "x"] # CASE-SPECIFIC

    # ------ Library Config ------
    const1 = torch.tensor(np.array(const1) .astype(float)).to(device)

    args_make_tokens = {
                    # operations
                    "op_names"             : L0,
                    "use_protected_ops"    : True,
                    # input variables
                    "input_var_ids"        : {"x" : 0         }, # CASE-SPECIFIC
                    "input_var_units"      : {"x" : [0, 0, 0] },
                    "input_var_complexity" : {"x" : 0.        },
                    # constants
                    "constants"            : {"1" : const1    },
                    "constants_units"      : {"1" : [0, 0, 0] },
                    "constants_complexity" : {"1" : 1.        },
                        }

    library_config = {"args_make_tokens"  : args_make_tokens,
                    "superparent_units" : [0, 0, 0],
                    "superparent_name"  : "f",
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


### ------------------------------ Nguyen-2 ------------------------------
def TestCase_Nguyen2 (device):
    # ------ Test case name ------
    test_case_name = "Nguyen-2" # CASE-SPECIFIC
    # ------ Constants ------
    const1 = 1.
    # ------ Data points ------
    n_dim = 1                           # CASE-SPECIFIC
    data_size = 20                      # CASE-SPECIFIC
    low, up = -1, 1 # CASE-SPECIFIC

    x0_array = np.sort(np.random.uniform(low, up, data_size))
    stack = [x0_array] + [np.random.uniform(low, up, data_size) for i in range (n_dim-1)]
    X_array = np.stack(stack, axis=0)
    y_array = x0_array**4 + x0_array**3 + x0_array**2 + x0_array**1 # CASE-SPECIFIC

    # Stack of all input variables
    X = torch.tensor(X_array).to(device)
    # Output of symbolic function to guess
    y = torch.tensor(y_array).to(device)

    # ------ Target Function ------
    target_program_str = ["add", "add", "add",
                          "mul", "mul", "mul", "x", "x", "x", "x",
                          "mul", "mul", "x", "x", "x",
                          "mul", "x", "x",
                          "x",]                     # CASE-SPECIFIC

    # ------ Library Config ------
    const1 = torch.tensor(np.array(const1) .astype(float)).to(device)

    args_make_tokens = {
                    # operations
                    "op_names"             : L0,
                    "use_protected_ops"    : True,
                    # input variables
                    "input_var_ids"        : {"x" : 0         }, # CASE-SPECIFIC
                    "input_var_units"      : {"x" : [0, 0, 0] },
                    "input_var_complexity" : {"x" : 0.        },
                    # constants
                    "constants"            : {"1" : const1    },
                    "constants_units"      : {"1" : [0, 0, 0] },
                    "constants_complexity" : {"1" : 1.        },
                        }

    library_config = {"args_make_tokens"  : args_make_tokens,
                    "superparent_units" : [0, 0, 0],
                    "superparent_name"  : "f",
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
    TestCase_Nguyen1,
    TestCase_Nguyen2,
    TestCase_Nguyen3,
    TestCase_Nguyen4,
    TestCase_Nguyen5,
    TestCase_Nguyen6,
    TestCase_Nguyen7,
    TestCase_Nguyen8,
    TestCase_Nguyen9,
    TestCase_Nguyen10,
    TestCase_Nguyen11,
    TestCase_Nguyen12,
]
