import time
import unittest
import numpy as np
import torch

# Internal imports
from physr.physym import Batch
from physr.physym import Program
from physr.physym.Functions import data_conversion
from physr.physym import Library
from physr.physym import ExecuteProgram
from physr.physym import Prior

class BatchTest(unittest.TestCase):
    def test_creation(self):

        # ------- TEST CASE -------
        DEVICE = 'cpu'
        if torch.cuda.is_available():
            DEVICE = 'cuda'

        # --- DATA ---
        N = int(1e6)
        x = data_conversion  (np.linspace(0.04, 4, N)  ).to(DEVICE)
        v = data_conversion  (np.linspace(0.10, 10, N) ).to(DEVICE)
        t = data_conversion  (np.linspace(0.06, 6, N)  ).to(DEVICE)
        X = torch.stack((x, v, t), axis=0)
        y = data_conversion(np.linspace(0.06, 6, N)).to(DEVICE)
        M  = data_conversion (1e6).to(DEVICE)
        c  = data_conversion (3e8).to(DEVICE)
        pi = data_conversion (np.pi).to(DEVICE)
        const1 = data_conversion (1.).to(DEVICE)

        # --- LIBRARY CONFIG ---
        args_make_tokens = {
                        # operations
                        "op_names"             : ["mul", "add", "neg", "inv", "cos"],
                        "use_protected_ops"    : True,
                        # input variables
                        "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                        "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                        "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                        # constants
                        "constants"            : {"pi" : pi        , "c" : c         , "M" : M         , "const1" : const1    },
                        "constants_units"      : {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] , "const1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        , "const1" : 1.        },
                            }
        library_args = {"args_make_tokens"  : args_make_tokens,
                        "superparent_units" : [1, -2, 1],
                        "superparent_name"  : "y",
                        }

        # --- PRIORS ---
        priors_config  = [ ("UniformArityPrior", None),
                           ("HardLengthPrior", {"min_length": 1,
                                               "max_length": 8, }),]

        # --- BATCH ---
        batch_size    = 1000
        max_time_step = 30

        my_batch = Batch.Batch(library_args    = library_args,
                               priors_config   = priors_config,
                               batch_size      = batch_size,
                               max_time_step   = max_time_step,
                               reward_function = ExecuteProgram.Reward_SquashedNRMSE,
                               X        = X,
                               y_target = y,
                               )
        return None

    def test_dummy_epoch(self):

        # ------- TEST CASE -------
        DEVICE = 'cpu'
        if torch.cuda.is_available():
            DEVICE = 'cuda'

        # --- DATA ---
        N = int(1e6)
        x = data_conversion  (np.linspace(0.04, 4, N)  ).to(DEVICE)
        v = data_conversion  (np.linspace(0.10, 10, N) ).to(DEVICE)
        t = data_conversion  (np.linspace(0.06, 6, N)  ).to(DEVICE)
        X = torch.stack((x, v, t), axis=0)
        y = data_conversion(np.linspace(0.06, 6, N)).to(DEVICE)
        M  = data_conversion (1e6).to(DEVICE)
        c  = data_conversion (3e8).to(DEVICE)
        pi = data_conversion (np.pi).to(DEVICE)
        const1 = data_conversion (1.).to(DEVICE)

        # --- LIBRARY CONFIG ---
        args_make_tokens = {
                        # operations
                        "op_names"             : ["mul", "add", "neg", "inv", "cos"],
                        "use_protected_ops"    : True,
                        # input variables
                        "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                        "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                        "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                        # constants
                        "constants"            : {"pi" : pi        , "c" : c         , "M" : M         , "const1" : const1    },
                        "constants_units"      : {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] , "const1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        , "const1" : 1.        },
                            }
        library_args = {"args_make_tokens"  : args_make_tokens,
                        "superparent_units" : [2, -2, 1],
                        "superparent_name"  : "E",
                        }

        # --- PRIORS ---
        priors_config  = [ ("UniformArityPrior", None),
                           ("HardLengthPrior", {"min_length": 1,
                                               "max_length": 8, }),
                           ("PhysicalUnitsPrior", {"prob_eps": np.finfo(np.float32).eps})]

        # --- BATCH ---
        batch_size    = 1000
        max_time_step = 30

        my_batch = Batch.Batch(library_args    = library_args,
                               priors_config   = priors_config,
                               batch_size      = batch_size,
                               max_time_step   = max_time_step,
                               reward_function = ExecuteProgram.Reward_SquashedNRMSE,
                               X        = X,
                               y_target = y,
                               )

        # --- DUMMY EPOCH ---
        t0 = time.perf_counter()
        for step in range(max_time_step):
            # Embedding output
            prior        = torch.tensor(my_batch.prior().astype(np.float32),   requires_grad=False)                          # (batch_size, n_choices)
            observations = torch.tensor(my_batch.get_obs().astype(np.float32), requires_grad=False)                          # (batch_size, 3*n_choices+1)
            # Dummy model output
            probs        = torch.tensor(np.random.rand(my_batch.batch_size, my_batch.library.n_choices).astype(np.float32))  # (batch_size, n_choices,)
            # Actions
            actions      = torch.multinomial(probs * prior, num_samples=1)[:, 0]
            my_batch.programs.append(actions)
        # Embedding output
        lengths = my_batch.programs.n_lengths
        R       = my_batch.get_rewards()
        # Computing model loss using (probs,actions, R, lengths)
        # ...
        t1 = time.perf_counter()
        print("Dummy epoch time = %f ms"%((t1-t0)*1e3))

if __name__ == '__main__':
    unittest.main(verbosity=2)
