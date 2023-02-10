import time
import unittest
import numpy as np
import torch

# Internal imports
from physo.physym import batch
from physo.physym import program
from physo.physym.functions import data_conversion
from physo.physym import library
from physo.physym import reward
from physo.physym import prior

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

        my_batch = batch.Batch(library_args     = library_args,
                               priors_config    = priors_config,
                               batch_size       = batch_size,
                               max_time_step    = max_time_step,
                               rewards_computer = reward.make_RewardsComputer (reward_function = reward.SquashedNRMSE),
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
        N = int(1e3)
        x_array = np.linspace(0.04, 4, N)
        x = data_conversion (x_array).to(DEVICE)
        X = torch.stack((x,), axis=0)
        pi = data_conversion (np.pi).to(DEVICE)
        const1 = data_conversion (1.).to(DEVICE)
        T_array  = 1.028
        v0_array = 0.995
        T = data_conversion (T_array).to(DEVICE)
        v0 = data_conversion (v0_array).to(DEVICE)
        y_target = data_conversion(x_array/T_array + v0_array).to(DEVICE)

        # --- LIBRARY CONFIG ---
        args_make_tokens = {
                        # operations
                        "op_names"             : ["add", "div"],
                        "use_protected_ops"    : True,
                        # input variables
                        "input_var_ids"        : {"x" : 0         },
                        "input_var_units"      : {"x" : [1, 0, 0] },
                        "input_var_complexity" : {"x" : 1.        },
                        # constants
                        "constants"            : {"pi" : pi        , "const1" : const1    , "T" : T         , "v0" : v0         },
                        "constants_units"      : {"pi" : [0, 0, 0] , "const1" : [0, 0, 0] , "T" : [0, 1, 0] , "v0" : [1, -1, 0] },
                        "constants_complexity" : {"pi" : 1.        , "const1" : 1.        , "T" : 1.        , "v0" : 1.         },
                            }
        library_args = {"args_make_tokens"  : args_make_tokens,
                        "superparent_units" : [1, -1, 0],
                        "superparent_name"  : "v",
                        }

        # --- PRIORS ---
        priors_config  = [ ("UniformArityPrior", None),
                           ("HardLengthPrior", {"min_length": 1,
                                               "max_length": 5, }),
                           ("PhysicalUnitsPrior", {"prob_eps": np.finfo(np.float32).eps})]

        # --- BATCH ---
        batch_size    = 1000
        max_time_step = 10

        my_batch = batch.Batch(library_args     = library_args,
                               priors_config    = priors_config,
                               batch_size       = batch_size,
                               max_time_step    = max_time_step,
                               rewards_computer = reward.make_RewardsComputer (reward_function = reward.SquashedNRMSE),
                               X        = X,
                               y_target = y_target,
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

    def test_dummy_epoch_duplicate_elimination (self):

        # ------- TEST CASE -------
        DEVICE = 'cpu'
        if torch.cuda.is_available():
            DEVICE = 'cuda'

        # --- DATA ---
        N = int(1e3)
        x_array = np.linspace(0.04, 4, N)
        x = data_conversion (x_array).to(DEVICE)
        X = torch.stack((x,), axis=0)
        pi = data_conversion (np.pi).to(DEVICE)
        const1 = data_conversion (1.).to(DEVICE)
        T_array  = 1.028
        v0_array = 0.995
        T = data_conversion (T_array).to(DEVICE)
        v0 = data_conversion (v0_array).to(DEVICE)
        y_target = data_conversion(x_array/T_array + v0_array).to(DEVICE)

        # --- LIBRARY CONFIG ---
        args_make_tokens = {
                        # operations
                        "op_names"             : ["add", "div"],
                        "use_protected_ops"    : True,
                        # input variables
                        "input_var_ids"        : {"x" : 0         },
                        "input_var_units"      : {"x" : [1, 0, 0] },
                        "input_var_complexity" : {"x" : 1.        },
                        # constants
                        "constants"            : {"pi" : pi        , "const1" : const1    , "T" : T         , "v0" : v0         },
                        "constants_units"      : {"pi" : [0, 0, 0] , "const1" : [0, 0, 0] , "T" : [0, 1, 0] , "v0" : [1, -1, 0] },
                        "constants_complexity" : {"pi" : 1.        , "const1" : 1.        , "T" : 1.        , "v0" : 1.         },
                            }
        library_args = {"args_make_tokens"  : args_make_tokens,
                        "superparent_units" : [1, -1, 0],
                        "superparent_name"  : "v",
                        }

        # --- PRIORS ---
        priors_config  = [ ("UniformArityPrior", None),
                           ("HardLengthPrior", {"min_length": 1,
                                               "max_length": 5, }),
                           ("PhysicalUnitsPrior", {"prob_eps": np.finfo(np.float32).eps})]

        # --- BATCH ---
        batch_size    = 1000
        max_time_step = 10

        my_batch = batch.Batch(library_args     = library_args,
                               priors_config    = priors_config,
                               batch_size       = batch_size,
                               max_time_step    = max_time_step,
                               rewards_computer = reward.make_RewardsComputer (reward_function = reward.SquashedNRMSE,
                                                                               zero_out_unphysical = True,
                                                                               zero_out_duplicates = True),
                               X        = X,
                               y_target = y_target,
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
        eps = 1e-5
        n_solutions = (np.abs(1-R) < eps).sum()
        n_kept = (R>0).sum()
        print("Dummy epoch time (w duplicate elimination) = %f ms (found %i/%i candidates with R > 0 "
              "and %i/%i with R = 1 +/- %f)"%((t1-t0)*1e3, n_kept, batch_size, n_solutions, batch_size, eps))

    def test_dummy_epoch_free_consts(self):

        # ------- TEST CASE -------
        DEVICE = 'cpu'
        if torch.cuda.is_available():
            DEVICE = 'cuda'

        # --- DATA ---
        N = int(1e3)
        x_array = np.linspace(0.04, 4, N)
        x = data_conversion (x_array).to(DEVICE)
        X = torch.stack((x,), axis=0)
        pi = data_conversion (np.pi).to(DEVICE)
        const1 = data_conversion (1.).to(DEVICE)
        T = 1.028
        v0 = 0.995
        y_target = data_conversion(x_array/T + v0).to(DEVICE)


        # --- LIBRARY CONFIG ---
        args_make_tokens = {
                        # operations
                        "op_names"             : ["add", "div"],
                        "use_protected_ops"    : True,
                        # input variables
                        "input_var_ids"        : {"x" : 0         },
                        "input_var_units"      : {"x" : [1, 0, 0] },
                        "input_var_complexity" : {"x" : 0.        },
                        # constants
                        "constants"            : {"pi" : pi        , "const1" : const1    },
                        "constants_units"      : {"pi" : [0, 0, 0] , "const1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 0.        , "const1" : 1.        },
                        # free constants
                        "free_constants"            : {"T"              , "v0"              ,},
                        "free_constants_init_val"   : {"T" : 1.         , "v0" : 1.         ,},
                        "free_constants_units"      : {"T" : [0, 1, 0] , "v0" : [1, -1, 0] ,},
                        "free_constants_complexity" : {"T" : 0.         , "v0" : 0.         ,},
                            }
        library_args = {"args_make_tokens"  : args_make_tokens,
                        "superparent_units" : [1, -1, 0],
                        "superparent_name"  : "v",
                        }

        # --- PRIORS ---
        priors_config  = [ ("UniformArityPrior", None),
                           ("HardLengthPrior", {"min_length": 1,
                                               "max_length": 5, }),
                           ("PhysicalUnitsPrior", {"prob_eps": np.finfo(np.float32).eps})]

        # --- FREE CONST OPTI ---
        free_const_opti_args = {
            'loss'   : "MSE",
            'method' : 'LBFGS',
            'method_args': {
                        'n_steps' : 30,
                        'tol'     : 1e-6,
                        'lbfgs_func_args' : {
                            'max_iter'       : 4,
                            'line_search_fn' : "strong_wolfe",
                                             },
                            },
        }

        # --- BATCH ---
        batch_size    = 1000
        max_time_step = 10

        my_batch = batch.Batch(library_args     = library_args,
                               priors_config    = priors_config,
                               batch_size       = batch_size,
                               max_time_step    = max_time_step,
                               rewards_computer = reward.make_RewardsComputer (reward_function     = reward.SquashedNRMSE,
                                                                               zero_out_unphysical = True),
                               free_const_opti_args = free_const_opti_args,
                               X        = X,
                               y_target = y_target,
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
        eps = 1e-5
        n_solutions = (np.abs(1-R) < eps).sum()
        print("Dummy epoch time (w free const) = %f ms (found %i/%i candidates with R = 1 +/- %f)"%((t1-t0)*1e3, n_solutions, batch_size, eps))

    def test_dummy_epoch_free_consts_and_duplicate_elimination(self):

        # ------- TEST CASE -------
        DEVICE = 'cpu'
        if torch.cuda.is_available():
            DEVICE = 'cuda'

        # --- DATA ---
        N = int(1e3)
        x_array = np.linspace(0.04, 4, N)
        x = data_conversion (x_array).to(DEVICE)
        X = torch.stack((x,), axis=0)
        pi = data_conversion (np.pi).to(DEVICE)
        const1 = data_conversion (1.).to(DEVICE)
        T = 1.028
        v0 = 0.995
        y_target = data_conversion(x_array/T + v0).to(DEVICE)


        # --- LIBRARY CONFIG ---
        args_make_tokens = {
                        # operations
                        "op_names"             : ["add", "div"],
                        "use_protected_ops"    : True,
                        # input variables
                        "input_var_ids"        : {"x" : 0         },
                        "input_var_units"      : {"x" : [1, 0, 0] },
                        "input_var_complexity" : {"x" : 0.        },
                        # constants
                        "constants"            : {"pi" : pi        , "const1" : const1    },
                        "constants_units"      : {"pi" : [0, 0, 0] , "const1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 0.        , "const1" : 1.        },
                        # free constants
                        "free_constants"            : {"T"              , "v0"              ,},
                        "free_constants_init_val"   : {"T" : 1.         , "v0" : 1.         ,},
                        "free_constants_units"      : {"T" : [0, 1, 0] , "v0" : [1, -1, 0] ,},
                        "free_constants_complexity" : {"T" : 0.         , "v0" : 0.         ,},
                            }
        library_args = {"args_make_tokens"  : args_make_tokens,
                        "superparent_units" : [1, -1, 0],
                        "superparent_name"  : "v",
                        }

        # --- PRIORS ---
        priors_config  = [ ("UniformArityPrior", None),
                           ("HardLengthPrior", {"min_length": 1,
                                               "max_length": 5, }),
                           ("PhysicalUnitsPrior", {"prob_eps": np.finfo(np.float32).eps})]

        # --- FREE CONST OPTI ---
        free_const_opti_args = {
            'loss'   : "MSE",
            'method' : 'LBFGS',
            'method_args': {
                        'n_steps' : 30,
                        'tol'     : 1e-6,
                        'lbfgs_func_args' : {
                            'max_iter'       : 4,
                            'line_search_fn' : "strong_wolfe",
                                             },
                            },
        }

        # --- BATCH ---
        batch_size    = 1000
        max_time_step = 10

        my_batch = batch.Batch(library_args     = library_args,
                               priors_config    = priors_config,
                               batch_size       = batch_size,
                               max_time_step    = max_time_step,
                               rewards_computer = reward.make_RewardsComputer (reward_function     = reward.SquashedNRMSE,
                                                                               zero_out_unphysical = True,
                                                                               zero_out_duplicates = True,),
                               free_const_opti_args = free_const_opti_args,
                               X        = X,
                               y_target = y_target,
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
        eps = 1e-5
        n_solutions = (np.abs(1-R) < eps).sum()
        n_kept = (R>0).sum()
        print("Dummy epoch time (w free const and duplicate elimination) = %f ms (found %i/%i candidates with R > 0 "
              "and %i/%i with R = 1 +/- %f)"%((t1-t0)*1e3, n_kept, batch_size, n_solutions, batch_size, eps))

    def test_dummy_epoch_free_consts_and_duplicate_elimination_and_lowest_complexity(self):

        # ------- TEST CASE -------
        DEVICE = 'cpu'
        if torch.cuda.is_available():
            DEVICE = 'cuda'

        # --- DATA ---
        N = int(1e3)
        x_array = np.linspace(0.04, 4, N)
        x = data_conversion (x_array).to(DEVICE)
        X = torch.stack((x,), axis=0)
        pi = data_conversion (np.pi).to(DEVICE)
        const1 = data_conversion (1.).to(DEVICE)
        T = 1.028
        v0 = 0.995
        y_target = data_conversion(x_array/T + v0).to(DEVICE)


        # --- LIBRARY CONFIG ---
        args_make_tokens = {
                        # operations
                        "op_names"             : ["add", "div"],
                        "use_protected_ops"    : True,
                        # input variables
                        "input_var_ids"        : {"x" : 0         },
                        "input_var_units"      : {"x" : [1, 0, 0] },
                        "input_var_complexity" : {"x" : 0.        },
                        # constants
                        "constants"            : {"pi" : pi        , "const1" : const1    },
                        "constants_units"      : {"pi" : [0, 0, 0] , "const1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 0.        , "const1" : 1.        },
                        # free constants
                        "free_constants"            : {"T"              , "v0"              ,},
                        "free_constants_init_val"   : {"T" : 1.         , "v0" : 1.         ,},
                        "free_constants_units"      : {"T" : [0, 1, 0] , "v0" : [1, -1, 0] ,},
                        "free_constants_complexity" : {"T" : 0.         , "v0" : 0.         ,},
                            }
        library_args = {"args_make_tokens"  : args_make_tokens,
                        "superparent_units" : [1, -1, 0],
                        "superparent_name"  : "v",
                        }

        # --- PRIORS ---
        priors_config  = [ ("UniformArityPrior", None),
                           ("HardLengthPrior", {"min_length": 1,
                                               "max_length": 5, }),
                           ("PhysicalUnitsPrior", {"prob_eps": np.finfo(np.float32).eps})]

        # --- FREE CONST OPTI ---
        free_const_opti_args = {
            'loss'   : "MSE",
            'method' : 'LBFGS',
            'method_args': {
                        'n_steps' : 30,
                        'tol'     : 1e-6,
                        'lbfgs_func_args' : {
                            'max_iter'       : 4,
                            'line_search_fn' : "strong_wolfe",
                                             },
                            },
        }

        # --- BATCH ---
        batch_size    = 1000
        max_time_step = 10

        my_batch = batch.Batch(library_args     = library_args,
                               priors_config    = priors_config,
                               batch_size       = batch_size,
                               max_time_step    = max_time_step,
                               rewards_computer = reward.make_RewardsComputer (reward_function     = reward.SquashedNRMSE,
                                                                               zero_out_unphysical = True,
                                                                               zero_out_duplicates = True,
                                                                               keep_lowest_complexity_duplicate = True),
                               free_const_opti_args = free_const_opti_args,
                               X        = X,
                               y_target = y_target,
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
        eps = 1e-5
        n_solutions = (np.abs(1-R) < eps).sum()
        n_kept = (R>0).sum()
        print("Dummy epoch time (w free const and duplicate elimination, keeping lowest complexity) = %f ms (found "
              "%i/%i candidates with R > 0 and %i/%i with R = 1 +/- %f)"%(
                (t1-t0)*1e3, n_kept, batch_size, n_solutions, batch_size, eps))

if __name__ == '__main__':
    unittest.main(verbosity=2)
