import time
import unittest
import numpy as np
import torch

# Internal imports
from physo.physym import batch
from physo.physym.functions import data_conversion
from physo.physym import reward

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

        multi_X = [X,]
        multi_y = [y,]

        my_batch = batch.Batch(library_args     = library_args,
                               priors_config    = priors_config,
                               batch_size       = batch_size,
                               max_time_step    = max_time_step,
                               rewards_computer = reward.make_RewardsComputer (reward_function = reward.SquashedNRMSE),
                               multi_X = multi_X,
                               multi_y = multi_y,
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

        multi_X = [X,]
        multi_y = [y_target,]

        my_batch = batch.Batch(library_args     = library_args,
                               priors_config    = priors_config,
                               batch_size       = batch_size,
                               max_time_step    = max_time_step,
                               rewards_computer = reward.make_RewardsComputer (reward_function = reward.SquashedNRMSE),
                               multi_X = multi_X,
                               multi_y = multi_y,
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
            actions      = actions.cpu().numpy()
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

        multi_X = [X,]
        multi_y = [y_target,]

        my_batch = batch.Batch(library_args     = library_args,
                               priors_config    = priors_config,
                               batch_size       = batch_size,
                               max_time_step    = max_time_step,
                               rewards_computer = reward.make_RewardsComputer (reward_function = reward.SquashedNRMSE,
                                                                               zero_out_unphysical = True,
                                                                               zero_out_duplicates = True),
                               multi_X = multi_X,
                               multi_y = multi_y,
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
            actions      = actions.cpu().numpy()
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

        multi_X = [X,]
        multi_y = [y_target,]

        my_batch = batch.Batch(library_args     = library_args,
                               priors_config    = priors_config,
                               batch_size       = batch_size,
                               max_time_step    = max_time_step,
                               rewards_computer = reward.make_RewardsComputer (reward_function     = reward.SquashedNRMSE,
                                                                               zero_out_unphysical = True),
                               free_const_opti_args = free_const_opti_args,
                               multi_X = multi_X,
                               multi_y = multi_y,
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
            actions      = actions.cpu().numpy()
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

    def test_dummy_epoch_spe_free_consts_mdho2d(self):

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

        #y_weights_per_dataset = np.array([0, 0.001, 1.0]*10) # Shows weights work
        y_weights_per_dataset = np.array([1., 1., 1.]*10)
        multi_y_weights = [np.full(shape=(n_samples_per_dataset[i],), fill_value=y_weights_per_dataset[i]) for i in range (n_realizations)]
        multi_y_weights = [torch.tensor(y_weights).to(DEVICE) for y_weights in multi_y_weights]
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
        multi_y_target   = unflatten_multi_data(y_ideals_flatten)                                         # (n_realizations,) of (n_samples depends on dataset,)

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
                        "input_var_units"      : {"t" : [0, 1, 0] , "l" : [1, 0, 0]  },
                        "input_var_complexity" : {"t" : 0.        , "l" : 1.         },
                        # constants
                        "constants"            : {"pi" : pi        , "const1" : const1    },
                        "constants_units"      : {"pi" : [0, 0, 0] , "const1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 1.        , "const1" : 1.        },
                        # free constants
                        "class_free_constants"            : {"c0"              , "c1"               },
                        "class_free_constants_init_val"   : {"c0" : 1.         , "c1"  : 1.         },
                        "class_free_constants_units"      : {"c0" : [0, -1, 0] , "c1"  : [0, -1, 0] },
                        "class_free_constants_complexity" : {"c0" : 1.         , "c1"  : 1.         },
                        # free constants
                        "spe_free_constants"            : {"k0"              , "k1"               , "k2"               },
                        "spe_free_constants_init_val"   : {"k0" : k0_init    , "k1"  : 1.         , "k2"  : 1.         },
                        "spe_free_constants_units"      : {"k0" : [1, -1, 0] , "k1"  : [0, -1, 0] , "k2"  : [0, 0, 0]  },
                        "spe_free_constants_complexity" : {"k0" : 1.         , "k1"  : 1.         , "k2"  : 1.         },
                           }

        library_args = {"args_make_tokens"  : args_make_tokens,
                                "superparent_units" : [1, -1, 0],
                                "superparent_name"  : "v",
                                }

        # --- PRIORS ---
        priors_config  = [ ("UniformArityPrior", None),
                           ("HardLengthPrior", {"min_length": 1,
                                                "max_length": 20, }),
                           ("PhysicalUnitsPrior", {"prob_eps": np.finfo(np.float32).eps})]

        # --- FREE CONST OPTI ---
        free_const_opti_args = {
            'loss'   : "MSE",
            'method' : 'LBFGS',
            'method_args': {
                        'n_steps' : 50,
                        'tol'     : 1e-6,
                        'lbfgs_func_args' : {
                            'max_iter'       : 4,
                            'line_search_fn' : "strong_wolfe",
                                             },
                            },
        }


        # --- BATCH ---
        batch_size    = 100
        max_time_step = 30

        my_batch = batch.Batch(library_args     = library_args,
                               priors_config    = priors_config,
                               batch_size       = batch_size,
                               max_time_step    = max_time_step,
                               rewards_computer = reward.make_RewardsComputer (reward_function     = reward.SquashedNRMSE,
                                                                               zero_out_unphysical = True,
                                                                               ),
                               free_const_opti_args = free_const_opti_args,
                               multi_X         = multi_X,
                               multi_y         = multi_y_target,
                               multi_y_weights = multi_y_weights,
                               )

        target_prog_str = ["add", "mul", "mul", "k0", "exp", "mul", "neg", "k1", "t", "cos", "add", "mul", "c0", "t",
                           "k2", "mul", "c1", "l", ]
        target_prog_idx = [my_batch.library.lib_name_to_idx[name] for name in target_prog_str]
        target_prog_idx += [0]*(max_time_step-len(target_prog_idx)) # Padding with zeros to max_time_step
        target_prog_idx = np.array(target_prog_idx)

        # --- DUMMY EPOCH ---
        t0 = time.perf_counter()
        for step in range(max_time_step):
            # Embedding output
            prior        = torch.tensor(my_batch.prior().astype(np.float32),   requires_grad=False)                          # (batch_size, n_choices)
            observations = torch.tensor(my_batch.get_obs().astype(np.float32), requires_grad=False)                          # (batch_size, 3*n_choices+1)
            # Dummy model output
            probs        = torch.tensor(np.random.rand(my_batch.batch_size, my_batch.library.n_choices).astype(np.float32))  # (batch_size, n_choices,)
            # Cheating by setting the target program in the first batch element of probs
            probs[0]                        = 0.  # Zeroing out all probs
            probs[0, target_prog_idx[step]] = 1.  # Setting the target program to 1
            # Actions
            actions      = torch.multinomial(probs * prior, num_samples=1)[:, 0]
            actions      = actions.cpu().numpy()
            my_batch.programs.append(actions)
        # Embedding output
        lengths = my_batch.programs.n_lengths
        R       = my_batch.get_rewards()
        # Computing model loss using (probs,actions, R, lengths)
        # ...
        t1 = time.perf_counter()
        eps = 1e-3
        n_solutions = (np.abs(1-R) < eps).sum()
        print("Dummy epoch time (mdho2d scenario) = %f ms (found %i/%i candidates with R = 1 +/- %f)"%((t1-t0)*1e3, n_solutions, batch_size, eps))
        return None

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

        multi_X = [X,]
        multi_y = [y_target,]

        my_batch = batch.Batch(library_args     = library_args,
                               priors_config    = priors_config,
                               batch_size       = batch_size,
                               max_time_step    = max_time_step,
                               rewards_computer = reward.make_RewardsComputer (reward_function     = reward.SquashedNRMSE,
                                                                               zero_out_unphysical = True,
                                                                               zero_out_duplicates = True,),
                               free_const_opti_args = free_const_opti_args,
                               multi_X = multi_X,
                               multi_y = multi_y,
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
            actions      = actions.cpu().numpy()
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

        multi_X = [X,]
        multi_y = [y_target,]

        my_batch = batch.Batch(library_args     = library_args,
                               priors_config    = priors_config,
                               batch_size       = batch_size,
                               max_time_step    = max_time_step,
                               rewards_computer = reward.make_RewardsComputer (reward_function     = reward.SquashedNRMSE,
                                                                               zero_out_unphysical = True,
                                                                               zero_out_duplicates = True,
                                                                               keep_lowest_complexity_duplicate = True),
                               free_const_opti_args = free_const_opti_args,
                               multi_X = multi_X,
                               multi_y = multi_y,
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
            actions      = actions.cpu().numpy()
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
