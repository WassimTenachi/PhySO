import unittest
import  torch
import numpy as np

import physo
# Internal imports
from physo.physym.functions import data_conversion
import physo.task.checks as checks

class Test_Checks(unittest.TestCase):
    def test_sanity_check_ClassSR(self):

        # Seed
        seed = 0
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

        library_config = {"args_make_tokens"  : args_make_tokens,
                                "superparent_units" : [1, -1, 0],
                                "superparent_name"  : "v",
                                }
        run_config = physo.task.class_sr.default_config
        run_config.update({
            "library_config": library_config,
        })

        target_prog_str = ["add", "mul", "mul", "k0"  , "exp", "mul", "neg", "k1", "t", "cos", "add", "mul", "c0", "t", "k2", "mul", "c1", "l", ]

        checks.sanity_check_ClassSR (multi_X         = multi_X,
                                     multi_y         = multi_y_target,
                                     multi_y_weights = multi_y_weights,
                                     run_config      = run_config,
                                     target_program_str = target_prog_str,
                                     expected_ideal_reward=0.999,
                                     )



if __name__ == '__main__':
    unittest.main()
