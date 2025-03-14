import unittest
import numpy as np
import torch

from physo.physym import dataset
from physo.physym import library as Lib
from physo.physym.functions import data_conversion, data_conversion_inv


class TestDataset(unittest.TestCase):

    def test_Dataset_assertions(self):

        DEVICE = 'cpu'
        if torch.cuda.is_available():
            DEVICE = 'cuda'

        # DATA
        N = int(1e6)
        x = data_conversion  (np.linspace(0.04, 4, N)  ).to(DEVICE)
        v = data_conversion  (np.linspace(0.10, 10, N) ).to(DEVICE)
        t = data_conversion  (np.linspace(0.06, 6, N)  ).to(DEVICE)
        M  = data_conversion (1e6).to(DEVICE)
        c  = data_conversion (3e8).to(DEVICE)
        pi = data_conversion (np.pi).to(DEVICE)
        const1 = data_conversion (1.).to(DEVICE)


        X = torch.stack((x, v, t), axis=0)
        y_target = data_conversion  (np.linspace(0.01, 6, N)  ).to(DEVICE)
        y_weights = data_conversion (np.random.rand(N)  ).to(DEVICE)

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
                        "constants"            : {"pi" : pi        , "c" : c         , "M" : M         , "const1" : const1    },
                        "constants_units"      : {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] , "const1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        , "const1" : 1.        },
                            }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [1, -2, 1], superparent_name = "y")

        def make_dataset_for_regular_SR(library, X, y, y_weights=1.):
            my_dataset = dataset.Dataset(multi_X=[X, ], multi_y=[y, ], multi_y_weights=[y_weights, ], library=library)
            return my_dataset

        # ------- TEST CREATION -------
        try:
            my_dataset = make_dataset_for_regular_SR(library=my_lib, X=X, y=y_target)
        except:
            self.fail("Dataset creation failed.")

        # ------- ASSERTIONS : FLOAT TYPE -------
        with self.assertRaises(AssertionError):
            my_dataset = make_dataset_for_regular_SR(library = my_lib, X = torch.ones((3, 100), dtype=int), y = torch.ones((100,)))
        with self.assertRaises(AssertionError):
            my_dataset = make_dataset_for_regular_SR(library = my_lib, X = torch.ones((3, 100)), y = torch.ones((100,), dtype=int))

        # ------- ASSERTIONS : SHAPE -------
        with self.assertRaises(AssertionError):
            my_dataset = make_dataset_for_regular_SR(library = my_lib, X = torch.ones((3, 100),), y = torch.ones((200,)))
        with self.assertRaises(AssertionError):
            my_dataset = make_dataset_for_regular_SR(library = my_lib, X = torch.ones((100, 3),), y = torch.ones((100,)))

        # ------- ASSERTIONS : VARIABLE ID -------
        with self.assertRaises(AssertionError):
            my_dataset = make_dataset_for_regular_SR(library=my_lib, X=torch.ones((1, 100), ), y=torch.ones((100,)))

        # ------- ASSERTIONS : ONE REALIZATION -------
        my_dataset = make_dataset_for_regular_SR(library=my_lib, X=X, y=y_target, y_weights=y_weights)
        self.assertTrue(my_dataset.n_realizations == 1)

        self.assertTrue((my_dataset.multi_X_flatten == X).all())
        self.assertTrue((my_dataset.multi_y_flatten == y_target).all())
        self.assertTrue((my_dataset.multi_y_weights_flatten == y_weights).all())

        return None

    def test_Dataset_assertions_multi_real(self):

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
        #y_weights_per_dataset = np.array([1., 1., 1.]*10)
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

        n_realizations = len(multi_X)

        # ------- TEST CREATION -------
        try:
            my_dataset = dataset.Dataset(multi_X=multi_X, multi_y=multi_y_target, library=my_lib)
        except:
            self.fail("Dataset creation failed.")

        # ------- TESTS -------

        # Wrong number of realizations between X and y_target
        with self.assertRaises(AssertionError):
            my_dataset = dataset.Dataset(multi_X=multi_X, multi_y=multi_y_target[:-1], library=my_lib)
        with self.assertRaises(AssertionError):
            my_dataset = dataset.Dataset(multi_X=multi_X[:-1], multi_y=multi_y_target, library=my_lib)

        # Sending data for one realization only / sending tensor type
        with self.assertRaises(AssertionError):
            my_dataset = dataset.Dataset(multi_X=multi_X[0], multi_y=multi_y_target[0], library=my_lib)

        # Test number of realizations
        my_dataset = dataset.Dataset(multi_X=multi_X, multi_y=multi_y_target, library=my_lib)
        self.assertEqual(my_dataset.n_realizations, n_realizations)

        # Test conversion to torch, when already torch tensors
        my_dataset = dataset.Dataset(multi_X=multi_X, multi_y=multi_y_target, library=my_lib)
        for i in range (n_realizations):
            self.assertTrue(torch.is_tensor(my_dataset.multi_X[i]))
            self.assertTrue(torch.is_tensor(my_dataset.multi_y[i]))

        # Test conversion to torch, when numpy arrays
        my_dataset = dataset.Dataset(multi_X=[X.cpu().numpy() for X in multi_X],
                                     multi_y=[y.cpu().numpy() for y in multi_y_target], library=my_lib)
        for i in range (n_realizations):
            self.assertTrue(torch.is_tensor(my_dataset.multi_X[i]))
            self.assertTrue(torch.is_tensor(my_dataset.multi_y[i]))

        # Wrong type
        with self.assertRaises(AssertionError):
            my_dataset = dataset.Dataset(multi_X=[X.cpu().numpy().astype(int) for X in multi_X], multi_y=multi_y_target,
                                         library=my_lib)
        with self.assertRaises(AssertionError):
            my_dataset = dataset.Dataset(multi_X=multi_X, multi_y=[y.cpu().numpy().astype(int) for y in multi_y_target],
                                         library=my_lib)
        # Containing NaNs
        wrong_multi_X = [X.cpu().numpy().copy() for X in multi_X]
        wrong_multi_X [0][0, 0] = float(np.nan)
        wrong_multi_y = [y.cpu().numpy().copy() for y in multi_y_target]
        wrong_multi_y [0][0] = float(np.nan)
        with self.assertRaises(AssertionError):
            my_dataset = dataset.Dataset(multi_X=wrong_multi_X, multi_y=multi_y_target, library=my_lib)
        with self.assertRaises(AssertionError):
            my_dataset = dataset.Dataset(multi_X=multi_X, multi_y=wrong_multi_y, library=my_lib)

        # Containing inconsistent n_dim
        wrong_multi_X = [X.cpu().numpy().copy() for X in multi_X]
        wrong_multi_X [0] = wrong_multi_X[0][:-1,:] # removing one dim in realization 0
        with self.assertRaises(AssertionError):
            my_dataset = dataset.Dataset(multi_X=wrong_multi_X, multi_y=multi_y_target, library=my_lib)

        # Containing too low dimension given library
        wrong_multi_X = [X.cpu().numpy().copy() for X in multi_X]
        wrong_multi_X = [np.stack([wrong_multi_X[i][0,:]]*1) for i in range(n_realizations)] # 1D per realization
        with self.assertRaises(AssertionError):
            my_dataset = dataset.Dataset(multi_X=wrong_multi_X, multi_y=multi_y_target, library=my_lib)

        # ------ Test weights as one single float ------
        # Creating dataset
        try:
            my_dataset = dataset.Dataset(multi_X=multi_X, multi_y=multi_y_target, multi_y_weights=2.0, library=my_lib)
        except:
            self.fail("Dataset creation failed.")
        # Tensor type and content
        for i, y_weights in enumerate(my_dataset.multi_y_weights):
            self.assertTrue(torch.is_tensor(y_weights))
            expected = torch.full_like(multi_y_target[i], fill_value=2.0)
            self.assertTrue((y_weights == expected).all())
        # NAN assertion
        with self.assertRaises(AssertionError):
            my_dataset = dataset.Dataset(multi_X=multi_X, multi_y=multi_y_target, multi_y_weights=np.nan,
                                         library=my_lib)
        # Wrong type -> Converts to float in this case
        my_dataset = dataset.Dataset(multi_X=multi_X, multi_y=multi_y_target, multi_y_weights=int(2), library=my_lib)
        for i, y_weights in enumerate(my_dataset.multi_y_weights):
            self.assertTrue(torch.is_tensor(y_weights))
            expected = torch.full_like(multi_y_target[i], fill_value=2.0)
            self.assertTrue((y_weights == expected).all())

        # ------ Test weights as (n_realizations,) of floats ------
        # Creating dataset
        try:
            my_dataset = dataset.Dataset(multi_X=multi_X, multi_y=multi_y_target, multi_y_weights=y_weights_per_dataset,
                                         library=my_lib)
        except:
            self.fail("Dataset creation failed.")
        # Tensor type and content
        for i, y_weights in enumerate(my_dataset.multi_y_weights):
            self.assertTrue(torch.is_tensor(y_weights))
            expected = torch.full_like(multi_y_target[i], fill_value=y_weights_per_dataset[i])
            self.assertTrue((y_weights == expected).all())
        self.assertTrue(torch.is_tensor(my_dataset.multi_y_weights_flatten))
        self.assertTrue((my_dataset.multi_y_weights_flatten == y_weights_flatten ).all())
        # NAN assertion
        with self.assertRaises(AssertionError):
            wrong_y_weights_per_dataset = y_weights_per_dataset.copy()
            wrong_y_weights_per_dataset[0] = np.nan
            my_dataset = dataset.Dataset(multi_X=multi_X, multi_y=multi_y_target,
                                         multi_y_weights=wrong_y_weights_per_dataset, library=my_lib)
        # Wrong (n_realizations,) length
        with self.assertRaises(AssertionError):
            wrong_y_weights_per_dataset = y_weights_per_dataset.copy()[:-1]
            my_dataset = dataset.Dataset(multi_X=multi_X, multi_y=multi_y_target,
                                         multi_y_weights=wrong_y_weights_per_dataset, library=my_lib)
        # Wrong type -> Converts to float in this case
        my_dataset = dataset.Dataset(multi_X=multi_X, multi_y=multi_y_target,
                                     multi_y_weights=y_weights_per_dataset.astype(int), library=my_lib)
        for i, y_weights in enumerate(my_dataset.multi_y_weights):
            self.assertTrue(torch.is_tensor(y_weights))
            expected = torch.full_like(multi_y_target[i], fill_value=float(int(y_weights_per_dataset[i])))
            self.assertTrue((y_weights == expected).all())

        # ------ Test weights as (n_realizations,) of ([n_samples depends on dataset]) ------
        # Creating dataset
        try:
            my_dataset = dataset.Dataset(multi_X=multi_X, multi_y=multi_y_target, multi_y_weights=multi_y_weights,
                                         library=my_lib)
        except:
            self.fail("Dataset creation failed.")
        # Tensor type and content
        for i, y_weights in enumerate(my_dataset.multi_y_weights):
            self.assertTrue(torch.is_tensor(y_weights))
            expected = multi_y_weights[i]
            self.assertTrue((y_weights == expected).all())
        self.assertTrue(torch.is_tensor(my_dataset.multi_y_weights_flatten))
        self.assertTrue((my_dataset.multi_y_weights_flatten == y_weights_flatten ).all())
        # NAN assertion
        with self.assertRaises(AssertionError):
            wrong_multi_y_weights = [y.cpu().numpy().copy() for y in multi_y_weights]
            wrong_multi_y_weights[0][0] = float(np.nan)
            my_dataset = dataset.Dataset(multi_X=multi_X, multi_y=multi_y_target, multi_y_weights=wrong_multi_y_weights,
                                         library=my_lib)
        # Wrong (n_realizations,) length
        with self.assertRaises(AssertionError):
            wrong_multi_y_weights = [y.cpu().numpy().copy() for y in multi_y_weights]
            wrong_multi_y_weights = wrong_multi_y_weights[:-1]
            my_dataset = dataset.Dataset(multi_X=multi_X, multi_y=multi_y_target, multi_y_weights=wrong_multi_y_weights,
                                         library=my_lib)
        # Inconsistent n_samples
        with self.assertRaises(AssertionError):
            wrong_multi_y_weights = [y.cpu().numpy().copy() for y in multi_y_weights]
            wrong_multi_y_weights[0] = wrong_multi_y_weights[0][:-1]
            my_dataset = dataset.Dataset(multi_X=multi_X, multi_y=multi_y_target, multi_y_weights=wrong_multi_y_weights,
                                         library=my_lib)
        # Conversion to torch
        my_dataset = dataset.Dataset(multi_X=multi_X, multi_y=multi_y_target,
                                     multi_y_weights=[y.cpu().numpy() for y in multi_y_weights], library=my_lib)
        for i, y_weights in enumerate(my_dataset.multi_y_weights):
            self.assertTrue(torch.is_tensor(y_weights))
            expected = multi_y_weights[i]
            self.assertTrue((y_weights == expected).all())
        # Wrong type
        with self.assertRaises(AssertionError):
            my_dataset = dataset.Dataset(multi_X=multi_X, multi_y=multi_y_target,
                                         multi_y_weights=[y.cpu().numpy().astype(int) for y in multi_y_weights],
                                         library=my_lib)

        # ----- Flattened values -----
        my_dataset = dataset.Dataset(multi_X=multi_X, multi_y=multi_y_target, multi_y_weights=multi_y_weights,
                                     library=my_lib)
        self.assertTrue(torch.is_tensor(my_dataset.multi_X_flatten))
        self.assertTrue(torch.is_tensor(my_dataset.multi_y_flatten))
        self.assertTrue(torch.is_tensor(my_dataset.multi_y_weights_flatten))

        self.assertTrue((my_dataset.multi_X_flatten         == multi_X_flatten   ).all())
        self.assertTrue((my_dataset.multi_y_flatten         == y_ideals_flatten  ).all())
        self.assertTrue((my_dataset.multi_y_weights_flatten == y_weights_flatten ).all())


        return None



if __name__ == '__main__':
    unittest.main(verbosity=2)
