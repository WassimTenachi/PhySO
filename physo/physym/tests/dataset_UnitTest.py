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

        # ------- TEST CREATION -------
        try:
            my_dataset = dataset.Dataset(library = my_lib, X = X, y_target = y_target)
        except:
            self.fail("Dataset creation failed.")

        # ------- ASSERTIONS : TENSOR TYPE -------
        with self.assertRaises(AssertionError):
            my_dataset = dataset.Dataset(library = my_lib, X = np.ones((3, 100)), y_target = torch.ones((100,)))
        with self.assertRaises(AssertionError):
            my_dataset = dataset.Dataset(library = my_lib, X = torch.ones((3, 100)), y_target = np.ones((100,)))

        # ------- ASSERTIONS : FLOAT TYPE -------
        with self.assertRaises(AssertionError):
            my_dataset = dataset.Dataset(library = my_lib, X = torch.ones((3, 100), dtype=int), y_target = torch.ones((100,)))
        with self.assertRaises(AssertionError):
            my_dataset = dataset.Dataset(library = my_lib, X = torch.ones((3, 100)), y_target = torch.ones((100,), dtype=int))

        # ------- ASSERTIONS : SHAPE -------
        with self.assertRaises(AssertionError):
            my_dataset = dataset.Dataset(library = my_lib, X = torch.ones((3, 100),), y_target = torch.ones((200,)))
        with self.assertRaises(AssertionError):
            my_dataset = dataset.Dataset(library = my_lib, X = torch.ones((100, 3),), y_target = torch.ones((100,)))

        # ------- ASSERTIONS : VARIABLE ID -------
        with self.assertRaises(AssertionError):
            my_dataset = dataset.Dataset(library=my_lib, X=torch.ones((1, 100), ), y_target=torch.ones((100,)))

    def test_Dataset_device_detection (self):

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

        # ------- TEST CREATION -------
        try:
            my_dataset = dataset.Dataset(library = my_lib, X = X, y_target = y_target)
        except:
            self.fail("Dataset creation failed.")

        # TEST
        detected_device = my_dataset.detected_device.type
        works_bool = (detected_device == DEVICE)
        self.assertEqual(works_bool, True)

    def test_MODataset_assertions(self):

        DEVICE = 'cpu'
        if torch.cuda.is_available():
            DEVICE = 'cuda'

        # DATA
        multi_X = []
        multi_y_target = []
        target_function = lambda X,c1,c2,c3 : c1*X[0] + c2*X[1] + c3

        # Object 1
        N = 128
        x0 = data_conversion  (np.linspace(-5, 5, N)  ).to(DEVICE)
        x1 = data_conversion  (np.linspace(-6, 6, N)  ).to(DEVICE)
        X = torch.stack((x0, x1), axis=0)
        y_target = target_function(X, 1, 2, 3)
        multi_X.append(X)
        multi_y_target.append(y_target)

        # Object 2
        N = 256
        x0 = data_conversion  (np.linspace(-7, 7, N)  ).to(DEVICE)
        x1 = data_conversion  (np.linspace(-8, 8, N)  ).to(DEVICE)
        X = torch.stack((x0, x1), axis=0)
        y_target = target_function(X, 4, 5, 6)
        multi_X.append(X)
        multi_y_target.append(y_target)

        # Object 3
        N = 512
        x0 = data_conversion  (np.linspace(-9, 9, N)  ).to(DEVICE)
        x1 = data_conversion  (np.linspace(-10, 10, N)  ).to(DEVICE)
        X = torch.stack((x0, x1), axis=0)
        y_target = target_function(X, 7, 8, 9)
        multi_X.append(X)
        multi_y_target.append(y_target)

        # LIBRARY CONFIG
        pi = data_conversion(np.pi).to(DEVICE)
        const1 = data_conversion(1.).to(DEVICE)
        args_make_tokens = {
                        # operations
                        "op_names"             : "all",  # or ["mul", "neg", "inv", "sin"]
                        "use_protected_ops"    : False,
                        "input_var_ids"        : {"x0" : 0         , "x1" : 1 },
                        "input_var_units"      : {"x0" : [0, 0, 0] , "x1" : [0, 0, 0] },
                        "constants"            : {"pi" : pi        , "const1" : const1    },
                        "constants_units"      : {"pi" : [0, 0, 0] , "const1" : [0, 0, 0] },
                            }

        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [1, -2, 1], superparent_name = "y")

        # ------- TEST CREATION -------
        try:
            my_modataset = dataset.MODataset(library = my_lib, multi_X = multi_X, multi_y_target = multi_y_target)
        except:
            self.fail("MODataset creation failed.")

        # TEST ASSERTIONS
        # Wrong number of objects between X and y_target
        with self.assertRaises(AssertionError):
            my_modataset = dataset.MODataset(library = my_lib, multi_X = multi_X, multi_y_target = multi_y_target[:-1])
        with self.assertRaises(AssertionError):
            my_modataset = dataset.MODataset(library = my_lib, multi_X = multi_X[:-1], multi_y_target = multi_y_target)
        # Sending data for one object only
        with self.assertRaises(AssertionError):
            my_modataset = dataset.MODataset(library = my_lib, multi_X = multi_X[0], multi_y_target = multi_y_target[0])


        return None



if __name__ == '__main__':
    unittest.main(verbosity=2)
