import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings

# Internal code import
import physo
import physo.learn.random_sampler as rs

import unittest

class RandomSamplerTest(unittest.TestCase):
    def test_sample_random_expressions_default(self):
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            eqs = rs.sample_random_expressions(
                    # X
                    X_names = ["f", "t"],
                    X_units = [[0,-1], [0,1]],
                    # y
                    y_name  = "y",
                    y_units = [0,0],
                    # Fixed constants
                    fixed_consts       = [1.],
                    fixed_consts_units = [[0,0]],
                    # Class free constants
                    class_free_consts_names    = ["c0", "c1",],
                    class_free_consts_units    = [[0,0], [0,0]],
                    class_free_consts_init_val = [1., 1.],
                    # Spe Free constants
                    spe_free_consts_names    = ["k0", "k1", "k2"],
                    spe_free_consts_units    = [[0,0], [0,0], [0,0]],
                    spe_free_consts_init_val = [1., 1., 1.],
                    # Operations to use
                    op_names = ["add", "sub", "mul", "div", "pow", "log", "exp", "cos"],
                    # Device to use
                    device="cpu",
                    # verbose
                    verbose=False,
                )
        except Exception as e:
            self.fail("Expression generation failed.")

        return None

    def test_sample_random_expressions_custom_args(self):
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            eqs = rs.sample_random_expressions(
                    # Batch size
                    batch_size=1000,
                    # Max length
                    max_length=30,

                    # Soft length prior
                    soft_length_loc = 12.,
                    soft_length_scale = 5.,
                    # X
                    X_names = ["x1", "x2"],
                    X_units = [[0,],[0,]],
                    # y
                    y_name = "y",
                    y_units = [0,],
                    # Fixed constants
                    fixed_consts       = [1.],
                    fixed_consts_units = [[0,]],
                    # Class free constants
                    class_free_consts_names    = ["c0", "c1"],
                    class_free_consts_units    = [[0,], [0,]],
                    # Operations to use
                    op_names          = ["add", "sub", "mul", "div", "pow", "log", "exp", "cos"],
                    # Priors configuration
                    priors_config = None,
                    # Device to use
                    device="cpu",
                    # verbose
                    verbose=False
                )
        except Exception as e:
            self.fail("Expression generation failed.")

    def test_sample_random_expressions_custom_configs(self):
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        MAX_LENGTH = 35
        priors_config = [
                #("UniformArityPrior", None),
                # LENGTH RELATED
                ("HardLengthPrior"  , {"min_length": 4, "max_length": MAX_LENGTH, }),
                ("SoftLengthPrior"  , {"length_loc": 8, "scale": 5, }),
                # RELATIONSHIPS RELATED
                ("NoUselessInversePrior"  , None),
                ("PhysicalUnitsPrior", {"prob_eps": np.finfo(np.float32).eps}), # PHYSICALITY
                ("NestedFunctions", {"functions":["exp",], "max_nesting" : 1}),
                ("NestedFunctions", {"functions":["log",], "max_nesting" : 1}),
                ("NestedTrigonometryPrior", {"max_nesting" : 1}),
                #("OccurrencesPrior", {"targets" : ["1",], "max" : [3,] }),
                 ]
        try:
            eqs = rs.sample_random_expressions(
                     # Batch size
                    batch_size=1000,
                    # Max length
                    max_length=None,
                    # Soft length prior
                    soft_length_loc = None,
                    soft_length_scale = 5.,

                    # X
                    X_names = ["x1", "x2"],
                    X_units = [[0,], [0,]],
                    # y
                    y_name = "y",
                    y_units = [0,],
                    # Fixed constants
                    fixed_consts       = [1.],
                    fixed_consts_units = [[0,]],
                    # Class free constants
                    class_free_consts_names    = ["c0", "c1"],
                    class_free_consts_units    = [[0,], [0,]],
                    class_free_consts_init_val = None,
                    # Spe Free constants
                    spe_free_consts_names    = None,
                    spe_free_consts_units    = None,
                    spe_free_consts_init_val = None,
                    # Operations to use
                    op_names          = ["add", "sub", "mul", "div", "pow", "log", "exp", "cos"],
                    use_protected_ops = True,

                    # Priors configuration
                    priors_config = priors_config,

                    # Number of realizations
                    n_realizations = 1,
                    # Device to use
                    device="cpu",

                    # verbose
                    verbose=False
                )
        except Exception as e:
            self.fail("Expression generation failed.")

    def test_sample_random_expressions_custom_configs_overrides(self):
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        MAX_LENGTH = 35
        priors_config = [
                #("UniformArityPrior", None),
                # LENGTH RELATED
                ("HardLengthPrior"  , {"min_length": 4, "max_length": MAX_LENGTH, }),
                ("SoftLengthPrior"  , {"length_loc": 8, "scale": 5, }),
                # RELATIONSHIPS RELATED
                ("NoUselessInversePrior"  , None),
                ("PhysicalUnitsPrior", {"prob_eps": np.finfo(np.float32).eps}), # PHYSICALITY
                ("NestedFunctions", {"functions":["exp",], "max_nesting" : 1}),
                ("NestedFunctions", {"functions":["log",], "max_nesting" : 1}),
                ("NestedTrigonometryPrior", {"max_nesting" : 1}),
                #("OccurrencesPrior", {"targets" : ["1",], "max" : [3,] }),
                 ]
        try:
            eqs = rs.sample_random_expressions(
                     # Batch size
                    batch_size=1000,
                    # Max length
                    max_length=45,
                    # Soft length prior
                    soft_length_loc = 60,
                    soft_length_scale = 5.,

                    # X
                    X_names = ["x1", "x2"],
                    X_units = [[0,], [0,]],
                    # y
                    y_name = "y",
                    y_units = [0,],
                    # Fixed constants
                    fixed_consts       = [1.],
                    fixed_consts_units = [[0,]],
                    # Class free constants
                    class_free_consts_names    = ["c0", "c1"],
                    class_free_consts_units    = [[0,], [0,]],
                    class_free_consts_init_val = None,
                    # Spe Free constants
                    spe_free_consts_names    = None,
                    spe_free_consts_units    = None,
                    spe_free_consts_init_val = None,
                    # Operations to use
                    op_names          = ["add", "sub", "mul", "div", "pow", "log", "exp", "cos"],
                    use_protected_ops = True,

                    # Priors configuration
                    priors_config = priors_config,

                    # Number of realizations
                    n_realizations = 1,
                    # Device to use
                    device="cpu",

                    # verbose
                    verbose=False
                )
        except Exception as e:
            self.fail("Expression generation failed.")

        return None

