import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
# Internal code import
import physo
import physo.learn.monitoring as monitoring

import unittest

class Test_ClassSR(unittest.TestCase):
    def test_ClassSR_task(self):

        run_logger = lambda : monitoring.RunLogger(
                                      save_path = 'SR.log',
                                      do_save   = False)
        run_visualiser = lambda : monitoring.RunVisualiser (
                                      epoch_refresh_rate = 1,
                                      save_path = 'SR_curves.png',
                                      do_show   = False,
                                      do_prints = True,
                                      do_save   = False, )

        # Seed
        seed = 0
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Dataset
        multi_X = []
        multi_y = []

        # Object 0
        x0 = np.random.uniform(-10, 10, 256)
        x1 = np.random.uniform(-10, 10, 256)
        X = np.stack((x0, x1), axis=0)
        y = 1.123*x0 + 1.123*x1 + 10.123
        multi_X.append(X)
        multi_y.append(y)

        # Object 1
        x0 = np.random.uniform(-11, 11, 500)
        x1 = np.random.uniform(-11, 11, 500)
        X = np.stack((x0, x1), axis=0)
        y = 2*1.123*x0 + 1.123*x1 + 10.123
        multi_X.append(X)
        multi_y.append(y)

        # Object 2
        x0 = np.random.uniform(-12, 12, 256)
        x1 = np.random.uniform(-12, 12, 256)
        X = np.stack((x0, x1), axis=0)
        y = 1.123*x0 + 2*1.123*x1 + 10.123
        multi_X.append(X)
        multi_y.append(y)

        # Object 3
        x0 = np.random.uniform(-13, 13, 256)
        x1 = np.random.uniform(-13, 13, 256)
        X = np.stack((x0, x1), axis=0)
        y = 1.3*1.123*x0 + 1.5*1.123*x1 + 10.123
        multi_X.append(X)
        multi_y.append(y)

        run_config = physo.config.config0b.config0b

        # Removing priors related to tokens that are not used in this example to avoid unnecessary warnings
        priors_name_to_remove = ["NestedFunctions", "NestedTrigonometryPrior"]
        priors_to_keep = []
        for prior_config in run_config["priors_config"]:
            if prior_config[0] not in priors_name_to_remove:
                priors_to_keep.append(prior_config)
        run_config["priors_config"] = priors_to_keep

        # Running SR task
        expression, logs = physo.ClassSR(multi_X = multi_X,
                                         multi_y = multi_y,
                                         # Giving names of variables (for display purposes)
                                         X_names = [ "x0"      , "x1"        ],
                                         # Giving units of input variables
                                         X_units = [ [0, 0, 0] , [0, 0, 0] ],
                                         # Giving name of root variable (for display purposes)
                                         y_name  = "y",
                                         # Giving units of the root variable
                                         y_units = [0, 0, 0],
                                         # Fixed constants
                                         fixed_consts       = [ 1.      ],
                                         # Units of fixed constants
                                         fixed_consts_units = [ [0, 0, 0] ],
                                         # Class free constants
                                         class_free_consts_names = [ "d"        ],
                                         class_free_consts_units = [ [0, 0, 0]  ],
                                         # Spe free constants
                                         spe_free_consts_names = [ "a"       , "b"        , "c"      ],
                                         spe_free_consts_units = [ [0, 0, 0] , [0, 0, 0]  , [0, 0, 0]  ],
                                         # Run config
                                         run_config = run_config,
                                         # FOR TESTING
                                         op_names = ["add", "sub", "mul", "div"],
                                         get_run_logger     = run_logger,
                                         get_run_visualiser = run_visualiser,
                                         parallel_mode = False,
                                         epochs = 5,
                                         )

        # Inspecting pareto front expressions
        pareto_front_complexities, pareto_front_expressions, pareto_front_r, pareto_front_rmse = logs.get_pareto_front()

        # Assert that solution expression was found
        assert pareto_front_r.max() > 0.9999, "Solution expression was not found."

        return None

if __name__ == '__main__':
    unittest.main()
