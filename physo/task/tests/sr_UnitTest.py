import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings
# Internal code import
import physo
import physo.learn.monitoring as monitoring

import unittest

class Test_SR(unittest.TestCase):
    def test_SR_task(self):

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
        z = np.random.uniform(-10, 10, 50)
        v = np.random.uniform(-10, 10, 50)
        X = np.stack((z, v), axis=0)
        y = 1.234*9.807*z + 1.234*v**2

        # Running SR task
        expression, logs = physo.SR(X, y,
                                    # Giving names of variables (for display purposes)
                                    X_names = [ "z"       , "v"        ],
                                    # Giving units of input variables
                                    X_units = [ [1, 0, 0] , [1, -1, 0] ],
                                    # Giving name of root variable (for display purposes)
                                    y_name  = "E",
                                    # Giving units of the root variable
                                    y_units = [2, -2, 1],
                                    # Fixed constants
                                    fixed_consts       = [ 1.      ],
                                    # Units of fixed constants
                                    fixed_consts_units = [ [0,0,0] ],
                                    # Free constants names (for display purposes)
                                    free_consts_names = [ "m"       , "g"        ],
                                    # Units of free constants
                                    free_consts_units = [ [0, 0, 1] , [1, -2, 0] ],
                                    # Run config
                                    run_config = physo.config.config0.config0,

                                    # FOR TESTING
                                    get_run_logger     = run_logger,
                                    get_run_visualiser = run_visualiser,
                                    parallel_mode=False,
                                    epochs = 5,
        )

        # Inspecting pareto front expressions
        pareto_front_complexities, pareto_front_expressions, pareto_front_r, pareto_front_rmse = logs.get_pareto_front()

        # Assert that solution expression was found
        assert pareto_front_r.max() > 0.9999, "Solution expression was not found."


        return None

    # Testing that SR task runs with few operators (ie. operators mentioned in default priors missing)
    def test_prior_not_in_op_names (self):

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
        z = np.random.uniform(-10, 10, 50)
        v = np.random.uniform(-10, 10, 50)
        X = np.stack((z, v), axis=0)
        y = 1.234*9.807*z + 1.234*v**2

        # Running SR task
        # Avoiding expected warnings caused by priors referring to operators not in op_names
        # We still want to keep those priors for the sake of this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expression, logs = physo.SR(X, y,
                                        X_names = [ "z"       , "v"        ],
                                        X_units = [ [1, 0, 0] , [1, -1, 0] ],
                                        y_name  = "E",
                                        y_units = [2, -2, 1],
                                        fixed_consts       = [ 1.      ],
                                        fixed_consts_units = [ [0,0,0] ],
                                        free_consts_names = [ "m"       , "g"        ],
                                        free_consts_units = [ [0, 0, 1] , [1, -2, 0] ],
                                        op_names = ["mul", "add", "sub", "div"],
                                        # Run config
                                        run_config = physo.config.config0.config0,
                                        # FOR TESTING
                                        get_run_logger     = run_logger,
                                        get_run_visualiser = run_visualiser,
                                        parallel_mode = False,
                                        epochs = 7,
            )
        # Inspecting pareto front expressions
        pareto_front_complexities, pareto_front_expressions, pareto_front_r, pareto_front_rmse = logs.get_pareto_front()

        # Assert that solution expression was found
        assert pareto_front_r.max() > 0.9999, "Solution expression was not found."

        return None

if __name__ == '__main__':
    unittest.main()
