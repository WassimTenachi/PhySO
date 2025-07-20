import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings
# Internal code import
import physo
import physo.learn.monitoring as monitoring
from physo.physym import execute as Exec

import unittest

class Test_structure_analysis(unittest.TestCase):
    def test_StructureAnalysis_task(self):

        # Seed
        seed = 0
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Dataset
        z = np.random.uniform(-10, 10, 50)
        v = np.random.uniform(-10, 10, 50)
        X = np.stack((z, v), axis=0)
        y = 1.234*9.807*z + 1.234*v**2

        # Running task
        structure = physo.StructureAnalysis(X, y,
                                    X_names = [ "z"       , "v"        ],
                                    # Run config
                                    run_config = physo.config.config0.config0,
                                    # Save path
                                    do_save   = True,
                                    save_path = None,
                                    # Device
                                    device        = 'cpu',
        )

        return None

    def test_SR_with_StructureAnalysis_task(self):

        run_logger = lambda : monitoring.RunLogger(
                                      save_path = 'SR.log',
                                      do_save   = True)
        run_visualiser = lambda : monitoring.RunVisualiser (
                                      epoch_refresh_rate = 1,
                                      save_path = 'SR_curves.png',
                                      do_show   = False,
                                      do_prints = True,
                                      do_save   = True, ) # todo: change back to False

        # Seed
        seed = 0
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Dataset
        z = np.random.uniform(-10, 10, 50)
        v = np.random.uniform(-10, 10, 50)
        X = np.stack((z, v), axis=0)
        y = 1.234*9.807*z + 1.234*v**2

        # Larger batch size and longer size when using structure analysis
        run_config = physo.config.config3_expA.config3

        # target_prog_str = ["add", "mul", "mul", "m", "g", "z", "mul", "m", "n2", "v"]
        # cheater_prior_config = ('SymbolicPrior', {'expression': target_prog_str})
        # run_config["priors_config"].append(cheater_prior_config)

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
                                    # Run with structure analysis
                                    #run_config = run_config,

                                    # FOR TESTING
                                    get_run_logger     = run_logger,
                                    get_run_visualiser = run_visualiser,
                                    parallel_mode=False,
                                    epochs = 5,

                                    # Run with structure analysis
                                    structure_analysis = True,
        )

        # Inspecting pareto front expressions
        pareto_front_complexities, pareto_front_expressions, pareto_front_r, pareto_front_rmse = logs.get_pareto_front()

        # Assert that solution expression was found
        assert pareto_front_r.max() > 0.9999, "Solution expression was not found."


        return None
if __name__ == '__main__':
    unittest.main()
