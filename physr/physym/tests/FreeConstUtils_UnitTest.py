import time
import unittest
import torch
import numpy as np

import physr.physym.FreeConstUtils as FreeConstUtils

class FreeConstUtilsTest(unittest.TestCase):

    def test_lgbs_optimizer (self):

        # ------ Test case ------
        r = torch.tensor(np.linspace(-10, 10, 100))
        v = torch.tensor(np.linspace(-10, 10, 100))
        X = torch.stack((r,v))

        func = lambda params, X: params[0] * X[1] ** 2 + (params[1] ** 2) * torch.log(X[0] ** 2 + params[2] ** 2)

        ideal_params = [0.5, 1.14, 0.936]
        func_params = lambda params: func(params, X)
        y_ideal = func_params(params=ideal_params)

        n_params = len(ideal_params)

        # ------ Run ------
        total_n_steps = 0
        t0 = time.perf_counter()

        N = 1000
        for _ in range (N):

            params_init = 1. * torch.ones(n_params, )
            params = params_init

            history = FreeConstUtils.optimize_free_const ( func     = func_params,
                                                           params   = params,
                                                           y_target = y_ideal,
                                                           loss        = "MSE",
                                                           method      = "LBFGS",
                                                           method_args = None)
            total_n_steps += history.shape[0]

        t1 = time.perf_counter()
        dt = ((t1-t0)*1e3)/total_n_steps
        print("LBFGS const opti: %f ms / step" %(dt))

        # ------ Test ------
        obs_params   = params.detach().cpu().numpy()
        ideal_params = np.array(ideal_params)
        for i in range (n_params):
            err = np.abs(obs_params[0] - ideal_params[0])
            works_bool = (err < 1e-6)
            self.assertEqual(works_bool, True)

        return None
    
if __name__ == '__main__':
    unittest.main(verbosity=2)
