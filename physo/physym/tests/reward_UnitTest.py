import unittest
import numpy as np
import time as time
import torch as torch

# Internal imports
from physo.physym import reward
from physo.physym.functions import data_conversion, data_conversion_inv

class RewardTest(unittest.TestCase):

    # Test Squashed NRMSE reward
    def test_Reward_SquashedNRMSE (self):

        DEVICE = 'cpu'
        if torch.cuda.is_available():
            DEVICE = 'cuda'

        # DATA
        N = int(1e6)
        y_target = data_conversion  (np.linspace(0.04, 4, N)  ).to(DEVICE)
        y_pred   = y_target

        # EXECUTION
        t0 = time.perf_counter()
        N = 100
        for _ in range (N):
            res = reward.SquashedNRMSE(y_target = y_target, y_pred = y_pred, )
        t1 = time.perf_counter()
        print("\nReward_SquashedNRMSE time = %.3f ms"%((t1-t0)*1e3/N))

        # TEST
        works_bool = np.array_equal(data_conversion_inv(res.cpu()), 1.)
        self.assertTrue(works_bool)
        return None

if __name__ == '__main__':
    unittest.main(verbosity=2)
