import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings

# Internal code import
import physo
import physo.toolkit.codec as codec

import unittest

class CodecTest(unittest.TestCase):
    def test_get_library_default(self):
        try:
            lib = codec.get_library(
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
                )
        except Exception as e:
            self.fail("Library generation failed.")

        return None
