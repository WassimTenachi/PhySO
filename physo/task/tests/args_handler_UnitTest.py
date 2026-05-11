import unittest

import numpy as np

import physo.task.args_handler as args_handler
from physo.physym.free_const import FreeConstantsTable
from physo.toolkit import codec


class ArgsHandlerTest(unittest.TestCase):

    def get_args_make_tokens(self, class_free_consts_init_val=None, spe_free_consts_init_val=None):
        library_config = args_handler.check_library_args(
            # X
            X_names=["x"],
            X_units=[[0, 0, 0]],
            # y
            y_name="y",
            y_units=[0, 0, 0],
            # Fixed constants
            fixed_consts=[],
            fixed_consts_units=[],
            # Class free constants
            class_free_consts_names=["c0", "c1"],
            class_free_consts_units=[[0, 0, 0], [0, 0, 0]],
            class_free_consts_init_val=class_free_consts_init_val,
            # Spe Free constants
            spe_free_consts_names=["k0", "k1"],
            spe_free_consts_units=[[0, 0, 0], [0, 0, 0]],
            spe_free_consts_init_val=spe_free_consts_init_val,
            # Operations to use
            op_names=["add"],
            use_protected_ops=True,
            # Number of realizations
            n_realizations=3,
            # Warn about units
            warn_about_units=False,
        )
        return library_config["args_make_tokens"]

    def assert_class_init_vals_equal(self, actual, expected):
        self.assertEqual(set(actual.keys()), set(expected.keys()))
        for name, value in expected.items():
            self.assertAlmostEqual(actual[name], value)

    def assert_spe_init_vals_equal(self, actual, expected):
        self.assertEqual(set(actual.keys()), set(expected.keys()))
        for name, value in expected.items():
            np.testing.assert_array_equal(actual[name], np.asarray(value, dtype=float))

    def test_class_free_consts_init_val_none(self):
        args_make_tokens = self.get_args_make_tokens(class_free_consts_init_val=None)

        self.assert_class_init_vals_equal(
            args_make_tokens["class_free_constants_init_val"],
            {"c0": 1.0, "c1": 1.0},
        )

    def test_class_free_consts_init_val_sequence_scalars(self):
        args_make_tokens = self.get_args_make_tokens(class_free_consts_init_val=[1.0, 2.0])

        self.assert_class_init_vals_equal(
            args_make_tokens["class_free_constants_init_val"],
            {"c0": 1.0, "c1": 2.0},
        )

    def test_class_free_consts_init_val_dict_scalars(self):
        args_make_tokens = self.get_args_make_tokens(class_free_consts_init_val={"c0": 1.0, "c1": 2.0})

        self.assert_class_init_vals_equal(
            args_make_tokens["class_free_constants_init_val"],
            {"c0": 1.0, "c1": 2.0},
        )

    def test_spe_free_consts_init_val_none(self):
        args_make_tokens = self.get_args_make_tokens(spe_free_consts_init_val=None)

        self.assert_spe_init_vals_equal(
            args_make_tokens["spe_free_constants_init_val"],
            {"k0": 1.0, "k1": 1.0},
        )

    def test_spe_free_consts_init_val_sequence_scalars(self):
        args_make_tokens = self.get_args_make_tokens(spe_free_consts_init_val=[1.0, 2.0])

        self.assert_spe_init_vals_equal(
            args_make_tokens["spe_free_constants_init_val"],
            {"k0": 1.0, "k1": 2.0},
        )

    def test_spe_free_consts_init_val_sequence_arrays(self):
        args_make_tokens = self.get_args_make_tokens(
            spe_free_consts_init_val=[
                np.array([1.0, 1.1, 1.2]),
                np.array([2.0, 2.1, 2.2]),
            ],
        )

        self.assert_spe_init_vals_equal(
            args_make_tokens["spe_free_constants_init_val"],
            {
                "k0": np.array([1.0, 1.1, 1.2]),
                "k1": np.array([2.0, 2.1, 2.2]),
            },
        )

    def test_spe_free_consts_init_val_dict_scalars(self):
        args_make_tokens = self.get_args_make_tokens(spe_free_consts_init_val={"k0": 1.0, "k1": 2.0})

        self.assert_spe_init_vals_equal(
            args_make_tokens["spe_free_constants_init_val"],
            {"k0": 1.0, "k1": 2.0},
        )

    def test_spe_free_consts_init_val_dict_arrays(self):
        args_make_tokens = self.get_args_make_tokens(
            spe_free_consts_init_val={
                "k0": np.array([1.0, 1.1, 1.2]),
                "k1": np.array([2.0, 2.1, 2.2]),
            },
        )

        self.assert_spe_init_vals_equal(
            args_make_tokens["spe_free_constants_init_val"],
            {
                "k0": np.array([1.0, 1.1, 1.2]),
                "k1": np.array([2.0, 2.1, 2.2]),
            },
        )

    def test_dict_init_vals_build_free_constants_table(self):
        lib = codec.get_library(
            # X
            X_names=["x"],
            X_units=[[0, 0, 0]],
            # y
            y_name="y",
            y_units=[0, 0, 0],
            # Fixed constants
            fixed_consts=[],
            fixed_consts_units=[],
            # Class free constants
            free_consts_names=["c0"],
            free_consts_units=[[0, 0, 0]],
            free_consts_init_val={"c0": 3.0},
            # Spe Free constants
            spe_free_consts_names=["k0", "k1"],
            spe_free_consts_units=[[0, 0, 0], [0, 0, 0]],
            spe_free_consts_init_val={"k0": np.array([1.0, 1.1, 1.2]), "k1": 2.0},
            # Operations to use
            op_names=["add"],
            # Number of realizations
            n_realizations=3,
        )

        FreeConstantsTable(batch_size=1, library=lib, n_realizations=3)

        np.testing.assert_array_equal(lib.class_free_constants_init_val, np.array([3.0]))
        np.testing.assert_array_equal(
            lib.spe_free_constants_init_val,
            np.array([[1.0, 1.1, 1.2], [2.0, 2.0, 2.0]]),
        )


if __name__ == "__main__":
    unittest.main()
