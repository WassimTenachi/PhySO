import time
import unittest
import numpy as np
import torch

# Internal imports
import physo.benchmark.FeynmanDataset.FeynmanProblem as Feyn

class FeynmanProblemTest(unittest.TestCase):
    def test_loading_csv(self):
        # Loading bulk equations
        try:
            bulk_df = Feyn.load_feynman_bulk_equations_csv(Feyn.PATH_FEYNMAN_EQS_CSV)
        except:
            self.fail("Failed to load bulk equations csv")
        # Loading bonus equations
        try:
            bonus_df = Feyn.load_feynman_bonus_equations_csv(Feyn.PATH_FEYNMAN_EQS_BONUS_CSV)
        except:
            self.fail("Failed to load bonus equations csv")

        assert len(bulk_df) == 100, "Bulk equations csv has wrong number of equations."
        assert len(bonus_df) == 20, "Bonus equations csv has wrong number of equations."

        assert np.array_equal(bulk_df.columns, bonus_df.columns), "Bulk and bonus equations dfs have different columns"

        # Loading all equations
        try:
            eqs_df = Feyn.EQS_FEYNMAN_DF
        except:
            self.fail("Failed to load all equations csv")

        assert len(eqs_df) == 120, "All equations df has wrong number of equations."

        expected_columns = np.array(['Filename', 'Name', 'Set', 'Number', 'Output', 'Formula', '# variables',
                           'v1_name', 'v1_low', 'v1_high', 'v2_name', 'v2_low', 'v2_high',
                           'v3_name', 'v3_low', 'v3_high', 'v4_name', 'v4_low', 'v4_high',
                           'v5_name', 'v5_low', 'v5_high', 'v6_name', 'v6_low', 'v6_high',
                           'v7_name', 'v7_low', 'v7_high', 'v8_name', 'v8_low', 'v8_high',
                           'v9_name', 'v9_low', 'v9_high', 'v10_name', 'v10_low', 'v10_high'])
        assert np.array_equal(eqs_df.columns,expected_columns)

        return None

    def test_get_units(self):

        # Test length
        assert len(Feyn.get_units("v")) == Feyn.FEYN_UNITS_VECTOR_SIZE, "Wrong length for units vector"
        # Test v
        assert np.array_equal(Feyn.get_units('v'), np.array([1., -1., 0., 0., 0.])), "Failed to get units for v"
        # Test a
        assert np.array_equal(Feyn.get_units('a'), np.array([1., -2., 0., 0., 0.])), "Failed to get units for a"
        # Test L_ind
        assert np.array_equal(Feyn.get_units('L_ind'), np.array([-2., 4., -1., 0., 2.])), "Failed to get units for L_ind"
        # Test mu_drift (corrected from original paper)
        assert np.array_equal(Feyn.get_units('mu_drift'), np.array([0., 1., -1., 0., 0.])), "Failed to get units for mu_drift"

        return None

    def test_FeynmanProblem(self):

        # Test loading a problem
        try:
            relatpb = Feyn.FeynmanProblem(eq_name ="I.15.1")
        except:
            self.fail("Failed to load a problem")
        assert relatpb.eq_name == "I.15.1", "Wrong eq_name."

        try:
            relatpb = Feyn.FeynmanProblem(18)
        except:
            self.fail("Failed to load a problem")
        assert relatpb.eq_name == "I.15.1", "Wrong eq_name."

        # Test variable names on sample problem
        expected_original_var_names = ['m_0', 'v', 'c']
        expected_original_y_name    = 'p'
        expected_standard_var_names = ['x0', 'x1', 'x2']
        expected_standard_y_name    = 'y'
        relatpb = Feyn.FeynmanProblem(18, original_var_names = True) # With original var names
        assert np.array_equal(relatpb.X_names,          expected_original_var_names), "Wrong X_names."
        assert np.array_equal(relatpb.X_names_original, expected_original_var_names), "Wrong X_names."
        assert np.array_equal(relatpb.y_name,          expected_original_y_name), "Wrong y_name."
        assert np.array_equal(relatpb.y_name_original, expected_original_y_name), "Wrong y_name."
        assert relatpb.n_vars == 3, "Wrong n_vars."

        relatpb = Feyn.FeynmanProblem(18, original_var_names = False) # Without original var names
        assert np.array_equal(relatpb.X_names,          expected_standard_var_names), "Wrong X_names."
        assert np.array_equal(relatpb.X_names_original, expected_original_var_names), "Wrong X_names."
        assert np.array_equal(relatpb.y_name,          expected_standard_y_name), "Wrong y_name."
        assert np.array_equal(relatpb.y_name_original, expected_original_y_name), "Wrong y_name."
        assert relatpb.n_vars == 3, "Wrong n_vars."

        # Test units on sample problem
        relatpb = Feyn.FeynmanProblem(18)
        assert np.array_equal(relatpb.X_units, np.array([[ 0.,  0.,  1.,  0.,  0.],
                                                         [ 1., -1.,  0.,  0.,  0.],
                                                         [ 1., -1.,  0.,  0.,  0.]])), "Wrong X_units."
        assert np.array_equal(relatpb.y_units, np.array([1., 0., 0., 0., 0.])), "Wrong y_units."

        # Test ranges on sample problem
        relatpb = Feyn.FeynmanProblem(18)
        assert np.array_equal(relatpb.X_lows , [1., 1., 3.]  ),  "Wrong X_lows."
        assert np.array_equal(relatpb.X_highs, [ 5.,  2., 10.]), "Wrong X_highs."

        return None

    def test_FeynmanProblem_datagen_all(self):
        verbose = False

        # Iterating through all Feynman problems (ie. equations)
        for i in range(Feyn.N_EQS):

            # Loading problem
            original_var_names = False  # replacing original symbols (e.g. theta, sigma etc.) by x0, x1 etc.
            # original_var_names = True  # using original symbols (e.g. theta, sigma etc.)
            pb = Feyn.FeynmanProblem(i, original_var_names=original_var_names)

            if verbose:
                print("\n------------------------ %i : %s ------------------------" % (pb.i_eq, pb.eq_name))
                print(pb)

                # Print expression with evaluated constants
                print("--- Expression with evaluated constants ---")
                print(pb.formula_sympy_eval)

                # Printing physical units of variables
                print("--- Units ---")
                print("X units : \n", pb.X_units)
                print("y units : \n", pb.y_units)

            # Loading data sample
            X_array, y_array = pb.generate_data_points(n_samples=100)

            # Printing min, max of data points and warning if absolute value is above WARN_LIM
            if verbose: print("--- min, max ---")
            WARN_LIM = 50
            xmin, xmax, ymin, ymax = X_array.min(), X_array.max(), y_array.min(), y_array.max()
            if verbose:
                print("X min = ", xmin)
                print("X max = ", xmax)
                print("y min = ", ymin)
                print("y max = ", ymax)
                if abs(xmin) > WARN_LIM:
                    print("-> xmin has high absolute value : %f" % (xmin))
                if abs(xmax) > WARN_LIM:
                    print("-> xmax has high absolute value : %f" % (xmax))
                if abs(ymin) > WARN_LIM:
                    print("-> ymin has high absolute value : %f" % (ymin))
                if abs(ymax) > WARN_LIM:
                    print("-> ymax has high absolute value : %f" % (ymax))


if __name__ == '__main__':
    unittest.main()
