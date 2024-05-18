import time
import unittest
import numpy as np
import torch

# Internal imports
import physo.benchmark.ClassDataset.ClassProblem as Class


class ClassProblemTest(unittest.TestCase):
    def test_loading_csv(self):
        # Loading  equations
        try:
            df = Class.load_class_equations_csv(Class.PATH_CLASS_EQS_CSV)
        except:
            self.fail("Failed to load equations csv")

        assert len(df) == 8, "Equations csv has wrong number of equations."

        return None
    def test_ClassProblem_datagen_all(self):
        verbose = False

        # Iterating through all Class problems (ie. equations)
        for i in range(Class.N_EQS):

            # Loading problem
            original_var_names = False  # replacing original symbols (e.g. theta, sigma etc.) by x0, x1 etc.
            # original_var_names = True  # using original symbols (e.g. theta, sigma etc.)
            pb = Class.ClassProblem(i, original_var_names=original_var_names)

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
                print("K units : \n", pb.K_units)

            # Loading data sample
            multi_X, multi_y = pb.generate_data_points(n_samples=100)
            if verbose:
                pb.show_sample()

            # Printing min, max of data points and warning if absolute value is above WARN_LIM
            if verbose: print("--- min, max ---")
            WARN_LIM = 50
            xmin, xmax, ymin, ymax = multi_X.min(), multi_X.max(), multi_y.min(), multi_y.max()
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
