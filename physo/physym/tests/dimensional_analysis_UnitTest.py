import time
import unittest
import numpy as np

from physo.physym import program as Prog
from physo.physym import library as Lib
from physo.physym import dimensional_analysis as phy


def hard_test_case():
    # LIBRARY CONFIG
    args_make_tokens = {
                    # operations
                    "op_names"             : "all",  # or ["mul", "neg", "inv", "sin"]
                    "use_protected_ops"    : True,
                    # input variables
                    "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                    "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                    "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                    # constants
                    "constants"            : {"pi" : np.pi     , "M" : 1e6       , "const1" : 1         },
                    "constants_units"      : {"pi" : [0, 0, 0] , "M" : [0, 0, 1] , "const1" : [0, 0, 0] },
                    "constants_complexity" : {"pi" : 0.        , "M" : 1.        , "const1" : 1.        },
                    # free constants
                    "free_constants"            : {"c"               },
                    "free_constants_init_val"   : {"c"  : 10.        },
                    "free_constants_units"      : {"c"  : [1, -1, 0] },
                    "free_constants_complexity" : {"c"  : 0.         },
                           }

    my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                         superparent_units = [2, -2, 1], superparent_name = "y")

    # TEST PROGRAM
    test_program_str = ["mul", "mul", "M", "n2", "c", "sub", "inv", "sqrt", "sub", "const1", "div", "n2", "v", "n2",
                        "c", "cos", "div", "sub", "div", "const1", "const1", "div", "v", "c", "div", "v", "c"]
    test_program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str])

    # ------------------- EXPECTED LIVE BEHAVIOR -------------------
    expected_constraints = np.array([
        (True, [2., -2., 1., 0., 0., 0., 0.],),
        (False, [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],),
        (False, [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],),
        (False, [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],),
        (False, [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],),
        (True, [0., 0., 0., 0., 0., 0., 0.],),
        (True, [0., 0., 0., 0., 0., 0., 0.],),
        (True, [-0., -0., -0., -0., -0., -0., -0.],),
        (True, [-0., -0., -0., -0., -0., -0., -0.],),
        (True, [-0., -0., -0., -0., -0., -0., -0.],),
        (True, [0., 0., 0., 0., 0., 0., 0.],),
        (False, [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],),
        (False, [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],),
        (True, [2., -2., 0., 0., 0., 0., 0.],),
        (True, [1., -1., 0., 0., 0., 0., 0.],),
        (True, [0., 0., 0., 0., 0., 0., 0.],),
        (True, [0., 0., 0., 0., 0., 0., 0.],),
        (False, [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],),
        (False, [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],),
        (False, [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],),
        (False, [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],),
        (True, [0., 0., 0., 0., 0., 0., 0.],),
        (False, [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],),
        (True, [1., -1., 0., 0., 0., 0., 0.],),
        (True, [0., 0., 0., 0., 0., 0., 0.],),
        (False, [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],),
        (True, [1., -1., 0., 0., 0., 0., 0.],),
    ], dtype=object)
    expected_tokens_cases_record = np.array([3, 70, 21, 72, 21, 73, 4, 5, 5, 4, 1, 70, 21, 73, 5, 1, 6, 70, 21, 21, 72,
                                             20, 70, 73, 73, 70, 73])
    expected_phy_units = np.array([x for x in expected_constraints[:, 1]], dtype=float)
    expected_is_constraining = expected_constraints[:, 0].astype(bool)

    # ------------------- EXPECTED UNITS AT FINAL STEP -------------------
    expected_constraints_final = np.array([
           (True, [ 2., -2.,  1.,  0.,  0.,  0.,  0.],),
           (True, [ 2., -2.,  1.,  0.,  0.,  0.,  0.],),
           (True, [ 0.,  0.,  1.,  0.,  0.,  0.,  0.],),
           (True, [ 2., -2.,  0.,  0.,  0.,  0.,  0.],),
           (True, [ 1., -1.,  0.,  0.,  0.,  0.,  0.],),
           (True, [-0., -0., -0., -0., -0., -0., -0.],),
           (True, [-0., -0., -0., -0., -0., -0., -0.],),
           (True, [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],),
           (True, [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],),
           (True, [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],),
           (True, [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],),
           (True, [ 2., -2.,  0.,  0.,  0.,  0.,  0.],),
           (True, [ 1., -1.,  0.,  0.,  0.,  0.,  0.],),
           (True, [ 2., -2.,  0.,  0.,  0.,  0.,  0.],),
           (True, [ 1., -1.,  0.,  0.,  0.,  0.,  0.],),
           (True, [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],),
           (True, [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],),
           (True, [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],),
           (True, [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],),
           (True, [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],),
           (True, [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],),
           (True, [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],),
           (True, [ 1., -1.,  0.,  0.,  0.,  0.,  0.],),
           (True, [ 1., -1.,  0.,  0.,  0.,  0.,  0.],),
           (True, [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],),
           (True, [ 1., -1.,  0.,  0.,  0.,  0.,  0.],),
           (True, [ 1., -1.,  0.,  0.,  0.,  0.,  0.],),
    ], dtype=object)

    expected_phy_units_final       = np.array([x for x in expected_constraints_final[:, 1]], dtype=float)
    expected_is_constraining_final = expected_constraints_final[:, 0].astype(bool)

    return (test_program_idx, my_lib, expected_tokens_cases_record, expected_phy_units, expected_is_constraining, \
           expected_phy_units_final, expected_is_constraining_final)

class DimensionalAnalysisTest(unittest.TestCase):

    def test_assign_units_bottom_up(self):
        test_program_idx, my_lib, expected_tokens_cases_record, expected_phy_units, expected_is_constraining, \
            expected_phy_units_final, expected_is_constraining_final = hard_test_case()

        test_program_length = len(test_program_idx)
        batch_size = 3
        test_programs_idx = np.tile(test_program_idx, reps=(batch_size, 1))

        my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=test_program_length, library=my_lib)
        my_programs.set_programs(test_programs_idx)

        # ------------------- TEST BOTTOM-UP -------------------
        try:
            phy.assign_units_bottom_up(my_programs,
                                   coords_start = my_programs.coords_of_step(0),
                                   coords_end   = my_programs.coords_of_step(my_programs.max_time_step-1),)
        except:
            self.fail("Bottom up run failed.")

        # ------------------- TEST BOTTOM-UP RESULTS-------------------
        # After bottom up
        observed_phy_units_final_after_BU       = my_programs.tokens.phy_units                 [0,:].copy()
        observed_is_constraining_final_after_BU = my_programs.tokens.is_constraining_phy_units [0,:].copy()

        # ASSERT BOTTOM UP GIVES RIGHT RESULT
        bool_works = np.array_equal(expected_is_constraining_final , observed_is_constraining_final_after_BU)
        self.assertEqual(bool_works, True)
        bool_works = np.array_equal(expected_phy_units_final       , observed_phy_units_final_after_BU, equal_nan = True )
        self.assertEqual(bool_works, True)

    def test_get_required_units(self):

        # ------------------- TEST CASE -------------------
        test_program_idx, my_lib, expected_tokens_cases_record, expected_phy_units, expected_is_constraining, \
            expected_phy_units_final, expected_is_constraining_final = hard_test_case()

        test_program_length = len(test_program_idx)
        batch_size = 3
        test_programs_idx = np.tile(test_program_idx, reps=(batch_size, 1)).transpose()

        # ------------------- TRY AT STEP: assign_required_units_at_step -------------------
        try:
            batch_size_large = int(1e3)
            test_programs_idx_large = np.tile(test_program_idx, reps=(batch_size_large, 1)).transpose()
            my_programs = Prog.VectPrograms(batch_size=batch_size_large, max_time_step=test_program_length, library=my_lib)
            t0 = time.perf_counter()
            for i, idx in enumerate(test_programs_idx_large):
                # Computing units requirements
                cases_record = phy.assign_required_units_at_step(my_programs)
                # Appending new token
                # In real world run, next token should be chosen based on constraints found above
                my_programs.append(idx)
            t1 = time.perf_counter()
            marginal_time = ((t1-t0)*1e3)/(test_program_length * batch_size_large)
            print("\nRequired units time (in ideal non-mixed cases) : %f ms / step / (prog in batch) " % marginal_time)
        except:
            self.fail("Unable to run assign_required_units_at_step")

        # ------------------- TEST AT STEP: assign_required_units_at_step -------------------
        my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=test_program_length, library=my_lib)

        observed_phy_units           = []
        observed_is_constraining     = []
        observed_tokens_cases_record = []
        for i, idx in enumerate(test_programs_idx) :
            # Computing units requirements
            cases_record = phy.assign_required_units_at_step(my_programs)
            # Saving observed results
            observed_tokens_cases_record.   append (cases_record                                 [0]  .copy())
            observed_phy_units.             append (my_programs.tokens.phy_units                 [0,i].copy())
            observed_is_constraining.       append (my_programs.tokens.is_constraining_phy_units [0,i].copy())
            # Assertions
            bool_works = np.array_equal(observed_tokens_cases_record[i], expected_tokens_cases_record[i])
            self.assertEqual(bool_works, True)
            bool_works = np.array_equal(observed_is_constraining    [i], expected_is_constraining    [i])
            self.assertEqual(bool_works, True)
            bool_works = np.array_equal(observed_phy_units          [i], expected_phy_units          [i], equal_nan = True )
            self.assertEqual(bool_works, True)
            # Appending new token
            # In real world run, next token should be chosen based on constraints found above
            my_programs.append(idx)

        # To numpy array
        observed_phy_units           = np.array(observed_phy_units)
        observed_is_constraining     = np.array(observed_is_constraining)
        observed_tokens_cases_record = np.array(observed_tokens_cases_record)

        # Final result after all idx were added (Results using assign_required_units_at_step function)
        observed_phy_units_final_w_step       = my_programs.tokens.phy_units                 [0,:].copy()
        observed_is_constraining_final_w_step = my_programs.tokens.is_constraining_phy_units [0,:].copy()

        # (Results using assign_required_units_at_step function)
        observed_phy_units_w_step           = observed_phy_units           .copy()
        observed_is_constraining_w_step     = observed_is_constraining     .copy()
        observed_tokens_cases_record_w_step = observed_tokens_cases_record .copy()

        # ------------------- TEST AT STEP: assign_required_units_at_step from scratch -------------------
        my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=test_program_length, library=my_lib)
        my_programs.set_programs(test_programs_idx.transpose())

        phy.assign_required_units_at_step(my_programs, from_scratch=True)

       # Final result after all idx were added (Results using assign_required_units_at_step function)
        observed_phy_units_final_w_step_from_scratch       = my_programs.tokens.phy_units                 [0,:].copy()
        observed_is_constraining_final_w_step_from_scratch = my_programs.tokens.is_constraining_phy_units [0,:].copy()

        # Assert re-computing everything from_scratch gives same result
        bool_works = np.array_equal(observed_phy_units_final_w_step_from_scratch       , observed_phy_units_final_w_step       , equal_nan=True)
        self.assertEqual(bool_works, True)
        bool_works = np.array_equal(observed_is_constraining_final_w_step_from_scratch , observed_is_constraining_final_w_step)
        self.assertEqual(bool_works, True)

        # ------------------- TEST AT COORDS: assign_required_units -------------------
        my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=test_program_length, library=my_lib)

        observed_phy_units           = []
        observed_is_constraining     = []
        observed_tokens_cases_record = []
        for i, idx in enumerate(test_programs_idx) :
            # Computing units requirements
            coords = my_programs.coords_of_step(i)[:, [0,2]] # doing 0-th and 2-th of batch but not 1-th
            cases_record = phy.assign_required_units(my_programs, coords=coords)
            # Saving observed results
            observed_tokens_cases_record.   append (cases_record                                 [0]  .copy())
            observed_phy_units.             append (my_programs.tokens.phy_units                 [0,i].copy())
            observed_is_constraining.       append (my_programs.tokens.is_constraining_phy_units [0,i].copy())
            # Assertions
            bool_works = np.array_equal(observed_tokens_cases_record[i], expected_tokens_cases_record[i])
            self.assertEqual(bool_works, True)
            bool_works = np.array_equal(observed_is_constraining    [i], expected_is_constraining    [i])
            self.assertEqual(bool_works, True)
            bool_works = np.array_equal(observed_phy_units          [i], expected_phy_units          [i], equal_nan = True )
            self.assertEqual(bool_works, True)
            # Appending new token
            # In real world run, next token should be chosen based on constraints found above
            my_programs.append(idx)

        # To numpy array
        observed_phy_units           = np.array(observed_phy_units)
        observed_is_constraining     = np.array(observed_is_constraining)
        observed_tokens_cases_record = np.array(observed_tokens_cases_record)

        # Final result after all idx were added
        observed_phy_units_final       = my_programs.tokens.phy_units                 [0,:].copy()
        observed_is_constraining_final = my_programs.tokens.is_constraining_phy_units [0,:].copy()

        # Assert that assign_required_units at coords gives same results as assign_required_units_at_step
        # Final result
        bool_works = np.array_equal(observed_phy_units_final       , observed_phy_units_final_w_step       , equal_nan=True)
        self.assertEqual(bool_works, True)
        bool_works = np.array_equal(observed_is_constraining_final , observed_is_constraining_final_w_step)
        self.assertEqual(bool_works, True)
        # Intermediate results
        bool_works = np.array_equal(observed_phy_units             , observed_phy_units_w_step             , equal_nan=True)
        self.assertEqual(bool_works, True)
        bool_works = np.array_equal(observed_is_constraining       , observed_is_constraining_w_step)
        self.assertEqual(bool_works, True)
        bool_works = np.array_equal(observed_tokens_cases_record   , observed_tokens_cases_record_w_step)
        self.assertEqual(bool_works, True)

        # ------------------- CHECK ALL THE UNITS FOUND ARE COHERENT WITH BOTTOM-UP -------------------
        try:
            phy.assign_units_bottom_up(my_programs,
                                   coords_start = my_programs.coords_of_step(0),
                                   coords_end   = my_programs.coords_of_step(my_programs.max_time_step-1),)
        except:
            self.fail("Final check bottom up failed.")

        # After bottom up
        observed_phy_units_final_after_BU       = my_programs.tokens.phy_units                 [0,:].copy()
        observed_is_constraining_final_after_BU = my_programs.tokens.is_constraining_phy_units [0,:].copy()

        # ASSERT BOTTOM UP GIVES RIGHT RESULT
        bool_works = np.array_equal(expected_is_constraining_final , observed_is_constraining_final_after_BU)
        self.assertEqual(bool_works, True)
        bool_works = np.array_equal(expected_phy_units_final       , observed_phy_units_final_after_BU, equal_nan = True )
        self.assertEqual(bool_works, True)

        # ASSERT THAT END STEP OF LIVE UNITS CONSTRAINTS GIVES SAME AS BOTTOM UP
        bool_works = np.array_equal(observed_is_constraining_final , observed_is_constraining_final_after_BU)
        self.assertEqual(bool_works, True)
        bool_works = np.array_equal(observed_phy_units_final       , observed_phy_units_final_after_BU, equal_nan = True )
        self.assertEqual(bool_works, True)

        return None


if __name__ == '__main__':
    unittest.main(verbosity=2)
