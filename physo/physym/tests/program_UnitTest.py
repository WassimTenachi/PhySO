import unittest

import matplotlib.pyplot as plt
import numpy as np
import time as time

# Internal imports
from physo.physym import library as Lib
from physo.physym import program as Prog
from physo.physym import dimensional_analysis as phy


def make_lib():
    # LIBRARY CONFIG
    args_make_tokens = {
                    # operations
                    "op_names"             : "all",  # or ["mul", "neg", "inv", "sin"]
                    "use_protected_ops"    : False,
                    # input variables
                    "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                    "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                    "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                    # constants
                    "constants"            : {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                    "constants_units"      : {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] },
                    "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                    # free constants
                    "free_constants"            : {"c0"             , "c1"               , "c2"             },
                    "free_constants_init_val"   : {"c0" : 1.        , "c1"  : 10.        , "c2" : 1.        },
                    "free_constants_units"      : {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                    "free_constants_complexity" : {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                           }

    my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                         superparent_units = [1, -2, 1], superparent_name = "y")
    return my_lib


class VectProgramsTest(unittest.TestCase):

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------ INIT ------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # Test VectPrograms init
    def test_make_VectPrograms(self):
        # BATCH CONFIG
        batch_size = 10000
        max_time_step = 32
        my_lib = make_lib()
        # BATCH
        try:
            my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=max_time_step, library=my_lib)
        except:
            self.fail("VectPrograms creation failed.")

    # Test VectPrograms init
    def test_make_VectPrograms_assertions(self):
        # BATCH CONFIG
        my_lib = make_lib()
        # BATCH
        with self.assertRaises(AssertionError, ):
            my_programs = Prog.VectPrograms(batch_size='1000', max_time_step=32, library=my_lib)
        with self.assertRaises(AssertionError, ):
            my_programs = Prog.VectPrograms(batch_size=1000.0, max_time_step=32, library=my_lib)
        with self.assertRaises(AssertionError, ):
            my_programs = Prog.VectPrograms(batch_size=-1, max_time_step=32, library=my_lib)
        with self.assertRaises(AssertionError, ):
            my_programs = Prog.VectPrograms(batch_size=1000, max_time_step='32', library=my_lib)
        with self.assertRaises(AssertionError, ):
            my_programs = Prog.VectPrograms(batch_size=1000, max_time_step=32.0, library=my_lib)
        with self.assertRaises(AssertionError, ):
            my_programs = Prog.VectPrograms(batch_size=1000, max_time_step=-1, library=my_lib)

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- APPEND ASSERTIONS ------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # Exceed max_time_step
    def test_append_exceed_time_step(self):
        # BATCH CONFIG
        batch_size = 6
        max_time_step = 5
        my_lib = make_lib()
        # BATCH
        my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=max_time_step, library=my_lib)
        # APPEND
        # Exceeding time step by appending terminal tokens to raise this error
        terminal_token_idx = my_lib.n_choices - 1
        next_tokens_idx = np.full(batch_size, terminal_token_idx, int)
        with self.assertRaises(IndexError, ):
            for _ in range (max_time_step+1):
                my_programs.append(next_tokens_idx)
        return None

    # Test append wrong type
    def test_append_wrong_arg_type(self):
        # BATCH CONFIG
        batch_size = 1000
        max_time_step = 32
        my_lib = make_lib()
        # BATCH
        my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=max_time_step, library=my_lib)
        # APPEND
        next_tokens_idx = np.random.randint(low=0, high=my_lib.n_choices, size=batch_size)
        with self.assertRaises(AssertionError, ):
            my_programs.append(next_tokens_idx.astype(float))

    # Test append wrong shape
    def test_append_wrong_arg_shape(self):
        # BATCH CONFIG
        batch_size = 1000
        max_time_step = 32
        my_lib = make_lib()
        # BATCH
        my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=max_time_step, library=my_lib)
        # APPEND
        next_tokens_idx = np.random.randint(low=0, high=my_lib.n_choices, size=batch_size + 99)
        with self.assertRaises(AssertionError, ):
            my_programs.append(next_tokens_idx)

    # Test append wrong min/max
    def test_append_wrong_arg_min_max(self):
        # BATCH CONFIG
        batch_size = 1000
        max_time_step = 32
        my_lib = make_lib()
        # BATCH
        my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=max_time_step, library=my_lib)
        # APPEND
        next_tokens_idx = np.random.randint(low=-1, high=my_lib.n_choices, size=batch_size)
        with self.assertRaises(AssertionError):
            my_programs.append(next_tokens_idx)
        next_tokens_idx = np.random.randint(low=0, high=my_lib.n_choices+1, size=batch_size)
        with self.assertRaises(AssertionError):
            my_programs.append(next_tokens_idx)

    # Test append too many dummies necessary to complete
    def test_append_not_enough_space_for_dummies(self):
        # BATCH CONFIG
        batch_size = int(1e5)
        max_time_step = 8
        my_lib = make_lib()
        np.random.seed(seed=42)
        # Not enough space for dummies with unsafe number of steps
        with self.assertRaises(IndexError):
            # BATCH
            my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=max_time_step, library=my_lib)
            # ADDING NEW TOKENS
            for step in range(1, my_programs.safe_max_time_step*4):
                next_tokens_idx = np.random.randint(low=0, high=my_lib.n_choices, size=batch_size)
                my_programs.append(next_tokens_idx)
        # Enough space for dummies with safe number of steps
        try:
            # BATCH
            my_programs = Prog.VectPrograms(batch_size=batch_size, max_time_step=max_time_step, library=my_lib)
            # ADDING NEW TOKENS
            for step in range(1, my_programs.safe_max_time_step):
                next_tokens_idx = np.random.randint(low=0, high=my_lib.n_choices, size=batch_size)
                my_programs.append(next_tokens_idx)
        except:
            self.fail("VectPrograms append failed.")
        return None

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------- FAMILY RELATIONSHIPS ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def test_family_relationships(self):
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
                        "constants"            : {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       , "const1" : 1         },
                        "constants_units"      : {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] , "const1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        , "const1" : 1.        },
                        # free constants
                        "free_constants"            : {"c0"             , "c1"               , "c2"             },
                        "free_constants_init_val"   : {"c0" : 1.        , "c1"  : 10.        , "c2" : 1.        },
                        "free_constants_units"      : {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                        "free_constants_complexity" : {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                           }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [1, -2, 1], superparent_name = "y")

        # TEST PROGRAM
        test_program_str = ["mul", "mul", "M", "n2", "c", "sub", "inv", "sqrt", "sub", "c1", "div", "n2", "v", "n2",
                            "c", "cos", "div", "sub", "const1", "div", "v", "c", "div", "v", "c"]
        test_program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str])
        test_program_length = len(test_program_str)
        test_program_idx = test_program_idx[np.newaxis, :]

        # BATCH
        my_programs = Prog.VectPrograms(batch_size=1, max_time_step=test_program_length, library=my_lib)
        my_programs.set_programs(test_program_idx)
        #for i in range (test_program_length):
        #    batch.append(test_program_idx[:, i])

        # CURSOR
        cursor = Prog.Cursor(programs = my_programs, prog_idx= 0, pos = 0)
        works_bool = cursor.set_pos(0).child(0).child(0).sibling.parent.sibling.child(0).child().child().child(1).child(
            0).child().parent.parent.sibling.__repr__() == "c1"

        # TEST
        self.assertTrue(works_bool)

        test = my_programs.get_infix_pretty(prog_idx=0)
        return None

    def test_ancestors_relationships(self):
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
                        "constants"            : {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       , "const1" : 1         },
                        "constants_units"      : {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] , "const1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        , "const1" : 1.        },
                            }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [1, -2, 1], superparent_name = "y")

        # TEST PROGRAM
        test_program_str = ["mul", "mul", "M", "n2", "c", "sub", "inv", "sqrt", "sub", "const1", "div", "n2", "v", "n2",
                            "c", "cos", "div", "sub", "const1", "div", "v", "c", "div", "v", "c"]
        test_program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str])
        test_program_length = len(test_program_str)
        test_program_idx = test_program_idx[np.newaxis, :]

        # BATCH
        my_programs = Prog.VectPrograms(batch_size=1, max_time_step=test_program_length, library=my_lib)
        my_programs.set_programs(test_program_idx)

        # CURSOR
        cursor = Prog.Cursor(programs = my_programs, prog_idx= 0, pos = 0)

        # TEST has_ancestors_mask
        np.array_equal(my_programs.tokens.has_ancestors_mask, np.full((1, test_program_length), True))

        # TEST n_ancestors and ancestors_pos
        for i in range (len(test_program_str)):
            cursor.set_pos(i)
            # TEST n_ancestors
            expected_n_ancestors = my_programs.tokens.depth[0, i] + 1
            computed_n_ancestors = my_programs.tokens.n_ancestors[0, i]
            # test
            works_bool = (expected_n_ancestors == computed_n_ancestors)
            self.assertTrue(works_bool)
            # TEST ancestors_pos
            # manual ancestors search
            expected_ancestors = [cursor.pos]
            for j in range (expected_n_ancestors-1):
                cursor.set_pos(expected_ancestors[-1])
                expected_ancestors.append(cursor.parent.pos)
            expected_ancestors = np.array(expected_ancestors[::-1])
            # computed ancestors
            computed_ancestors = my_programs.tokens.ancestors_pos[0, i, :expected_n_ancestors]
            # test
            works_bool = np.array_equal(expected_ancestors, computed_ancestors)
            self.assertTrue(works_bool)
        return None

    # Test program management regarding units (units tests are in dimensional_analysis_UnitTest.py)
    def test_units_related(self):
        # LIBRARY CONFIG
        args_make_tokens = {
                        # operations
                        "op_names"             : "all",  # or ["mul", "neg", "inv", "sin"]
                        "use_protected_ops"    : True,
                        # input variables
                        "input_var_ids"        : {"z" : 0         , "v" : 1          , "t" : 2,        },
                        "input_var_units"      : {"z" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                        "input_var_complexity" : {"z" : 0.        , "v" : 1.         , "t" : 0.,       },
                        # constants
                        "constants"            : {"g" : 9.8        , "c" : 3e8       , "m" : 1e6       , "E0" : 1         },
                        "constants_units"      : {"g" : [1, -2, 0] , "c" : [1, -1, 0], "m" : [0, 0, 1] , "E0" : [2, -2, 1] },
                        "constants_complexity" : {"g" : 0.         , "c" : 0.        , "m" : 1.        , "E0" : 1.        },
                            }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [2, -2, 1], superparent_name = "y")

        # TEST PROGRAMS
        test_programs_idx = []
        test_programs_str = [
            ["add", "mul", "mul", "m" , "z", "z" , "E0",],
            ["add", "mul", "mul", "m" , "g", "z" , "E0",],
            ["add", "mul", "m"  , "n2", "z", "E0", "-" ,],
        ]
        # Using terminal token placeholder that will be replaced by '-' void token in append function
        test_programs_str = np.char.replace(test_programs_str, '-', 't')

        # Expected behavior
        expected_is_physical = np.array([False,  True, False])
        # Only expressing expectations for cases
        o = phy.UNITS_ANALYSIS_NOT_PERFORMED_CASE_CODE
        n = np.NAN
        expected_units_analysis_cases = np.array([
            [ n,  n, n, n, n, n, o],
            [ n,  n, n, n, n, n, n],
            [ n,  n, n, n, n, o, o]])
        coords_expected_no_case = np.where(expected_units_analysis_cases == o)

        # Converting into idx
        for test_program_str in test_programs_str :
            test_programs_idx.append(np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str]))
        test_programs_idx = np.array(test_programs_idx)

        # Initializing programs
        my_programs = Prog.VectPrograms(batch_size=test_programs_idx.shape[0], max_time_step=test_programs_idx.shape[1], library=my_lib)

        # Appending tokens
        for i in range (test_programs_idx.shape[1]):
            my_programs.assign_required_units(ignore_unphysical = True)
            my_programs.append(test_programs_idx[:,i])

        coords_observed_no_case = np.where(my_programs.units_analysis_cases == o)

        # Test that unphysical programs were properly detected
        bool_works = np.array_equal(my_programs.is_physical, expected_is_physical)
        self.assertEqual(bool_works, True)

        # Test that units requirements was not performed where it is useless to perform it
        bool_works = np.array_equal(coords_expected_no_case, coords_observed_no_case)
        self.assertEqual(bool_works, True)

        return None


    def test_get_family_relationship_idx_interface(self):
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
                        "constants"            : {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       , "const1" : 1         },
                        "constants_units"      : {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] , "const1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        , "const1" : 1.        },
                            }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [1, -2, 1], superparent_name = "y")

        # TEST PROGRAM
        test_program_str = ["mul", "mul", "M", "n2", "c", "sub", "inv", "sqrt", "sub", "const1", "div", "n2", "v", "n2",
                            "c", "cos", "div", "sub", "const1", "div", "v", "c", "div", "v", "c"]
        test_program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str])
        test_program_length = len(test_program_str)
        test_program_idx = test_program_idx[np.newaxis, :]

        # BATCH
        my_programs = Prog.VectPrograms(batch_size=1, max_time_step=test_program_length, library=my_lib)
        my_programs.set_programs(test_program_idx)

        # ----- TEST GET PARENT IDX -----
        no_parent_idx_filler = 8888
        # parent of token at step = last step (c) should be a div
        test_step = len(test_program_str) - 1
        parent_idx   = my_programs.get_parent_idx(my_programs.coords_of_step(test_step))
        parent_token = my_lib.lib_tokens[parent_idx][0]
        works_bool = parent_token.name == "div"
        self.assertTrue(works_bool)
        # parent of token at step = 1 (mul) should be a mul
        test_step = 1
        parent_idx   = my_programs.get_parent_idx(my_programs.coords_of_step(test_step))
        parent_token = my_lib.lib_tokens[parent_idx][0]
        works_bool = parent_token.name == "mul"
        self.assertTrue(works_bool)
        # parent of token at step = 0 (mul) should no exit (superparent)
        test_step = 0
        parent_idx   = my_programs.get_parent_idx(coords = my_programs.coords_of_step(test_step),
                                                  no_parent_idx_filler = no_parent_idx_filler)
        works_bool = (parent_idx == no_parent_idx_filler)
        self.assertTrue(works_bool)

        # ----- TEST GET SIBLING IDX -----
        no_sibling_idx_filler = 8888
        # sibling of token at step = last step (c) should be a v
        test_step = len(test_program_str) - 1
        sibling_idx   = my_programs.get_sibling_idx(my_programs.coords_of_step(test_step))
        sibling_token = my_lib.lib_tokens[sibling_idx][0]
        works_bool = sibling_token.name == "v"
        self.assertTrue(works_bool)
        # sibling of token at step = 0 should not exist
        test_step = 0
        sibling_idx   = my_programs.get_sibling_idx(coords = my_programs.coords_of_step(test_step),
                                                   no_sibling_idx_filler = no_parent_idx_filler)
        works_bool = (sibling_idx == no_sibling_idx_filler)
        self.assertTrue(works_bool)

        # ----- TEST GET ANCESTORS IDX ----
        #is_ancestor = my_programs.get_is_ancestor()

        return None

if __name__ == '__main__':
    unittest.main(verbosity=2)
