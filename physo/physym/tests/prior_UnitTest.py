import unittest
import numpy as np
import warnings

# Internal imports
from physo.physym import library as Lib
from physo.physym import program as Prog
from physo.physym import prior as Prior
from physo.physym import vect_programs as VProg
from physo.physym import token as Tok
from physo.physym import functions as Func

class PriorTest(unittest.TestCase):

    def test_UniformArityPrior(self):
        # Test case
        args_make_tokens = {
                        # operations
                        "op_names"             : ["mul", "add", "neg", "inv", "sin"],
                        "use_protected_ops"    : False,
                        # input variables
                        "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                        "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                        "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                        # constants
                        "constants"            : {"pi" : np.pi     ,  },
                        "constants_units"      : {"pi" : [0, 0, 0] ,  },
                        "constants_complexity" : {"pi" : 0.        ,  },
                        # free constants
                        "free_constants"            : {"M"             , "c"             , },
                        "free_constants_init_val"   : {"M" : 1.        , "c" : 1.        , },
                        "free_constants_units"      : {"M" : [0, 0, 1] , "c" : [1, -1, 0] , },
                        "free_constants_complexity" : {"M" : 1.        , "c" : 1.        , },
                           }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [1, -2, 1], superparent_name = "y")
        my_programs = VProg.VectPrograms(batch_size=1, max_time_step=8, library=my_lib, n_realizations=1)
        # Creation
        try:
            my_prior = Prior.UniformArityPrior (library = my_lib, programs = my_programs)
        except:
            self.fail("Prior creation failed.")
        # Test
        bool_works = my_prior()[0,0] == my_prior()[0,1] == (1./2)
        self.assertEqual(bool_works, True)
        bool_works = my_prior()[0,2] == my_prior()[0,3]  == my_prior()[0,4] == (1./3)
        self.assertEqual(bool_works, True)
        bool_works = my_prior()[0,5] == my_prior()[0,6] == (1./6)
        self.assertEqual(bool_works, True)

    def test_HardLengthPrior(self):

        # ------- TEST CASE -------
        # Library
        args_make_tokens = {
                        # operations
                        "op_names"             : ["mul", "add", "neg", "inv", "cos"],
                        "use_protected_ops"    : False,
                        # input variables
                        "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                        "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                        "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                        # constants
                        "constants"            : {"pi" : np.pi     ,  },
                        "constants_units"      : {"pi" : [0, 0, 0] ,  },
                        "constants_complexity" : {"pi" : 0.        ,  },
                        # free constants
                        "free_constants"            : {"M"             , "c"             , },
                        "free_constants_init_val"   : {"M" : 1.        , "c" : 1.        , },
                        "free_constants_units"      : {"M" : [0, 0, 1] , "c" : [1, -1, 0] , },
                        "free_constants_complexity" : {"M" : 1.        , "c" : 1.        , },
                           }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [1, -2, 1], superparent_name = "y")
        # Programs test case
        max_length = 5
        min_length = 3
        test_case_str = np.array([
            # 0      1      2      3      4
            ["add", "cos", "x"  , "cos", "c"  ],  # -> should begin enforcing arity == 0 tokens at pos = 3
                                                  # -> should begin enforcing arity <= 1 tokens at pos = 1
                                                  # -> should enforce arity >= 1 tokens until pos = 0

            ["cos", "cos", "cos", "cos", "x"  ],  # -> should begin enforcing arity == 0 tokens at pos = 3
                                                  # -> should begin enforcing arity <= 1 tokens at pos = 2
                                                  # -> should enforce arity >= 1 tokens until pos = 1

            ["add", "add", "x"  , "pi" , "c"  ],  # -> should begin enforcing arity == 0 tokens at pos = 1
                                                  # -> should begin enforcing arity <= 1 tokens at pos = 1
                                                  # -> should enforce arity >= 1 tokens until pos = 0

            ["add", "x"  , "c"  , "-"  , "-"  ],  # -> should begin enforcing arity == 0 tokens at pos = inf
                                                  # -> should begin enforcing arity <= 1 tokens at pos = inf
                                                  # -> should enforce arity >= 1 tokens until pos = 0

            ["add", "cos", "x"  , "c"  , "-"  ],  # -> should begin enforcing arity == 0 tokens at pos = inf
                                                  # -> should begin enforcing arity <= 1 tokens at pos = 1
                                                  # -> should enforce arity >= 1 tokens until pos = 0

            ["cos", "add", "x"  , "v"  , "-"  ],  # -> should begin enforcing arity == 0 tokens at pos = inf
                                                  # -> should begin enforcing arity <= 1 tokens at pos = 1
                                                  # -> should enforce arity >= 1 tokens until pos = 1
        ])
        pos_begin_max_arity_is_0 = np.array([3, 3, 1, np.inf, np.inf, np.inf])
        pos_begin_max_arity_is_1 = np.array([1, 2, 1, np.inf, 1     , 1     ])
        pos_end_min_arity_is_1   = np.array([0, 1, 0, 0     , 0     , 1     ])

        # Using a valid placeholder existing in the library that will be ignored anyway instead of '-'
        test_case_str = np.where(test_case_str == "-", "x", test_case_str)

        # Creating idx that will be appended
        n_progs, n_steps = test_case_str.shape
        test_case = np.zeros((n_progs, n_steps)).astype(int)
        for i in range (n_progs):
            for j in range (n_steps):
                tok_str = test_case_str[i,j]
                test_case[i,j] = my_lib.lib_name_to_idx[tok_str]

        # VectPrograms
        my_programs = VProg.VectPrograms(batch_size = n_progs, max_time_step=n_steps, library=my_lib, n_realizations=1)

        # ------- TEST CREATION -------
        try:
            my_prior = Prior.HardLengthPrior (library = my_lib, programs = my_programs, min_length = min_length, max_length = max_length)
        except:
            self.fail("Prior creation failed.")

        # # ------- TEST ASSERTION -------
        with self.assertRaises(TypeError, msg = "min_length must be cast-able to a float"):
            my_prior = Prior.HardLengthPrior (library = my_lib, programs = my_programs, min_length = 'n_steps', max_length = max_length)
        with self.assertRaises(TypeError, msg = "max_length must be cast-able to a float"):
            my_prior = Prior.HardLengthPrior (library = my_lib, programs = my_programs, min_length = n_steps  , max_length = 'max_length')
        with self.assertRaises(AssertionError, msg = "min_length must be such as: min_length <= max_time_step"):
            my_prior = Prior.HardLengthPrior (library = my_lib, programs = my_programs, min_length = n_steps+1, max_length = max_length)
        with self.assertRaises(AssertionError, msg = "max_length must be such as: max_length <= max_time_step"):
            my_prior = Prior.HardLengthPrior (library = my_lib, programs = my_programs, min_length = min_length, max_length = n_steps+1)
        with self.assertRaises(AssertionError, msg = "max_length must be such as: max_length >= 1"):
            my_prior = Prior.HardLengthPrior (library = my_lib, programs = my_programs, min_length = min_length, max_length = 0)
        with self.assertRaises(AssertionError, msg = "Must be: min_length <= max_length"):
            my_prior = Prior.HardLengthPrior (library = my_lib, programs = my_programs, min_length = 8, max_length = 2)

        # ------- TEST PRIOR -------
        prior = my_prior()

        for step in range (n_steps):

             # TESTS
            for i_prog in range (n_progs):

                # If step > max step : tokens with arity > 0 are forbidden
                if step > pos_begin_max_arity_is_0[i_prog]:
                    works_bool = (prior[i_prog][my_lib.get_choosable_prop("arity") > 0]).all() == False
                    self.assertTrue(works_bool, msg = "Test case prog %i failed at step = %i "%(i_prog,step ))

                # If step > max step : tokens with arity > 1 are forbidden
                if step > pos_begin_max_arity_is_1[i_prog]:
                    works_bool = (prior[i_prog][my_lib.get_choosable_prop("arity") > 1]).all() == False
                    self.assertTrue(works_bool, msg = "Test case prog %i failed at step = %i "%(i_prog,step ))

                # Tokens with arity < 1 are allowed   while step > min_step
                # Tokens with arity < 1 are forbidden while step <= min_step
                works_bool = (prior[i_prog][my_lib.get_choosable_prop("arity") < 1]).all() == (step > pos_end_min_arity_is_1[i_prog])
                self.assertTrue(works_bool, msg = "Test case prog %i failed at step = %i "%(i_prog,step ))

            # NEXT STEP
            my_programs.append(test_case[:, step])
            display = my_lib.lib_name[my_programs.tokens.idx]
            prior = my_prior()

    def test_SoftLengthPrior(self):

        # ------- TEST CASE -------
        # Library
        args_make_tokens = {
                        # operations
                        "op_names"             : ["mul", "add", "neg", "inv", "cos"],
                        "use_protected_ops"    : False,
                        # input variables
                        "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                        "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                        "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                        # constants
                        "constants"            : {"pi" : np.pi     ,  },
                        "constants_units"      : {"pi" : [0, 0, 0] ,  },
                        "constants_complexity" : {"pi" : 0.        ,  },
                        # free constants
                        "free_constants"            : {"M"             , "c"             , },
                        "free_constants_init_val"   : {"M" : 1.        , "c" : 1.        , },
                        "free_constants_units"      : {"M" : [0, 0, 1] , "c" : [1, -1, 0] , },
                        "free_constants_complexity" : {"M" : 1.        , "c" : 1.        , },
                           }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [1, -2, 1], superparent_name = "y")
        # Programs test case
        scale = 1.
        length_loc  = 3.
        step_loc = length_loc - 1
        test_case_str = np.array([
        #     0      1      2      3      4
            ["add", "x"  , "cos", "cos", "c"  ],  # prog 0
            ["cos", "x"  , "-"  , "-"  , "-"  ],  # prog 1
        ])

        # --- Expected prior values for these programs assuming length_loc = 3. ---
        # gaussian_vals[step_loc] = 1.
        #
        steps         = np.array([ 0               , 1               , 2               , 3               , 4               , ])
        gaussian_vals = np.exp(-(steps - step_loc)**2 / (2*scale) )
        # No need to discourage terminal tokens after 1st token as they would not result in end of prog before loc
        prog_0_arity_0_true    = [gaussian_vals[0], 1.              , 1.              , 1.              , 1.              , ]
        # Non-terminal tokens should be discouraged after loc
        prog_0_arity_non0_true = [1.              , 1.              , 1.              , gaussian_vals[3], gaussian_vals[4], ]
        # Should discourage terminal tokens that would result in end of prog before loc
        prog_1_arity_0_true    = [gaussian_vals[0], gaussian_vals[1], ]
        # Should be neutral toward non-terminal tokens before loc
        prog_1_arity_non0_true = [1.              , 1.              , ]

        # Using a valid placeholder existing in the library that will be ignored anyway instead of '-'
        test_case_str = np.where(test_case_str == "-", "x", test_case_str)

        # Creating idx that will be appended
        n_progs, n_steps = test_case_str.shape
        test_case = np.zeros((n_progs, n_steps)).astype(int)
        for i in range (n_progs):
            for j in range (n_steps):
                tok_str = test_case_str[i,j]
                test_case[i,j] = my_lib.lib_name_to_idx[tok_str]

        # VectPrograms
        my_programs = VProg.VectPrograms(batch_size = n_progs, max_time_step=n_steps, library=my_lib, n_realizations=1)

        # ------- TEST CREATION -------
        try:
            my_prior = Prior.SoftLengthPrior (library = my_lib, programs = my_programs, length_loc= length_loc, scale= scale)
        except:
            self.fail("Prior creation failed.")

        # ------- TEST ASSERTION -------
        with self.assertRaises(ValueError, ):
            my_prior = Prior.SoftLengthPrior (library = my_lib, programs = my_programs, length_loc = 'hi', scale= 'hi')

        # ------- TEST PRIOR -------
        prior = my_prior()

        for step in range (n_steps):

            # TESTS
            # Prog 0
            prog_0_arity_0    = prior[0][my_lib.get_choosable_prop("arity") == 0]
            works_bool = (prog_0_arity_0 == prog_0_arity_0_true[step]).all()
            self.assertTrue(works_bool, msg = "prog_0_arity_0 arity tokens prior has wrong values at step = %i"%(step))

            prog_0_arity_non0 = prior[0][my_lib.get_choosable_prop("arity") != 0]
            works_bool = (prog_0_arity_non0 == prog_0_arity_non0_true[step]).all()
            self.assertTrue(works_bool, msg = "prog_0_arity_non0    tokens prior has wrong values at step = %i"%(step))

            # Prog 1 : test for step < 2
            if step < 2:
                prog_1_arity_0    = prior[1][my_lib.get_choosable_prop("arity") == 0]
                works_bool = (prog_1_arity_0 == prog_1_arity_0_true[step]).all()
                self.assertTrue(works_bool, msg = "prog_1_arity_0 arity tokens prior has wrong values at step = %i"%(step))

                prog_1_arity_non0 = prior[1][my_lib.get_choosable_prop("arity") != 0]
                works_bool = (prog_1_arity_non0 == prog_1_arity_non0_true[step]).all()
                self.assertTrue(works_bool, msg = "prog_1_arity_non0    tokens prior has wrong values at step = %i"%(step))

            # NEXT STEP
            my_programs.append(test_case[:, step])
            display = my_lib.lib_name[my_programs.tokens.idx]
            prior = my_prior()

    def test_RelationshipConstraintPrior(self):

        # -------------------- LIB TEST CASE --------------------
        args_make_tokens = {
                        # operations
                        "op_names"             : ["add", "sub", "neg", "inv", "n2", "sqrt", "cos", "sin", "exp", "log"],
                        "use_protected_ops"    : False,
                        # input variables
                        "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                        "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                        "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                        # constants
                        "constants"            : {"pi" : np.pi     ,  },
                        "constants_units"      : {"pi" : [0, 0, 0] ,  },
                        "constants_complexity" : {"pi" : 0.        ,  },
                        # free constants
                        "free_constants"            : {"M"             , "c"             , },
                        "free_constants_init_val"   : {"M" : 1.        , "c" : 1.        , },
                        "free_constants_units"      : {"M" : [0, 0, 1] , "c" : [1, -1, 0] , },
                        "free_constants_complexity" : {"M" : 1.        , "c" : 1.        , },
                           }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [1, -2, 1], superparent_name = "y")
        my_programs = VProg.VectPrograms(batch_size=4, max_time_step=10, library=my_lib, n_realizations=1)

        # -------------------- CREATION --------------------
        try:
            my_prior = Prior.RelationshipConstraintPrior (library = my_lib, programs = my_programs,
                                                effectors = ["exp" , "log", "sin", "exp", "sub"],
                                                targets   = ["log" , "exp", "sin", "cos", "neg"],
                                                relationship = "child",)
        except:
            self.fail("Prior creation failed.")

        # ---------- ASSERTIONS ----------
        # Assertion test : relationship
        with self.assertRaises(AssertionError, msg="relationship arg must be valid"):
            my_prior = Prior.RelationshipConstraintPrior (library = my_lib, programs = my_programs,
                                                effectors = ["n2" , "sin"],
                                                relationship = "not_a_valid_relationship",
                                                targets = ["sqrt" , "sin"])

        # Assertion test : effectors
        with self.assertRaises(AssertionError, msg="effectors arg must be valid"):
            my_prior = Prior.RelationshipConstraintPrior (library = my_lib, programs = my_programs,
                                                effectors = "n2",
                                                relationship = "child",
                                                targets = ["sqrt" , "sin"])
        with self.assertRaises(AssertionError, msg="effectors arg must be valid"):
            my_prior = Prior.RelationshipConstraintPrior (library = my_lib, programs = my_programs,
                                                effectors = [["add"], ["n2"]],
                                                relationship = "child",
                                                targets = ["sqrt" , "sin"])
        # Assertion test : effectors
        with self.assertRaises(AssertionError, msg="effectors arg must be valid"):
            my_prior = Prior.RelationshipConstraintPrior (library = my_lib, programs = my_programs,
                                                effectors = ["token_not_in_lib", "token_not_in_lib"],
                                                relationship = "child",
                                                targets = ["sqrt" , "sin"])

        # Assertion test : targets
        with self.assertRaises(AssertionError, msg="targets arg must be valid"):
            my_prior = Prior.RelationshipConstraintPrior (library = my_lib, programs = my_programs,
                                                targets = "n2",
                                                relationship = "child",
                                                effectors = ["sqrt" , "sin"])
        with self.assertRaises(AssertionError, msg="targets arg must be valid"):
            my_prior = Prior.RelationshipConstraintPrior (library = my_lib, programs = my_programs,
                                                targets = [["add"], ["n2"]],
                                                relationship = "child",
                                                effectors = ["sqrt" , "sin"])
        with self.assertRaises(AssertionError, msg="targets arg must be valid"):
            my_prior = Prior.RelationshipConstraintPrior (library = my_lib, programs = my_programs,
                                                targets = ["token_not_in_lib", "token_not_in_lib"],
                                                relationship = "child",
                                                effectors = ["sqrt" , "sin"])

        # Assertion test : targets and effectors
        with self.assertRaises(AssertionError, msg="targets ad effectors must have the same size"):
            my_prior = Prior.RelationshipConstraintPrior (library = my_lib, programs = my_programs,
                                                effectors=["sqrt", "sin"],
                                                targets = ["n2"],
                                                relationship = "child",)

        # Assertion test : max_nb_violations
        try:
            my_prior = Prior.RelationshipConstraintPrior (library = my_lib, programs = my_programs,
                                                effectors=["cos" , "cos"],
                                                targets = ["cos" , "sin"],
                                                relationship = "descendant",)
        except:
            self.fail("Prior creation failed.")
        with self.assertRaises(AssertionError, msg="max_nb_violations and effectors must have the same size"):
            my_prior = Prior.RelationshipConstraintPrior (library = my_lib, programs = my_programs,
                                                effectors=["cos" , "cos"],
                                                targets = ["cos" , "sin"],
                                                relationship = "descendant",
                                                max_nb_violations = [1, 0, 1])
        with self.assertRaises(AssertionError, msg="max_nb_violations should contain positive integers"):
            my_prior = Prior.RelationshipConstraintPrior (library = my_lib, programs = my_programs,
                                                effectors=["cos" , "cos"],
                                                targets = ["cos" , "sin"],
                                                relationship = "descendant",
                                                max_nb_violations = [-3, 0,])
        with self.assertRaises(AssertionError, msg="max_nb_violations should contain positive integers"):
            my_prior = Prior.RelationshipConstraintPrior (library = my_lib, programs = my_programs,
                                                effectors=["cos" , "cos"],
                                                targets = ["cos" , "sin"],
                                                relationship = "descendant",
                                                max_nb_violations = [1.439, 0,])
        with self.assertRaises(AssertionError, msg="max_nb_violations have shape of effectors"):
            my_prior = Prior.RelationshipConstraintPrior (library = my_lib, programs = my_programs,
                                                effectors=["cos" , "cos"],
                                                targets = ["cos" , "sin"],
                                                relationship = "descendant",
                                                max_nb_violations = [[1], [0],])

        # -------------------- TEST WITH CHILD RELATIONSHIP (1) --------------------

        test_progs_str = np.array([
            ["add" , "x"   , "exp" , ],
            ["cos" , "sub" , "v"   , ],
            ["add" , "add" , "sin" , ],
            ["sub" , "cos" , "cos" , ],
            ["add" , "add" , "add" , ],
        ])
        # max_time_steps at least must be at least 6 as some of these test programs require 3 dummies
        test_progs = []
        for prog_str in test_progs_str:
            prog = [my_lib.lib_name_to_idx[tok_str] for tok_str in prog_str]
            test_progs.append(prog)
        test_progs = np.array(test_progs)
        my_programs = VProg.VectPrograms(batch_size=test_progs_str.shape[0], max_time_step=10, library=my_lib, n_realizations=1)
        my_programs.set_programs(test_progs)

        # Test Case --------
        my_prior = Prior.RelationshipConstraintPrior (library = my_lib, programs = my_programs,
                                                effectors = ["exp" , "log", "sin", "exp", "sub"],
                                                targets   = ["log" , "exp", "sin", "cos", "neg"],
                                                relationship = "child",)
        # Expected forbidden tokens
        expected_forbidden_tokens = [
            ["log", "cos"],
            ["neg"],
            ["sin"],
            [], # parent of next token is last cos -> no constraints
            [], # parent of next token is 1st add -> no constraints
        ]

        # Test --------
        mask_prob = my_prior()
        for i, mask_prob_i in enumerate(mask_prob):
            mask_is_forbidden = np.logical_not(mask_prob_i)
            idx = np.arange(my_lib.n_choices)[mask_is_forbidden]
            # print("prog", my_lib.lib_name[my_programs.tokens.idx][i])
            # print("forbidding:", my_lib.lib_name[idx])
            found_forbidden_tokens = np.sort(my_lib.lib_name[idx]).astype(str)
            expected = np.sort(expected_forbidden_tokens[i]).astype(str)
            bool_works = np.array_equal(found_forbidden_tokens, expected)
            self.assertEqual(bool_works, True)

        # -------------------- TEST WITH CHILD RELATIONSHIP (2) --------------------

        test_progs_str = np.array([
            [], # at step = 0
        ])
        # max_time_steps at least must be at least 6 as some of these test programs require 3 dummies
        test_progs = []
        for prog_str in test_progs_str:
            prog = [my_lib.lib_name_to_idx[tok_str] for tok_str in prog_str]
            test_progs.append(prog)
        test_progs = np.array(test_progs)
        my_programs = VProg.VectPrograms(batch_size=test_progs_str.shape[0], max_time_step=10, library=my_lib, n_realizations=1)
        my_programs.set_programs(test_progs)

        # Test Case --------
        my_prior = Prior.RelationshipConstraintPrior (library = my_lib, programs = my_programs,
                                                effectors = ["exp" , "log", "sin", "exp", "sub"],
                                                targets   = ["log" , "exp", "sin", "cos", "neg"],
                                                relationship = "child",)
        # Expected forbidden tokens
        expected_forbidden_tokens = [
            [],
        ]

        # Test --------
        mask_prob = my_prior()
        for i, mask_prob_i in enumerate(mask_prob):
            mask_is_forbidden = np.logical_not(mask_prob_i)
            idx = np.arange(my_lib.n_choices)[mask_is_forbidden]
            # print("prog", my_lib.lib_name[my_programs.tokens.idx][i])
            # print("forbidding:", my_lib.lib_name[idx])
            found_forbidden_tokens = np.sort(my_lib.lib_name[idx]).astype(str)
            expected = np.sort(expected_forbidden_tokens[i]).astype(str)
            bool_works = np.array_equal(found_forbidden_tokens, expected)
            self.assertEqual(bool_works, True)

        # -------------------- TEST WITH SIBLING RELATIONSHIP --------------------

        test_progs_str = np.array([
            ["add" , "x"   , "exp" , ],
            ["cos" , "sub" , "v"   , ],
            ["add" , "add" , "x"   , ],
            ["add" , "exp" , "x"   , ],
            ["add" , "x"   , "x"   , ],

        ])
        # max_time_steps at least must be at least 6 as some of these test programs require 3 dummies
        test_progs = []
        for prog_str in test_progs_str:
            prog = [my_lib.lib_name_to_idx[tok_str] for tok_str in prog_str]
            test_progs.append(prog)
        test_progs = np.array(test_progs)
        my_programs = VProg.VectPrograms(batch_size=test_progs_str.shape[0], max_time_step=10, library=my_lib, n_realizations=1)
        my_programs.set_programs(test_progs)

        # Test Case --------
        my_prior = Prior.RelationshipConstraintPrior (library = my_lib, programs = my_programs,
                                                effectors = ["x"   , "x"  , "exp", "add"],
                                                targets   = ["v"   , "exp", "sin", "neg"],
                                                relationship = "sibling",)
        # Expected forbidden tokens
        expected_forbidden_tokens = [
            [], # no sibling -> no constraints
            [],
            ["v", "exp"],
            ["sin"], # parent of next token is last cos -> no constraints
            [], # finished prog -> no constraints
        ]

        # Test --------
        mask_prob = my_prior()
        for i, mask_prob_i in enumerate(mask_prob):
            mask_is_forbidden = np.logical_not(mask_prob_i)
            idx = np.arange(my_lib.n_choices)[mask_is_forbidden]
            # print("prog", my_lib.lib_name[my_programs.tokens.idx][i])
            # print("forbidding:", my_lib.lib_name[idx])
            found_forbidden_tokens = np.sort(my_lib.lib_name[idx]).astype(str)
            expected = np.sort(expected_forbidden_tokens[i]).astype(str)
            bool_works = np.array_equal(found_forbidden_tokens, expected)
            self.assertEqual(bool_works, True)

        # -------------------- TEST WITH ANCESTOR RELATIONSHIP --------------------

        test_progs_str = np.array([
            ["add" , "x"   , "exp" , ],
            ["cos" , "add" , "v"   , ],
            ["add" , "add" , "sin" , ],
            ["add" , "cos" , "cos" , ],
            ["exp" , "exp" , "exp" , ],
            ["sub" , "x"   , "log" , ],
        ])
        # max_time_steps at least must be at least 6 as some of these test programs require 3 dummies
        test_progs = []
        for prog_str in test_progs_str:
            prog = [my_lib.lib_name_to_idx[tok_str] for tok_str in prog_str]
            test_progs.append(prog)
        test_progs = np.array(test_progs)
        my_programs = VProg.VectPrograms(batch_size=test_progs_str.shape[0], max_time_step=10, library=my_lib, n_realizations=1)
        my_programs.set_programs(test_progs)

        # Test Case --------
        my_prior = Prior.RelationshipConstraintPrior (library = my_lib, programs = my_programs,
                                                effectors = ["cos" , "cos", "exp", "exp", "sub"],
                                                targets   = ["cos" , "sin", "cos", "sin", "v"  ],
                                                relationship = "descendant",
                                                max_nb_violations = [1, 0, 0, 2, 0],
                                                      )
        # Expected forbidden tokens
        expected_forbidden_tokens = [
            ["cos"], # 0 tolerance on exp ancestors for cos, 2 ancestors tolerance on exp for sin -> it is allowed
            ["sin"], # 0 tolerance on cos ancestors for sin, 1 ancestor tolerance on cos for cos -> it is allowed
            [],      # add, add, sin ancestors can have any tokens as descendants
            ["cos", "sin"], # 2 cos ancestors but tolerance is 1 for cos -> forbidding it, tolerance for sin is 0 -> forbidding it
            ["cos", "sin"], # 3 exp ancestors but tolerance is 2 for sin -> forbidding it, tolerance for cos is 0 -> forbidding it
            ["v"], # 0 tolerance on sub being ancestors of v
        ]

        # Test --------
        mask_prob = my_prior()
        for i, mask_prob_i in enumerate(mask_prob):
            mask_is_forbidden = np.logical_not(mask_prob_i)
            idx = np.arange(my_lib.n_choices)[mask_is_forbidden]
            # print("prog", my_lib.lib_name[my_programs.tokens.idx][i])
            # print("forbidding:", my_lib.lib_name[idx])
            found_forbidden_tokens = np.sort(my_lib.lib_name[idx]).astype(str)
            expected = np.sort(expected_forbidden_tokens[i]).astype(str)
            bool_works = np.array_equal(found_forbidden_tokens, expected)
            self.assertEqual(bool_works, True)

        return None

    def test_NoUselessInversePrior(self):
        # -------------------- LIB TEST CASE --------------------
        args_make_tokens = {
                        # operations
                        "op_names"             : ["add", "sub", "neg", "inv", "n2", "sqrt", "cos", "sin", "exp", "log"],
                        "use_protected_ops"    : False,
                        # input variables
                        "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                        "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                        "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                        # constants
                        "constants"            : {"pi" : np.pi     ,  },
                        "constants_units"      : {"pi" : [0, 0, 0] ,  },
                        "constants_complexity" : {"pi" : 0.        ,  },
                        # free constants
                        "free_constants"            : {"M"             , "c"             , },
                        "free_constants_init_val"   : {"M" : 1.        , "c" : 1.        , },
                        "free_constants_units"      : {"M" : [0, 0, 1] , "c" : [1, -1, 0] , },
                        "free_constants_complexity" : {"M" : 1.        , "c" : 1.        , },
                           }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [1, -2, 1], superparent_name = "y")
        my_programs = VProg.VectPrograms(batch_size=4, max_time_step=10, library=my_lib, n_realizations=1)

        # -------------------- CREATION --------------------
        try:
            my_prior = Prior.NoUselessInversePrior (library = my_lib, programs = my_programs, )
        except:
            self.fail("Prior creation failed.")

        # -------------------- TEST --------------------
        # Test cases progs
        test_progs_str = np.array([
            ["add", "cos" , "n2"  , "neg"   , ],
            ["add", "add" , "n2"  , "sqrt"  , ],  # Previously violated prior
            ["add", "cos" , "add" , "n2"    , ],  # Testing f -> f-1 and f-1 -> f
            ["add", "sub" , "sin" , "inv"   , ],
            ["add", "add" , "cos" , "x"     , ],  # No constraints expected
            ["add", "sub" , "x"   , "log"   , ],
            ["add", "add" , "v"   , "exp"   , ],
            ["add", "add" , "x"   , "x"     , ],  # Already finished program
        ])
        # max_time_steps at least must be at least 6 as some of these test programs require 3 dummies
        test_progs = []
        for prog_str in test_progs_str:
            prog = [my_lib.lib_name_to_idx[tok_str] for tok_str in prog_str]
            test_progs.append(prog)
        test_progs = np.array(test_progs)
        my_programs = VProg.VectPrograms(batch_size=test_progs_str.shape[0], max_time_step=10, library=my_lib, n_realizations=1)
        my_programs.set_programs(test_progs)

        # Test Case --------
        my_prior = Prior.NoUselessInversePrior (library = my_lib, programs = my_programs, )

        # Expected forbidden tokens for each prog
        expected_forbidden_tokens = [
            ["neg"  ,],
            ["n2"   ,],
            ["sqrt" ,],
            ["inv"  ,],
            [],
            ["exp"],
            ["log"],
            [],
        ]

        # Test --------
        mask_prob = my_prior()
        for i, mask_prob_i in enumerate(mask_prob):
            mask_is_forbidden = np.logical_not(mask_prob_i)
            idx = np.arange(my_lib.n_choices)[mask_is_forbidden]
            # print("prog", my_lib.lib_name[my_programs.tokens.idx][i])
            # print("forbidding:", my_lib.lib_name[idx])
            found_forbidden_tokens = np.sort(my_lib.lib_name[idx]).astype(str)
            expected = np.sort(expected_forbidden_tokens[i]).astype(str)
            bool_works = np.array_equal(found_forbidden_tokens, expected)
            self.assertEqual(bool_works, True)


    def test_NestedFunctions(self):

        # -------------------- LIB TEST CASE --------------------
        args_make_tokens = {
                        # operations
                        "op_names"             : ["add", "sub", "neg", "inv", "n2", "sqrt", "cos", "sin", "exp", "log"],
                        "use_protected_ops"    : False,
                        # input variables
                        "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                        "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                        "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                        # constants
                        "constants"            : {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                        "constants_units"      : {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] },
                        "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                            }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [1, -2, 1], superparent_name = "y")
        my_programs = VProg.VectPrograms(batch_size=4, max_time_step=10, library=my_lib, n_realizations=1)

        # -------------------- CREATION --------------------
        try:
            my_prior = Prior.NestedFunctions (library = my_lib, programs = my_programs,
                                              functions = ["cos" , "sin", "exp", "log",],
                                              max_nesting = 1)
        except:
            self.fail("Prior creation failed.")

        # ---------- ASSERTIONS ----------

        # Assertion test : functions
        with self.assertRaises(AssertionError, msg="functions arg must be valid"):
            my_prior = Prior.NestedFunctions(library=my_lib, programs=my_programs,
                                             functions="n2",
                                             max_nesting=1, )
        with self.assertRaises(AssertionError, msg="functions arg must be valid"):
            my_prior = Prior.NestedFunctions(library=my_lib, programs=my_programs,
                                             functions=[["add"], ["n2"]],
                                             max_nesting=1, )
        with self.assertRaises(AssertionError, msg="functions arg must be valid"):
            my_prior = Prior.NestedFunctions(library=my_lib, programs=my_programs,
                                             functions=["token_not_in_lib", "token_not_in_lib"],
                                             max_nesting=1, )

        # Assertion test : max_nesting
        with self.assertRaises(AssertionError, msg="max_nesting arg must be valid"):
            my_prior = Prior.NestedFunctions(library=my_lib, programs=my_programs,
                                             functions = ["cos" , "sin", "exp", "log",],
                                             max_nesting = -1, )
        with self.assertRaises(AssertionError, msg="max_nesting arg must be valid"):
            my_prior = Prior.NestedFunctions(library=my_lib, programs=my_programs,
                                             functions = ["cos" , "sin", "exp", "log",],
                                             max_nesting = 0, )
        with self.assertRaises(AssertionError, msg="max_nesting arg must be valid"):
            my_prior = Prior.NestedFunctions(library=my_lib, programs=my_programs,
                                             functions = ["cos" , "sin", "exp", "log",],
                                             max_nesting = 1.4, )

        # -------------------- TEST WITH max_nesting = 1 --------------------

        test_progs_str = np.array([
            ["cos" , "neg" , "sin" , ],  # Previously violated prior
            ["add" , "cos" , "cos" , ],  # Previously violated prior
            ["cos" , "add" , "x"   , ],
            ["sub" , "sin" , "x"   , ],
            ["add" , "x"   , "x"   , ],  # Already finished program
        ])
        # max_time_steps at least must be at least 6 as some of these test programs require 3 dummies
        test_progs = []
        for prog_str in test_progs_str:
            prog = [my_lib.lib_name_to_idx[tok_str] for tok_str in prog_str]
            test_progs.append(prog)
        test_progs = np.array(test_progs)
        my_programs = VProg.VectPrograms(batch_size=test_progs_str.shape[0], max_time_step=10, library=my_lib, n_realizations=1)
        my_programs.set_programs(test_progs)

        # Test Case --------
        my_prior = Prior.NestedFunctions (library = my_lib, programs = my_programs,
                                          functions = ["cos" , "sin", "exp", "log",],
                                          max_nesting = 1)
        # Expected forbidden tokens
        expected_forbidden_tokens = [
            ["cos" , "sin", "exp", "log",],
            ["cos" , "sin", "exp", "log",],
            ["cos" , "sin", "exp", "log",],
            [],
            [],
        ]

        # Test --------
        mask_prob = my_prior()
        for i, mask_prob_i in enumerate(mask_prob):
            mask_is_forbidden = np.logical_not(mask_prob_i)
            idx = np.arange(my_lib.n_choices)[mask_is_forbidden]
            # print("prog", my_lib.lib_name[my_programs.tokens.idx][i])
            # print("forbidding:", my_lib.lib_name[idx])
            found_forbidden_tokens = np.sort(my_lib.lib_name[idx]).astype(str)
            expected = np.sort(expected_forbidden_tokens[i]).astype(str)
            bool_works = np.array_equal(found_forbidden_tokens, expected)
            self.assertEqual(bool_works, True)

    # -------------------- TEST WITH max_nesting = 2 --------------------

        test_progs_str = np.array([
            ["cos" , "neg" , "sin" , ],
            ["add" , "cos" , "cos" , ],
            ["cos" , "add" , "x"   , ],
            ["sub" , "sin" , "x"   , ],
            ["add" , "x"   , "x"   , ],  # Already finished program
            ["add" , "x"   , "sin" , ],
        ])
        # max_time_steps at least must be at least 6 as some of these test programs require 3 dummies
        test_progs = []
        for prog_str in test_progs_str:
            prog = [my_lib.lib_name_to_idx[tok_str] for tok_str in prog_str]
            test_progs.append(prog)
        test_progs = np.array(test_progs)
        my_programs = VProg.VectPrograms(batch_size=test_progs_str.shape[0], max_time_step=10, library=my_lib, n_realizations=1)
        my_programs.set_programs(test_progs)

        # Test Case --------
        my_prior = Prior.NestedFunctions (library = my_lib, programs = my_programs,
                                          functions = ["cos" , "sin", "exp", "log",],
                                          max_nesting = 2)
        # Expected forbidden tokens
        expected_forbidden_tokens = [
            ["cos" , "sin", "exp", "log",],
            ["cos" , "sin", "exp", "log",],
            [],
            [],
            [],
            [],
        ]

        # Test --------
        mask_prob = my_prior()
        for i, mask_prob_i in enumerate(mask_prob):
            mask_is_forbidden = np.logical_not(mask_prob_i)
            idx = np.arange(my_lib.n_choices)[mask_is_forbidden]
            # print("prog", my_lib.lib_name[my_programs.tokens.idx][i])
            # print("forbidding:", my_lib.lib_name[idx])
            found_forbidden_tokens = np.sort(my_lib.lib_name[idx]).astype(str)
            expected = np.sort(expected_forbidden_tokens[i]).astype(str)
            bool_works = np.array_equal(found_forbidden_tokens, expected)
            self.assertEqual(bool_works, True)

    def test_NestedTrigonometryPrior(self):

        # -------------------- LIB TEST CASE --------------------
        args_make_tokens = {
                        # operations
                        "op_names"             : ["add", "sub", "neg", "inv", "n2", "sqrt", "cos", "sin", "exp", "log"],
                        "use_protected_ops"    : False,
                        # input variables
                        "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                        "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                        "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                        # constants
                        "constants"            : {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                        "constants_units"      : {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] },
                        "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                            }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [1, -2, 1], superparent_name = "y")
        my_programs = VProg.VectPrograms(batch_size=4, max_time_step=10, library=my_lib, n_realizations=1)

        # -------------------- CREATION --------------------
        try:
            my_prior = Prior.NestedTrigonometryPrior (library = my_lib, programs = my_programs, max_nesting = 1)
        except:
            self.fail("Prior creation failed.")

        # ---------- ASSERTIONS ----------

        # Assertion test : max_nesting
        with self.assertRaises(AssertionError, msg="max_nesting arg must be valid"):
            my_prior = Prior.NestedTrigonometryPrior(library=my_lib, programs=my_programs, max_nesting = -1, )
        with self.assertRaises(AssertionError, msg="max_nesting arg must be valid"):
            my_prior = Prior.NestedTrigonometryPrior(library=my_lib, programs=my_programs, max_nesting = 0, )
        with self.assertRaises(AssertionError, msg="max_nesting arg must be valid"):
            my_prior = Prior.NestedTrigonometryPrior(library=my_lib, programs=my_programs, max_nesting = 1.4, )

        # -------------------- TEST WITH max_nesting = 1 --------------------

        test_progs_str = np.array([
            ["cos" , "neg" , "sin" , ],  # Previously violated prior
            ["add" , "cos" , "cos" , ],  # Previously violated prior
            ["cos" , "add" , "x"   , ],
            ["sub" , "sin" , "x"   , ],
            ["add" , "x"   , "x"   , ],  # Already finished program
        ])
        # max_time_steps at least must be at least 6 as some of these test programs require 3 dummies
        test_progs = []
        for prog_str in test_progs_str:
            prog = [my_lib.lib_name_to_idx[tok_str] for tok_str in prog_str]
            test_progs.append(prog)
        test_progs = np.array(test_progs)
        my_programs = VProg.VectPrograms(batch_size=test_progs_str.shape[0], max_time_step=10, library=my_lib, n_realizations=1)
        my_programs.set_programs(test_progs)

        # Test Case --------
        my_prior = Prior.NestedTrigonometryPrior (library = my_lib, programs = my_programs, max_nesting = 1)

        # Expected forbidden tokens
        expected_forbidden_tokens = [
            ["cos" , "sin",],
            ["cos" , "sin",],
            ["cos" , "sin",],
            [],
            [],
        ]

        # Test --------
        mask_prob = my_prior()
        for i, mask_prob_i in enumerate(mask_prob):
            mask_is_forbidden = np.logical_not(mask_prob_i)
            idx = np.arange(my_lib.n_choices)[mask_is_forbidden]
            # print("prog", my_lib.lib_name[my_programs.tokens.idx][i])
            # print("forbidding:", my_lib.lib_name[idx])
            found_forbidden_tokens = np.sort(my_lib.lib_name[idx]).astype(str)
            expected = np.sort(expected_forbidden_tokens[i]).astype(str)
            bool_works = np.array_equal(found_forbidden_tokens, expected)
            self.assertEqual(bool_works, True)

    # -------------------- TEST WITH max_nesting = 2 --------------------

        test_progs_str = np.array([
            ["cos" , "neg" , "sin" , ],
            ["add" , "cos" , "cos" , ],
            ["cos" , "add" , "x"   , ],
            ["sub" , "sin" , "x"   , ],
            ["add" , "x"   , "x"   , ],  # Already finished program
            ["add" , "x"   , "sin" , ],
        ])
        # max_time_steps at least must be at least 6 as some of these test programs require 3 dummies
        test_progs = []
        for prog_str in test_progs_str:
            prog = [my_lib.lib_name_to_idx[tok_str] for tok_str in prog_str]
            test_progs.append(prog)
        test_progs = np.array(test_progs)
        my_programs = VProg.VectPrograms(batch_size=test_progs_str.shape[0], max_time_step=10, library=my_lib, n_realizations=1)
        my_programs.set_programs(test_progs)

        # Test Case --------
        my_prior = Prior.NestedTrigonometryPrior (library = my_lib, programs = my_programs, max_nesting = 2)

        # Expected forbidden tokens
        expected_forbidden_tokens = [
            ["cos" , "sin",],
            ["cos" , "sin",],
            [],
            [],
            [],
            [],
        ]

        # Test --------
        mask_prob = my_prior()
        for i, mask_prob_i in enumerate(mask_prob):
            mask_is_forbidden = np.logical_not(mask_prob_i)
            idx = np.arange(my_lib.n_choices)[mask_is_forbidden]
            # print("prog", my_lib.lib_name[my_programs.tokens.idx][i])
            # print("forbidding:", my_lib.lib_name[idx])
            found_forbidden_tokens = np.sort(my_lib.lib_name[idx]).astype(str)
            expected = np.sort(expected_forbidden_tokens[i]).astype(str)
            bool_works = np.array_equal(found_forbidden_tokens, expected)
            self.assertEqual(bool_works, True)


    def test_OccurencesPrior (self):

        # -------------------- LIB TEST CASE --------------------
        args_make_tokens = {
                        # operations
                        "op_names"             : ["add", "sub", "mul", "neg", "inv", "n2", "sqrt", "cos", "sin", "exp", "log"],
                        "use_protected_ops"    : False,
                        # input variables
                        "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                        "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                        "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                        # constants
                        "constants"            : {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                        "constants_units"      : {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] },
                        "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                            }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [1, -2, 1], superparent_name = "y")
        my_programs = VProg.VectPrograms(batch_size=4, max_time_step=10, library=my_lib, n_realizations=1)

        # -------------------- CREATION --------------------
        try:
            my_prior = Prior.OccurrencesPrior (library = my_lib, programs = my_programs,
                                               targets = ["add", "x"],
                                               max     = [1, 2],)
        except:
            self.fail("Prior creation failed.")

        # ---------- ASSERTIONS ----------

        # Assertion test : targets
        with self.assertRaises(AssertionError, msg="targets arg must be valid"):
            my_prior = Prior.OccurrencesPrior  (library = my_lib, programs = my_programs,
                                                targets = "x",
                                                max     = [2, 2],)
        with self.assertRaises(AssertionError, msg="targets arg must be valid"):
            my_prior = Prior.OccurrencesPrior (library = my_lib, programs = my_programs,
                                                targets = [["add"], ["x"]],
                                                max     = [2, 2],)
        with self.assertRaises(AssertionError, msg="targets arg must be valid"):
            my_prior = Prior.OccurrencesPrior  (library = my_lib, programs = my_programs,
                                                targets = ["token_not_in_lib", "token_not_in_lib"],
                                                max     = [2, 2],)

        # Assertion test : max
        with self.assertRaises(AssertionError, msg="max arg must be valid"):
            my_prior = Prior.OccurrencesPrior  (library = my_lib, programs = my_programs,
                                                targets = ["add", "x"],
                                                max     = [np.nan, 2],)
        with self.assertRaises(AssertionError, msg="max arg must be valid"):
            my_prior = Prior.OccurrencesPrior (library = my_lib, programs = my_programs,
                                               targets = ["add", "x"],
                                               max     = [2.578, 2],)
        with self.assertRaises(AssertionError, msg="max arg must be valid"):
            my_prior = Prior.OccurrencesPrior (library = my_lib, programs = my_programs,
                                               targets = ["add", "x"],
                                               max     = [2, -1],)
        with self.assertRaises(AssertionError, msg="max arg must be valid"):
            my_prior = Prior.OccurrencesPrior  (library = my_lib, programs = my_programs,
                                                targets = ["add", "x"],
                                                max     = [2, 2, 2,],)

        # -------------------- TEST WITH CHILD RELATIONSHIP (1) --------------------

        test_progs_str = np.array([
            ["add" , "x"   , "exp" , "cos" ,],
            ["cos" , "sub" , "v"   , "sin" ,],
            ["sub" , "add" , "x"   , "x"   ,],
            ["sub" , "x"   , "mul" , "x"   ,],
            ["mul" , "add" , "mul" , "x"   ,],
        ])
        # max_time_steps at least must be at least 6 as some of these test programs require 3 dummies
        test_progs = []
        for prog_str in test_progs_str:
            prog = [my_lib.lib_name_to_idx[tok_str] for tok_str in prog_str]
            test_progs.append(prog)
        test_progs = np.array(test_progs)
        my_programs = VProg.VectPrograms(batch_size=test_progs_str.shape[0], max_time_step=10, library=my_lib, n_realizations=1)
        my_programs.set_programs(test_progs)

        # Test Case --------
        my_prior = Prior.OccurrencesPrior (library = my_lib, programs = my_programs,
                                                targets   = ["add" , "x",],
                                                max = [1, 2],)
        # Expected forbidden tokens
        expected_forbidden_tokens = [
            ["add",],
            [],
            ["add", "x"],
            ["x"],
            ["add"],
        ]

        # Test --------
        mask_prob = my_prior()
        for i, mask_prob_i in enumerate(mask_prob):
            mask_is_forbidden = np.logical_not(mask_prob_i)
            idx = np.arange(my_lib.n_choices)[mask_is_forbidden]
            # print("prog", my_lib.lib_name[my_programs.tokens.idx][i])
            # print("forbidding:", my_lib.lib_name[idx])
            found_forbidden_tokens = np.sort(my_lib.lib_name[idx]).astype(str)
            expected = np.sort(expected_forbidden_tokens[i]).astype(str)
            bool_works = np.array_equal(found_forbidden_tokens, expected)
            self.assertEqual(bool_works, True)


    def test_PhysicalUnitsPrior(self):
        # LIBRARY CONFIG
        args_make_tokens = {
                        # operations
                        "op_names"             : ["mul", "add", "neg", "n2", "inv", "cos",],
                        "use_protected_ops"    : True,
                        # input variables
                        "input_var_ids"        : {"z" : 0         ,"x" : 0         , "v" : 1          , "t" : 2,        },
                        "input_var_units"      : {"z" : [1, 0, 0] ,"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                        "input_var_complexity" : {"z" : 0.        ,"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                        # constants
                        "constants"            : {"g" : 9.8        , "E0" : 1         },
                        "constants_units"      : {"g" : [1, -2, 0] , "E0" : [2, -2, 1] },
                        "constants_complexity" : {"g" : 0.         , "E0" : 1.        },
                        # free constants
                        "free_constants"            : {"m"             , "c"              , },
                        "free_constants_init_val"   : {"m" : 1.        , "c" : 1.         , },
                        "free_constants_units"      : {"m" : [0, 0, 1] , "c" : [1, -1, 0] , },
                        "free_constants_complexity" : {"m" : 1.        , "c" : 1.         , },
                           }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [2, -2, 1], superparent_name = "y")

        # ------------------------- TEST PROGRAMS -------------------------
        test_programs_idx = []
        test_programs_str = [
            ["add", "mul", "mul", "m" , "z", "z" , "E0",],
            ["add", "mul", "mul", "m" , "g", "z" , "E0",],
            ["add", "mul", "m"  , "n2", "z", "E0", "-" ,],
        ]
        # Using terminal token placeholder that will be replaced by '-' void token in append function
        test_programs_str = np.char.replace(test_programs_str, '-', 't')

        # Converting into idx
        for test_program_str in test_programs_str :
            test_programs_idx.append(np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str]))
        test_programs_idx = np.array(test_programs_idx)

        # ------------------------- TEST PROGRAMS : EXPECTED BEHAVIOR -------------------------

        expected_allowed = [
            # Prog 0
            [
            ['mul', 'add', 'neg', 'n2', 'inv', 'E0'],  # step 0
            ['mul', 'add', 'neg', 'n2', 'inv', 'E0'],  # step 1
            ['mul', 'add', 'neg', 'n2', 'inv', 'cos', 'g', 'c', 'm', 'E0', 'z', 'x', 'v', 't'], # step 2
            ['mul', 'add', 'neg', 'n2', 'inv', 'cos', 'g', 'c', 'm', 'E0', 'z', 'x', 'v', 't'],
            ['mul', 'add', 'neg', 'n2', 'inv', 'cos', 'g', 'c', 'm', 'E0', 'z', 'x', 'v', 't'],
            ['mul', 'add', 'neg', 'n2', 'inv', 'g' ],
            ['mul', 'add', 'neg', 'n2', 'inv', 'cos', 'g', 'c', 'm', 'E0', 'z', 'x', 'v', 't'], # -> not physical anymore, all tokens are allowed
            ],
            # Prog 1
            [
            ['mul', 'add', 'neg', 'n2', 'inv', 'E0'],  # step 0
            ['mul', 'add', 'neg', 'n2', 'inv', 'E0'],  # step 1
            ['mul', 'add', 'neg', 'n2', 'inv', 'cos', 'g', 'c', 'm', 'E0', 'z', 'x', 'v', 't'], # step 2
            ['mul', 'add', 'neg', 'n2', 'inv', 'cos', 'g', 'c', 'm', 'E0', 'z', 'x', 'v', 't'],
            ['mul', 'add', 'neg', 'n2', 'inv', 'cos', 'g', 'c', 'm', 'E0', 'z', 'x', 'v', 't'],
            ['mul', 'add', 'neg', 'n2', 'inv', 'z', 'x'],
            ['mul', 'add', 'neg', 'n2', 'inv', 'E0'],
            ],
            # Prog 2
            [
            ['mul', 'add', 'neg', 'n2', 'inv', 'E0'],  # step 0
            ['mul', 'add', 'neg', 'n2', 'inv', 'E0'],  # step 1
            ['mul', 'add', 'neg', 'n2', 'inv', 'cos', 'g', 'c', 'm', 'E0', 'z', 'x', 'v', 't'], # step 2
            ['mul', 'add', 'neg', 'n2', 'inv'],
            ['mul', 'add', 'neg', 'n2', 'inv', 'c', 'v'],
            ['mul', 'add', 'neg', 'n2', 'inv', 'cos', 'g', 'c', 'm', 'E0', 'z', 'x', 'v', 't'], # -> not physical anymore, all tokens are allowed
            ['mul', 'add', 'neg', 'n2', 'inv', 'cos', 'g', 'c', 'm', 'E0', 'z', 'x', 'v', 't'], # -> prog complete, all tokens are allowed
            ],
        ]

        # ------------------------- TEST & EXPECTED VS OBSERVED -------------------------

        # Initializing programs & prior
        my_programs = VProg.VectPrograms(batch_size=test_programs_idx.shape[0], max_time_step=test_programs_idx.shape[1], library=my_lib, n_realizations=1)
        my_prior    = Prior.PhysicalUnitsPrior(library=my_lib, programs=my_programs, )

        # Appending tokens
        mask_prob = my_prior()

        for i in range (test_programs_idx.shape[1]):
             mask_prob = my_prior()
             for prog_i in range(len(expected_allowed)):
                # observed
                prog_step_obs_allowed = my_lib.lib_name[:my_lib.n_choices][mask_prob.astype(bool)[prog_i]]
                # expected
                prog_step_exp_allowed = np.array(expected_allowed[prog_i][i])
                #print("observed legal tokens for prog %i, step = %i:" % (prog_i, i), prog_step_obs_allowed.tolist())
                #print("expected legal tokens for prog %i, step = %i:" % (prog_i, i), prog_step_exp_allowed.tolist())
                bool_works = np.array_equal(sorted(prog_step_exp_allowed), sorted(prog_step_obs_allowed))
                self.assertTrue(bool_works)
             # prog_i = 2
             # print("observed legal tokens for prog %i, step = %i:" % (prog_i, i), prog_step_obs_allowed.tolist())
             # my_programs.get_tree_image(prog_i, fpath="%i.png"%(i))
             my_programs.append(test_programs_idx[:,i])

        return None

    def test_SymbolicPrior(self):

        # LIBRARY CONFIG
        args_make_tokens = {
                        # operations
                        "op_names"             : ["mul", "add", "sub", "div", "inv", "n2", "sqrt", "neg", "exp", "log", "sin", "cos"],
                        "use_protected_ops"    : True,
                        # input variables
                        "input_var_ids"        : {"r" : 0         },
                        "input_var_units"      : {"r" : [1, 0, 0] },
                        # constants
                        "constants"            : {"1" : 1.         },
                        "constants_units"      : {"1" : [0, 0 , 0] },
                        # free constants
                        "free_constants"            : { "R"             , "E_t"              , "A"              , "c"             },
                        "free_constants_units"      : { "R" : [1, 0, 0] , "E_t" : [2, -2, 0] , "A" : [2, -2, 0] , "c" : [0, 0, 0] },
                           }

        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [2, -2, 0], superparent_name = "y")

        # ------------------------- TEST PROGRAMS -------------------------
        test_prog_str = ["add", "E_t", "mul", "A", "mul", "div", "R", "r", "log", "add", "1", "div", "r", "R"]
        test_prog_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_prog_str])

        target_prog_str = ["add", "E_t", "mul", "A", "mul", "div", "R", "r", "log", "add", "1", "div", "-", "R"]
        target_prog_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in target_prog_str])

        max_time_step = len(target_prog_str) + 10 # Extra steps to check behavior outside target
        batch_size    = 128

        # ------------------------- TEST & EXPECTED VS OBSERVED -------------------------

        # Initializing programs & prior
        my_programs = VProg.VectPrograms(batch_size=batch_size, max_time_step=max_time_step, library=my_lib, n_realizations=1)
        my_prior    = Prior.SymbolicPrior(library=my_lib, programs=my_programs, expression=target_prog_str)

        target_prog_idx_padded = np.array(target_prog_idx.tolist() + [my_lib.invalid_idx,] * (max_time_step - len(target_prog_str)))
        self.assertTrue(np.array_equal(my_prior.expression,
                                       # Expecting expression to have been padded with invalid tokens
                                       target_prog_idx_padded))

        # Appending tokens
        mask_prob = my_prior()

        for i in range (max_time_step):
            mask_prob = my_prior()

            # In cases where we are inside range of target program
            if i < len(target_prog_str):
                # In cases where token is not invalid
                if target_prog_idx[i] != my_lib.invalid_idx:
                    favored_tokens = np.tile(my_lib.lib_choosable_name, (batch_size, 1))[mask_prob.astype(bool)]
                    # Asserting that all favored tokens are the same
                    self.assertTrue(np.unique(favored_tokens).shape[0] == 1)
                    favored_token = favored_tokens[0]
                    # Asserting that favored token is the same as the target token
                    self.assertTrue(favored_token == target_prog_str[i])
                # In cases where token is invalid
                if target_prog_idx[i] == my_lib.invalid_idx:
                    # Asserting that all tokens are choosable
                    self.assertTrue(np.all(mask_prob == True))
            # In cases where we are outside range of target program
            else:
                # Asserting that all tokens are choosable
                self.assertTrue(np.all(mask_prob == True))

            if i < len(test_prog_idx):
                next_tokens_idx = np.full( (batch_size,), test_prog_idx[i] )                        # (batch_size,)
            else:
                next_tokens_idx = np.random.randint(low=0, high=my_lib.n_choices, size=batch_size)  # (batch_size,)

            my_programs.append(next_tokens_idx)

        return None

    def test_StructurePrior(self):

        # -------------------- LIB TEST CASE --------------------
        args_make_tokens = {
                        # operations
                        "op_names"             : ["add", "sub", "mul", "div", "neg", "inv", "n2", "sqrt", "cos", "sin", "exp", "log"],
                        "use_protected_ops"    : False,
                        # input variables
                        "input_var_ids"        : {"x1" : 0, "x2" : 1, "x3" : 2, "x4" : 3, "t" : 4},
                        # constants
                        "constants"            : {"pi" : np.pi    },
                        "constants_complexity" : {"pi" : 0.       },
                            }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Raises some warnings due to some units provided (this is ok)
            my_lib = Lib.Library(args_make_tokens = args_make_tokens, superparent_name = "y")

        # -------------------- TEST CASES--------------------
        # TEST CASE A
        def get_test_case_A ():
            # --- Test structure ---
            # Target is like f(x1,x2) * ( f(t) * (f(x3)+f(x4)) )
            test_structure = ["mul", ["x1","x2"], "mul", ["t"], "add", ["x3"], ["x4"]]

            # --- Test progs ---
            test_programs_idx = []
            test_programs_str = [
                # Target is sin(x1*x2)*t*(1/x3 + 1/x4)
                ["mul", "sin", "mul", "x1", "x2", "mul", "t", "add", "inv", "x3", "inv", "x4", "-"],
                # Target is (x1/x2)*t*(1/x3 + 1/x4)
                ["mul", "div", "x1", "x2", "mul", "t", "add", "inv", "x3", "inv", "x4", "-", "-"],
                # Target is (log(x1)/exp(x2))*t*(1/x3 + 1/x4)
                ["mul", "div", "log", "x1", "exp", "x2", "mul", "t", "add", "inv", "x3", "inv", "x4"],
            ]
            # Using terminal token placeholder that will be replaced by '-' void token in append function
            test_programs_str = np.char.replace(test_programs_str, '-', 't')
            # Converting into idx
            for test_program_str in test_programs_str :
                test_programs_idx.append(np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str]))
            test_programs_idx = np.array(test_programs_idx)
            max_time_step = test_programs_idx.shape[1]

            # --- Expected output ---
            subf1 = ["x1","x2"] # subfunction 1 depends on x1 and x2
            subf2 = ["t",]
            subf3 = ["x3",]
            subf4 = ["x4",]
            test_programs_expected_struct = [
                # Target is sin(x1*x2)*t*(1/x3 + 1/x4)
                ["mul", subf1, subf1, subf1, subf1, "mul", subf2, "add", subf3, subf3, subf4, subf4, "-"],
                # Target is (x1/x2)*t*(1/x3 + 1/x4)
                ["mul", subf1, subf1, subf1, "mul", subf2, "add",  subf3, subf3, subf4, subf4, "-", "-"],
                # Target is (log(x1)/exp(x2))*t*(1/x3 + 1/x4)
                ["mul", subf1, subf1, subf1, subf1, subf1, "mul", subf2, "add", subf3, subf3, subf4, subf4]
            ]
            test_programs_expected_prior = []
            for prog in test_programs_expected_struct:
                prog_priors = []
                for node in prog:
                    if isinstance(node,str):
                        # "mul" or "add"
                        if node == "mul" or node == "add":
                            if node == "mul":
                                legal_tokens = Func.MULTIPLICATIVE_SEP_OPS  # eg. ["mul","div"]
                            elif node == "add":
                                legal_tokens = Func.ADDITIVE_SEP_OPS  # eg. ["add","sub"]
                            else:
                                raise ValueError("Invalid node")
                            # Zeroes everywhere except at legal tokens
                            prior = np.zeros(my_lib.n_choices)
                            for tok in legal_tokens:
                                idx = my_lib.lib_name_to_idx[tok]
                                prior[idx] = 1
                        # "-"
                        elif node == "-":
                            # Everything is allowed
                            prior = np.ones(my_lib.n_choices)
                        else:
                            raise ValueError("Invalid node")
                    # ["x1","x2"] or ["t"] or ["x3"] or ["x4"]
                    elif isinstance(node,list):
                        # No input variables are allowed except the ones in the list
                        prior = ~(my_lib.var_type == Tok.VAR_TYPE_INPUT_VAR)[:my_lib.n_choices]
                        for var in node:
                            prior[my_lib.lib_name_to_idx[var]] = True
                    else:
                        raise ValueError("Invalid node")

                    prog_priors.append(prior.astype(float))
                test_programs_expected_prior.append(prog_priors)
            test_programs_expected_prior = np.array(test_programs_expected_prior) # (n_progs, max_time_step, n_choices)

            return test_programs_idx, test_structure, test_programs_expected_prior, max_time_step

        # TEST CASE B
        def get_test_case_B():
            # --- Test structure ---
            # Target is like f(x1,x2)
            test_structure = [["x1","x2"],]

            # --- Test progs ---
            test_programs_idx = []
            test_programs_str = [
                # Target is sin(x1*x2)
                ["sin", "mul", "x1", "x2", "-", "-"],
                # Target is (x1/x2)
                ["div", "x1", "x2", "-", "-", "-"],
                # Target is (log(x1)/exp(x2))
                ["div", "log", "x1", "exp", "x2", "-"],
            ]
            # Using terminal token placeholder that will be replaced by '-' void token in append function
            test_programs_str = np.char.replace(test_programs_str, '-', 't')
            # Converting into idx
            for test_program_str in test_programs_str :
                test_programs_idx.append(np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str]))
            test_programs_idx = np.array(test_programs_idx)
            max_time_step = test_programs_idx.shape[1]

            # --- Expected output ---
            subf1 = ["x1","x2"]
            test_programs_expected_struct = [
                # Target is sin(x1*x2)
                [subf1, subf1, subf1, subf1, "-", "-"],
                # Target is (x1/x2)
                [subf1, subf1, subf1, "-", "-", "-"],
                # Target is (log(x1)/exp(x2))
                [subf1, subf1, subf1, subf1, subf1, "-"]
            ]
            test_programs_expected_prior = []
            for prog in test_programs_expected_struct:
                prog_priors = []
                for node in prog:
                    if isinstance(node, str):
                        # "mul" or "add"
                        if node == "mul" or node == "add":
                            if node == "mul":
                                legal_tokens = Func.MULTIPLICATIVE_SEP_OPS  # eg. ["mul","div"]
                            elif node == "add":
                                legal_tokens = Func.ADDITIVE_SEP_OPS  # eg. ["add","sub"]
                            else:
                                raise ValueError("Invalid node")
                            # Zeroes everywhere except at legal tokens
                            prior = np.zeros(my_lib.n_choices)
                            for tok in legal_tokens:
                                idx = my_lib.lib_name_to_idx[tok]
                                prior[idx] = 1
                        # "-"
                        elif node == "-":
                            # Everything is allowed
                            prior = np.ones(my_lib.n_choices)
                        else:
                            raise ValueError("Invalid node")
                    # ["x1","x2"] or ["t"] or ["x3"] or ["x4"]
                    elif isinstance(node, list):
                        # No input variables are allowed except the ones in the list
                        prior = ~(my_lib.var_type == Tok.VAR_TYPE_INPUT_VAR)[:my_lib.n_choices]
                        for var in node:
                            prior[my_lib.lib_name_to_idx[var]] = True
                    else:
                        raise ValueError("Invalid node")

                    prog_priors.append(prior.astype(float))
                test_programs_expected_prior.append(prog_priors)
            test_programs_expected_prior = np.array(test_programs_expected_prior)  # (n_progs, max_time_step, n_choices)
            return test_programs_idx, test_structure, test_programs_expected_prior, max_time_step

        # TEST CASE C
        def get_test_case_C():
            # --- Test structure ---
            # Target is like f(t)*f(x1)+f(x2,x3)+f()
            test_structure = ["add", "mul", ["t"], ["x1"], "add", ["x2", "x3"], []] # structure can use only part of the input vars and can use non dependent subfuncs

            # --- Test progs ---
            test_programs_idx = []
            test_programs_str = [
                # Target is sin(t)*cos(x1)+exp(x2*x3)+pi
                ["add", "mul", "sin", "t", "cos", "x1", "add", "exp", "mul", "x2", "x3", "pi", "-",],
                # Target is exp(t)/x1 + (x2+x3) + cos(pi) # multiplicative does not mean just mul, could be div
                ["add", "div", "exp", "t", "x1", "add", "add", "x2", "x3", "cos", "pi", "-", "-",],
                # Target is t/x1 + (x2) + sin(pi) # subfunc can use only part of its input vars
                ["add", "div", "t", "x1", "add", "x2", "sin", "pi", "-", "-", "-", "-", "-",],
            ]
            # Using terminal token placeholder that will be replaced by '-' void token in append function
            test_programs_str = np.char.replace(test_programs_str, '-', 't')
            # Converting into idx
            for test_program_str in test_programs_str:
                test_programs_idx.append(
                    np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str]))
            test_programs_idx = np.array(test_programs_idx)
            max_time_step = test_programs_idx.shape[1]

            # --- Expected output ---
            subf1 = ["t"]  # subfunction 1 depends on t
            subf2 = ["x1", ]
            subf3 = ["x2","x3" ]
            subf4 = []
            test_programs_expected_struct = [
                # Target is sin(t)*cos(x1)+exp(x2*x3)+pi
                ["add", "mul", subf1, subf1, subf2, subf2, "add", subf3, subf3, subf3, subf3, subf4, "-",],
                # Target is exp(t)/x1 + (x2+x3) + cos(pi) # multiplicative does not mean just mul, could be div
                ["add", "mul", subf1, subf1, subf2, "add", subf3, subf3, subf3, subf4, subf4, "-", "-",],
                # Target is t/x1 + (x2) + sin(pi) # subfunc can use only part of its input vars
                ["add", "mul", subf1, subf2, "add", subf3, subf4, subf4, "-", "-", "-", "-", "-",],
            ]
            test_programs_expected_prior = []
            for prog in test_programs_expected_struct:
                prog_priors = []
                for node in prog:
                    if isinstance(node, str):
                        # "mul" or "add"
                        if node == "mul" or node == "add":
                            if node == "mul":
                                legal_tokens = Func.MULTIPLICATIVE_SEP_OPS  # eg. ["mul","div"]
                            elif node == "add":
                                legal_tokens = Func.ADDITIVE_SEP_OPS  # eg. ["add","sub"]
                            else:
                                raise ValueError("Invalid node")
                            # Zeroes everywhere except at legal tokens
                            prior = np.zeros(my_lib.n_choices)
                            for tok in legal_tokens:
                                idx = my_lib.lib_name_to_idx[tok]
                                prior[idx] = 1
                        # "-"
                        elif node == "-":
                            # Everything is allowed
                            prior = np.ones(my_lib.n_choices)
                        else:
                            raise ValueError("Invalid node")
                    # ["x1","x2"] or ["t"] or ["x3"] or ["x4"]
                    elif isinstance(node, list):
                        # No input variables are allowed except the ones in the list
                        prior = ~(my_lib.var_type == Tok.VAR_TYPE_INPUT_VAR)[:my_lib.n_choices]
                        for var in node:
                            prior[my_lib.lib_name_to_idx[var]] = True
                    else:
                        raise ValueError("Invalid node")

                    prog_priors.append(prior.astype(float))
                test_programs_expected_prior.append(prog_priors)
            test_programs_expected_prior = np.array(
                test_programs_expected_prior)  # (n_progs, max_time_step, n_choices)

            return test_programs_idx, test_structure, test_programs_expected_prior, max_time_step


        # All test cases
        test_cases = [get_test_case_A, get_test_case_B, get_test_case_C]

        # -------------------- CREATION TEST --------------------

        for get_test_case in test_cases:
            test_programs_idx, test_structure, test_programs_expected_prior, max_time_step = get_test_case()

            try:
                my_programs = VProg.VectPrograms(batch_size=3, max_time_step=max_time_step, library=my_lib, n_realizations=1)
                my_prior = Prior.StructurePrior (library = my_lib, programs = my_programs,
                                                   structure = test_structure,
                                                   )
            except:
                self.fail("Prior creation failed.")

        # -------------------- ASSERTIONS --------------------

        test_programs_idx, test_structure, test_programs_expected_prior, max_time_step = get_test_case_A()

        # Test that non-existing sep raises error
        with self.assertRaises(AssertionError, msg="Invalid sep node must raise error"):
            wrong_test_structure = ["mul", ["x1","x2"], "mul", ["t"], "invalid_sep", ["x3"], ["x4"]]
            my_programs = VProg.VectPrograms(batch_size=3, max_time_step=max_time_step, library=my_lib, n_realizations=1)
            my_prior = Prior.StructurePrior(library=my_lib, programs=my_programs, structure=wrong_test_structure)

        # Test that add sep not in lib but mentionned in structure raises error
        with self.assertRaises(AssertionError, msg="Invalid sep node must raise error"):
            weird_args_make_tokens = {
                # operations
                "op_names": ["mul", "div", "neg", "inv", "n2", "sqrt", "cos", "sin", "exp", "log"],
                "use_protected_ops": False,
                # input variables
                "input_var_ids": {"x1": 0, "x2": 1, "x3": 2, "x4": 3, "t": 4},
                # constants
                "constants": {"pi": np.pi},
                "constants_complexity": {"pi": 0.},
            }
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Raises some warnings due to some units provided (this is ok)
                my_weird_lib = Lib.Library(args_make_tokens=weird_args_make_tokens, superparent_name="y")
            my_programs = VProg.VectPrograms(batch_size=3, max_time_step=max_time_step, library=my_weird_lib, n_realizations=1)
            my_prior = Prior.StructurePrior(library=my_weird_lib, programs=my_programs, structure=test_structure)

        # Test that wrong subfunc node raises error (not a list of str)
        with self.assertRaises(AssertionError, msg="Invalid subfunc node must raise error"):
            wrong_test_structure = ["mul", [1,2], "mul", ["t"], "invalid_sep", ["x3"], ["x4"]]
            my_programs = VProg.VectPrograms(batch_size=3, max_time_step=max_time_step, library=my_lib, n_realizations=1)
            my_prior = Prior.StructurePrior(library=my_lib, programs=my_programs, structure=wrong_test_structure)

        # Test that wrong subfunc node raises error (non existing input var)
        with self.assertRaises(AssertionError, msg="Invalid subfunc node must raise error"):
            wrong_test_structure = ["mul", ["x1","x2"], "mul", ["t"], "add", ["x9999999"], ["x4"],]
            my_programs = VProg.VectPrograms(batch_size=3, max_time_step=max_time_step, library=my_lib, n_realizations=1)
            my_prior = Prior.StructurePrior(library=my_lib, programs=my_programs, structure=wrong_test_structure)

        # Test that wrong node type raises error
        with self.assertRaises(AssertionError, msg="Invalid node type must raise error"):
            wrong_test_structure = ["mul", ["x1","x2"], "mul", "t", "add", ["x3"], ["x4"],]
            my_programs = VProg.VectPrograms(batch_size=3, max_time_step=max_time_step, library=my_lib, n_realizations=1)
            my_prior = Prior.StructurePrior(library=my_lib, programs=my_programs, structure=wrong_test_structure)
        with self.assertRaises(ValueError, msg="Invalid node type must raise error"):
            wrong_test_structure = ["mul", ["x1","x2"], "mul", ["t",], "add", ["x3"], 4.12,]
            my_programs = VProg.VectPrograms(batch_size=3, max_time_step=max_time_step, library=my_lib, n_realizations=1)
            my_prior = Prior.StructurePrior(library=my_lib, programs=my_programs, structure=wrong_test_structure)

        # Test that unbalanced structure raises error
        with self.assertRaises(AssertionError, msg="Unbalanced structure must raise error"):
            wrong_test_structure = ["mul", ["x1","x2"], "mul", ["t",], "add", ["x3"],]
            my_programs = VProg.VectPrograms(batch_size=3, max_time_step=max_time_step, library=my_lib, n_realizations=1)
            my_prior = Prior.StructurePrior(library=my_lib, programs=my_programs, structure=wrong_test_structure)
        with self.assertRaises(AssertionError, msg="Unbalanced structure must raise error"):
            wrong_test_structure = ["mul", ["x1","x2"], "mul", "add", ["x3"], ["x4"],]
            my_programs = VProg.VectPrograms(batch_size=3, max_time_step=max_time_step, library=my_lib, n_realizations=1)
            my_prior = Prior.StructurePrior(library=my_lib, programs=my_programs, structure=wrong_test_structure)
        with self.assertRaises(AssertionError, msg="Unbalanced structure must raise error"):
            wrong_test_structure = ["mul", ["x1","x2"], "mul", ["t"], "add", "mul", ["x3"], ["x4"]]
            my_programs = VProg.VectPrograms(batch_size=3, max_time_step=max_time_step, library=my_lib, n_realizations=1)
            my_prior = Prior.StructurePrior(library=my_lib, programs=my_programs, structure=wrong_test_structure)


        # -------------------- REPR TEST --------------------

        for get_test_case in test_cases:
            test_programs_idx, test_structure, test_programs_expected_prior, max_time_step = get_test_case()

            my_programs = VProg.VectPrograms(batch_size=3, max_time_step=max_time_step, library=my_lib, n_realizations=1)
            my_prior = Prior.StructurePrior(library=my_lib, programs=my_programs,
                                            structure=test_structure,
                                            )
            try:
                a = my_prior.__repr__()
            except:
                self.fail("Prior __repr__ failed.")

        # -------------------- NORMAL USAGE --------------------

        for get_test_case in test_cases:
            test_programs_idx, test_structure, test_programs_expected_prior, max_time_step = get_test_case()

            my_programs = VProg.VectPrograms(batch_size=3, max_time_step=max_time_step, library=my_lib, n_realizations=1)
            my_prior = Prior.StructurePrior(library=my_lib, programs=my_programs,
                                            structure=test_structure,
                                            )
            for i in range(max_time_step):
                mask_prob = my_prior()                                      # (n_progs, n_choices)
                expected_mask_prob = test_programs_expected_prior[:, i, :]  # (n_progs, n_choices)
                self.assertTrue(np.all(mask_prob == expected_mask_prob))
                my_programs.append(test_programs_idx[:, i])

        # -------------------- NORMAL USAGE WITH EPS --------------------
        prob_eps = 1e-6

        for get_test_case in test_cases:
            test_programs_idx, test_structure, test_programs_expected_prior, max_time_step = get_test_case()

            my_programs = VProg.VectPrograms(batch_size=3, max_time_step=max_time_step, library=my_lib, n_realizations=1)
            my_prior = Prior.StructurePrior(library=my_lib, programs=my_programs,
                                            structure=test_structure,
                                            prob_eps = prob_eps,
                                            )
            for i in range(max_time_step):
                mask_prob = my_prior()                                      # (n_progs, n_choices)
                expected_mask_prob = test_programs_expected_prior[:, i, :]  # (n_progs, n_choices)
                # We expect eps instead of 0
                expected_mask_prob[expected_mask_prob == 0] = prob_eps
                self.assertTrue(np.all(mask_prob == expected_mask_prob))
                my_programs.append(test_programs_idx[:, i])

        # -------------------- UNREGULAR CALLS --------------------

        for get_test_case in test_cases:
            test_programs_idx, test_structure, test_programs_expected_prior, max_time_step = get_test_case()

            my_programs = VProg.VectPrograms(batch_size=3, max_time_step=max_time_step, library=my_lib, n_realizations=1)
            my_prior = Prior.StructurePrior(library=my_lib, programs=my_programs,
                                            structure=test_structure,
                                            )
            for i in range(max_time_step):
                if i%3 == 0:
                    mask_prob = my_prior()                                      # (n_progs, n_choices)
                    expected_mask_prob = test_programs_expected_prior[:, i, :]  # (n_progs, n_choices)
                    self.assertTrue(np.all(mask_prob == expected_mask_prob))
                my_programs.append(test_programs_idx[:, i])

        # -------------------- UN-OBEYED PRIOR --------------------

        test_programs_idx_A, test_structure_A, _, max_time_step = get_test_case_A()
        test_programs_idx_C, test_structure_C, _, max_time_step = get_test_case_C() # have same max_time_step

        # Let's say we have prior structure from A but progs from C are being generated instead
        test_structure    = test_structure_A
        test_programs_idx = test_programs_idx_C

        my_programs = VProg.VectPrograms(batch_size=3, max_time_step=max_time_step, library=my_lib, n_realizations=1)
        my_prior = Prior.StructurePrior(library=my_lib, programs=my_programs,
                                        structure=test_structure,
                                        )

        for i in range(max_time_step):
            mask_prob = my_prior()  # (n_progs, n_choices)
            my_programs.append(test_programs_idx[:, i])

        return None



    # Test prior collection using (UniformArityPrior, HardLengthPrior)
    def test_PriorCollection(self):

        # ------- TEST CASE -------
        # Library
        args_make_tokens = {
                        # operations
                        "op_names"             : ["mul", "add", "neg", "inv", "cos"],
                        "use_protected_ops"    : False,
                        # input variables
                        "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                        "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                        "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                        # constants
                        "constants"            : {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                        "constants_units"      : {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] },
                        "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                            }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [1, -2, 1], superparent_name = "y")
        # Programs test case
        max_length = 5
        min_length = 3
        test_case_str = np.array([
            # 0      1      2      3      4
            ["add", "cos", "x"  , "cos", "c"  ],  # -> should begin enforcing arity == 0 tokens at pos = 3
                                                  # -> should begin enforcing arity <= 1 tokens at pos = 1
                                                  # -> should enforce arity >= 1 tokens until pos = 0

            ["cos", "cos", "cos", "cos", "x"  ],  # -> should begin enforcing arity == 0 tokens at pos = 3
                                                  # -> should begin enforcing arity <= 1 tokens at pos = 2
                                                  # -> should enforce arity >= 1 tokens until pos = 1

            ["add", "add", "x"  , "pi" , "c"  ],  # -> should begin enforcing arity == 0 tokens at pos = 1
                                                  # -> should begin enforcing arity <= 1 tokens at pos = 1
                                                  # -> should enforce arity >= 1 tokens until pos = 0

            ["add", "x"  , "c"  , "-"  , "-"  ],  # -> should begin enforcing arity == 0 tokens at pos = inf
                                                  # -> should begin enforcing arity <= 1 tokens at pos = inf
                                                  # -> should enforce arity >= 1 tokens until pos = 0

            ["add", "cos", "x"  , "c"  , "-"  ],  # -> should begin enforcing arity == 0 tokens at pos = inf
                                                  # -> should begin enforcing arity <= 1 tokens at pos = 1
                                                  # -> should enforce arity >= 1 tokens until pos = 0

            ["cos", "add", "x"  , "v"  , "-"  ],  # -> should begin enforcing arity == 0 tokens at pos = inf
                                                  # -> should begin enforcing arity <= 1 tokens at pos = 1
                                                  # -> should enforce arity >= 1 tokens until pos = 1
        ])
        pos_begin_max_arity_is_0 = np.array([3, 3, 1, np.inf, np.inf, np.inf])
        pos_begin_max_arity_is_1 = np.array([1, 2, 1, np.inf, 1     , 1     ])
        pos_end_min_arity_is_1   = np.array([0, 1, 0, 0     , 0     , 1     ])

        # Using a valid placeholder existing in the library that will be ignored anyway instead of '-'
        test_case_str = np.where(test_case_str == "-", "x", test_case_str)

        # Creating idx that will be appended
        n_progs, n_steps = test_case_str.shape
        test_case = np.zeros((n_progs, n_steps)).astype(int)
        for i in range (n_progs):
            for j in range (n_steps):
                tok_str = test_case_str[i,j]
                test_case[i,j] = my_lib.lib_name_to_idx[tok_str]

        # VectPrograms
        my_programs = VProg.VectPrograms(batch_size = n_progs, max_time_step=n_steps, library=my_lib, n_realizations=1)

        # ------- TEST ASSERTIONS -------
        # Assert wrong prior name
        with self.assertRaises(AssertionError,):
            my_collection = Prior.make_PriorCollection(
                library       = my_lib,
                programs      = my_programs,
                priors_config = [ ("A_prior_that_does_not_exist" , None),
                                  ("HardLengthPrior", {"min_length": min_length,
                                                       "max_length": max_length}),
                                ],
            )
        # Assert missing prior args
        with self.assertRaises(AssertionError,):
            my_collection = Prior.make_PriorCollection(
                library       = my_lib,
                programs      = my_programs,
                priors_config = [("UniformArityPrior" , None),
                                 ("HardLengthPrior"   , None),],
            )
        # ------- TEST CREATION -------
        my_prior_HardLength      = Prior.HardLengthPrior  (library = my_lib, programs = my_programs, min_length = min_length, max_length = max_length)
        my_prior_UniformArity    = Prior.UniformArityPrior(library = my_lib, programs = my_programs)
        try:
            my_collection = Prior.make_PriorCollection(
                library       = my_lib,
                programs      = my_programs,
                priors_config = [("UniformArityPrior" , None),
                                 ("HardLengthPrior"   , {"min_length": min_length,
                                                         "max_length": max_length, }),],
            )
        except:
            self.fail("PriorCollection creation failed.")

        # ------- TEST PRIOR -------
        expected_prior_val = np.multiply(my_prior_HardLength(), my_prior_UniformArity())

        for step in range (n_steps):
            works_bool = np.array_equal(expected_prior_val, my_collection())
            self.assertTrue(works_bool)
            # NEXT STEP
            my_programs.append(test_case[:, step])
            expected_prior_val = np.multiply(my_prior_HardLength(), my_prior_UniformArity())

        # ------- TEST CREATION (MULTIPLE PRIORS OF SAME TYPE) -------
        priors_config = [("UniformArityPrior", None),
                         ("SoftLengthPrior", {"length_loc": 10,
                                             "scale": 5, }),
                         ("SoftLengthPrior", {"length_loc": 11,
                                             "scale": 6, }),
                         ("HardLengthPrior", {"min_length": 3,
                                             "max_length": 5, })
                        ]
        try:
            my_collection = Prior.make_PriorCollection(
                library       = my_lib,
                programs      = my_programs,
                priors_config = priors_config,
            )
        except:
            self.fail("PriorCollection creation failed.")

        for i, config in enumerate(priors_config):
            name, args = config[0], config[1]
            assert my_collection.priors[i].__class__.__name__ == name, "Prior %s not in PriorCollection object"%(name)
            if args is not None:
                for arg_name, arg_value in args.items():
                    assert my_collection.priors[i].__getattribute__(arg_name) == arg_value, "Wrong prior parameter for " \
                                                "parameter %s of prior %s of PriorCollection object"%(arg_name, name)

        return None

if __name__ == '__main__':
    unittest.main(verbosity=2)