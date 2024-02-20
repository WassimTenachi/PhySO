import time
import unittest
import numpy as np
import torch as torch
import matplotlib.pyplot as plt
import warnings

# Internal imports
from physo.physym import functions as Func
from physo.physym.functions import data_conversion, data_conversion_inv
import physo.physym.token as Tok

# Test token and output shapes
def test_one_token(tester, token):
    data0 = data_conversion ( np.arange(-5, 5, 0.5)     )
    data1 = data_conversion ( np.arange(-5, 5, 0.5) + 1 )
    data2 = data_conversion ( np.arange(-5, 5, 0.5) * 2 )   # 0 in same place as data0
    pi    = data_conversion ( np.array(np.pi) )
    large = data_conversion ( np.array(1e10)  )   # large float
    n_data = len(data0)

    # Binary
    if token.arity == 2:
        tester.assertEqual(len( data_conversion_inv ( token(data0, data1)              )) , n_data)   # np.array    , np.array
        tester.assertEqual(len( data_conversion_inv ( token(data0, data2)              )) , n_data)   # np.array    , np.array with (0,0)
        tester.assertEqual(len( data_conversion_inv ( token(data0, pi   )              )) , n_data)   # np.array    , float
        tester.assertEqual(len( data_conversion_inv ( token(data0, large)              )) , n_data)   # np.array    , large float
        tester.assertEqual(len( data_conversion_inv ( token(large, data0)              )) , n_data)   # large float , np.array
        tester.assertEqual(len( data_conversion_inv ( token(*torch.stack((data0, data1))) )) , n_data)  # *[np. array    , np.array]
        # large float , large float
        # expecting length = 1 or n_data to be able to compute afterward
        out_len = np.shape(np.atleast_1d(
                                data_conversion_inv ( token(large, large)              )))
        tester.assertEqual(out_len == n_data or out_len == (1,), True)
    # Unary
    if token.arity == 1:
        tester.assertEqual(len( data_conversion_inv ( token(data0)                     )) , n_data)  # np.array
        # large float
        # expecting length = 1 or n_data to be able to compute afterward
        out_len = np.shape(np.atleast_1d(
                                data_conversion_inv ( token(large)                     )))
        tester.assertEqual(out_len == n_data or out_len == (1,), True)
    # Zero-arity
    if token.arity == 0:
        out_len = np.shape(np.atleast_1d(
                                data_conversion_inv( token()                           )))
        bool_works = (out_len == (n_data,) or out_len == (1,))
        tester.assertEqual(bool_works, True)


class FuncTest(unittest.TestCase):

    # Test make tokens function
    def test_make_tokens(self):
        op_names = ["mul", "add", "neg", "inv", "sin"]
        try:
            # Raises some warnings due to lack of initial values etc (not a problem)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                my_tokens = Func.make_tokens(op_names          = op_names,
                                             input_var_ids     = {"x0" : 0     , "x1" : 1 },
                                             constants         = {"pi" : np.pi , "c"  : 3e8},
                                             free_constants    = {"c0", "c1"},
                                             use_protected_ops = False,
                                             )
        except Exception: self.fail("Make tokens function failed")

    # Test make tokens function for Class SR
    def test_make_tokens_for_classSR(self):
        op_names = ["mul", "add", "neg", "inv", "sin"]
        try:
            # Raises some warnings due to lack of initial values etc (not a problem)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                my_tokens = Func.make_tokens(op_names             = op_names,
                                             input_var_ids        = {"x0" : 0     , "x1" : 1 },
                                             constants            = {"pi" : np.pi , "c"  : 3e8},
                                             free_constants       = {"c0", "c1"},
                                             class_free_constants = {"c3",},
                                             spe_free_constants   = {"k0", "k1"},
                                             use_protected_ops    = False,
                                             )
        except Exception: self.fail("Make tokens function failed")

        # Check that types are correct
        my_tokens_dict = {token.name: token for token in my_tokens}
        self.assertEqual(my_tokens_dict["add"] .var_type, Tok.VAR_TYPE_OP               )
        self.assertEqual(my_tokens_dict["pi"]  .var_type, Tok.VAR_TYPE_FIXED_CONST      )
        self.assertEqual(my_tokens_dict["x0"]  .var_type, Tok.VAR_TYPE_INPUT_VAR        )
        self.assertEqual(my_tokens_dict["c0"]  .var_type, Tok.VAR_TYPE_CLASS_FREE_CONST )
        self.assertEqual(my_tokens_dict["c3"]  .var_type, Tok.VAR_TYPE_CLASS_FREE_CONST )
        self.assertEqual(my_tokens_dict["k0"]  .var_type, Tok.VAR_TYPE_SPE_FREE_CONST   )
        return None

    # Test make tokens function with units and complexity
    def test_make_tokens_units_and_complexity(self):
        # Test creation
        try:
            my_tokens = Func.make_tokens(
                # operations
                op_names             = ["mul", "add", "neg", "inv", "sin"],
                use_protected_ops    = False,
                # input variables
                input_var_ids        = {"x" : 0         , "v" : 1          , "t" : 2,        },
                input_var_units      = {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                input_var_complexity = {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                # constants
                constants            = {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                constants_units      = {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] },
                constants_complexity = {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                # free constants
                free_constants            = {"c0"             , "c1"               , "c2"             },
                free_constants_init_val   = {"c0" : 1.        , "c1"  : 10.        , "c2" : 1.        },
                free_constants_units      = {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                free_constants_complexity = {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                                                )
        except Exception:
            self.fail("Make tokens function failed")
        # Test that properties were encoded
        my_tokens_dict = {token.name: token for token in my_tokens}
        # Checking sample units # Checking 3 first values because phy_units is padded to match Lib.UNITS_VECTOR_SIZE
        is_equal = np.array_equal(my_tokens_dict["x"].phy_units[:3], [1, 0, 0])
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["v"].phy_units[:3], [1, -1, 0])
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["c"].phy_units[:3], [1, -1, 0])
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["M"].phy_units[:3], [0, 0, 1])
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["c0"].phy_units[:3], [0, 0, 0])
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["c1"].phy_units[:3], [1, -1, 0])
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["c2"].phy_units[:3], [0, 0, 1])
        self.assertEqual(is_equal, True)
        # Checking sample complexities
        is_equal = np.array_equal(my_tokens_dict["x"].complexity, 0.)
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["v"].complexity, 1.)
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["c"].complexity, 0.)
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["M"].complexity, 1.)
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["c1"].complexity, 0.)
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["c2"].complexity, 1.)
        self.assertEqual(is_equal, True)

    # Test make tokens function with units and complexity for Class SR
    def test_make_tokens_units_and_complexity_classSR(self):
        # Test creation
        try:
            my_tokens = Func.make_tokens(
                # operations
                op_names             = ["mul", "add", "neg", "inv", "sin"],
                use_protected_ops    = False,
                # input variables
                input_var_ids        = {"x" : 0         , "v" : 1          , "t" : 2,        },
                input_var_units      = {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                input_var_complexity = {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                # constants
                constants            = {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                constants_units      = {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] },
                constants_complexity = {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                # class free constants
                free_constants            = {"c0"             , "c1"               , "c2"             },
                free_constants_init_val   = {"c0" : 1.        , "c1"  : 10.        , "c2" : 1.        },
                free_constants_units      = {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                free_constants_complexity = {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                # spe free constants
                spe_free_constants            = {"k0"              , "k1"                , "k2"               },
                spe_free_constants_init_val   = {"k0" : 1.         , "k1"  : 10.         , "k2" : 1.          },
                spe_free_constants_units      = {"k0" : [6, 0, 11] , "k1"  : [1, -14, 3] , "k2" : [5, 13, -2] },
                spe_free_constants_complexity = {"k0" : 0.         , "k1"  : 12.         , "k2" : 0.          },
                                                )
        except Exception:
            self.fail("Make tokens function failed")
        # Test that properties were encoded
        my_tokens_dict = {token.name: token for token in my_tokens}
        # Checking sample units # Checking 3 first values because phy_units is padded to match Lib.UNITS_VECTOR_SIZE
        is_equal = np.array_equal(my_tokens_dict["x"].phy_units[:3], [1, 0, 0])
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["v"].phy_units[:3], [1, -1, 0])
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["c"].phy_units[:3], [1, -1, 0])
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["M"].phy_units[:3], [0, 0, 1])
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["c0"].phy_units[:3], [0, 0, 0])
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["c1"].phy_units[:3], [1, -1, 0])
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["c2"].phy_units[:3], [0, 0, 1])
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["k0"].phy_units[:3], [6, 0, 11])
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["k1"].phy_units[:3], [1, -14, 3])
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["k2"].phy_units[:3], [5, 13, -2])
        self.assertEqual(is_equal, True)
        # Checking sample complexities
        is_equal = np.array_equal(my_tokens_dict["x"].complexity, 0.)
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["v"].complexity, 1.)
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["c"].complexity, 0.)
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["M"].complexity, 1.)
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["c1"].complexity, 0.)
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["c2"].complexity, 1.)
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["k1"].complexity, 12.)
        self.assertEqual(is_equal, True)
        is_equal = np.array_equal(my_tokens_dict["k2"].complexity, 0.)
        self.assertEqual(is_equal, True)

    # Test make tokens function with units and complexity, missing units or complexity in dict
    def test_make_tokens_units_and_complexity_missing_info_warnings(self):

        # Test missing units in input variables
        with self.assertWarns(Warning):
            my_tokens = Func.make_tokens(
                # operations
                op_names             = ["mul", "add", "neg", "inv", "sin"],
                use_protected_ops    = False,
                # input variables
                input_var_ids        = {"x" : 0         , "v" : 1          , "t" : 2,        },
                input_var_units      = {"x" : [1, 0, 0] , "v" : [1, -1, 0]},
                input_var_complexity = {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                # constants
                constants            = {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                constants_units      = {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] },
                constants_complexity = {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                # free constants
                free_constants            = {"c0"             , "c1"               , "c2"             },
                free_constants_init_val   = {"c0" : 1.        , "c1"  : 10.        , "c2" : 1.        },
                free_constants_units      = {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                free_constants_complexity = {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                                                )
        # Test missing complexity in input variables
        with self.assertWarns(Warning):
            my_tokens = Func.make_tokens(
                # operations
                op_names             = ["mul", "add", "neg", "inv", "sin"],
                use_protected_ops    = False,
                # input variables
                input_var_ids        = {"x" : 0         , "v" : 1          , "t" : 2,        },
                input_var_units      = {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                input_var_complexity = {"x" : 0.        , "v" : 1.         },
                # constants
                constants            = {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                constants_units      = {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] },
                constants_complexity = {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                # free constants
                free_constants            = {"c0"             , "c1"               , "c2"             },
                free_constants_init_val   = {"c0" : 1.        , "c1"  : 10.        , "c2" : 1.        },
                free_constants_units      = {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                free_constants_complexity = {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                                                )
        # Test missing units in constants
        with self.assertWarns(Warning):
            my_tokens = Func.make_tokens(
                # operations
                op_names             = ["mul", "add", "neg", "inv", "sin"],
                use_protected_ops    = False,
                # input variables
                input_var_ids        = {"x" : 0         , "v" : 1          , "t" : 2,        },
                input_var_units      = {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                input_var_complexity = {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                # constants
                constants            = {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                constants_units      = {"pi" : [0, 0, 0] , "c" : [1, -1, 0] },
                constants_complexity = {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                # free constants
                free_constants            = {"c0"             , "c1"               , "c2"             },
                free_constants_init_val   = {"c0" : 1.        , "c1"  : 10.        , "c2" : 1.        },
                free_constants_units      = {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                free_constants_complexity = {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                                                )
        # Test missing complexity in constants
        with self.assertWarns(Warning):
            my_tokens = Func.make_tokens(
                # operations
                op_names             = ["mul", "add", "neg", "inv", "sin"],
                use_protected_ops    = False,
                # input variables
                input_var_ids        = {"x" : 0         , "v" : 1          , "t" : 2,        },
                input_var_units      = {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                input_var_complexity = {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                # constants
                constants            = {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                constants_units      = {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] },
                constants_complexity = {"pi" : 0.        , "c" : 0.        },
                # free constants
                free_constants            = {"c0"             , "c1"               , "c2"             },
                free_constants_init_val   = {"c0" : 1.        , "c1"  : 10.        , "c2" : 1.        },
                free_constants_units      = {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                free_constants_complexity = {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                                                )
        # Test missing units in free constants
        with self.assertWarns(Warning):
            my_tokens = Func.make_tokens(
                # operations
                op_names             = ["mul", "add", "neg", "inv", "sin"],
                use_protected_ops    = False,
                # input variables
                input_var_ids        = {"x" : 0         , "v" : 1          , "t" : 2,        },
                input_var_units      = {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                input_var_complexity = {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                # constants
                constants            = {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                constants_units      = {"pi" : [0, 0, 0] , "c" : [1, -1, 0] },
                constants_complexity = {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                # free constants
                free_constants            = {"c0"             , "c1"               , "c2"             },
                free_constants_init_val   = {"c0" : 1.        , "c1"  : 10.        , "c2" : 1.        },
                free_constants_units      = {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] ,                  },
                free_constants_complexity = {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                                                )
        # Test missing complexity in free constants
        with self.assertWarns(Warning):
            my_tokens = Func.make_tokens(
                # operations
                op_names             = ["mul", "add", "neg", "inv", "sin"],
                use_protected_ops    = False,
                # input variables
                input_var_ids        = {"x" : 0         , "v" : 1          , "t" : 2,        },
                input_var_units      = {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                input_var_complexity = {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                # constants
                constants            = {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                constants_units      = {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] },
                constants_complexity = {"pi" : 0.        , "c" : 0.        },
                # free constants
                free_constants            = {"c0"             , "c1"               , "c2"             },
                free_constants_init_val   = {"c0" : 1.        , "c1"  : 10.        , "c2" : 1.        },
                free_constants_units      = {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                free_constants_complexity = {"c0" : 0.        , "c1"  : 0.         ,                  },
                                                )
        # Test missing init_val in free constants
        with self.assertWarns(Warning):
            my_tokens = Func.make_tokens(
                # operations
                op_names             = ["mul", "add", "neg", "inv", "sin"],
                use_protected_ops    = False,
                # input variables
                input_var_ids        = {"x" : 0         , "v" : 1          , "t" : 2,        },
                input_var_units      = {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                input_var_complexity = {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                # constants
                constants            = {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                constants_units      = {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] },
                constants_complexity = {"pi" : 0.        , "c" : 0.                          },
                # free constants
                free_constants            = {"c0"             , "c1"               , "c2"             },
                free_constants_init_val   = {"c0" : 1.        , "c1"  : 10.        ,                  },
                free_constants_units      = {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                free_constants_complexity = {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                                                )
        # Test missing units in spe free constants
        with self.assertWarns(Warning):
            my_tokens = Func.make_tokens(
                # operations
                op_names             = ["mul", "add", "neg", "inv", "sin"],
                use_protected_ops    = False,
                # input variables
                input_var_ids        = {"x" : 0         , "v" : 1          , "t" : 2,        },
                input_var_units      = {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                input_var_complexity = {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                # constants
                constants            = {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                constants_units      = {"pi" : [0, 0, 0] , "c" : [1, -1, 0] },
                constants_complexity = {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                # spe free constants
                spe_free_constants            = {"c0"             , "c1"               , "c2"             },
                spe_free_constants_init_val   = {"c0" : 1.        , "c1"  : 10.        , "c2" : 1.        },
                spe_free_constants_units      = {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] ,                  },
                spe_free_constants_complexity = {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                                                )
        # Test missing complexity in spe free constants
        with self.assertWarns(Warning):
            my_tokens = Func.make_tokens(
                # operations
                op_names             = ["mul", "add", "neg", "inv", "sin"],
                use_protected_ops    = False,
                # input variables
                input_var_ids        = {"x" : 0         , "v" : 1          , "t" : 2,        },
                input_var_units      = {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                input_var_complexity = {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                # constants
                constants            = {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                constants_units      = {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] },
                constants_complexity = {"pi" : 0.        , "c" : 0.        },
                # spe free constants
                spe_free_constants            = {"c0"             , "c1"               , "c2"             },
                spe_free_constants_init_val   = {"c0" : 1.        , "c1"  : 10.        , "c2" : 1.        },
                spe_free_constants_units      = {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                spe_free_constants_complexity = {"c0" : 0.        , "c1"  : 0.         ,                  },
                                                )
        # Test missing init_val in free constants
        with self.assertWarns(Warning):
            my_tokens = Func.make_tokens(
                # operations
                op_names             = ["mul", "add", "neg", "inv", "sin"],
                use_protected_ops    = False,
                # input variables
                input_var_ids        = {"x" : 0         , "v" : 1          , "t" : 2,        },
                input_var_units      = {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                input_var_complexity = {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                # constants
                constants            = {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                constants_units      = {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] },
                constants_complexity = {"pi" : 0.        , "c" : 0.                          },
                # spe free constants
                spe_free_constants            = {"c0"             , "c1"               , "c2"             },
                spe_free_constants_init_val   = {"c0" : 1.        , "c1"  : 10.        ,                  },
                spe_free_constants_units      = {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                spe_free_constants_complexity = {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                                                )

    # Test make tokens function with wrong units
    def test_make_tokens_units_and_complexity_wrong_unit(self):
        # Test unit too large in input variables
        with self.assertRaises(AssertionError):
            my_tokens = Func.make_tokens(
                # operations
                op_names             = ["mul", "add", "neg", "inv", "sin"],
                use_protected_ops    = False,
                # input variables
                input_var_ids        = {"x" : 0         , "v" : 1          , "t" : 2,        },
                input_var_units      = {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : np.ones(10000) },
                input_var_complexity = {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                # constants
                constants            = {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                constants_units      = {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] },
                constants_complexity = {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                # free constants
                free_constants            = {"c0"             , "c1"               , "c2"             },
                free_constants_init_val   = {"c0" : 1.        , "c1"  : 10.        , "c2" : 1.        },
                free_constants_units      = {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                free_constants_complexity = {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                                                )
        # Test unit too large in constants
        with self.assertRaises(AssertionError):
            my_tokens = Func.make_tokens(
                # operations
                op_names             = ["mul", "add", "neg", "inv", "sin"],
                use_protected_ops    = False,
                # input variables
                input_var_ids        = {"x" : 0         , "v" : 1          , "t" : 2,        },
                input_var_units      = {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                input_var_complexity = {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                # constants
                constants            = {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                constants_units      = {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : np.ones(10000) },
                constants_complexity = {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                # free constants
                free_constants            = {"c0"             , "c1"               , "c2"             },
                free_constants_init_val   = {"c0" : 1.        , "c1"  : 10.        , "c2" : 1.        },
                free_constants_units      = {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                free_constants_complexity = {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                                                )
        # Test unit too large in free constants
        with self.assertRaises(AssertionError):
            my_tokens = Func.make_tokens(
                # operations
                op_names             = ["mul", "add", "neg", "inv", "sin"],
                use_protected_ops    = False,
                # input variables
                input_var_ids        = {"x" : 0         , "v" : 1          , "t" : 2,        },
                input_var_units      = {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                input_var_complexity = {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                # constants
                constants            = {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                constants_units      = {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] },
                constants_complexity = {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                # free constants
                free_constants            = {"c0"             , "c1"               , "c2"             },
                free_constants_init_val   = {"c0" : 1.        , "c1"  : 10.        , "c2" : 1.        },
                free_constants_units      = {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : np.ones(10000) },
                free_constants_complexity = {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                                                )
        # Test unit too large in spe free constants
        with self.assertRaises(AssertionError):
            my_tokens = Func.make_tokens(
                # operations
                op_names             = ["mul", "add", "neg", "inv", "sin"],
                use_protected_ops    = False,
                # input variables
                input_var_ids        = {"x" : 0         , "v" : 1          , "t" : 2,        },
                input_var_units      = {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                input_var_complexity = {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                # constants
                constants            = {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                constants_units      = {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] },
                constants_complexity = {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                # spe free constants
                spe_free_constants            = {"c0"             , "c1"               , "c2"             },
                spe_free_constants_init_val   = {"c0" : 1.        , "c1"  : 10.        , "c2" : 1.        },
                spe_free_constants_units      = {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : np.ones(10000) },
                spe_free_constants_complexity = {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                                                )

        # Test units having wrong variable type
        with self.assertRaises(AssertionError):
            my_tokens = Func.make_tokens(
                # operations
                op_names             = ["mul", "add", "neg", "inv", "sin"],
                use_protected_ops    = False,
                # input variables
                input_var_ids        = {"x" : 0         , "v" : 1          , "t" : 2,        },
                input_var_units      = {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                input_var_complexity = {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                # constants
                constants            = {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                constants_units      = {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : ['a', 'b', 1] },
                constants_complexity = {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                # free constants
                free_constants            = {"c0"             , "c1"               , "c2"             },
                free_constants_init_val   = {"c0" : 1.        , "c1"  : 10.        , "c2" : 1.        },
                free_constants_units      = {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                free_constants_complexity = {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                                                )
        # Test units having wrong shape
        with self.assertRaises(AssertionError):
            my_tokens = Func.make_tokens(
                # operations
                op_names             = ["mul", "add", "neg", "inv", "sin"],
                use_protected_ops    = False,
                # input variables
                input_var_ids        = {"x" : 0         , "v" : 1          , "t" : 2,        },
                input_var_units      = {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                input_var_complexity = {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                # constants
                constants            = {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                constants_units      = {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : np.ones((7,7)) },
                constants_complexity = {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                # free constants
                free_constants            = {"c0"             , "c1"               , "c2"             },
                free_constants_init_val   = {"c0" : 1.        , "c1"  : 10.        , "c2" : 1.        },
                free_constants_units      = {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                free_constants_complexity = {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                                                )



    # Test make tokens function with units and complexity for Class SR
    def test_make_tokens_init_values(self):
        n_realizations = 10
        a = 12.
        b = 4.
        c = 1.
        aa = np.random.rand (n_realizations,)
        bb = np.ones        (n_realizations,)
        cc = 1.
        # Test creation
        try:
            my_tokens = Func.make_tokens(
                # operations
                op_names             = ["mul", "add", "neg", "inv", "sin"],
                use_protected_ops    = False,
                # input variables
                input_var_ids        = {"x" : 0         , "v" : 1          , "t" : 2,        },
                input_var_units      = {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                input_var_complexity = {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                # constants
                constants            = {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                constants_units      = {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] },
                constants_complexity = {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                # class free constants
                free_constants            = {"c0"             , "c1"               , "c2"             },
                free_constants_init_val   = {"c0" : a         , "c1"  : b          , "c2" : c         },
                free_constants_units      = {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                free_constants_complexity = {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                # spe free constants
                spe_free_constants            = {"k0"              , "k1"                , "k2"               },
                spe_free_constants_init_val   = {"k0" : aa         , "k1"  : bb          , "k2" : cc          },
                spe_free_constants_units      = {"k0" : [6, 0, 11] , "k1"  : [1, -14, 3] , "k2" : [5, 13, -2] },
                spe_free_constants_complexity = {"k0" : 0.         , "k1"  : 12.         , "k2" : 0.          },
                                                )
        except Exception:
            self.fail("Make tokens function failed")

        # Testing initial values
        my_tokens_dict = {token.name: token for token in my_tokens}

        self.assertEqual(my_tokens_dict["c0"].init_val, a)
        self.assertEqual(my_tokens_dict["c1"].init_val, b)
        self.assertEqual(my_tokens_dict["c2"].init_val, c)
        self.assertTrue(np.all(my_tokens_dict["k0"].init_val == aa))
        self.assertTrue(np.all(my_tokens_dict["k1"].init_val == bb))
        self.assertEqual(my_tokens_dict["k2"].init_val, cc)
        return None

    # Test unknown function exception
    def test_make_tokens_unknown_function(self):
        with self.assertRaises(Func.UnknownFunction, ):
            my_tokens = Func.make_tokens(op_names=["mul", "function_that_does_not_exist"])

    # Test all protected tokens and output shapes of their underlying functions
    def test_protected_tokens(self):
        my_tokens = Func.make_tokens(op_names="all",  #
                                     constants={"pi": np.pi, "c": 3e8},
                                     use_protected_ops=True, )
        for token in my_tokens:
            test_one_token(tester=self, token=token)

    # Test all unprotected tokens and output shapes of their underlying functions
    def test_unprotected_tokens(self):
        my_tokens = Func.make_tokens(op_names="all",
                                     constants={"pi": np.pi, "c": 3e8},
                                     use_protected_ops=False, )
        for token in my_tokens:
            test_one_token(tester=self, token=token)

    # Test that arity and complexity have the same values in protected and unprotected modes
    def test_protected_unprotected_same_attributes(self):
        unprotected_tokens = Func.make_tokens(op_names="all", use_protected_ops=False, )
        protected_tokens   = Func.make_tokens(op_names="all", use_protected_ops=True, )
        unprotected_tokens_names = np.array([token.name for token in unprotected_tokens])
        protected_tokens_names   = np.array([token.name for token in protected_tokens])
        for name in protected_tokens_names:
            # ------------------------------------ protected token ------------------------------------
            protected_token = protected_tokens[np.argwhere(protected_tokens_names == name)]
            # Check that there is only one version of current token in protected tokens
            self.assertEqual(np.array_equal(protected_token.shape, [1, 1]), True)
            protected_token = protected_token[0, 0]
            # ------------------------------------ unprotected token ------------------------------------
            unprotected_token = unprotected_tokens[np.argwhere(unprotected_tokens_names == name)]
            # Check that there is only one version of current token in unprotected tokens
            self.assertEqual(np.array_equal(unprotected_token.shape, [1, 1]), True)
            unprotected_token = unprotected_token[0, 0]
            # ---------------------------- check that attributes are the same ----------------------------
            for attribute_name, attribute_val in protected_token.__dict__.items():
                attribute_val_in_protected   = protected_token  .__dict__[attribute_name]
                attribute_val_in_unprotected = unprotected_token.__dict__[attribute_name]
                # Checking all attributes except function which is bound to be different in protected vs unprotected
                if attribute_name != "function":
                    # Do regular comparison for str (can not compare str using equal_nan=True)
                    if isinstance(attribute_val_in_protected   , str) or \
                       isinstance(attribute_val_in_unprotected , str):
                        is_equal = np.array_equal(attribute_val_in_protected  ,
                                                  attribute_val_in_unprotected,
                                                  equal_nan=False)
                    else:
                        is_equal = np.array_equal(attribute_val_in_protected  ,
                                                  attribute_val_in_unprotected,
                                                  equal_nan=True)
                    self.assertEqual(is_equal, True)

    # Test that arity of functions does not exceed the max nb of children
    def test_max_arity(self):
        protected_tokens = Func.make_tokens(op_names="all", use_protected_ops=True, )
        for token in protected_tokens:
            self.assertIs(token.arity <= Func.Tok.MAX_ARITY, True)

    # Test that tokens pointing to data work
    def test_data_pointers_work(self):
        const_data0 = data_conversion ( np.random.rand() )
        const_data1 = data_conversion ( np.random.rand() )
        my_tokens = Func.make_tokens(op_names="all",
                                     constants={"pi": np.pi, "const1": 1., "data0": const_data0,
                                                           "data1": const_data1}, )
        my_tokens_dict = {token.name: token for token in my_tokens}
        # test that tokens point to data
        bool = np.array_equal(data_conversion_inv ( my_tokens_dict["pi"]()    ) , np.pi)
        self.assertEqual(bool, True)
        bool = np.array_equal(data_conversion_inv ( my_tokens_dict["const1"]() ) , 1.)
        self.assertEqual(bool, True)
        bool = np.array_equal(data_conversion_inv ( my_tokens_dict["data0"]()  ) , const_data0)
        self.assertEqual(bool, True)
        bool = np.array_equal(data_conversion_inv ( my_tokens_dict["data1"]()  ) , const_data1)
        self.assertEqual(bool, True)
        ## test mul(data0,data1) === np.multiply(const_data0, const_data1)
        bool = np.array_equal(data_conversion_inv ( my_tokens_dict["mul"](my_tokens_dict["data0"](), my_tokens_dict["data1"]())),
                              data_conversion_inv ( my_tokens_dict["mul"].function(const_data0, const_data1)))
        self.assertEqual(bool, True)

    # Test that behavior objects contain different operation names
    # (eg. "add" must only have one unique behavior)
    def test_behavior_contain_different_ops(self):
        unprotected_tokens = Func.make_tokens(op_names="all", use_protected_ops=False, )
        protected_tokens = Func.make_tokens(op_names="all", use_protected_ops=True, )
        unprotected_tokens_names = [token.name for token in unprotected_tokens]
        protected_tokens_names = [token.name for token in protected_tokens]
        for name in unprotected_tokens_names+protected_tokens_names:
            count = 0
            for _, behavior in Func.OP_UNIT_BEHAVIORS_DICT.items():
                if (name in behavior.op_names): count+=1
            if count >1: self.fail("Token named %s appears in more than one behavior."%(name))

    # Test that each behavior has a unique identifier
    def test_behavior_have_unique_ids(self):
        ids = [behavior.behavior_id for _, behavior in Func.OP_UNIT_BEHAVIORS_DICT.items()]
        if not len(ids) == len(np.unique(ids)):
            self.fail("Behaviors ids are not unique, ids = %s"%(str(ids)))

    # Test that tokens encoded with dimensionless behavior id are dimensionless
    def test_behavior_dimensionless_are_dimensionless(self):
        dimensionless_id = Func.OP_UNIT_BEHAVIORS_DICT["UNARY_DIMENSIONLESS_OP"].behavior_id
        unprotected_tokens = Func.make_tokens(op_names="all", use_protected_ops=False, )
        protected_tokens = Func.make_tokens(op_names="all", use_protected_ops=True, )
        for token in unprotected_tokens.tolist() + protected_tokens.tolist():
            if token.behavior_id == dimensionless_id:
                self.assertEqual(token.is_constraining_phy_units, True)
                works_bool = np.array_equal(token.phy_units, np.zeros(Func.Tok.UNITS_VECTOR_SIZE))
                self.assertEqual(works_bool, True)

    # Test GroupUnitsBehavior
    def test_GroupUnitsBehavior (self):
        behavior_id_mul = Func.UNIT_BEHAVIORS_DICT["MULTIPLICATION_OP"].behavior_id
        behavior_id_div = Func.UNIT_BEHAVIORS_DICT["DIVISION_OP"]      .behavior_id
        group = Func.UNIT_BEHAVIORS_DICT["BINARY_MULTIPLICATIVE_OP"]
        # --- Vect 1 ---
        test_vect_ids   = np.array([behavior_id_mul]*int(1e4) + [behavior_id_div]*int(1e4),)
        batch_size = test_vect_ids.shape[0]
        # Should contain all True statements
        t0 = time.perf_counter()
        equal_test = group.is_id(test_vect_ids)
        t1 = time.perf_counter()
        # print("Eq time = %f ms"%((t1-t0)*1e3))
        works_bool = (equal_test.dtype == bool)
        self.assertEqual(works_bool, True)
        works_bool = np.array_equal(equal_test, np.full(shape=batch_size, fill_value=True))
        self.assertEqual(works_bool, True)
        # --- Vect 2 ---
        equal_test = group.is_id([behavior_id_mul, behavior_id_div, 999999999999])
        works_bool = (equal_test.dtype == bool)
        self.assertEqual(works_bool, True)
        works_bool = np.array_equal(equal_test, [True, True, False])
        self.assertEqual(works_bool, True)
        # --- Single value ---
        equal_test = group.is_id(behavior_id_mul)
        works_bool = (equal_test.dtype == bool)
        self.assertEqual(works_bool, True)
        works_bool = np.array_equal(equal_test, True)
        self.assertEqual(works_bool, True)
        # --- Single value ---
        equal_test = group.is_id(999999999999)
        works_bool = (equal_test.dtype == bool)
        self.assertEqual(works_bool, True)
        works_bool = np.array_equal(equal_test, False)
        self.assertEqual(works_bool, True)


    def test_protected_functions_plots (self):

        do_show = False
        do_save = False
        make_plots = do_show or do_save  # if either is true, make plots

        n_plot = int(1e4)

        if make_plots:
            # protected_div
            x1 = torch.linspace(-1, 1, n_plot)
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            ax.plot(x1, Func.protected_div(x1, 4 * x1), label="protected_div", color="k", linestyle="solid")
            ax.plot(x1, Func.protected_div(x1, -4 * x1), label="protected_div (neg num)", color="r", linestyle="dotted")
            ax.legend()
            if do_show: plt.show()
            if do_save: fig.savefig("protected_div.png")

            # protected_exp
            x1 = torch.linspace(0, 1.1*Func.EXP_THRESHOLD, n_plot)
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            ax.plot(x1, Func.protected_exp(x1), label="protected_exp", color="k")
            ax.legend()
            if do_show: plt.show()
            if do_save: fig.savefig("protected_exp.png")

            # protected_log
            x1 = torch.linspace(-10*Func.EPSILON, 10*Func.EPSILON, n_plot)
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            ax.plot(x1, Func.protected_log(x1), label="protected_log", color="k")
            ax.legend()
            if do_show: plt.show()
            if do_save: fig.savefig("protected_log.png")

            # protected_logabs
            x1 = torch.linspace(-10*Func.EPSILON, 10*Func.EPSILON, n_plot)
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            ax.plot(x1, Func.protected_logabs(x1), label="protected_logabs", color="k")
            ax.legend()
            if do_show: plt.show()
            if do_save: fig.savefig("protected_logabs.png")

            # protected_sqrt
            x1 = torch.linspace(-10, 10, n_plot)
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            ax.plot(x1, Func.protected_sqrt(x1), label="protected_sqrt", color="k")
            ax.legend()
            if do_show: plt.show()
            if do_save: fig.savefig("protected_sqrt.png")

            # protected_inv
            x1 = torch.linspace(-10*Func.EPSILON, 10*Func.EPSILON, n_plot)
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            ax.plot(x1, Func.protected_inv(x1), label="protected_inv", color="k")
            ax.legend()
            if do_show: plt.show()
            if do_save: fig.savefig("protected_inv.png")

            # protected_expneg
            x1 = torch.linspace(-1.1*Func.EXP_THRESHOLD, 0, n_plot)
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            ax.plot(x1, Func.protected_expneg(x1), label="protected_expneg", color="k")
            ax.legend()
            if do_show: plt.show()
            if do_save: fig.savefig("protected_expneg.png")

            # protected_n2
            x1 = torch.linspace(-2 * Func.INF, 2 * Func.INF, n_plot)
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            ax.plot(x1, Func.protected_n2(x1), label="protected_n2", color="k")
            ax.legend()
            if do_show: plt.show()
            if do_save: fig.savefig("protected_n2.png")

            # protected_n3
            x1 = torch.linspace(-2 * Func.INF, 2 * Func.INF, n_plot)
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            ax.plot(x1, Func.protected_n3(x1), label="protected_n3", color="k")
            ax.legend()
            if do_show: plt.show()
            if do_save: fig.savefig("protected_n3.png")

            # protected_n4
            x1 = torch.linspace(-2 * Func.INF, 2 * Func.INF, n_plot)
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            ax.plot(x1, Func.protected_n4(x1), label="protected_n4", color="k")
            ax.legend()
            if do_show: plt.show()
            if do_save: fig.savefig("protected_n4.png")

            # protected_arcsin
            x1 = torch.linspace(-2, 2, n_plot)
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            ax.plot(x1, Func.protected_arcsin(x1), label="protected_arcsin", color="k")
            ax.legend()
            if do_show: plt.show()
            if do_save: fig.savefig("protected_arcsin.png")

            # protected_arccos
            x1 = torch.linspace(-2, 2, n_plot)
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            ax.plot(x1, Func.protected_arccos(x1), label="protected_arccos", color="k")
            ax.legend()
            if do_show: plt.show()
            if do_save: fig.savefig("protected_arccos.png")

            # protected_torch_pow
            x1 = torch.linspace(-0.1*Func.INF, Func.INF, n_plot)
            x2 = torch.linspace(-8, 8, n_plot)
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            ax.plot(x1, Func.protected_torch_pow(x1, x2), label="protected_div", color="k", linestyle="solid")
            ax.legend()
            if do_show: plt.show()
            if do_save: fig.savefig("protected_torch_pow.png")

        return None

if __name__ == '__main__':
    unittest.main(verbosity=2)
