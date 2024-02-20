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
from physo.physym.tokenize import make_tokens

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


class TokenizeTest(unittest.TestCase):

    # Test make tokens function
    def test_make_tokens(self):
        op_names = ["mul", "add", "neg", "inv", "sin"]
        try:
            # Raises some warnings due to lack of initial values etc (not a problem)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                my_tokens = make_tokens(op_names          = op_names,
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
                my_tokens = make_tokens(op_names             = op_names,
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
            my_tokens = make_tokens(
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
            my_tokens = make_tokens(
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
            my_tokens = make_tokens(
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
            my_tokens = make_tokens(
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
            my_tokens = make_tokens(
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
            my_tokens = make_tokens(
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
            my_tokens = make_tokens(
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
            my_tokens = make_tokens(
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
            my_tokens = make_tokens(
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
            my_tokens = make_tokens(
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
            my_tokens = make_tokens(
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
            my_tokens = make_tokens(
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
            my_tokens = make_tokens(
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
            my_tokens = make_tokens(
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
            my_tokens = make_tokens(
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
            my_tokens = make_tokens(
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
            my_tokens = make_tokens(
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
            my_tokens = make_tokens(
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
            my_tokens = make_tokens(
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
            my_tokens = make_tokens(op_names=["mul", "function_that_does_not_exist"])

if __name__ == '__main__':
    unittest.main(verbosity=2)
