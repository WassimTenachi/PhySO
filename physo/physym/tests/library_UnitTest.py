import unittest
import numpy as np
import warnings

# Internal imports
from physo.physym import library as Lib
from physo.physym import token as Tok


class LibraryTest(unittest.TestCase):

    # Check all library creation + append configuration, sizes, types of data contained within library
    def test_library_creation_and_types(self):
        def test_lib_types_and_size(my_lib, expected_n_lib):
            # Idx of superparent
            self.assertEqual(my_lib.lib_name[my_lib.superparent_idx], my_lib.superparent.name)
            self.assertEqual(my_lib.lib_name[my_lib.dummy_idx      ], my_lib.dummy.name)
            # Sizes
            self.assertEqual(my_lib.n_library, expected_n_lib + len(my_lib.placeholders))
            self.assertEqual(my_lib.n_choices, expected_n_lib)
            # Shapes
            bool_works = np.array_equal(my_lib.phy_units, my_lib.properties.phy_units[0,:], equal_nan=True)
            self.assertEqual(bool_works, True)
            bool_works = np.array_equal(my_lib.arity, my_lib.properties.arity[0, :], equal_nan=True)
            self.assertEqual(bool_works, True)
            # Test properties vectors types # https://numpy.org/doc/stable/reference/arrays.scalars.html
            self.assertEqual(                my_lib            .lib_function                 .dtype , np.object_ )
            self.assertTrue( np.issubdtype ( my_lib.properties .arity                        .dtype , np.integer    ))
            self.assertTrue( np.issubdtype ( my_lib.properties .complexity                   .dtype , np.floating   ))
            self.assertTrue( np.issubdtype ( my_lib.properties .var_type                     .dtype , np.integer    ))
            self.assertTrue( np.issubdtype ( my_lib.properties .var_id                       .dtype , np.integer    ))
            self.assertTrue( np.issubdtype ( my_lib.properties .is_constraining_phy_units    .dtype , np.bool_      ))
            self.assertTrue( np.issubdtype ( my_lib.properties .phy_units                    .dtype , np.floating   ))
            self.assertTrue( np.issubdtype ( my_lib.properties .behavior_id                  .dtype , np.integer    ))
            self.assertTrue( np.issubdtype ( my_lib.properties .is_power                     .dtype , np.bool_      ))
            self.assertTrue( np.issubdtype ( my_lib.properties .power                        .dtype , np.floating   ))

        # -------- Test args --------
        custom_tokens = [
        Tok.TokenInputVar(name='x0', sympy_repr='x0', complexity=0,
                       var_id=0,
                       is_constraining_phy_units=True,
                       phy_units=[1.,0.,0.,0.,0.,0.,0.]),
        Tok.TokenInputVar(name='x1', sympy_repr='x1', complexity=0,
                       var_id=1),
        Tok.TokenOp(name='add', sympy_repr='add', arity=2, complexity=0,
                        function=np.add,),
        Tok.TokenOp(name='cos', sympy_repr='cos', arity=1, complexity=0,
                        function=np.cos,),
        Tok.TokenOp(name='pi', sympy_repr='pi', arity=0, complexity=0,
                       function=lambda const=np.pi: const,),
                        ]
        n_tokens_via_custom = len(custom_tokens)
        # Initial values
        a,b,c = 1.,10.,1.
        n_realizations = 5
        aa, bb, cc = [1.,2.,3.,4.,5.], [10.,20.,30.,40.,50.] , 1.
        expected_class_init_vals = np.array([a, b, c,])
        expected_spe_init_vals   = np.array([np.array(aa), np.array(bb), np.array([cc]),], dtype=object)
        expected_spe_init_vals_after_pad = np.array([aa, bb, np.full((n_realizations,), cc)])
        # -------- Test args_make_tokens --------
        args_make_tokens = {
                # operations
                "op_names"             : ["mul", "neg", "inv", "sin"],
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
                "free_constants_init_val"   : {"c0" : a         , "c1"  : b          , "c2" : c         },
                "free_constants_units"      : {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                "free_constants_complexity" : {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                # free constants
                "spe_free_constants"            : {"k0"              , "k1"                   , "k2"             },
                "spe_free_constants_init_val"   : {"k0" : aa         , "k1"  : bb             , "k2" : cc        },
                "spe_free_constants_units"      : {"k0" : [0, 0, 0]  , "k1"  : [1, -1, 0]     , "k2" : [0, 0, 1] },
                "spe_free_constants_complexity" : {"k0" : 0.         , "k1"  : 0.             , "k2" : 1.        },
                           }
        n_tokens_via_make = len(args_make_tokens["op_names"]) + len(args_make_tokens["input_var_ids"])\
                            + len(args_make_tokens["constants"]) + len(args_make_tokens["free_constants"]) \
                            + len(args_make_tokens["spe_free_constants"])

        # -------- Test args_make_tokens only --------
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Raises some warnings due to some units provided (this is ok)
                my_lib = Lib.Library(args_make_tokens = args_make_tokens)
        except:
            self.fail("Library creation failed.")
        # Test lib
        test_lib_types_and_size(my_lib=my_lib, expected_n_lib=n_tokens_via_make)
        # Test superparent units
        bool_works = np.array_equal(my_lib.superparent.phy_units, np.full(Tok.UNITS_VECTOR_SIZE, np.nan), equal_nan=True)
        self.assertEqual(bool_works, True)
        # Test sample token units
        bool_works = np.array_equal(my_lib.phy_units[my_lib.lib_name_to_idx["v"]][0:3],
                                    args_make_tokens["input_var_units"]["v"])
        self.assertEqual(bool_works, True)
        # Test initial values
        bool_works = np.array_equal(my_lib.class_free_constants_init_val, expected_class_init_vals)
        self.assertEqual(bool_works, True)
        bool_works = np.array([np.array_equal(a,b) for i, (a,b) in enumerate(zip(my_lib.spe_free_constants_init_val, expected_spe_init_vals))]).all()
        self.assertEqual(bool_works, True)
        my_lib.check_and_pad_spe_free_const_init_val(n_realizations=n_realizations)
        bool_works = np.array_equal(my_lib.spe_free_constants_init_val, expected_spe_init_vals_after_pad)
        self.assertEqual(bool_works, True)

        # -------- Test custom_tokens only --------
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Raises some warnings due to some units provided (this is ok)
                my_lib = Lib.Library(custom_tokens = custom_tokens)
        except:
            self.fail("Library creation failed.")
        # Test lib
        test_lib_types_and_size(my_lib=my_lib, expected_n_lib=n_tokens_via_custom)
        # Test superparent units
        bool_works = np.array_equal(my_lib.superparent.phy_units, np.full(Tok.UNITS_VECTOR_SIZE, np.nan), equal_nan=True)
        self.assertEqual(bool_works, True)
        # Test sample token units
        bool_works = np.array_equal(my_lib.phy_units[my_lib.lib_name_to_idx["x0"]][0:3],
                                    custom_tokens[0].phy_units[0:3])
        self.assertEqual(bool_works, True)

        # -------- Test append custom_tokens then args_make_tokens --------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Raises some warnings due to some units provided (this is ok)
            my_lib = Lib.Library(custom_tokens=custom_tokens)
            my_lib.append_from_tokenize(args_make_tokens=args_make_tokens)
        test_lib_types_and_size(my_lib=my_lib, expected_n_lib=n_tokens_via_custom+n_tokens_via_make)

        # Test superparent units
        bool_works = np.array_equal(my_lib.superparent.phy_units, np.full(Tok.UNITS_VECTOR_SIZE, np.nan),
                                    equal_nan=True)
        self.assertEqual(bool_works, True)
        # Test sample token units
        bool_works = np.array_equal(my_lib.phy_units[my_lib.lib_name_to_idx["x0"]][0:3],
                                    custom_tokens[0].phy_units[0:3])
        self.assertEqual(bool_works, True)
        bool_works = np.array_equal(my_lib.phy_units[my_lib.lib_name_to_idx["v"]][0:3],
                                    args_make_tokens["input_var_units"]["v"])
        self.assertEqual(bool_works, True)

        # -------- Test append args_make_tokens then custom_tokens --------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Raises some warnings due to some units provided (this is ok)
            my_lib = Lib.Library(args_make_tokens=args_make_tokens)
            my_lib.append_custom_tokens(custom_tokens=custom_tokens)
        test_lib_types_and_size(my_lib=my_lib, expected_n_lib=n_tokens_via_custom + n_tokens_via_make)

        # Test superparent units
        bool_works = np.array_equal(my_lib.superparent.phy_units, np.full(Tok.UNITS_VECTOR_SIZE, np.nan),
                                    equal_nan=True)
        self.assertEqual(bool_works, True)
        # Test sample token units
        bool_works = np.array_equal(my_lib.phy_units[my_lib.lib_name_to_idx["x0"]][0:3],
                                    custom_tokens[0].phy_units[0:3])
        self.assertEqual(bool_works, True)
        bool_works = np.array_equal(my_lib.phy_units[my_lib.lib_name_to_idx["v"]][0:3],
                                    args_make_tokens["input_var_units"]["v"])
        self.assertEqual(bool_works, True)

        # -------- Test custom_tokens and args_make_tokens --------
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Raises some warnings due to some units provided (this is ok)
                my_lib = Lib.Library(custom_tokens=custom_tokens, args_make_tokens=args_make_tokens)
        except:
            self.fail("Library creation failed.")
        # Test lib
        test_lib_types_and_size(my_lib=my_lib, expected_n_lib=n_tokens_via_custom + n_tokens_via_make)
        # Test superparent units
        bool_works = np.array_equal(my_lib.superparent.phy_units, np.full(Tok.UNITS_VECTOR_SIZE, np.nan),
                                    equal_nan=True)
        self.assertEqual(bool_works, True)
        # Test sample token units
        bool_works = np.array_equal(my_lib.phy_units[my_lib.lib_name_to_idx["x0"]][0:3],
                                    custom_tokens[0].phy_units[0:3])
        self.assertEqual(bool_works, True)
        bool_works = np.array_equal(my_lib.phy_units[my_lib.lib_name_to_idx["v"]][0:3],
                                    args_make_tokens["input_var_units"]["v"])
        self.assertEqual(bool_works, True)

        # -------- Test custom_tokens and args_make_tokens and customized superparent --------
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Raises some warnings due to some units provided (this is ok)
                my_lib = Lib.Library(custom_tokens=custom_tokens, args_make_tokens=args_make_tokens,
                                     superparent_units=[1, -1, 0], superparent_name="v")
        except:
            self.fail("Library creation failed.")
        # Test lib
        test_lib_types_and_size(my_lib=my_lib, expected_n_lib=n_tokens_via_custom + n_tokens_via_make)
        # Test superparent units
        bool_works = np.array_equal(my_lib.superparent.phy_units[0:3], [1, -1, 0], equal_nan=True)
        self.assertEqual(bool_works, True)
        # Test sample token units
        bool_works = np.array_equal(my_lib.phy_units[my_lib.lib_name_to_idx["x0"]][0:3],
                                    custom_tokens[0].phy_units[0:3])
        self.assertEqual(bool_works, True)
        bool_works = np.array_equal(my_lib.phy_units[my_lib.lib_name_to_idx["v"]][0:3],
                                    args_make_tokens["input_var_units"]["v"])
        self.assertEqual(bool_works, True)
        # Superparent name
        self.assertEqual(my_lib.superparent.name, "v")

    # Check that library containing free units terminal tokens raises error
    def test_some_units_not_provided_warning(self):
        # -------- Test args --------

        x0 = Tok.TokenInputVar(name='x0', sympy_repr='x0', complexity=0,
                       var_id=0,
                       is_constraining_phy_units=True,
                       phy_units=[1.,0.,0.,0.,0.,0.,0.])
        x1 = Tok.TokenInputVar(name='x1', sympy_repr='x1', complexity=0,
                       var_id=1)
        add = Tok.TokenOp(name='add', sympy_repr='add', arity=2, complexity=0,
                        function=np.add,)
        cos = Tok.TokenOp(name='cos', sympy_repr='cos', arity=1, complexity=0,
                        function=np.cos,)
        pi = Tok.TokenOp(name='pi', sympy_repr='pi', arity=0, complexity=0,
                       function=lambda const=np.pi: const,)
        c0 = Tok.TokenClassFreeConst(name='c0', sympy_repr='c0', complexity=0,
                       init_val=1.,
                       var_id=0)
        args_make_tokens = {
                # operations
                "op_names"             : ["mul", "neg", "inv", "sin"],
                "use_protected_ops"    : False,
                # input variables
                "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                # constants
                "constants"            : {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                           }
        with self.assertWarns(Warning):
            my_lib = Lib.Library(custom_tokens = [c0,add] , args_make_tokens = None,)
        with self.assertWarns(Warning):
            my_lib = Lib.Library(custom_tokens = [x0,x1,add] , args_make_tokens = None,)
        with self.assertWarns(Warning):
            my_lib = Lib.Library(custom_tokens = None        , args_make_tokens = args_make_tokens,)
        with self.assertWarns(Warning):
            my_lib = Lib.Library(custom_tokens = [x0,x1,add] , args_make_tokens = args_make_tokens,)
        with self.assertWarns(Warning):
            my_lib = Lib.Library(custom_tokens = [x0]        , args_make_tokens = None,)
            my_lib.append_from_tokenize(args_make_tokens = args_make_tokens)
        with self.assertWarns(Warning):
            my_lib = Lib.Library(custom_tokens=[x0], args_make_tokens=None,
                                 superparent_units=None, superparent_name="v")
        with self.assertWarns(Warning):
            my_lib = Lib.Library(custom_tokens=[x0, x1, add], args_make_tokens=args_make_tokens,
                                 superparent_units=None, superparent_name="v")

    def test_spe_free_const_init_val_consistency_check(self):
        # Initial values
        a,b,c = 1.,10.,1.
        n_realizations = 5
        def get_args_make_tokens (aa, bb, cc):
            args_make_tokens = {
                # operations
                "op_names"             : ["mul", "neg", "inv", "sin"],
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
                "free_constants_init_val"   : {"c0" : a         , "c1"  : b          , "c2" : c         },
                "free_constants_units"      : {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                "free_constants_complexity" : {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                # free constants
                "spe_free_constants"            : {"k0"              , "k1"                   , "k2"             },
                "spe_free_constants_init_val"   : {"k0" : aa         , "k1"  : bb             , "k2" : cc        },
                "spe_free_constants_units"      : {"k0" : [0, 0, 0]  , "k1"  : [1, -1, 0]     , "k2" : [0, 0, 1] },
                "spe_free_constants_complexity" : {"k0" : 0.         , "k1"  : 0.             , "k2" : 1.        },
                       }
            return args_make_tokens

        # Test that everything works fine in the nominal case
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Raises some warnings due to some units provided (this is ok)
                my_lib = Lib.Library(args_make_tokens = get_args_make_tokens([1.,2.,3.,4.,5.], [10.,20.,30.,40.,50.] , 1.))
        except:
            self.fail("Library creation failed.")
        try:
            my_lib.check_and_pad_spe_free_const_init_val(n_realizations=n_realizations)
        except:
            self.fail("Library check_and_pad_spe_free_const_init_val failed.")
        bool_works = np.array_equal(my_lib.spe_free_constants_init_val, np.array([[1.,2.,3.,4.,5.], [10.,20.,30.,40.,50.] , np.full((n_realizations,), 1.)]))
        self.assertEqual(bool_works, True)

        # Test that everything works fine in the nominal case
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Raises some warnings due to some units provided (this is ok)
                my_lib = Lib.Library(args_make_tokens = get_args_make_tokens(1., 1. , 1.))
        except:
            self.fail("Library creation failed.")
        try:
            my_lib.check_and_pad_spe_free_const_init_val(n_realizations=n_realizations)
        except:
            self.fail("Library check_and_pad_spe_free_const_init_val failed.")
        bool_works = np.array_equal(my_lib.spe_free_constants_init_val, np.array([np.full((n_realizations,), 1.), np.full((n_realizations,), 1.) , np.full((n_realizations,), 1.)]))
        self.assertEqual(bool_works, True)

        # Inconsistent number of realizations
        with self.assertRaises(AssertionError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                my_lib = Lib.Library(args_make_tokens = get_args_make_tokens([1.,2.,3.,4.,5.], [10.,20.,30.,40.,50.] , 1.))
            my_lib.check_and_pad_spe_free_const_init_val(n_realizations=2)

        # Inconsistent shapes of initial values
        with self.assertRaises(AssertionError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                my_lib = Lib.Library(args_make_tokens = get_args_make_tokens([1.,2.,3.,4.,5.], [10.,20.,30.,] , 1.))
            my_lib.check_and_pad_spe_free_const_init_val(n_realizations=5)

        return None

if __name__ == '__main__':
    unittest.main(verbosity=2)
