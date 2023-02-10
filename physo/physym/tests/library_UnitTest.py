import unittest
import numpy as np

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
            self.assertEqual( my_lib            .lib_function                 .dtype , np.object_ )
            self.assertEqual( my_lib.properties .arity                        .dtype , np.int_    )
            self.assertEqual( my_lib.properties .complexity                   .dtype , np.float_  )
            self.assertEqual( my_lib.properties .var_type                     .dtype , np.int_   )
            self.assertEqual( my_lib.properties .var_id                       .dtype , np.int_    )
            self.assertEqual( my_lib.properties .is_constraining_phy_units    .dtype , np.bool_   )
            self.assertEqual( my_lib.properties .phy_units                    .dtype , np.float_  )
            self.assertEqual( my_lib.properties .behavior_id                  .dtype , np.int_    )
            self.assertEqual( my_lib.properties .is_power                     .dtype , np.bool_   )
            self.assertEqual( my_lib.properties .power                        .dtype , np.float_  )
        # -------- Test args --------
        custom_tokens = [
        Tok.Token(name='x0', sympy_repr='x0', arity=0, complexity=0, var_type=1,
                       function=None,
                       var_id=0,
                       is_constraining_phy_units=True,
                       phy_units=[1.,0.,0.,0.,0.,0.,0.]),
        Tok.Token(name='x1', sympy_repr='x1', arity=0, complexity=0, var_type=1,
                       function=None,
                       var_id=1),
        Tok.Token(name='add', sympy_repr='add', arity=2, complexity=0, var_type=0,
                        function=np.add,
                        var_id=None),
        Tok.Token(name='cos', sympy_repr='cos', arity=1, complexity=0, var_type=0,
                        function=np.cos,
                        var_id=None),
        Tok.Token(name='pi', sympy_repr='pi', arity=0, complexity=0, var_type=0,
                       function=lambda const=np.pi: const,
                       var_id=None,),
                        ]
        n_tokens_via_custom = len(custom_tokens)
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
                "free_constants_init_val"   : {"c0" : 1.        , "c1"  : 10.        , "c2" : 1.        },
                "free_constants_units"      : {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                "free_constants_complexity" : {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                           }
        n_tokens_via_make = len(args_make_tokens["op_names"]) + len(args_make_tokens["input_var_ids"])\
                            + len(args_make_tokens["constants"]) + len(args_make_tokens["free_constants"])

        # -------- Test args_make_tokens only --------
        try:
            my_lib = Lib.Library(args_make_tokens = args_make_tokens)
        except:
            self.fail("Library creation failed.")
        # Test lib
        test_lib_types_and_size(my_lib=my_lib, expected_n_lib=n_tokens_via_make)
        # Test superparent units
        bool_works = np.array_equal(my_lib.superparent.phy_units, np.full(Tok.UNITS_VECTOR_SIZE, np.NAN), equal_nan=True)
        self.assertEqual(bool_works, True)
        # Test sample token units
        bool_works = np.array_equal(my_lib.phy_units[my_lib.lib_name_to_idx["v"]][0:3],
                                    args_make_tokens["input_var_units"]["v"])
        self.assertEqual(bool_works, True)

        # -------- Test custom_tokens only --------
        try:
            my_lib = Lib.Library(custom_tokens = custom_tokens)
        except:
            self.fail("Library creation failed.")
        # Test lib
        test_lib_types_and_size(my_lib=my_lib, expected_n_lib=n_tokens_via_custom)
        # Test superparent units
        bool_works = np.array_equal(my_lib.superparent.phy_units, np.full(Tok.UNITS_VECTOR_SIZE, np.NAN), equal_nan=True)
        self.assertEqual(bool_works, True)
        # Test sample token units
        bool_works = np.array_equal(my_lib.phy_units[my_lib.lib_name_to_idx["x0"]][0:3],
                                    custom_tokens[0].phy_units[0:3])
        self.assertEqual(bool_works, True)

        # -------- Test append custom_tokens then args_make_tokens --------
        my_lib = Lib.Library(custom_tokens=custom_tokens)
        my_lib.append_tokens_from_names(args_make_tokens=args_make_tokens)
        test_lib_types_and_size(my_lib=my_lib, expected_n_lib=n_tokens_via_custom+n_tokens_via_make)

        # Test superparent units
        bool_works = np.array_equal(my_lib.superparent.phy_units, np.full(Tok.UNITS_VECTOR_SIZE, np.NAN),
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

        my_lib = Lib.Library(args_make_tokens=args_make_tokens)
        my_lib.append_custom_tokens(custom_tokens=custom_tokens)
        test_lib_types_and_size(my_lib=my_lib, expected_n_lib=n_tokens_via_custom + n_tokens_via_make)

        # Test superparent units
        bool_works = np.array_equal(my_lib.superparent.phy_units, np.full(Tok.UNITS_VECTOR_SIZE, np.NAN),
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
            my_lib = Lib.Library(custom_tokens=custom_tokens, args_make_tokens=args_make_tokens)
        except:
            self.fail("Library creation failed.")
        # Test lib
        test_lib_types_and_size(my_lib=my_lib, expected_n_lib=n_tokens_via_custom + n_tokens_via_make)
        # Test superparent units
        bool_works = np.array_equal(my_lib.superparent.phy_units, np.full(Tok.UNITS_VECTOR_SIZE, np.NAN),
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
        x0 = Tok.Token(name='x0', sympy_repr='x0', arity=0, complexity=0, var_type=1,
                       function=None,
                       var_id=0,
                       is_constraining_phy_units=True,
                       phy_units=[1.,0.,0.,0.,0.,0.,0.])
        x1 = Tok.Token(name='x1', sympy_repr='x1', arity=0, complexity=0, var_type=1,
                       function=None,
                       var_id=1)
        add = Tok.Token(name='add', sympy_repr='add', arity=2, complexity=0, var_type=0,
                        function=np.add,
                        var_id=None)
        cos = Tok.Token(name='cos', sympy_repr='cos', arity=1, complexity=0, var_type=0,
                        function=np.cos,
                        var_id=None)
        pi = Tok.Token(name='pi', sympy_repr='pi', arity=0, complexity=0, var_type=0,
                       function=lambda const=np.pi: const,
                       var_id=None)
        c0 = Tok.Token(name='c0', sympy_repr='c0', arity=0, complexity=0, var_type=2,
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
            my_lib.append_tokens_from_names(args_make_tokens = args_make_tokens)
        with self.assertWarns(Warning):
            my_lib = Lib.Library(custom_tokens=[x0], args_make_tokens=None,
                                 superparent_units=None, superparent_name="v")
        with self.assertWarns(Warning):
            my_lib = Lib.Library(custom_tokens=[x0, x1, add], args_make_tokens=args_make_tokens,
                                 superparent_units=None, superparent_name="v")

if __name__ == '__main__':
    unittest.main(verbosity=2)
