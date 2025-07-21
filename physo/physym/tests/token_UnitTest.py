import unittest
import numpy as np

# Internal imports
from physo.physym import token as Tok

class TokenTest(unittest.TestCase):

    # ------------------------------------------------------------------------
    # ------------------------------ TEST TOKEN ------------------------------
    # ------------------------------------------------------------------------

    # --------------------- Token representing operation --------------------

    # Test token creation
    def test_token_operation_creation(self):
        # Test token creation
        try:
            add = Tok.Token(name='add', sympy_repr='add', arity=2, complexity=0, var_type=Tok.VAR_TYPE_OP,
                            function=np.add,
                            var_id=None)
        except:
            self.fail("Token creation failed")
        # Same test with type specific class
        try:
            add = Tok.TokenOp(name='add', sympy_repr='add', arity=2, complexity=0, function=np.add,)
        except:
            self.fail("Token creation failed")

    # Test token creation exceptions
    def test_token_operation_creation_exceptions(self):

        # Test exception: function is supposed to be callable
        with self.assertRaises(AssertionError):
            add = Tok.Token(name='add', sympy_repr='add', arity=2, complexity=0, var_type=Tok.VAR_TYPE_OP,
                            function=None,
                            var_id=None)
        # Same test with type specific class
        with self.assertRaises(AssertionError):
            add = Tok.Token(name='add', sympy_repr='add', arity=2, complexity=0,
                            function=None,)

        # Test exception: var_id is supposed to be Nan
        with self.assertRaises(AssertionError):
            add = Tok.Token(name='add', sympy_repr='add', arity=2, complexity=0, var_type=Tok.VAR_TYPE_OP,
                            function=np.add,
                            var_id=0)

        # Test exception: arity is supposed to be >= 0
        with self.assertRaises(AssertionError):
            x0 = Tok.Token(name='x0', sympy_repr='x0', arity=-1, complexity=0, var_type=Tok.VAR_TYPE_INPUT_VAR,
                           function=None,
                           var_id=0)
        # Same test with type specific class
        with self.assertRaises(AssertionError):
            x0 = Tok.Token(name='x0', sympy_repr='x0', arity=-1, complexity=0,
                           function=None,)

    # ------------------ Token representing input variable ------------------

    # Test token creation
    def test_token_input_var_creation(self):
        # Test token creation
        try:
            x0 = Tok.Token(name='x0', sympy_repr='x0', arity=0, complexity=0, var_type=Tok.VAR_TYPE_INPUT_VAR,
                           function=None,
                           var_id=0)
        except:
            self.fail("Token creation failed")

        # Same test with type specific class
        try:
            x0 = Tok.TokenInputVar(name='x0', sympy_repr='x0', complexity=0,var_id=0)
        except:
            self.fail("Token creation failed")

    # Test token creation exceptions
    def test_token_input_var_creation_exceptions(self):

        # Test exception: function is supposed to be None
        with self.assertRaises(AssertionError):
            x0 = Tok.Token(name='x0', sympy_repr='x0', arity=0, complexity=0, var_type=Tok.VAR_TYPE_INPUT_VAR,
                           function=np.multiply,
                           var_id=0)

        # Test exception: var_id is supposed to be int
        with self.assertRaises(AssertionError):
            x0 = Tok.Token(name='x0', sympy_repr='x0', arity=0, complexity=0, var_type=Tok.VAR_TYPE_INPUT_VAR,
                           function=None,
                           var_id='0')
        # Same test with type specific class
        with self.assertRaises(AssertionError):
            x0 = Tok.TokenInputVar(name='x0', sympy_repr='x0', complexity=0,
                           var_id='0')

        # Test exception: arity is supposed to be = 0
        with self.assertRaises(AssertionError):
            x0 = Tok.Token(name='x0', sympy_repr='x0', arity=1, complexity=0, var_type=Tok.VAR_TYPE_INPUT_VAR,
                           function=None,
                           var_id=0)

    # Test token usage
    def test_token_input_var_usage(self):
        data_x0 = np.linspace(start=-10, stop=10, num=1000)
        data_x1 = np.linspace(start=-5, stop=15, num=1000)
        dataset = np.array([data_x0, data_x1])
        x0 = Tok.Token(name='x0', sympy_repr='x0', arity=0, complexity=0, var_type=Tok.VAR_TYPE_INPUT_VAR,
                       function=None,
                       var_id=0)
        works_bool = np.array_equal(dataset[x0.var_id], data_x0)
        self.assertEqual(works_bool, True)

        # Same test with type specific class
        data_x0 = np.linspace(start=-10, stop=10, num=1000)
        data_x1 = np.linspace(start=-5, stop=15, num=1000)
        dataset = np.array([data_x0, data_x1])
        x0 = Tok.TokenInputVar(name='x0', sympy_repr='x0', complexity=0, var_id=0)
        works_bool = np.array_equal(dataset[x0.var_id], data_x0)
        self.assertEqual(works_bool, True)


    # --------------------- Token representing fixed constant ---------------------

    # Test token creation
    def test_token_fixed_constant_creation(self):
        # Test token creation
        try:
            pi = Tok.Token(name='pi', sympy_repr='pi', arity=0, complexity=0, var_type=Tok.VAR_TYPE_FIXED_CONST,
                           fixed_const=np.pi,
                           function=None,
                           var_id=None)
        except:
            self.fail("Token creation failed")
        # Same test with type specific class
        try:
            pi = Tok.TokenFixedConst(name='pi', sympy_repr='pi', complexity=0, fixed_const=np.pi,)
        except:
            self.fail("Token creation failed")

    # Test token creation exceptions
    def test_token_fixed_constant_creation_exceptions(self):
        # Test exception: fixed_const is supposed to be non nan
        with self.assertRaises(AssertionError):
            pi = Tok.Token(name='pi', sympy_repr='pi', arity=0, complexity=0, var_type=Tok.VAR_TYPE_FIXED_CONST,
                           fixed_const=np.nan,
                           function=None,
                           var_id=None)
        # Same test with type specific class
        with self.assertRaises(AssertionError):
            pi = Tok.TokenFixedConst(name='pi', sympy_repr='pi', complexity=0,
                           fixed_const=np.nan,)

        # Test exception: function is supposed to be None
        with self.assertRaises(AssertionError):
            pi = Tok.Token(name='pi', sympy_repr='pi', arity=0, complexity=0, var_type=Tok.VAR_TYPE_FIXED_CONST,
                           fixed_const=np.pi,
                           function=lambda const = np.pi: const,
                           var_id=None)
        # Test exception: var_id is supposed to be Nan
        with self.assertRaises(AssertionError):
            pi = Tok.Token(name='pi', sympy_repr='pi', arity=0, complexity=0, var_type=Tok.VAR_TYPE_FIXED_CONST,
                           fixed_const=np.pi,
                           function=None,
                           var_id=0)
        # Test exception: arity is supposed to be = 0
        with self.assertRaises(AssertionError):
            pi = Tok.Token(name='pi', sympy_repr='pi', arity=1, complexity=0, var_type=Tok.VAR_TYPE_FIXED_CONST,
                           fixed_const=np.pi,
                           function=None,
                           var_id=None)

    # --------------------- Token representing class free constant ---------------------

    # Test token creation
    def test_token_class_free_constant_creation(self):
        # Test token creation
        try:
            c0 = Tok.Token(name='c0', sympy_repr='c0', arity=0, complexity=0, var_type=Tok.VAR_TYPE_CLASS_FREE_CONST,
                           function=None,
                           var_id=0,
                           init_val=1.)
        except:
            self.fail("Token creation failed")

        # Same test with type specific class
        try:
            c0 = Tok.TokenClassFreeConst(name='c0', sympy_repr='c0', complexity=0,
                           var_id=0,
                           init_val=1.)
        except:
            self.fail("Token creation failed")

    # Test token creation exceptions
    def test_token_class_free_constant_creation_exceptions(self):

        # Test exception: function is supposed to be None
        with self.assertRaises(AssertionError):
            c0 = Tok.Token(name='c0', sympy_repr='c0', arity=0, complexity=0, var_type=Tok.VAR_TYPE_CLASS_FREE_CONST,
                           function=np.multiply,
                           var_id=0,
                           init_val=1.)

        # Test exception: var_id is supposed to be int
        with self.assertRaises(AssertionError):
            c0 = Tok.Token(name='c0', sympy_repr='c0', arity=0, complexity=0, var_type=Tok.VAR_TYPE_CLASS_FREE_CONST,
                           function=None,
                           var_id='0',
                           init_val=1.)
        # Same test with type specific class
        with self.assertRaises(AssertionError):
            c0 = Tok.TokenClassFreeConst(name='c0', sympy_repr='c0', complexity=0,
                           var_id='0',
                           init_val=1.)

        # Test exception: arity is supposed to be = 0
        with self.assertRaises(AssertionError):
            c0 = Tok.Token(name='c0', sympy_repr='c0', arity=1, complexity=0, var_type=Tok.VAR_TYPE_CLASS_FREE_CONST,
                           function=None,
                           var_id=0,
                           init_val=1.)

    # --------------------- Token representing spe free constant ---------------------

    # Test token creation
    def test_token_spe_free_constant_creation(self):
        # Test token creation
        try:
            c0 = Tok.Token(name='c0', sympy_repr='c0', arity=0, complexity=0, var_type=Tok.VAR_TYPE_SPE_FREE_CONST,
                           function=None,
                           var_id=0,
                           init_val=1.)
        except:
            self.fail("Token creation failed")

        # Same test with type specific class
        try:
            c0 = Tok.TokenSpeFreeConst(name='c0', sympy_repr='c0', complexity=0,
                           var_id=0,
                           init_val=1.)
        except:
            self.fail("Token creation failed")

    # Test token creation exceptions
    def test_token_spe_free_constant_creation_exceptions(self):
        # Test exception: function is supposed to be None
        with self.assertRaises(AssertionError):
            c0 = Tok.Token(name='c0', sympy_repr='c0', arity=0, complexity=0, var_type=Tok.VAR_TYPE_SPE_FREE_CONST,
                           function=np.multiply,
                           var_id=0,
                           init_val=1.)

        # Test exception: var_id is supposed to be int
        with self.assertRaises(AssertionError):
            c0 = Tok.Token(name='c0', sympy_repr='c0', arity=0, complexity=0, var_type=Tok.VAR_TYPE_SPE_FREE_CONST,
                           function=None,
                           var_id='0',
                           init_val=1.)
        # Same test with type specific class
        with self.assertRaises(AssertionError):
            c0 = Tok.TokenSpeFreeConst(name='c0', sympy_repr='c0', complexity=0,
                           var_id='0',
                           init_val=1.)

        # Test exception: arity is supposed to be = 0
        with self.assertRaises(AssertionError):
            c0 = Tok.Token(name='c0', sympy_repr='c0', arity=1, complexity=0, var_type=Tok.VAR_TYPE_SPE_FREE_CONST,
                           function=None,
                           var_id=0,
                           init_val=1.)

    def test_token_spe_free_constant_creation_exceptions_init_val_related(self):
        # Test exception: init_val is supposed to be a non-nan
        with self.assertRaises(AssertionError):
            c0 = Tok.Token(name='c0', sympy_repr='c0', arity=0, complexity=0, var_type=Tok.VAR_TYPE_SPE_FREE_CONST,
                           function=None,
                           var_id=0,
                           init_val=np.nan)
        # Same test with type specific class
        with self.assertRaises(AssertionError):
            c0 = Tok.TokenSpeFreeConst(name='c0', sympy_repr='c0', complexity=0,
                           var_id=0,
                           init_val=np.nan)

        # Test exception: init_val is supposed to be a non-nan (multi realization)
        with self.assertRaises(AssertionError):
            c0 = Tok.Token(name='c0', sympy_repr='c0', arity=0, complexity=0, var_type=Tok.VAR_TYPE_SPE_FREE_CONST,
                           function=None,
                           var_id=0,
                           init_val=[np.nan,2.,3.])
        # Same test with type specific class
        with self.assertRaises(AssertionError):
            c0 = Tok.TokenSpeFreeConst(name='c0', sympy_repr='c0', complexity=0,
                           var_id=0,
                           init_val=[np.nan,2.,3.])

        # Test exception: init_val is supposed to be a float or a 1D array of floats
        with self.assertRaises(AssertionError):
            c0 = Tok.Token(name='c0', sympy_repr='c0', arity=0, complexity=0, var_type=Tok.VAR_TYPE_SPE_FREE_CONST,
                           function=None,
                           var_id=0,
                           init_val=np.ones((2,2)))
        # Same test with type specific class
        with self.assertRaises(AssertionError):
            c0 = Tok.TokenSpeFreeConst(name='c0', sympy_repr='c0', complexity=0,
                           var_id=0,
                           init_val=np.ones((2,2)))

    def test_token_spe_free_constant_init_val_related (self):
            n_realizations = 10
            aa = 2.1
            aa_expected = np.array([aa])
            bb = np.random.rand(n_realizations,)
            bb_expected = bb


            # Test creation (single float init_val)
            try:
                c0 = Tok.Token(name='c0', sympy_repr='c0', arity=0, complexity=0, var_type=Tok.VAR_TYPE_SPE_FREE_CONST,
                               function = None,
                               var_id   = 0,
                               init_val = aa)
            except:
                self.fail("Token creation failed")
            bool_works = np.array_equal(c0.init_val, aa_expected)
            self.assertEqual(bool_works, True)
            # Same test with type specific class
            try:
                c0 = Tok.TokenSpeFreeConst(name='c0', sympy_repr='c0', complexity=0,
                                           var_id=0,
                                           init_val=aa,)
            except:
                self.fail("Token creation failed")
            bool_works = np.array_equal(c0.init_val, aa_expected)
            self.assertEqual(bool_works, True)


            # Test creation (multi realization init_val)
            try:
                c0 = Tok.Token(name='c0', sympy_repr='c0', arity=0, complexity=0, var_type=Tok.VAR_TYPE_SPE_FREE_CONST,
                               function = None,
                               var_id   = 0,
                               init_val = bb,
                             )
            except:
                self.fail("Token creation failed")
            bool_works = np.array_equal(c0.init_val, bb_expected)
            self.assertEqual(bool_works, True)
            # Same test with type specific class
            try:
                c0 = Tok.TokenSpeFreeConst(name='c0', sympy_repr='c0', complexity=0,
                                           var_id=0,
                                           init_val=bb,)
            except:
                self.fail("Token creation failed")
            bool_works = np.array_equal(c0.init_val, bb_expected)
            self.assertEqual(bool_works, True)


            return None


    # ----------------------------- Token call ------------------------------

    # Test token call method (all types of tokens)
    def test_token_call(self):
        # ----- Data for tests -----
        data_x0 = np.linspace(start=-10, stop=10, num=1000)
        data_x1 = np.linspace(start=-5, stop=15, num=1000)
        dataset = np.array([data_x0, data_x1])
        add = Tok.Token(name='add', sympy_repr='add', arity=2, complexity=0, var_type=Tok.VAR_TYPE_OP,
                        function=np.add,
                        var_id=None)
        cos = Tok.Token(name='cos', sympy_repr='cos', arity=1, complexity=0, var_type=Tok.VAR_TYPE_OP,
                        function=np.cos,
                        var_id=None)
        x0 = Tok.Token(name='x0', sympy_repr='x0', arity=0, complexity=0, var_type=Tok.VAR_TYPE_INPUT_VAR,
                       function=None,
                       var_id=0)
        pi = Tok.Token(name='pi', sympy_repr='pi', arity=0, complexity=0, var_type=Tok.VAR_TYPE_OP,
                       function=lambda const=np.pi: const,
                       var_id=None)
        # ----- Test legal calls -----
        works_bool = np.array_equal(add(data_x0, data_x1), np.add(data_x0, data_x1))
        self.assertEqual(works_bool, True)
        works_bool = np.array_equal(cos(data_x0), np.cos(data_x0))
        self.assertEqual(works_bool, True)
        works_bool = np.array_equal(pi(), np.pi)
        self.assertEqual(works_bool, True)
        works_bool = np.array_equal(add(cos(pi()), dataset[x0.var_id]), np.add(np.cos(np.pi), data_x0))
        self.assertEqual(works_bool, True)
        # ----- Test exception: n args is supposed to be = arity -----
        with self.assertRaises(AssertionError): res = add(data_x0)
        with self.assertRaises(AssertionError): res = cos(data_x0, data_x1)
        with self.assertRaises(AssertionError): res = cos()
        # ----- Test exception: token representing data is not supposed to be called -----
        with self.assertRaises(AssertionError): res = x0(data_x0, data_x1)
        with self.assertRaises(AssertionError): res = x0()

        # Same test with type specific class
        # ----- Data for tests -----
        data_x0 = np.linspace(start=-10, stop=10, num=1000)
        data_x1 = np.linspace(start=-5, stop=15, num=1000)
        dataset = np.array([data_x0, data_x1])
        add = Tok.TokenOp(name='add', sympy_repr='add', arity=2, complexity=0,
                        function=np.add,)
        cos = Tok.TokenOp(name='cos', sympy_repr='cos', arity=1, complexity=0,
                        function=np.cos,)
        x0 = Tok.TokenInputVar(name='x0', sympy_repr='x0', complexity=0,
                       var_id=0)
        pi = Tok.TokenOp(name='pi', sympy_repr='pi', arity=0, complexity=0,
                       function=lambda const=np.pi: const,)
        # ----- Test legal calls -----
        works_bool = np.array_equal(add(data_x0, data_x1), np.add(data_x0, data_x1))
        self.assertEqual(works_bool, True)
        works_bool = np.array_equal(cos(data_x0), np.cos(data_x0))
        self.assertEqual(works_bool, True)
        works_bool = np.array_equal(pi(), np.pi)
        self.assertEqual(works_bool, True)
        works_bool = np.array_equal(add(cos(pi()), dataset[x0.var_id]), np.add(np.cos(np.pi), data_x0))
        self.assertEqual(works_bool, True)
        # ----- Test exception: n args is supposed to be = arity -----
        with self.assertRaises(AssertionError): res = add(data_x0)
        with self.assertRaises(AssertionError): res = cos(data_x0, data_x1)
        with self.assertRaises(AssertionError): res = cos()
        # ----- Test exception: token representing data is not supposed to be called -----
        with self.assertRaises(AssertionError): res = x0(data_x0, data_x1)
        with self.assertRaises(AssertionError): res = x0()

    # Test token with too long name
    def test_token_name_too_long(self):
        too_large_name = "blabla" * Tok.MAX_NAME_SIZE
        with self.assertRaises(AssertionError):
            add = Tok.Token(name=too_large_name, sympy_repr='add', arity=2, complexity=0, var_type=Tok.VAR_TYPE_OP,
                            function=np.add,
                            var_id=None)
        with self.assertRaises(AssertionError):
            add = Tok.Token(name='add', sympy_repr=too_large_name, arity=2, complexity=0, var_type=Tok.VAR_TYPE_OP,
                            function=np.add,
                            var_id=None)

    # ----------------------------- Token units ------------------------------
    # Test token creation
    def test_token_creation_units(self):
        try:
            physical_quantity = Tok.Token(name='x0', sympy_repr='x0', arity=0, complexity=0,
                                          var_type=Tok.VAR_TYPE_INPUT_VAR,
                                          function=None,
                                          var_id=0,
                                          is_constraining_phy_units=True,
                                          phy_units=np.ones(Tok.UNITS_VECTOR_SIZE))

            cos = Tok.Token(name='cos', sympy_repr='cos', arity=1, complexity=0, var_type=Tok.VAR_TYPE_OP,
                            function=np.cos,
                            var_id=None,
                            is_constraining_phy_units = True,
                            phy_units = np.zeros(Tok.UNITS_VECTOR_SIZE), )
            sqrt = Tok.Token(name='sqrt', sympy_repr='sqrt', arity=1, complexity=0, var_type=Tok.VAR_TYPE_OP,
                             function=np.sqrt,
                             var_id=None,
                             is_power=True,
                             power=0.5)
        except:
            self.fail("Token creation failed")

    # Test token creation exceptions : phy_units
    def test_token_creation_units_exceptions_phy_units(self):
        # Test exception: token with is_constraining_phy_units = True must have units
        with self.assertRaises(AssertionError):
            physical_quantity = Tok.Token(name='x0', sympy_repr='x0', arity=0, complexity=0,
                                          var_type=Tok.VAR_TYPE_INPUT_VAR,
                                          function=None,
                                          var_id=0,
                                          is_constraining_phy_units=True,
                                          phy_units=None)
        # Test exception: token with is_constraining_phy_units = True must have units vector of size Lib.UNITS_VECTOR_SIZE
        with self.assertRaises(AssertionError):
            physical_quantity = Tok.Token(name='x0', sympy_repr='x0', arity=0, complexity=0,
                                          var_type=Tok.VAR_TYPE_INPUT_VAR,
                                          function=None,
                                          var_id=0,
                                          is_constraining_phy_units=True,
                                          phy_units=np.ones(Tok.UNITS_VECTOR_SIZE + 1))
        # Test exception: token with is_constraining_phy_units = True must not contain any NAN
        wrong_units = np.ones(Tok.UNITS_VECTOR_SIZE)
        wrong_units[0] = np.nan
        with self.assertRaises(AssertionError):
            physical_quantity = Tok.Token(name='x0', sympy_repr='x0', arity=0, complexity=0,
                                          var_type=Tok.VAR_TYPE_INPUT_VAR,
                                          function=None,
                                          var_id=0,
                                          is_constraining_phy_units=True,
                                          phy_units=wrong_units)
        # Test exception: token with is_constraining_phy_units = False must not have units
        with self.assertRaises(AssertionError):
            add = Tok.Token(name='add', sympy_repr='add', arity=2, complexity=0, var_type=Tok.VAR_TYPE_OP,
                            function=np.add,
                            var_id=None,
                            is_constraining_phy_units=False,
                            phy_units=np.ones(Tok.UNITS_VECTOR_SIZE))

    # Test token creation exceptions : behavior_id
    def test_token_creation_units_exceptions_behavior_id(self):
        # Test exception: behavior_id must be an int
        with self.assertRaises(AssertionError):
            cos = Tok.Token(name='cos', sympy_repr='cos', arity=1, complexity=0, var_type=Tok.VAR_TYPE_OP,
                            function=np.cos,
                            var_id=None,
                            is_constraining_phy_units=True,
                            phy_units=np.zeros(Tok.UNITS_VECTOR_SIZE),
                            behavior_id="not an int")

    # Test token creation exceptions : power
    def test_token_creation_units_exceptions_power(self):
        # Test exception: power must be a float
        with self.assertRaises(AssertionError):
            sqrt = Tok.Token(name='sqrt', sympy_repr='sqrt', arity=1, complexity=0, var_type=Tok.VAR_TYPE_OP,
                             function=np.sqrt,
                             var_id=None,
                             is_power=True,
                             power="not a float")

        with self.assertRaises(AssertionError):
            sqrt = Tok.Token(name='sqrt', sympy_repr='sqrt', arity=1, complexity=0, var_type=Tok.VAR_TYPE_OP,
                             function=np.sqrt,
                             var_id=None,
                             is_power=False,
                             power=0.5)

    # Test that type specific token classes are using correct type under the hood
    def test_token_types_of_subclasses(self):
        # Creating a token of each type and testing its type

        my_tok = Tok.TokenInputVar(name='x0', sympy_repr='x0', complexity=0,var_id=0)
        self.assertEqual(my_tok.var_type, Tok.VAR_TYPE_INPUT_VAR)

        my_tok = Tok.TokenOp(name='add', sympy_repr='add', arity=2, complexity=0, function=np.add,)
        self.assertEqual(my_tok.var_type, Tok.VAR_TYPE_OP)

        my_tok = Tok.TokenFixedConst(name='pi', sympy_repr='pi', complexity=0, fixed_const=np.pi,)
        self.assertEqual(my_tok.var_type, Tok.VAR_TYPE_FIXED_CONST)

        my_tok = Tok.TokenClassFreeConst(name='c0', sympy_repr='c0', complexity=0,var_id=0,init_val=1.)
        self.assertEqual(my_tok.var_type, Tok.VAR_TYPE_CLASS_FREE_CONST)

        my_tok = Tok.TokenSpeFreeConst(name='c0', sympy_repr='c0', complexity=0,var_id=0,init_val=1.)
        self.assertEqual(my_tok.var_type, Tok.VAR_TYPE_SPE_FREE_CONST)



if __name__ == '__main__':
    unittest.main(verbosity = 2)

