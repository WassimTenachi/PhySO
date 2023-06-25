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
            add = Tok.Token(name='add', sympy_repr='add', arity=2, complexity=0, var_type=0,
                            function=np.add,
                            var_id=None)
        except:
            self.fail("Token creation failed")

    # Test token creation exceptions
    def test_token_operation_creation_exceptions(self):
        # Test exception: function is supposed to be callable
        with self.assertRaises(AssertionError):
            add = Tok.Token(name='add', sympy_repr='add', arity=2, complexity=0, var_type=0,
                            function=None,
                            var_id=None)
        # Test exception: var_id is supposed to be Nan
        with self.assertRaises(AssertionError):
            add = Tok.Token(name='add', sympy_repr='add', arity=2, complexity=0, var_type=0,
                            function=np.add,
                            var_id=0)
        # Test exception: arity is supposed to be >= 0
        with self.assertRaises(AssertionError):
            x0 = Tok.Token(name='x0', sympy_repr='x0', arity=-1, complexity=0, var_type=1,
                           function=None,
                           var_id=0)

    # ------------------ Token representing input variable ------------------

    # Test token creation
    def test_token_input_var_creation(self):
        # Test token creation
        try:
            x0 = Tok.Token(name='x0', sympy_repr='x0', arity=0, complexity=0, var_type=1,
                           function=None,
                           var_id=0)
        except:
            self.fail("Token creation failed")

    # Test token creation exceptions
    def test_token_input_var_creation_exceptions(self):
        # Test exception: function is supposed to be None
        with self.assertRaises(AssertionError):
            x0 = Tok.Token(name='x0', sympy_repr='x0', arity=0, complexity=0, var_type=1,
                           function=np.multiply,
                           var_id=0)
        # Test exception: var_id is supposed to be int
        with self.assertRaises(AssertionError):
            x0 = Tok.Token(name='x0', sympy_repr='x0', arity=0, complexity=0, var_type=1,
                           function=None,
                           var_id='0')
        # Test exception: arity is supposed to be = 0
        with self.assertRaises(AssertionError):
            x0 = Tok.Token(name='x0', sympy_repr='x0', arity=1, complexity=0, var_type=1,
                           function=None,
                           var_id=0)

    # Test token usage
    def test_token_input_var_usage(self):
        data_x0 = np.linspace(start=-10, stop=10, num=1000)
        data_x1 = np.linspace(start=-5, stop=15, num=1000)
        dataset = np.array([data_x0, data_x1])
        x0 = Tok.Token(name='x0', sympy_repr='x0', arity=0, complexity=0, var_type=1,
                       function=None,
                       var_id=0)
        works_bool = np.array_equal(dataset[x0.var_id], data_x0)
        self.assertEqual(works_bool, True)


    # --------------------- Token representing fixed constant ---------------------

    # Test token creation
    def test_token_constant_creation(self):
        # Test token creation
        try:
            pi = Tok.Token(name='pi', sympy_repr='pi', arity=0, complexity=0, var_type=3,
                           fixed_const=np.pi,
                           function=None,
                           var_id=None)
        except:
            self.fail("Token creation failed")

    # Test token creation exceptions
    def test_token_constant_creation_exceptions(self):
        # Test exception: fixed_const is supposed to be non nan
        with self.assertRaises(AssertionError):
            pi = Tok.Token(name='pi', sympy_repr='pi', arity=0, complexity=0, var_type=3,
                           fixed_const=np.NaN,
                           function=None,
                           var_id=None)
        # Test exception: function is supposed to be None
        with self.assertRaises(AssertionError):
            pi = Tok.Token(name='pi', sympy_repr='pi', arity=0, complexity=0, var_type=3,
                           fixed_const=np.pi,
                           function=lambda const = np.pi: const,
                           var_id=None)
        # Test exception: var_id is supposed to be Nan
        with self.assertRaises(AssertionError):
            pi = Tok.Token(name='pi', sympy_repr='pi', arity=0, complexity=0, var_type=3,
                           fixed_const=np.pi,
                           function=None,
                           var_id=0)
        # Test exception: arity is supposed to be = 0
        with self.assertRaises(AssertionError):
            pi = Tok.Token(name='pi', sympy_repr='pi', arity=1, complexity=0, var_type=3,
                           fixed_const=np.pi,
                           function=None,
                           var_id=None)

    # --------------------- Token representing free constant ---------------------

    # Test token creation
    def test_token_free_constant_creation(self):
        # Test token creation
        try:
            c0 = Tok.Token(name='c0', sympy_repr='c0', arity=0, complexity=0, var_type=2,
                           function=None,
                           var_id=0,
                           init_val=1.)
        except:
            self.fail("Token creation failed")

    # Test token creation exceptions
    def test_token_free_constant_creation_exceptions(self):
        # Test exception: function is supposed to be None
        with self.assertRaises(AssertionError):
            c0 = Tok.Token(name='c0', sympy_repr='c0', arity=0, complexity=0, var_type=2,
                           function=np.multiply,
                           var_id=0,
                           init_val=1.)
        # Test exception: var_id is supposed to be int
        with self.assertRaises(AssertionError):
            c0 = Tok.Token(name='c0', sympy_repr='c0', arity=0, complexity=0, var_type=2,
                           function=None,
                           var_id='0',
                           init_val=1.)
        # Test exception: arity is supposed to be = 0
        with self.assertRaises(AssertionError):
            c0 = Tok.Token(name='c0', sympy_repr='c0', arity=1, complexity=0, var_type=2,
                           function=None,
                           var_id=0,
                           init_val=1.)

    # ----------------------------- Token call ------------------------------

    # Test token call method (all types of tokens)
    def test_token_call(self):
        # ----- Data for tests -----
        data_x0 = np.linspace(start=-10, stop=10, num=1000)
        data_x1 = np.linspace(start=-5, stop=15, num=1000)
        dataset = np.array([data_x0, data_x1])
        add = Tok.Token(name='add', sympy_repr='add', arity=2, complexity=0, var_type=0,
                        function=np.add,
                        var_id=None)
        cos = Tok.Token(name='cos', sympy_repr='cos', arity=1, complexity=0, var_type=0,
                        function=np.cos,
                        var_id=None)
        x0 = Tok.Token(name='x0', sympy_repr='x0', arity=0, complexity=0, var_type=1,
                       function=None,
                       var_id=0)
        pi = Tok.Token(name='pi', sympy_repr='pi', arity=0, complexity=0, var_type=0,
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

    # Test token with too long name
    def test_token_name_too_long(self):
        too_large_name = "blabla" * Tok.MAX_NAME_SIZE
        with self.assertRaises(AssertionError):
            add = Tok.Token(name=too_large_name, sympy_repr='add', arity=2, complexity=0, var_type=0,
                            function=np.add,
                            var_id=None)
        with self.assertRaises(AssertionError):
            add = Tok.Token(name='add', sympy_repr=too_large_name, arity=2, complexity=0, var_type=0,
                            function=np.add,
                            var_id=None)

    # ----------------------------- Token units ------------------------------
    # Test token creation
    def test_token_creation_units(self):
        try:
            physical_quantity = Tok.Token(name='x0', sympy_repr='x0', arity=0, complexity=0,
                                          var_type=1,
                                          function=None,
                                          var_id=0,
                                          is_constraining_phy_units=True,
                                          phy_units=np.ones(Tok.UNITS_VECTOR_SIZE))

            cos = Tok.Token(name='cos', sympy_repr='cos', arity=1, complexity=0, var_type=0,
                            function=np.cos,
                            var_id=None,
                            is_constraining_phy_units = True,
                            phy_units = np.zeros(Tok.UNITS_VECTOR_SIZE), )
            sqrt = Tok.Token(name='sqrt', sympy_repr='sqrt', arity=1, complexity=0, var_type=0,
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
                                          var_type=1,
                                          function=None,
                                          var_id=0,
                                          is_constraining_phy_units=True,
                                          phy_units=None)
        # Test exception: token with is_constraining_phy_units = True must have units vector of size Lib.UNITS_VECTOR_SIZE
        with self.assertRaises(AssertionError):
            physical_quantity = Tok.Token(name='x0', sympy_repr='x0', arity=0, complexity=0,
                                          var_type=1,
                                          function=None,
                                          var_id=0,
                                          is_constraining_phy_units=True,
                                          phy_units=np.ones(Tok.UNITS_VECTOR_SIZE + 1))
        # Test exception: token with is_constraining_phy_units = True must not contain any NAN
        wrong_units = np.ones(Tok.UNITS_VECTOR_SIZE)
        wrong_units[0] = np.NAN
        with self.assertRaises(AssertionError):
            physical_quantity = Tok.Token(name='x0', sympy_repr='x0', arity=0, complexity=0,
                                          var_type=1,
                                          function=None,
                                          var_id=0,
                                          is_constraining_phy_units=True,
                                          phy_units=wrong_units)
        # Test exception: token with is_constraining_phy_units = False must not have units
        with self.assertRaises(AssertionError):
            add = Tok.Token(name='add', sympy_repr='add', arity=2, complexity=0, var_type=0,
                            function=np.add,
                            var_id=None,
                            is_constraining_phy_units=False,
                            phy_units=np.ones(Tok.UNITS_VECTOR_SIZE))

    # Test token creation exceptions : behavior_id
    def test_token_creation_units_exceptions_behavior_id(self):
        # Test exception: behavior_id must be an int
        with self.assertRaises(AssertionError):
            cos = Tok.Token(name='cos', sympy_repr='cos', arity=1, complexity=0, var_type=0,
                            function=np.cos,
                            var_id=None,
                            is_constraining_phy_units=True,
                            phy_units=np.zeros(Tok.UNITS_VECTOR_SIZE),
                            behavior_id="not an int")

    # Test token creation exceptions : power
    def test_token_creation_units_exceptions_power(self):
        # Test exception: power must be a float
        with self.assertRaises(AssertionError):
            sqrt = Tok.Token(name='sqrt', sympy_repr='sqrt', arity=1, complexity=0, var_type=0,
                             function=np.sqrt,
                             var_id=None,
                             is_power=True,
                             power="not a float")

        with self.assertRaises(AssertionError):
            sqrt = Tok.Token(name='sqrt', sympy_repr='sqrt', arity=1, complexity=0, var_type=0,
                             function=np.sqrt,
                             var_id=None,
                             is_power=False,
                             power=0.5)

if __name__ == '__main__':
    unittest.main(verbosity = 2)

