import warnings as warnings
import numpy as np
import torch as torch

# Internal imports
from physo.physym import token as Tok
from physo.physym.token import Token

# ------------------------------------------------------------------------------------------------------
# ------------------------------------------ UTILS MANAGEMENT ------------------------------------------
# ------------------------------------------------------------------------------------------------------

# Error to raise when a function is unknown
class UnknownFunction(Exception):
    pass

# Units behavior
class OpUnitsBehavior(object):
    """
    Encodes a single behavior (in the context of dimensional analysis) that concerns multiple operations.
    Attributes
    ----------
    op_names : list of str
        List of operation names having this behavior (eg. [sqrt, n2, n3] for power tokens)
    behavior_id : int
        Unique id of behavior.
    """
    def __init__(self, behavior_id, op_names,):
        self.behavior_id = behavior_id
        # Using list for op_names to be able to use if x in list in the future
        self.op_names    = np.array(op_names).tolist()

    def is_id (self, array):
        """
        Compares array to behavior id.
        Parameters
        ----------
        array      : int or numpy.array of shape = (whatever) of int
        Returns
        -------
        comparison : bool or numpy.array of shape = (whatever) of bool
        """
        comparison = np.equal.outer(array, [self.behavior_id,] ).any(axis=-1)
        return comparison

    def __repr__(self):
        return "OpUnitsBehavior (id = %s, op_names=%s)"%(self.behavior_id, self.op_names)

# Group of units behavior
class GroupUnitsBehavior(object):
    """
    Encodes a master-behavior (in the context of dimensional analysis) that is common among several sub-behaviors.
    Eg: mul and div tokens have their one unique behaviors regarding physical units (units of mul = units of arg1 +
    units of arg2 whereas units of div = units of arg1 - units of arg2) but they also share common behaviors that can
    be encoded here (in both cases it is impossible to guess the units of the token unless both args' units  are known
    etc.).
    Attributes
    ----------
    behaviors : list of OpUnitsBehavior
        Sub-behaviors that are part of this group.
    op_names : list of str
        List of operation names having this behavior (eg. [mul, div,] for multiplicative tokens)
    __behavior_ids : list of int
        Unique ids of each sub-behaviors.
    """
    def __init__(self, behaviors,):
        self.behaviors      = np.array(behaviors)
        # Using list for op_names to be able to use if x in list in the future
        self.op_names       = np.concatenate           ([behavior.op_names    for behavior in self.behaviors]).tolist()
        # Preventing __behavior_ids from being accessed as is_id should be called rather than doing direct comparisons
        # (Otherwise this could lead to eg: ([20, 21, 20, 20] == ids) = False even if ids = [20, 21]).
        self.__behavior_ids = np.array                 ([behavior.behavior_id for behavior in self.behaviors])

    def is_id (self, array):
        """
        Compares array to behavior ids of operations concerned by this group of behaviors to array.
        For each value of array, returns True if at least one sub-behavior id is equal to value and False otherwise.
        Eg: if mul's behavior id is 20 and div's behavior id is 21 and they are both represented in a group of behavior
        by a group = GroupUnitsBehavior, group.is_id([20, 20, 21]) = [True, True, True] ; group.is_id(20) = True ;
        group.is_id([20, 20, 99999]) = [True, True, False] ; group.is_id(99999) = False.
        Parameters
        ----------
        array      : int or numpy.array of shape = (whatever) of int
        Returns
        -------
        comparison : bool or numpy.array of shape = (whatever) of bool
        """
        comparison = np.equal.outer(array, self.__behavior_ids).any(axis=-1)
        return comparison

# ------------------------------------------------------------------------------------------------------
# --------------------------------------------- UTILS INFO ---------------------------------------------
# ------------------------------------------------------------------------------------------------------

# BEHAVIOR DURING DIMENSIONAL ANALYSIS
# Unique (token-wise) behaviors dict (ie. each token must appear in only one of those behavior, if necessary use a
# group of unit behavior).
OP_UNIT_BEHAVIORS_DICT = {
    "DEFAULT_BEHAVIOR"          : OpUnitsBehavior(behavior_id = Tok.DEFAULT_BEHAVIOR_ID, op_names = []),
    # Operations taking two arguments and having an additive behavior: units of both args and of op are the same.
    "BINARY_ADDITIVE_OP"        : OpUnitsBehavior(behavior_id = 1 , op_names = ["add", "sub"]),
    # Multiplication operation (units of op = units of arg 0 + units of arg 1)
    "MULTIPLICATION_OP"         : OpUnitsBehavior(behavior_id = 20, op_names = ["mul",]),
    # Division operation (units of op = units of arg 0 - units of arg 1).
    "DIVISION_OP"               : OpUnitsBehavior(behavior_id = 21, op_names = ["div",]),
    # Power operations taking one argument.
    "UNARY_POWER_OP"            : OpUnitsBehavior(behavior_id = 3 , op_names = ["n2", "sqrt", "n3", "n4", "inv"]),
    # Operations taking one argument and having an additive behavior: units of arg and parent should be the same).
    "UNARY_ADDITIVE_OP"         : OpUnitsBehavior(behavior_id = 4 , op_names = ["neg", "abs", "max", "min"]),
    # Dimensionless operations taking one dimensionless argument.
    "UNARY_DIMENSIONLESS_OP"    : OpUnitsBehavior(behavior_id = 5 , op_names = ["sin", "cos", "tan", "exp", "log", "expneg", "logabs", "sigmoid", "tanh", "sinh", "cosh", "harmonic", "arctan", "arccos", "arcsin", "erf", "pow"]),
            }
# Group of behaviors (tokens can appear in more than one of them)
GROUP_UNIT_BEHAVIOR = {
    "BINARY_MULTIPLICATIVE_OP": GroupUnitsBehavior(behaviors=[OP_UNIT_BEHAVIORS_DICT["MULTIPLICATION_OP"],
                                                              OP_UNIT_BEHAVIORS_DICT["DIVISION_OP"      ]]),
                      }
# All behaviors (tokens can appear in more than one of them)
UNIT_BEHAVIORS_DICT = {}
UNIT_BEHAVIORS_DICT.update(OP_UNIT_BEHAVIORS_DICT)
UNIT_BEHAVIORS_DICT.update(GROUP_UNIT_BEHAVIOR)

# TRIGONOMETRIC OPS
TRIGONOMETRIC_OP = ["sin", "cos", "tan", "tanh", "sinh", "cosh", "arctan", "arccos", "arcsin"]

# INVERSE OP
INVERSE_OP_DICT = {
    "inv": "inv",
    "neg": "neg",
    "exp": "log",
    "log": "exp",
    "sqrt": "n2",
    "n2": "sqrt",
    "arctan" : "tan",
    "tan"    : "arctan",
    "arcsin" : "sin",
    "sin"    : "arcsin",
    "arccos" : "cos",
    "cos"    : "arccos",
                  }

# POWER VALUES OF POWER TOKENS
OP_POWER_VALUE_DICT = {
     "n2"   : 2,
     "sqrt" : 0.5,
     "n3"   : 3,
     "n4"   : 4,
     "inv"  : -1,
}

# Data conversion to perform before being able to use functions
def data_conversion (data):
    if isinstance(data, float):
        return torch.tensor(np.array(data))
    else:
        return torch.tensor(data)

# Inverse of data conversion
def data_conversion_inv(data):
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    else:
        return data

# ------------------------------------------------------------------------------------------------------
# ---------------------------------------------- FUNCTIONS ---------------------------------------------
# ------------------------------------------------------------------------------------------------------
# All functions must be pickable (defined at highest level) to be able to use parallel computation

# ------------- unprotected functions -------------
def torch_pow(x0, x1):
    if not torch.is_tensor(x0):
        x0 = torch.ones_like(x1) * x0
    return torch.pow(x0, x1)

OPS_UNPROTECTED = [
    #  Binary operations
    Token (name = "add"    , sympy_repr = "+"      , arity = 2 , complexity = 1 , var_type = 0, function = torch.add        ),
    Token (name = "sub"    , sympy_repr = "-"      , arity = 2 , complexity = 1 , var_type = 0, function = torch.subtract   ),
    Token (name = "mul"    , sympy_repr = "*"      , arity = 2 , complexity = 1 , var_type = 0, function = torch.multiply   ),
    Token (name = "div"    , sympy_repr = "/"      , arity = 2 , complexity = 1 , var_type = 0, function = torch.divide     ),
    # Unary operations
    Token (name = "sin"    , sympy_repr = "sin"    , arity = 1 , complexity = 1 , var_type = 0, function = torch.sin        ),
    Token (name = "cos"    , sympy_repr = "cos"    , arity = 1 , complexity = 1 , var_type = 0, function = torch.cos        ),
    Token (name = "tan"    , sympy_repr = "tan"    , arity = 1 , complexity = 1 , var_type = 0, function = torch.tan        ),
    Token (name = "exp"    , sympy_repr = "exp"    , arity = 1 , complexity = 1 , var_type = 0, function = torch.exp        ),
    Token (name = "log"    , sympy_repr = "log"    , arity = 1 , complexity = 1 , var_type = 0, function = torch.log        ),
    Token (name = "sqrt"   , sympy_repr = "sqrt"   , arity = 1 , complexity = 1 , var_type = 0, function = torch.sqrt       ),
    Token (name = "n2"     , sympy_repr = "n2"     , arity = 1 , complexity = 1 , var_type = 0, function = torch.square     ),
    Token (name = "neg"    , sympy_repr = "-"      , arity = 1 , complexity = 1 , var_type = 0, function = torch.negative   ),
    Token (name = "abs"    , sympy_repr = "abs"    , arity = 1 , complexity = 1 , var_type = 0, function = torch.abs        ),
    Token (name = "inv"    , sympy_repr = "1/"     , arity = 1 , complexity = 1 , var_type = 0, function = torch.reciprocal ),
    Token (name = "tanh"   , sympy_repr = "tanh"   , arity = 1 , complexity = 1 , var_type = 0, function = torch.tanh       ),
    Token (name = "sinh"   , sympy_repr = "sinh"   , arity = 1 , complexity = 1 , var_type = 0, function = torch.sinh       ),
    Token (name = "cosh"   , sympy_repr = "cosh"   , arity = 1 , complexity = 1 , var_type = 0, function = torch.cosh       ),
    Token (name = "arctan" , sympy_repr = "arctan" , arity = 1 , complexity = 1 , var_type = 0, function = torch.arctan     ),
    Token (name = "arccos" , sympy_repr = "arccos" , arity = 1 , complexity = 1 , var_type = 0, function = torch.arccos     ),
    Token (name = "arcsin" , sympy_repr = "arcsin" , arity = 1 , complexity = 1 , var_type = 0, function = torch.arcsin     ),
    Token (name = "erf"    , sympy_repr = "erf"    , arity = 1 , complexity = 1 , var_type = 0, function = torch.erf        ),

    # Custom unary operations
    Token (name = "logabs" , sympy_repr = "logabs" , arity = 1 , complexity = 1 , var_type = 0, function = lambda x :torch.log(torch.abs(x)) ),
    Token (name = "expneg" , sympy_repr = "expneg" , arity = 1 , complexity = 1 , var_type = 0, function = lambda x :torch.exp(-x)           ),
    Token (name = "n3"     , sympy_repr = "n3"     , arity = 1 , complexity = 1 , var_type = 0, function = lambda x :torch.pow(x, 3)         ),
    Token (name = "n4"     , sympy_repr = "n4"     , arity = 1 , complexity = 1 , var_type = 0, function = lambda x :torch.pow(x, 4)         ),

    # Custom binary operations
    Token (name = "pow"     , sympy_repr = "pow"   , arity = 2 , complexity = 1 , var_type = 0, function = torch_pow                         ),
]

# ------------- protected functions -------------

def protected_div(x1, x2):
    #with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
    return torch.where(torch.abs(x2) > 0.001, torch.divide(x1, x2), 1.)

def protected_exp(x1):
    #with np.errstate(over='ignore'):
    return torch.where(x1 < 100, torch.exp(x1), 0.0)

def protected_log(x1):
    #with np.errstate(divide='ignore', invalid='ignore'):
    return torch.where(torch.abs(x1) > 0.001, torch.log(torch.abs(x1)), 0.)

protected_logabs = protected_log

def protected_sqrt(x1):
    return torch.sqrt(torch.abs(x1))

def protected_inv(x1):
    # with np.errstate(divide='ignore', invalid='ignore'):
    return torch.where(torch.abs(x1) > 0.001, 1. / x1, 0.)

def protected_expneg(x1):
    # with np.errstate(over='ignore'):
    return torch.where(x1 > -100, torch.exp(-x1), 0.0)

def protected_n2(x1):
    # with np.errstate(over='ignore'):
    return torch.where(torch.abs(x1) < 1e6, torch.square(x1), 0.0)

def protected_n3(x1):
    # with np.errstate(over='ignore'):
    return torch.where(torch.abs(x1) < 1e6, torch.pow(x1, 3), 0.0)

def protected_n4(x1):
    # with np.errstate(over='ignore'):
    return torch.where(torch.abs(x1) < 1e6, torch.pow(x1, 4), 0.0)

def protected_arcsin (x1):
    inf = 1e6
    return torch.where(torch.abs(x1) < 0.999, torch.arcsin(x1), torch.sign(x1)*inf)

def protected_arccos (x1):
    inf = 1e6
    return torch.where(torch.abs(x1) < 0.999, torch.arccos(x1), torch.sign(x1)*inf)

def protected_torch_pow(x0, x1):
    inf = 1e6
    if not torch.is_tensor(x0):
       x0 = torch.ones_like(x1)*x0
    y = torch.pow(x0, x1)
    y = torch.where(y > inf, inf, y)
    return y

OPS_PROTECTED = [
    # Binary operations
    Token (name = "div"    , sympy_repr = "/"      , arity = 2 , complexity = 1 , var_type = 0, function = protected_div    ),
    # Unary operations
    Token (name = "exp"    , sympy_repr = "exp"    , arity = 1 , complexity = 1 , var_type = 0, function = protected_exp    ),
    Token (name = "log"    , sympy_repr = "log"    , arity = 1 , complexity = 1 , var_type = 0, function = protected_log    ),
    Token (name = "sqrt"   , sympy_repr = "sqrt"   , arity = 1 , complexity = 1 , var_type = 0, function = protected_sqrt   ),
    Token (name = "n2"     , sympy_repr = "n2"     , arity = 1 , complexity = 1 , var_type = 0, function = protected_n2     ),
    Token (name = "inv"    , sympy_repr = "1/"     , arity = 1 , complexity = 1 , var_type = 0, function = protected_inv    ),
    Token (name = "arccos" , sympy_repr = "arccos" , arity = 1 , complexity = 1 , var_type = 0, function = protected_arccos ),
    Token (name = "arcsin" , sympy_repr = "arcsin" , arity = 1 , complexity = 1 , var_type = 0, function = protected_arcsin ),

    # Custom unary operations
    Token (name = "logabs" , sympy_repr = "logabs" , arity = 1 , complexity = 1 , var_type = 0, function = protected_logabs ),
    Token (name = "expneg" , sympy_repr = "expneg" , arity = 1 , complexity = 1 , var_type = 0, function = protected_expneg ),
    Token (name = "n3"     , sympy_repr = "n3"     , arity = 1 , complexity = 1 , var_type = 0, function = protected_n3     ),
    Token (name = "n4"     , sympy_repr = "n4"     , arity = 1 , complexity = 1 , var_type = 0, function = protected_n4     ),

    # Custom binary operations
    Token (name = "pow"     , sympy_repr = "pow"   , arity = 2 , complexity = 1 , var_type = 0, function = protected_torch_pow   ),

]

# ------------- encoding additional attributes (for units analysis) -------------

# iterating through all available tokens
for token_op in OPS_PROTECTED + OPS_UNPROTECTED:
    # encoding token behavior in dimensional analysis
    for _, behavior in OP_UNIT_BEHAVIORS_DICT.items():
        # Filtering out objects in the dict that are not meant to affect tokens' behavior id
        if token_op.name in behavior.op_names:
            token_op.behavior_id = behavior.behavior_id
    # encoding dimensionless tokens units
    if token_op.name in OP_UNIT_BEHAVIORS_DICT["UNARY_DIMENSIONLESS_OP"].op_names:
        token_op.is_constraining_phy_units = True
        token_op.phy_units                 = np.zeros((Tok.UNITS_VECTOR_SIZE))
    # encoding power tokens values
    if token_op.name in OP_UNIT_BEHAVIORS_DICT["UNARY_POWER_OP"].op_names:
        token_op.is_power = True
        try: token_op.power    = OP_POWER_VALUE_DICT[token_op.name]
        except KeyError: raise UnknownFunction("Token %s is a power token as it is listed in UNARY_POWER_OP "
            "(containing : %s) but the value of its power is not defined in dict OP_POWER_VALUE_DICT = %s"
            % (token_op.name, OP_UNIT_BEHAVIORS_DICT["UNARY_POWER_OP"].op_names, OP_POWER_VALUE_DICT))

# ------------- protected functions -------------

OPS_UNPROTECTED_DICT = {op.name: op for op in OPS_UNPROTECTED}
# Copy unprotected operations
OPS_PROTECTED_DICT = OPS_UNPROTECTED_DICT.copy()
# Update protected operations when defined
OPS_PROTECTED_DICT.update( {op.name: op for op in OPS_PROTECTED} )

# ------------------------------------------------------------------------------------------------------
# --------------------------------------------- MAKE TOKENS --------------------------------------------
# ------------------------------------------------------------------------------------------------------


# -------------------------------- Utils functions --------------------------------

def retrieve_complexity(complexity_dict, curr_name):
    """
    Helper function to safely retrieve complexity of token named curr_name from a dictionary of complexities
    (complexity_dict).
    Parameters
    ----------
    complexity_dict : dict of {str : float} or None
        If dictionary is None, returns token.DEFAULT_COMPLEXITY.
    curr_name : str
        If curr_name is not in units_dict keys, returns token.DEFAULT_COMPLEXITY.
    Returns
    -------
    curr_complexity : float
        Complexity of token.
    """
    curr_complexity = Tok.DEFAULT_COMPLEXITY
    if complexity_dict is not None:
        try:
            curr_complexity = complexity_dict[curr_name]
        except KeyError:
            warnings.warn(
                "Complexity of token %s not found in complexity dictionary %s, using complexity = %f" %
                (curr_name, complexity_dict, curr_complexity))
    curr_complexity = float(curr_complexity)
    return curr_complexity

def retrieve_init_val (init_val_dict, curr_name):
    """
    Helper function to safely retrieve value of token named curr_name from a dictionary of initial values.
    (init_val_dict).
    Parameters
    ----------
    init_val_dict : dict of {str : float} or None
        If dictionary is None, returns token.DEFAULT_FREE_CONST_INIT_VAL.
    curr_name : str
        If curr_name is not in units_dict keys, returns token.DEFAULT_FREE_CONST_INIT_VAL.
    Returns
    -------
    curr_init_val : float
        Initial value of token.
    """
    curr_init_val = Tok.DEFAULT_FREE_CONST_INIT_VAL
    if init_val_dict is not None:
        try:
            curr_init_val = init_val_dict[curr_name]
        except KeyError:
            warnings.warn(
                "Initial value of token %s not found in initial value dictionary %s, using complexity = %f" %
                (curr_name, init_val_dict, curr_init_val))
    curr_init_val = float(curr_init_val)
    return curr_init_val


def retrieve_units(units_dict, curr_name):
    """
    Helper function to safely retrieve units of token named curr_name from a dictionary of units (units_dict).
    Parameters
    ----------
    units_dict : dict of {str : array_like} or None
        If dictionary is None, returned curr_is_constraining_phy_units is False and curr_phy_units is None.
        (Note: creating a token.Token using None in place of units will result in a Token with units = vector of np.NAN)
    curr_name : str
        If curr_name is not in units_dict keys, returned curr_phy_units correspond to that of  dimensionless token
        (ie. vector of zeros).
    Returns
    -------
    curr_is_constraining_phy_units, curr_phy_units : (bool, numpy.array)
        Does the token require physical units, numpy array containing units
    """
    # Not working with units by default
    curr_is_constraining_phy_units = False
    curr_phy_units = None
    # retrieving units if user is using units dictionary
    if units_dict is not None:
        curr_is_constraining_phy_units = True
        try:
            curr_phy_units = units_dict[curr_name]
        except KeyError:
            curr_phy_units = np.zeros(Tok.UNITS_VECTOR_SIZE)
            warnings.warn(
                "Physical units of token %s not found in units dictionary %s, assuming it is \
                dimensionless ie units=%s" % (curr_name, units_dict, curr_phy_units))
        # Padding to match Lib.UNITS_VECTOR_SIZE + conversion to numpy array of float if necessary
        try:
            curr_phy_units = np.array(curr_phy_units).astype(float)
        except Exception:
            raise AssertionError("Physical units vector must be castable to numpy array of floats")
        assert len(curr_phy_units.shape) == 1, 'Physical units vector must have 1 dimension not %i' % (
            len(curr_phy_units.shape))
        curr_size = len(curr_phy_units)
        assert curr_size <= Tok.UNITS_VECTOR_SIZE, 'Physical units vector has size = %i which exceeds max size = %i \
            (Lib.UNITS_VECTOR_SIZE, can be changed)' % (curr_size, Tok.UNITS_VECTOR_SIZE)
        curr_phy_units = np.pad(curr_phy_units, (0, Tok.UNITS_VECTOR_SIZE - curr_size), 'constant')
    return curr_is_constraining_phy_units, curr_phy_units


def make_tokens(
                # operations
                op_names             = "all",
                use_protected_ops    = False,
                # input variables
                input_var_ids        = None,
                input_var_units      = None,
                input_var_complexity = None,
                # constants
                constants            = None,
                constants_units      = None,
                constants_complexity = None,
                # free constants
                free_constants            = None,
                free_constants_init_val   = None,
                free_constants_units      = None,
                free_constants_complexity = None,
                ):
    """
        Makes a list of tokens for a run based on a list of operation names, input variables ids and constants values.
        Parameters
        ----------
        -------- operations --------
        op_names : list of str or str, optional
            List of names of operations that will be used for a run (eg. ["mul", "add", "neg", "inv", "sin"]), or "all"
            to use all available tokens. By default, op_names = "all".
        use_protected_ops : bool, optional
            If True safe functions defined in functions.OPS_PROTECTED_DICT in place when available (eg. sqrt(abs(x))
            instead of sqrt(x)). False by default.
        -------- input variables --------
        input_var_ids : dict of { str : int } or None, optional
            Dictionary containing input variables names as keys (eg. 'x', 'v', 't') and corresponding input variables
            ids in dataset (eg. 0, 1, 2). None if no input variables to create. None by default.
        input_var_units : dict of { str : array_like of float } or None, optional
            Dictionary containing input variables names as keys (eg. 'x', 'v', 't') and corresponding physical units
            (eg. [1, 0, 0], [1, -1, 0], [0, 1, 0]). With x representing a distance, v a velocity and t a time assuming a
            convention such as [m, s, kg,...]). None if not using physical units. None by default.
        input_var_complexity : dict of { str : float } or None, optional
            Dictionary containing input variables names as keys (eg. 'x', 'v', 't') and corresponding complexities
            (eg. 0., 1., 0.). If None, complexity = token.DEFAULT_COMPLEXITY will be encoded to tokens. None by default.
        -------- constants --------
        constants : dict of { str : float } or None, optional
            Dictionary containing constant names as keys (eg. 'pi', 'c', 'M') and corresponding float values
            (eg. np.pi, 3e8, 1e6). None if no constants to create. None by default.
        constants_units : dict of { str : array_like of float } or None, optional
            Dictionary containing constants names as keys (eg. 'pi', 'c', 'M') and corresponding physical units
            (eg. [0, 0, 0], [1, -1, 0], [0, 0, 1]). With pi representing a dimensionless number, c a velocity and M a
            mass assuming a convention such as [m, s, kg,...]). None if not using physical units. None by default.
        constants_complexity : dict of { str : float } or None, optional
            Dictionary containing constants names as keys (eg. 'pi', 'c', 'M') and corresponding complexities
            (eg. 0., 0., 1.). If None, complexity = token.DEFAULT_COMPLEXITY will be encoded to tokens. None by default.
        -------- free constants --------
        free_constants : set of { str } or None, optional
            Set containing free constant names (eg. 'c0', 'c1', 'c2'). None if no free constants to create.
            None by default.
        free_constants_init_val : dict of { str : float } or None, optional
            Dictionary containing free constants names as keys (eg. 'c0', 'c1', 'c2') and corresponding float initial
            values to use during optimization process (eg. 1., 1., 1.). None will result in the usage of
            token.DEFAULT_FREE_CONST_INIT_VAL as initial values. None by default.
        free_constants_units : dict of { str : array_like of float } or None, optional
            Dictionary containing free constants names as keys (eg. 'c0', 'c1', 'c2') and corresponding physical units
            (eg. [0, 0, 0], [1, -1, 0], [0, 0, 1]). With c0 representing a dimensionless number, c1 a velocity and c2 a
            mass assuming a convention such as [m, s, kg,...]). None if not using physical units. None by default.
        free_constants_complexity : dict of { str : float } or None, optional
            Dictionary containing free constants names as keys (eg. 'c0', 'c1', 'c2') and corresponding complexities
            (eg. 1., 1., 1.). If None, complexity = token.DEFAULT_COMPLEXITY will be encoded to tokens. None by default.
        Returns
        -------
        list of token.Token
            List of tokens used for this run.

        Examples
        -------
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
                                    )
    """
    # -------------------------------- Handling ops --------------------------------
    tokens_ops = []
    # Use protected functions or not
    ops_dict = OPS_PROTECTED_DICT if use_protected_ops else OPS_UNPROTECTED_DICT
    # Using all available tokens
    if op_names == "all":
        tokens_ops = list(ops_dict.values())
    # Appending desired functions tokens
    else:
        # Iterating through desired functions names
        for name in op_names:
            # appending token function if available
            try:
                tokens_ops.append(ops_dict[name])
            except KeyError:
                raise UnknownFunction("%s is unknown, define a custom token function in functions.py or use a function \
                listed in %s"% (name, ops_dict))

    # -------------------------------- Handling input variables --------------------------------
    tokens_input_var = []
    if input_var_ids is not None:
        # Iterating through input variables
        for var_name, var_id in input_var_ids.items():
            # ------------- Units -------------
            is_constraining_phy_units, phy_units = retrieve_units (units_dict=input_var_units, curr_name=var_name)
            # ------------- Complexity -------------
            complexity = retrieve_complexity (complexity_dict=input_var_complexity, curr_name=var_name)
            # ------------- Token creation -------------
            tokens_input_var.append(Token(name         = var_name,
                                          sympy_repr   = var_name,
                                          arity        = 0,
                                          complexity   = complexity,
                                          var_type     = 1,
                                          # Input variable specific
                                          var_id       = var_id,
                                          # ---- Physical units : units ----
                                          is_constraining_phy_units = is_constraining_phy_units,
                                          phy_units                 = phy_units,))

    # -------------------------------- Handling constants --------------------------------
    tokens_constants = []
    if constants is not None:
        # Iterating through constants
        for const_name, const_val in constants.items():
            # ------------- Units -------------
            is_constraining_phy_units, phy_units = retrieve_units (units_dict=constants_units, curr_name=const_name)
            # ------------- Complexity -------------
            complexity = retrieve_complexity (complexity_dict=constants_complexity, curr_name=const_name)
            # ------------- Token creation -------------
            # Very important to put const as a default arg of lambda function
            # https://stackoverflow.com/questions/19837486/lambda-in-a-loop
            # or use def MakeConstFunc(x): return lambda: x
            tokens_constants.append(Token(name         = const_name,
                                          sympy_repr   = const_name,
                                          arity        = 0,
                                          complexity   = complexity,
                                          var_type     = 3,
                                          # Fixed const specific
                                          fixed_const  = const_val,
                                          # ---- Physical units : units ----
                                          is_constraining_phy_units = is_constraining_phy_units,
                                          phy_units                 = phy_units,))

    # -------------------------------- Handling free constants --------------------------------
    tokens_free_constants = []
    if free_constants is not None:
        free_constants_sorted = list(sorted(free_constants))  # (enumerating on sorted list rather than set)
        # Iterating through free constants
        for i, free_const_name in enumerate(free_constants_sorted):
            # ------------- Initial value -------------
            init_val = retrieve_init_val(init_val_dict=free_constants_init_val, curr_name=free_const_name)
            # ------------- Units -------------
            is_constraining_phy_units, phy_units = retrieve_units (units_dict=free_constants_units, curr_name=free_const_name)
            # ------------- Complexity -------------
            complexity = retrieve_complexity (complexity_dict=free_constants_complexity, curr_name=free_const_name)
            # ------------- Token creation -------------
            tokens_free_constants.append(Token(name         = free_const_name,
                                               sympy_repr   = free_const_name,
                                               arity        = 0,
                                               complexity   = complexity,
                                               var_type     = 2,
                                               # Free constant specific
                                               var_id       = i,
                                               init_val     = init_val,
                                               # ---- Physical units : units ----
                                               is_constraining_phy_units = is_constraining_phy_units,
                                               phy_units                 = phy_units,))

    # -------------------------------- Result --------------------------------
    return np.array(tokens_ops + tokens_constants + tokens_free_constants + tokens_input_var)

