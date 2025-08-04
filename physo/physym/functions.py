import warnings as warnings
import numpy as np
import torch as torch

# Internal imports
from physo.physym import token as Tok
from physo.physym.token import TokenOp

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
        List of operation names having this behavior (eg. [sqrt, cbrt, n2, n3] for power tokens)
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
    "UNARY_POWER_OP"            : OpUnitsBehavior(behavior_id = 3 , op_names = ["n2", "sqrt", "n3", "cbrt", "n4", "inv"]),
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
    "inv" : "inv",
    "neg" : "neg",
    "exp" : "log",
    "log" : "exp",
    "sqrt" : "n2",
    "n2"   : "sqrt",
    "cbrt" : "n3",
    "n3"   : "cbrt",
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
     "cbrt" : 1./3,
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
    TokenOp (name = "add"    , sympy_repr = "+"      , arity = 2 , complexity = 1 , function = torch.add        ),
    TokenOp (name = "sub"    , sympy_repr = "-"      , arity = 2 , complexity = 1 , function = torch.subtract   ),
    TokenOp (name = "mul"    , sympy_repr = "*"      , arity = 2 , complexity = 1 , function = torch.multiply   ),
    TokenOp (name = "div"    , sympy_repr = "/"      , arity = 2 , complexity = 1 , function = torch.divide     ),
    # Unary operations
    TokenOp (name = "sin"    , sympy_repr = "sin"    , arity = 1 , complexity = 1 , function = torch.sin        ),
    TokenOp (name = "cos"    , sympy_repr = "cos"    , arity = 1 , complexity = 1 , function = torch.cos        ),
    TokenOp (name = "tan"    , sympy_repr = "tan"    , arity = 1 , complexity = 1 , function = torch.tan        ),
    TokenOp (name = "exp"    , sympy_repr = "exp"    , arity = 1 , complexity = 1 , function = torch.exp        ),
    TokenOp (name = "log"    , sympy_repr = "log"    , arity = 1 , complexity = 1 , function = torch.log        ),
    TokenOp (name = "sqrt"   , sympy_repr = "sqrt"   , arity = 1 , complexity = 1 , function = torch.sqrt       ),
    TokenOp (name = "cbrt"   , sympy_repr = "cbrt"   , arity = 1 , complexity = 1 , function = lambda x :torch.pow(x, 1./3) ),
    TokenOp (name = "n2"     , sympy_repr = "n2"     , arity = 1 , complexity = 1 , function = torch.square     ),
    TokenOp (name = "neg"    , sympy_repr = "-"      , arity = 1 , complexity = 1 , function = torch.negative   ),
    TokenOp (name = "abs"    , sympy_repr = "abs"    , arity = 1 , complexity = 1 , function = torch.abs        ),
    TokenOp (name = "inv"    , sympy_repr = "1/"     , arity = 1 , complexity = 1 , function = torch.reciprocal ),
    TokenOp (name = "tanh"   , sympy_repr = "tanh"   , arity = 1 , complexity = 1 , function = torch.tanh       ),
    TokenOp (name = "sinh"   , sympy_repr = "sinh"   , arity = 1 , complexity = 1 , function = torch.sinh       ),
    TokenOp (name = "cosh"   , sympy_repr = "cosh"   , arity = 1 , complexity = 1 , function = torch.cosh       ),
    TokenOp (name = "arctan" , sympy_repr = "arctan" , arity = 1 , complexity = 1 , function = torch.arctan     ),
    TokenOp (name = "arccos" , sympy_repr = "arccos" , arity = 1 , complexity = 1 , function = torch.arccos     ),
    TokenOp (name = "arcsin" , sympy_repr = "arcsin" , arity = 1 , complexity = 1 , function = torch.arcsin     ),
    TokenOp (name = "erf"    , sympy_repr = "erf"    , arity = 1 , complexity = 1 , function = torch.erf        ),

    # Custom unary operations
    TokenOp (name = "logabs" , sympy_repr = "logabs" , arity = 1 , complexity = 1 , function = lambda x :torch.log(torch.abs(x)) ),
    TokenOp (name = "expneg" , sympy_repr = "expneg" , arity = 1 , complexity = 1 , function = lambda x :torch.exp(-x)           ),
    TokenOp (name = "n3"     , sympy_repr = "n3"     , arity = 1 , complexity = 1 , function = lambda x :torch.pow(x, 3)         ),
    TokenOp (name = "n4"     , sympy_repr = "n4"     , arity = 1 , complexity = 1 , function = lambda x :torch.pow(x, 4)         ),

    # Custom binary operations
    TokenOp (name = "pow"     , sympy_repr = "**"    , arity = 2 , complexity = 1 , function = torch_pow                         ),
]

# ------------- protected functions -------------
EPSILON = 0.001
EXP_THRESHOLD = 80.
INF = 1e6

def protected_div(x1, x2):
    #with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
    return torch.where(torch.abs(x2) > EPSILON, torch.divide(x1, x2), 1.)

exp_plateau = np.exp(EXP_THRESHOLD)
def protected_exp(x1):
    #with np.errstate(over='ignore'):
    return torch.where(x1 <= EXP_THRESHOLD, torch.exp(x1), exp_plateau)

log_plateau = np.log(np.abs(EPSILON))
def protected_log(x1):
    #with np.errstate(divide='ignore', invalid='ignore'):
    return torch.where(torch.abs(x1) >= EPSILON, torch.log(torch.abs(x1)), log_plateau)

protected_logabs = protected_log

def protected_sqrt(x1):
    return torch.sqrt(torch.abs(x1))

def protected_cbrt(x1):
    return torch.pow(torch.abs(x1), 1./3)

def protected_inv(x1):
    # with np.errstate(divide='ignore', invalid='ignore'):
    return torch.where(torch.abs(x1) > EPSILON, 1. / x1, 0.)

expneg_plateau = np.exp(--EXP_THRESHOLD)
def protected_expneg(x1):
    # with np.errstate(over='ignore'):
    return torch.where(x1 >= -EXP_THRESHOLD, torch.exp(-x1), expneg_plateau)

n2_plateau = np.square(INF)
def protected_n2(x1):
    # with np.errstate(over='ignore'):
    return torch.where(torch.abs(x1) <= INF, torch.square(x1), n2_plateau)

n3_plateau = np.power(INF, 3)
def protected_n3(x1):
    # with np.errstate(over='ignore'):
    return torch.where(torch.abs(x1) <= INF, torch.pow(x1, 3), torch.sign(x1)*n3_plateau)

n4_plateau = np.power(INF, 4)
def protected_n4(x1):
    # with np.errstate(over='ignore'):
    return torch.where(torch.abs(x1) <= INF, torch.pow(x1, 4), n4_plateau)

def protected_arcsin (x1):
    return torch.where(torch.abs(x1) < (1.-EPSILON), torch.arcsin(x1), torch.sign(x1)*INF)

def protected_arccos (x1):
    return torch.where(torch.abs(x1) < (1.-EPSILON), torch.arccos(x1), torch.sign(x1)*INF)

def protected_torch_pow(x0, x1):
    if not torch.is_tensor(x0):
       x0 = torch.ones_like(x1)*x0
    y = torch.pow(x0, x1)
    y = torch.where(y > INF, INF, y)
    return y

OPS_PROTECTED = [
    # Binary operations
    TokenOp (name = "div"    , sympy_repr = "/"      , arity = 2 , complexity = 1 , function = protected_div    ),
    # Unary operations
    TokenOp (name = "exp"    , sympy_repr = "exp"    , arity = 1 , complexity = 1 , function = protected_exp    ),
    TokenOp (name = "log"    , sympy_repr = "log"    , arity = 1 , complexity = 1 , function = protected_log    ),
    TokenOp (name = "sqrt"   , sympy_repr = "sqrt"   , arity = 1 , complexity = 1 , function = protected_sqrt   ),
    TokenOp (name = "cbrt"   , sympy_repr = "cbrt"   , arity = 1 , complexity = 1 , function = protected_cbrt   ),
    TokenOp (name = "n2"     , sympy_repr = "n2"     , arity = 1 , complexity = 1 , function = protected_n2     ),
    TokenOp (name = "inv"    , sympy_repr = "1/"     , arity = 1 , complexity = 1 , function = protected_inv    ),
    TokenOp (name = "arccos" , sympy_repr = "arccos" , arity = 1 , complexity = 1 , function = protected_arccos ),
    TokenOp (name = "arcsin" , sympy_repr = "arcsin" , arity = 1 , complexity = 1 , function = protected_arcsin ),

    # Custom unary operations
    TokenOp (name = "logabs" , sympy_repr = "logabs" , arity = 1 , complexity = 1 , function = protected_logabs ),
    TokenOp (name = "expneg" , sympy_repr = "expneg" , arity = 1 , complexity = 1 , function = protected_expneg ),
    TokenOp (name = "n3"     , sympy_repr = "n3"     , arity = 1 , complexity = 1 , function = protected_n3     ),
    TokenOp (name = "n4"     , sympy_repr = "n4"     , arity = 1 , complexity = 1 , function = protected_n4     ),

    # Custom binary operations
    TokenOp (name = "pow"     , sympy_repr = "**"    , arity = 2 , complexity = 1 , function = protected_torch_pow   ),

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
