import numpy as np

# --------------------- TOKEN DEFAULT VALUES ---------------------
# Max size for token names
MAX_NAME_SIZE = 10
# Number of units in SI system
UNITS_VECTOR_SIZE = 7
# Default behavior ID in dimensional analysis
DEFAULT_BEHAVIOR_ID = 9999999
# Element used in place of a NAN (which is a float) as var id in int arrays
INVALID_VAR_ID = 9999999  # NAN only exists for floats
# Default complexity
DEFAULT_COMPLEXITY = 1.
# Default initial value for free const token
DEFAULT_FREE_CONST_INIT_VAL = 1.

# --------------------- POSITIONAL TOKENS DEFAULT VALUES IN PROGRAMS ---------------------
# VectPrograms.append, VectPrograms.update_relationships_pos only work with MAX_NB_CHILDREN = 2
MAX_NB_CHILDREN = 2
# VectPrograms.append, VectPrograms.update_relationships_pos, VectPrograms.get_sibling_idx,
# VectPrograms.get_sibling_idx_of_step prior.RelationshipConstraintPrior get_property_of_relative,
# only work with MAX_NB_SIBLINGS = 1
MAX_NB_SIBLINGS = MAX_NB_CHILDREN - 1
# Max arity value
MAX_ARITY = MAX_NB_CHILDREN
# Out of range tokens, pos >= (n_lengths + n_dangling)
INVALID_TOKEN_NAME = "-"
INVALID_POS   = 9999999
INVALID_DEPTH = 9999999
# Dummy tokens, n_lengths <= pos < (n_lengths + n_dangling)
DUMMY_TOKEN_NAME = "dummy"

# --------------------- TOKEN VAR TYPES IDs ---------------------
# Token representing an operation / function
VAR_TYPE_OP = 0
# Token representing an input variable
VAR_TYPE_INPUT_VAR = 1
# Token representing a class free constant
VAR_TYPE_CLASS_FREE_CONST = 20
# Token representing a spe free constant
VAR_TYPE_SPE_FREE_CONST = 21
# Token representing a fixed constant
VAR_TYPE_FIXED_CONST = 3
# Special token
VAR_TYPE_SPECIAL = -1

# List of all var types
VAR_TYPES = [VAR_TYPE_OP, VAR_TYPE_INPUT_VAR, VAR_TYPE_CLASS_FREE_CONST, VAR_TYPE_SPE_FREE_CONST, VAR_TYPE_FIXED_CONST, VAR_TYPE_SPECIAL]

# Old deprecated var types
# Token representing a free constant
VAR_TYPE_FREE_CONST_DEPRECATED = 2

# --------------------- Token class ---------------------
class Token:
    """
        An object representing a unique mathematical symbol (non_positional & semi_positional infos), except idx (which
        represents the token's idx in the library and is not encoded here).
        Using only one class for all token types so they can be vectorized regardless of their type in VectTokens.
        Attributes :
        ----------
        See token.Token.__init__ for full description of parameters.

        name                      :  str (<MAX_NAME_SIZE)
        sympy_repr                :  str (<MAX_NAME_SIZE)
        arity                     :  int
        complexity                :  float
        var_type                  :  int
        function                  :  callable or None
        init_val                  :  float or array_like of floats (for multiple realizations)
        var_id                    :  int
        fixed_const               :  float-like
        behavior_id               :  int
        is_power                  :  bool
        power                     :  float

        is_constraining_phy_units :  bool
        phy_units                 :  numpy.array of shape (UNITS_VECTOR_SIZE,) of float

        Methods
        -------
        __call__(args)
            Calls the token's function.
    """
    def __init__(self,
                 # ---- Token representation ----
                 name,
                 sympy_repr,
                 # ---- Token main properties ----
                 arity,
                 complexity  = DEFAULT_COMPLEXITY,
                 var_type    = 0,
                 # Function specific
                 function = None,
                 # Free constant specific
                 init_val = np.nan,
                 # Input variable / free constant specific
                 var_id   = None,
                 # Fixed constant specific
                 fixed_const = np.nan,
                 # ---- Physical units : behavior id ----
                 behavior_id               = None,
                 # ---- Physical units : power ----
                 is_power                  = False,
                 power                     = np.nan,
                 # ---- Physical units : units (semi_positional) ----
                 is_constraining_phy_units = False,
                 phy_units                 = None,
                 ):
        """
        Note: __init__ accepts None for some parameters for ease of use which are then converted to the right value and
        type as attributes.
        Parameters
        ----------
        name : str
            A short name for the token (eg. 'add' for addition).
        sympy_repr : str
            Sympy representation of mathematical operation.

        arity : int
            Number of argument of token (eg. 2 for addition, 1 for sinus, 0 for input variables or constants).
            - This token represents a function (ie. var_type = VAR_TYPE_OP)  <=> arity >= 0
            - This token represents input_var or class free const or spe free const or a fixed const
            (ie. var_type = VAR_TYPE_INPUT_VAR, VAR_TYPE_CLASS_FREE_CONST, VAR_TYPE_SPE_FREE_CONST, VAR_TYPE_FIXED_CONST) <=> arity = 0
        complexity : float
            Complexity of token.
        var_type : int
            - If this token represents a function          : var_type = VAR_TYPE_OP  (eg. add, mul, cos, exp).
            - If this token represents an input_var        : var_type = VAR_TYPE_INPUT_VAR  (input variable, eg. x0, x1).
            - If this token represents a class free const  : var_type = VAR_TYPE_CLASS_FREE_CONST (free constant,  eg. c0, c1).
            - If this token represents a spe free const    : var_type = VAR_TYPE_SPE_FREE_CONST (free constant,  eg. k0, k1).
            - If this token represents a fixed const       : var_type = VAR_TYPE_FIXED_CONST  (eg. pi, 1)
            - If this token represents a special token      : var_type = VAR_TYPE_SPECIAL  (eg. placeholder, superparent etc.). Same behavior as VAR_TYPE_OP.
            Distinction between class free const and spe free const is important in Class SR context only: class free const
            values are shared between all realizations of a single program whereas spe free const values are specific to each
            dataset.
        function : callable or None
            - This token represents a function (ie. var_type = VAR_TYPE_OP ) <=> this represents the function associated with the
            token. Function of arity = n must be callable using n arguments, each argument consisting in eg. an array
            of floats of shape (int,) or a single float number.
            - This token represents an input_var or a constant (ie. var_type = VAR_TYPE_INPUT_VAR, VAR_TYPE_CLASS_FREE_CONST,
            VAR_TYPE_SPE_FREE_CONST or VAR_TYPE_FIXED_CONST) <=> function = None
        init_val : float or array_like of floats or np.nan
            - This token represents a function, a fixed const or an input variable (ie. var_type = VAR_TYPE_OP, VAR_TYPE_FIXED_CONST or VAR_TYPE_INPUT_VAR)
            <=> init_val = np.nan
            - This token represents a class free const (ie. var_type = VAR_TYPE_CLASS_FREE_CONST) <=>  init_val = non NaN float
            - This token represents a spe free const (ie. var_type = VAR_TYPE_SPE_FREE_CONST) <=>  init_val = non NaN float or array_like of floats (for multiple realizations)
        var_id : int or None
            - This token represents an input_var or a free constant (ie. var_type = VAR_TYPE_INPUT_VAR, VAR_TYPE_CLASS_FREE_CONST or VAR_TYPE_SPE_FREE_CONST) <=> var_id is an
            integer representing the id of the input_var in the dataset or the id of the free const in the free const
            array.
            - This token represents a function or a fixed constant (ie. var_type = VAR_TYPE_OP or VAR_TYPE_FIXED_CONST) <=> var_id = None.
            (converted to INVALID_VAR_ID in __init__)
        fixed_const : float or np.nan
            - This token represents a fixed constant (ie. var_type = VAR_TYPE_FIXED_CONST) <=> fixed_const = non NaN float
            - This token represents a function, an input_var or a free const (ie. var_type = VAR_TYPE_OP, VAR_TYPE_INPUT_VAR, VAR_TYPE_CLASS_FREE_CONST or VAR_TYPE_SPE_FREE_CONST )
            <=>  fixed_const = NaN float

        behavior_id : int
            Id encoding behavior of token in the context of dimensional analysis (see functions for details).

        is_power : bool
            True if token is a power token (n2, sqrt, n3 etc.), False else.
        power : float or np.nan
            - is_power = True <=> power is a float representing the power of a token (0.5 for sqrt, 2 for n2 etc.)
            - is_power = False <=> power is np.nan

        is_constraining_phy_units : bool
            - True if there are hard constraints regarding with this token's physical units (eg. dimensionless op such
            as cos, sin exp, log etc. or input variable / constant representing physical quantity such as speed, etc.)
            - False if this token's units are free ie there are no constraints associated with this token's physical
            units (eg. add, mul tokens).
        phy_units : numpy.array of size UNITS_VECTOR_SIZE of float or None
            - is_constraining_phy_units = False <=> phy_units = None (converted to vector of np.nan in __init__)
            - is_constraining_phy_units = True  <=> phy_units = vector containing power of units.
            Ie. vector of zeros for dimensionless operations (eg. cos, sin, exp, log), vector containing power of units
            for constants or input variable (eg. [1, -1, 0, 0, 0, 0, 0] for a token representing a velocity with the
            convention [m, s, kg, ...]).
        """

        # ---------------------------- Token representation ----------------------------
        # ---- Assertions ----
        assert isinstance(name,       str), "name       must be a string, %s is not a string" % (str(name))
        assert isinstance(sympy_repr, str), "sympy_repr must be a string, %s is not a string" % (str(sympy_repr))
        assert len(name)       < MAX_NAME_SIZE, "Token name       must be < than %i, MAX_NAME_SIZE can be changed." % (MAX_NAME_SIZE)
        assert len(sympy_repr) < MAX_NAME_SIZE, "Token sympy_repr must be < than %i, MAX_NAME_SIZE can be changed." % (MAX_NAME_SIZE)
        # ---- Attribution ----
        self.name       = name                                     # str (<MAX_NAME_SIZE)
        self.sympy_repr = sympy_repr                               # str (<MAX_NAME_SIZE)

        # ---------------------------- Token main properties ----------------------------
        # ---- Assertions ----
        assert isinstance(arity,      int),   "arity must be an int, %s is not an int" % (str(arity))
        assert isinstance(float(complexity), float), "complexity must be castable to float"
        assert isinstance(int(var_type), int) and int(var_type) in VAR_TYPES, "var_type must be castable as an int in %s" % (str(VAR_TYPES))
        assert isinstance(float(fixed_const), float), "fixed_const must be castable to a float"

        # Token representing input_var (eg. x0, x1 etc.)
        if var_type == VAR_TYPE_INPUT_VAR:
            assert function is None,        'Token representing input_var (var_type = %i) must have function = None'   %(VAR_TYPE_INPUT_VAR)
            assert arity == 0,              'Token representing input_var (var_type = %i) must have arity = 0'         %(VAR_TYPE_INPUT_VAR)
            assert isinstance(var_id, int), 'Token representing input_var (var_type = %i) must have an int var_id'     %(VAR_TYPE_INPUT_VAR)
            assert np.isnan(init_val),      'Token representing input_var (var_type = %i) must have init_val = NaN'    %(VAR_TYPE_INPUT_VAR)
            assert np.isnan(float(fixed_const)), \
                                            'Token representing input_var (var_type = %i) must have a nan fixed_const' %(VAR_TYPE_INPUT_VAR)

        # Token representing function (eg. add, mul, exp, etc.)
        elif var_type == VAR_TYPE_OP:
            assert callable(function), 'Token representing function (var_type = %i) must have callable function'  %(VAR_TYPE_OP)
            assert arity >= 0,         'Token representing function (var_type = %i) must have arity >= 0'         %(VAR_TYPE_OP)
            assert var_id is None,     'Token representing function (var_type = %i) must have var_id = None'      %(VAR_TYPE_OP)
            assert np.isnan(init_val), 'Token representing function (var_type = %i) must have init_val = NaN'     %(VAR_TYPE_OP)
            assert np.isnan(float(fixed_const)), \
                                       'Token representing function (var_type = %i) must have a nan fixed_const'  %(VAR_TYPE_OP)

        # DEPRECATED: old way of representing free constant (eg. c0, c1 etc.)
        elif var_type == VAR_TYPE_FREE_CONST_DEPRECATED:
            raise NotImplementedError('Plain free const representation (var_type = %i) is deprecated. Use class free const (var_type = %i) or spe free const (var_type = %i) instead.'%(VAR_TYPE_FREE_CONST_DEPRECATED, VAR_TYPE_CLASS_FREE_CONST, VAR_TYPE_SPE_FREE_CONST))

        # Token representing class free constant (eg. c0, c1 etc.) (where each constant value is the same across all datasets/realizations in Class SR context)
        elif var_type == VAR_TYPE_CLASS_FREE_CONST:
            assert function is None,        'Token representing class free const (var_type = %i) must have function = None'          %(VAR_TYPE_CLASS_FREE_CONST)
            assert arity == 0,              'Token representing class free const (var_type = %i) must have arity == 0'               %(VAR_TYPE_CLASS_FREE_CONST)
            assert isinstance(var_id, int), 'Token representing class free const (var_type = %i) must have an int var_id'            %(VAR_TYPE_CLASS_FREE_CONST)
            assert isinstance(init_val, float) and not np.isnan(init_val), \
                                            'Token representing class free const (var_type = %i) must have a non-nan float init_val' %(VAR_TYPE_CLASS_FREE_CONST)
            assert np.isnan(float(fixed_const)), \
                                            'Token representing class free const (var_type = %i) must have a nan fixed_const'        %(VAR_TYPE_CLASS_FREE_CONST)

        # Token representing (realization specific) spe free constant (eg. k0, k1 etc.) (where each constant has a different value across each dataset/realization in Class SR context)
        elif var_type == VAR_TYPE_SPE_FREE_CONST:
            assert function is None,        'Token representing spe free const (var_type = %i) must have function = None'             %(VAR_TYPE_SPE_FREE_CONST)
            assert arity == 0,              'Token representing spe free const (var_type = %i) must have arity == 0'                  %(VAR_TYPE_SPE_FREE_CONST)
            assert isinstance(var_id, int), 'Token representing spe free const (var_type = %i) must have an int var_id'               %(VAR_TYPE_SPE_FREE_CONST)
            assert np.isnan(float(fixed_const)), \
                                            'Token representing spe free const (var_type = %i) must have a nan fixed_const'           %(VAR_TYPE_SPE_FREE_CONST)
            # Checking that init_val is a float or an array_like of floats
            init_val = np.array(init_val)
            assert init_val.dtype == float,           'Token representing spe free const (var_type = %i) must have init_val containing float(s)' %(VAR_TYPE_SPE_FREE_CONST)
            assert np.isnan(init_val).any() == False, 'Token representing spe free const (var_type = %i) must have init_val containing no NaNs'  %(VAR_TYPE_SPE_FREE_CONST)
            # Single float case
            if init_val.shape == ():
                init_val = np.array([init_val])
            # Multiple realizations case
            else:
                assert len(init_val.shape) == 1, 'Token representing spe free const (var_type = %i) must have a single float init_val or 1D array of floats' %(VAR_TYPE_SPE_FREE_CONST)

        # Token representing a fixed constant (eg. 1, pi etc.)
        elif var_type == VAR_TYPE_FIXED_CONST:
            assert function is None,   'Token representing fixed const (var_type = %i) must have function = None'        %(VAR_TYPE_FIXED_CONST)
            assert arity == 0,         'Token representing fixed const (var_type = %i) must have arity == 0'             %(VAR_TYPE_FIXED_CONST)
            assert var_id is None,     'Token representing fixed const (var_type = %i) must have var_id = None'          %(VAR_TYPE_FIXED_CONST)
            assert np.isnan(init_val), 'Token representing fixed const (var_type = %i) must have init_val = NaN'         %(VAR_TYPE_FIXED_CONST)
            assert not np.isnan(float(fixed_const)), \
                                       'Token representing fixed const (var_type = %i) must have a non-nan fixed_const'  %(VAR_TYPE_FIXED_CONST)
            # not checking isinstance(fixed_const, float) as fixed_const can be a torch.tensor(float) or a float
            # ie. "float-like"

        # Token representing a special token (eg. placeholder, superparent etc.)
        elif var_type == VAR_TYPE_SPECIAL:
            assert callable(function), 'Token representing special token (var_type = %i) must have callable function'  %(VAR_TYPE_SPECIAL)
            assert arity >= 0,         'Token representing special token (var_type = %i) must have arity >= 0'         %(VAR_TYPE_SPECIAL)
            assert var_id is None,     'Token representing special token (var_type = %i) must have var_id = None'      %(VAR_TYPE_SPECIAL)
            assert np.isnan(init_val), 'Token representing special token (var_type = %i) must have init_val = NaN'     %(VAR_TYPE_SPECIAL)
            assert np.isnan(float(fixed_const)), \
                                       'Token representing special token (var_type = %i) must have a nan fixed_const'  %(VAR_TYPE_SPECIAL)

        else:
            raise ValueError("Unknown var_type %s" % (str(var_type)))

        # ---- Attribution ----
        self.arity       = arity                                   # int
        self.complexity  = float(complexity)                       # float
        self.var_type    = int(var_type)                           # int
        # Function specific
        self.function    = function                                # object (callable or None)
        # Free const specific
        self.init_val = init_val                                   # float or array_like of floats
        # Input variable / free const specific
        if self.var_type == VAR_TYPE_INPUT_VAR or self.var_type == VAR_TYPE_CLASS_FREE_CONST or self.var_type == VAR_TYPE_SPE_FREE_CONST:
            self.var_id = var_id                                   # int
        else:
            self.var_id = INVALID_VAR_ID                           # int
        # Fixed const spevific
        self.fixed_const = fixed_const                             # float-like

        # ---------------------------- Physical units : behavior id ----------------------------
        # ---- Assertions ----
        if behavior_id is not None:
            assert isinstance(behavior_id, int), 'Token behavior_id must be an int (%s is not an int)' % (str(behavior_id))
        # ---- Attribution ----
        if behavior_id is None:
            self.behavior_id = DEFAULT_BEHAVIOR_ID                 # int
        else:
            self.behavior_id = behavior_id                         # int

        # ---------------------------- Physical units : power ----------------------------
        assert isinstance(bool(is_power), bool), "is_power must be castable to bool"
        # ---- Assertions ----
        if is_power:
            assert isinstance(power, float) and not np.isnan(power), \
                        "Token with is_power=True must have a non nan float power (%s is not a float)" % (str(power))
        else:
            assert np.isnan(power), "Token with is_power=False must have a np.nan power"
        # ---- Attribution ----
        self.is_power = is_power                               # bool
        self.power    = power                                  # float

        # ---------------------------- Physical units : phy_units (semi_positional) ----------------------------
        assert isinstance(bool(is_constraining_phy_units), bool), "is_constraining_phy_units must be castable to bool"
        # ---- Assertions ----
        if is_constraining_phy_units:
            assert phy_units is not None, 'Token having physical units constraint (is_constraining_phy_units = True) must contain physical units.'
            assert np.array(phy_units).shape == (UNITS_VECTOR_SIZE,), 'Physical units vectors must be of shape (%s,) not %s, pad with zeros you are not using all elements.' % (UNITS_VECTOR_SIZE, np.array(phy_units).shape)
            assert np.array(phy_units).dtype == float, 'Physical units vectors must contain float.'
            assert not np.isnan(np.array(phy_units)).any(), 'No NaN allowed in phy_units, to create a free constraint token, use is_constraining_phy_units = False and phy_units = None (will result in phy_units = vect of np.nan)'
        else:
            assert phy_units is None, 'Token not having physical units constraint (is_constraining_phy_units = False) can not contain physical units.'
        # ---- Attribution ----
        self.is_constraining_phy_units = bool(is_constraining_phy_units)  # bool
        if phy_units is None:
            # no list definition in default arg
            self.phy_units = np.full((UNITS_VECTOR_SIZE), np.nan)  # (UNITS_VECTOR_SIZE,) of float
        else:
            # must be a numpy.array to support operations
            self.phy_units = np.array(phy_units)                   # (UNITS_VECTOR_SIZE,) of float
    @property
    def is_function(self):
        return self.var_type == VAR_TYPE_OP
    @property
    def is_input_var(self):
        return self.var_type == VAR_TYPE_INPUT_VAR
    @property
    def is_class_free_const(self):
        return self.var_type == VAR_TYPE_CLASS_FREE_CONST
    @property
    def is_spe_free_const(self):
        return self.var_type == VAR_TYPE_SPE_FREE_CONST
    @property
    def is_fixed_const(self):
        return self.var_type == VAR_TYPE_FIXED_CONST

    @property
    def is_special(self):
        return self.var_type == VAR_TYPE_SPECIAL


    def __call__(self, *args):
        # Assert number of args vs arity
        assert len(args) == self.arity, '%i arguments were passed to token %s during call but token has arity = %i' \
            % (len(args), self.name, self.arity,)

        if self.var_type == VAR_TYPE_OP or self.var_type == VAR_TYPE_SPECIAL:
            return self.function(*args)

        elif self.var_type == VAR_TYPE_FIXED_CONST:
            return self.fixed_const

        # Raise error for input_var and free const tokens
        # x0(data_x0, data_x1) would trigger both errors -> use AssertionError for both for simplicity
        else:
            raise AssertionError("Token %s does not represent a function or a fixed constant (var_type=%s), it can not "
                                 "be called."% (self.name, str(self.var_type)))

    def __repr__(self):
        return self.name

# --------------------- Type specific Token classes ---------------------

class TokenOp (Token):
    """
    An object inheriting from Token representing an operation / function (see Token documentation for details).
    """
    def __init__(self,
                 # ---- Token representation ----
                 name,
                 sympy_repr,
                 # ---- Token main properties ----
                 arity,
                 complexity  = DEFAULT_COMPLEXITY,
                 # Function specific
                 function = None,
                 # ---- Physical units : behavior id ----
                 behavior_id               = None,
                 # ---- Physical units : power ----
                 is_power                  = False,
                 power                     = np.nan,
                 # ---- Physical units : units (semi_positional) ----
                 is_constraining_phy_units = False,
                 phy_units                 = None,
                 ):
        """
        See Token.__init__ documentation for details.
        """

        # These properties should always have these values for this type of token (operation / function).
        var_type    = VAR_TYPE_OP
        init_val    = np.nan
        var_id      = None
        fixed_const = np.nan

        # Passing arguments parametrizing this type of token to parent object and filling in the rest.
        super().__init__(
                 # ---- Token representation ----
                 name       = name,
                 sympy_repr = sympy_repr,
                 # ---- Token main properties ----
                 arity       = arity,
                 complexity  = complexity,
                 var_type    = var_type,
                 # Function specific
                 function = function,
                 # Free constant specific
                 init_val = init_val,
                 # Input variable / free constant specific
                 var_id   = var_id,
                 # Fixed constant specific
                 fixed_const = fixed_const,
                 # ---- Physical units : behavior id ----
                 behavior_id               = behavior_id,
                 # ---- Physical units : power ----
                 is_power                  = is_power,
                 power                     = power,
                 # ---- Physical units : units (semi_positional) ----
                 is_constraining_phy_units = is_constraining_phy_units,
                 phy_units                 = phy_units,
        )

class TokenInputVar (Token):
    """
    An object inheriting from Token representing an input variable (see Token documentation for details).
    """
    def __init__(self,
                 # ---- Token representation ----
                 name,
                 sympy_repr,
                 # ---- Token main properties ----
                 complexity  = DEFAULT_COMPLEXITY,
                 # Input variable / free constant specific
                 var_id   = None,
                 # ---- Physical units : behavior id ----
                 behavior_id               = None,
                 # ---- Physical units : power ----
                 is_power                  = False,
                 power                     = np.nan,
                 # ---- Physical units : units (semi_positional) ----
                 is_constraining_phy_units = False,
                 phy_units                 = None,
                 ):
        """
        See Token.__init__ documentation for details.
        """

        # These properties should always have these values for this type of token (input_var).
        arity       = 0
        var_type    = VAR_TYPE_INPUT_VAR
        function    = None
        init_val    = np.nan
        fixed_const = np.nan

        # Passing arguments parametrizing this type of token to parent object and filling in the rest.
        super().__init__(
                 # ---- Token representation ----
                 name       = name,
                 sympy_repr = sympy_repr,
                 # ---- Token main properties ----
                 arity       = arity,
                 complexity  = complexity,
                 var_type    = var_type,
                 # Function specific
                 function = function,
                 # Free constant specific
                 init_val = init_val,
                 # Input variable / free constant specific
                 var_id   = var_id,
                 # Fixed constant specific
                 fixed_const = fixed_const,
                 # ---- Physical units : behavior id ----
                 behavior_id               = behavior_id,
                 # ---- Physical units : power ----
                 is_power                  = is_power,
                 power                     = power,
                 # ---- Physical units : units (semi_positional) ----
                 is_constraining_phy_units = is_constraining_phy_units,
                 phy_units                 = phy_units,
        )

class TokenClassFreeConst (Token):
    """
    An object inheriting from Token representing a class free constant (see Token documentation for details).
    Distinction between class free const and spe free const is important in Class SR context only: class free const
    values are shared between all realizations of a single program whereas spe free const values are specific to each
    dataset.
    """
    def __init__(self,
                 # ---- Token representation ----
                 name,
                 sympy_repr,
                 # ---- Token main properties ----
                 complexity  = DEFAULT_COMPLEXITY,
                 # Free constant specific
                 init_val = np.nan,
                 # Input variable / free constant specific
                 var_id   = None,
                 # ---- Physical units : behavior id ----
                 behavior_id               = None,
                 # ---- Physical units : power ----
                 is_power                  = False,
                 power                     = np.nan,
                 # ---- Physical units : units (semi_positional) ----
                 is_constraining_phy_units = False,
                 phy_units                 = None,
                 ):
        """
        See Token.__init__ documentation for details.
        """

        # These properties should always have these values for this type of token (class_free_const).
        arity       = 0
        var_type    = VAR_TYPE_CLASS_FREE_CONST
        function    = None
        fixed_const = np.nan

        # Passing arguments parametrizing this type of token to parent object and filling in the rest.
        super().__init__(
                 # ---- Token representation ----
                 name       = name,
                 sympy_repr = sympy_repr,
                 # ---- Token main properties ----
                 arity       = arity,
                 complexity  = complexity,
                 var_type    = var_type,
                 # Function specific
                 function = function,
                 # Free constant specific
                 init_val = init_val,
                 # Input variable / free constant specific
                 var_id   = var_id,
                 # Fixed constant specific
                 fixed_const = fixed_const,
                 # ---- Physical units : behavior id ----
                 behavior_id               = behavior_id,
                 # ---- Physical units : power ----
                 is_power                  = is_power,
                 power                     = power,
                 # ---- Physical units : units (semi_positional) ----
                 is_constraining_phy_units = is_constraining_phy_units,
                 phy_units                 = phy_units,
        )

class TokenSpeFreeConst (Token):
    """
    An object inheriting from Token representing a spe free constant (see Token documentation for details).
    Distinction between class free const and spe free const is important in Class SR context only: class free
    const values are shared between all realizations of a single program whereas spe free const values are specific to each
    dataset.
    """
    def __init__(self,
                 # ---- Token representation ----
                 name,
                 sympy_repr,
                 # ---- Token main properties ----
                 complexity  = DEFAULT_COMPLEXITY,
                 # Free constant specific
                 init_val = np.nan,
                 # Input variable / free constant specific
                 var_id   = None,
                 # ---- Physical units : behavior id ----
                 behavior_id               = None,
                 # ---- Physical units : power ----
                 is_power                  = False,
                 power                     = np.nan,
                 # ---- Physical units : units (semi_positional) ----
                 is_constraining_phy_units = False,
                 phy_units                 = None,
                 ):
        """
        See Token.__init__ documentation for details.
        """

        # These properties should always have these values for this type of token (spe_free_const).
        arity       = 0
        var_type    = VAR_TYPE_SPE_FREE_CONST
        function    = None
        fixed_const = np.nan

        # Passing arguments parametrizing this type of token to parent object and filling in the rest.
        super().__init__(
                 # ---- Token representation ----
                 name       = name,
                 sympy_repr = sympy_repr,
                 # ---- Token main properties ----
                 arity       = arity,
                 complexity  = complexity,
                 var_type    = var_type,
                 # Function specific
                 function = function,
                 # Free constant specific
                 init_val = init_val,
                 # Input variable / free constant specific
                 var_id   = var_id,
                 # Fixed constant specific
                 fixed_const = fixed_const,
                 # ---- Physical units : behavior id ----
                 behavior_id               = behavior_id,
                 # ---- Physical units : power ----
                 is_power                  = is_power,
                 power                     = power,
                 # ---- Physical units : units (semi_positional) ----
                 is_constraining_phy_units = is_constraining_phy_units,
                 phy_units                 = phy_units,
        )

class TokenFixedConst (Token):
    """
    An object inheriting from Token representing a fixed constant (see Token documentation for details).
    """
    def __init__(self,
                 # ---- Token representation ----
                 name,
                 sympy_repr,
                 # ---- Token main properties ----
                 complexity  = DEFAULT_COMPLEXITY,
                 # Fixed constant specific
                 fixed_const = np.nan,
                 # ---- Physical units : behavior id ----
                 behavior_id               = None,
                 # ---- Physical units : power ----
                 is_power                  = False,
                 power                     = np.nan,
                 # ---- Physical units : units (semi_positional) ----
                 is_constraining_phy_units = False,
                 phy_units                 = None,
                 ):
        """
        See Token.__init__ documentation for details.
        """

        # These properties should always have these values for this type of token (fixed_const).
        arity       = 0
        var_type    = VAR_TYPE_FIXED_CONST
        function    = None
        init_val    = np.nan
        var_id      = None

        # Passing arguments parametrizing this type of token to parent object and filling in the rest.
        super().__init__(
                 # ---- Token representation ----
                 name       = name,
                 sympy_repr = sympy_repr,
                 # ---- Token main properties ----
                 arity       = arity,
                 complexity  = complexity,
                 var_type    = var_type,
                 # Function specific
                 function = function,
                 # Free constant specific
                 init_val = init_val,
                 # Input variable / free constant specific
                 var_id   = var_id,
                 # Fixed constant specific
                 fixed_const = fixed_const,
                 # ---- Physical units : behavior id ----
                 behavior_id               = behavior_id,
                 # ---- Physical units : power ----
                 is_power                  = is_power,
                 power                     = power,
                 # ---- Physical units : units (semi_positional) ----
                 is_constraining_phy_units = is_constraining_phy_units,
                 phy_units                 = phy_units,
        )

class TokenSpecial (Token):
    """
    An object inheriting from Token representing a special token (eg. placeholders, superparents etc.).
    Same behavior as TokenOp.
    (see Token documentation for details).
    """
    def __init__(self,
                 # ---- Token representation ----
                 name,
                 sympy_repr,
                 # ---- Token main properties ----
                 arity,
                 complexity  = DEFAULT_COMPLEXITY,
                 # Function specific
                 function = None,
                 # ---- Physical units : behavior id ----
                 behavior_id               = None,
                 # ---- Physical units : power ----
                 is_power                  = False,
                 power                     = np.nan,
                 # ---- Physical units : units (semi_positional) ----
                 is_constraining_phy_units = False,
                 phy_units                 = None,
                 ):
        """
        See Token.__init__ documentation for details.
        """

        # These properties should always have these values for this type of token (special).
        var_type    = VAR_TYPE_SPECIAL
        init_val    = np.nan
        var_id      = None
        fixed_const = np.nan

        # Passing arguments parametrizing this type of token to parent object and filling in the rest.
        super().__init__(
                 # ---- Token representation ----
                 name       = name,
                 sympy_repr = sympy_repr,
                 # ---- Token main properties ----
                 arity       = arity,
                 complexity  = complexity,
                 var_type    = var_type,
                 # Function specific
                 function = function,
                 # Free constant specific
                 init_val = init_val,
                 # Input variable / free constant specific
                 var_id   = var_id,
                 # Fixed constant specific
                 fixed_const = fixed_const,
                 # ---- Physical units : behavior id ----
                 behavior_id               = behavior_id,
                 # ---- Physical units : power ----
                 is_power                  = is_power,
                 power                     = power,
                 # ---- Physical units : units (semi_positional) ----
                 is_constraining_phy_units = is_constraining_phy_units,
                 phy_units                 = phy_units,
        )

# --------------------- Vectorized tokens ---------------------

class VectTokens:
    """
    Object representing a matrix of positional tokens (positional) ie:
     - non_positional properties: idx + token properties attributes, see token.Token.__init__ for full description.
     - semi_positional properties: See token.Token.__init__ for full description of token properties attributes.
     - positional properties which are contextual (family relationships, depth etc.).
    This only contains properties expressed as float, int, bool to be jit-able.

    Attributes : In their non-vectorized shapes (types are vectorized)
    ----------
    idx                       : int
        Encodes token's nature, token index in the library.

    ( name                    :  str (<MAX_NAME_SIZE) )
    ( sympy_repr              :  str (<MAX_NAME_SIZE) )
    arity                     :  int
    complexity                :  float
    var_type                  :  int
    ( function                :  callable or None               )
    ( init_val                :  float or array_like of floats  )
    var_id                    :  int
    ( fixed_const             : float                           )
    behavior_id               :  int
    is_power                  :  bool
    power                     :  float

    is_constraining_phy_units :  bool
    phy_units                 :  numpy.array of shape (UNITS_VECTOR_SIZE,) of float

    pos                      : int
        Position in the program ie in time dim (eg. 0 for mul in program = [mul, x0, x1] )
    pos_batch                : int
        Position in the batch ie in batch dim.
    depth                    : int
        Depth in tree representation of program.
    has_parent_mask          : bool
        True if token has parent, False else.
    has_siblings_mask         : bool
        True if token has at least one sibling, False else.
    has_children_mask         : bool
        True if token has at least one child, False else.
    has_ancestors_mask        : bool
        True if token has at least one ancestor, False else. This is always true for valid tokens as the token itself
        counts as its own ancestor.
    parent_pos               : int
        Parent position in the program ie in time dim (eg. 0 for mul in program = [mul, x0, x1] )
    siblings_pos              : numpy.array of shape (MAX_NB_SIBLINGS,) of int
        Siblings position in the program ie in time dim (eg. 1 for x0 in program = [mul, x0, x1] )
    children_pos              : numpy.array of shape (MAX_NB_CHILDREN,) of int
        Children position in the program ie in time dim (eg. 2 for x1 in program = [mul, x0, x1] )
    ancestors_pos              : numpy.array of shape (shape[1],) of int`
        Ancestors positions in the program ie in time dim counting the token itself as itw own ancestor.
        (eg. [0, 1, 4, 5, INVALID_POS, INVALID_POS] for x1 in program = [mul, add, sin, x0, log, x1]).
    n_siblings                : int
        Number of siblings.
    n_children                : int
        Number of children.
    n_ancestors               : int
        Number of ancestors. This is equal to depth+1 as the token itself counts as its own ancestor.
    """

    def __init__(self, shape, invalid_token_idx):
        """
        Parameters
        ----------
        shape : (int, int)
            Shape of the matrix.
        invalid_token_idx : int
            Index of the invalid token in the library of tokens.

        """

        # -------------------------------------------------------------------------------------------------------
        # -------------------------------------- non_positional properties --------------------------------------
        # -------------------------------------------------------------------------------------------------------

        # ---- Shape ----
        assert len(shape)==2, "Shape of VectTokens object must be 2D." # remove line when jit-ing class ?
        self.shape = shape                          # (int, int)
        self.invalid_token_idx = invalid_token_idx  # int

        # ---- Index in library ----
        # Default value
        self.default_idx = self.invalid_token_idx
        # Property
        self.idx = np.full(shape=self.shape, fill_value=self.default_idx, dtype=int )

        # -------------------------------------------------------------------------------------------------------
        # -------------------------------------- non_positional properties --------------------------------------
        # -------------------------------------------------------------------------------------------------------
        # Same as ones in Token attributes

        # ---- Token representation ----
        # ( name                    :  str (<MAX_NAME_SIZE) )
        # self.tokens_names    = np.full((self.batch_size, self.max_time_step), INVALID_TOKEN_NAME, dtype="S%i"%(Tok.MAX_NAME_SIZE))
        # ( sympy_repr              :  str (<MAX_NAME_SIZE) )

        # ---- Token main properties ----
        # Default values
        self.default_arity        = 0
        self.default_complexity   = DEFAULT_COMPLEXITY
        self.default_var_type     = 0
        self.default_var_id       = INVALID_VAR_ID
        # Properties
        self.arity        = np.full(shape=self.shape, fill_value=self.default_arity        , dtype=int)
        self.complexity   = np.full(shape=self.shape, fill_value=self.default_complexity   , dtype=float)
        self.var_type     = np.full(shape=self.shape, fill_value=self.default_var_type     , dtype=int)
        # ( function                :  callable or None              )
        # ( init_val                :  float or array_like of floats )
        self.var_id       = np.full(shape=self.shape, fill_value=self.default_var_id       , dtype=int)
        # ( fixed_const                :  float         )

        # ---- Physical units : behavior id ----
        # Default value
        self.default_behavior_id = DEFAULT_BEHAVIOR_ID
        # Property
        self.behavior_id = np.full(shape=self.shape, fill_value=self.default_behavior_id, dtype=int)

        # ---- Physical units : power ----
        # Default values
        self.default_is_power = False
        self.default_power    = np.nan
        # Properties
        self.is_power = np.full(shape=self.shape, fill_value=self.default_is_power ,  dtype=bool)
        self.power    = np.full(shape=self.shape, fill_value=self.default_power    ,  dtype=float)

        # -------------------------------------------------------------------------------------------------------
        # -------------------------------------- semi_positional properties --------------------------------------
        # -------------------------------------------------------------------------------------------------------

        # ---- Physical units : units ----
        # Default values
        self.default_is_constraining_phy_units = False
        self.default_phy_units                 = np.nan
        # Properties
        self.is_constraining_phy_units = np.full(shape=self.shape,                        fill_value=self.default_is_constraining_phy_units  ,  dtype=bool)
        self.phy_units                 = np.full(shape=self.shape + (UNITS_VECTOR_SIZE,), fill_value=self.default_phy_units                  ,  dtype=float)

        # -------------------------------------------------------------------------------------------------------
        # ---------------------------------------- Positional properties ----------------------------------------
        # -------------------------------------------------------------------------------------------------------

        # ---- Position ----
        # Default values
        self.default_pos       = INVALID_POS
        self.default_pos_batch = INVALID_POS
        # Properties : position is the same in all elements of batch
        self.pos               = np.tile(np.arange(0, self.shape[1]), (self.shape[0], 1)).astype(int)
        self.pos_batch         = np.tile(np.arange(0, self.shape[0]), (self.shape[1], 1)).transpose().astype(int)

        # ---- Depth ----
        # Default value
        self.default_depth = INVALID_DEPTH
        # Property
        self.depth = np.full(shape=self.shape, fill_value=self.default_depth, dtype=int )

        # ---- Family relationships ----

        # Token family relationships: family mask
        # Default values
        self.default_has_parent_mask    = False
        self.default_has_siblings_mask  = False
        self.default_has_children_mask  = False
        self.default_has_ancestors_mask = False
        # Properties
        self.has_parent_mask    = np.full(shape=self.shape, fill_value=self.default_has_parent_mask    ,           dtype=bool)
        self.has_siblings_mask  = np.full(shape=self.shape, fill_value=self.default_has_siblings_mask  ,           dtype=bool)
        self.has_children_mask  = np.full(shape=self.shape, fill_value=self.default_has_children_mask  ,           dtype=bool)
        self.has_ancestors_mask = np.full(shape=self.shape, fill_value=self.default_has_ancestors_mask ,           dtype=bool)

        # Token family relationships: pos
        # Default values
        self.default_parent_pos    = INVALID_POS
        self.default_siblings_pos  = INVALID_POS
        self.default_children_pos  = INVALID_POS
        self.default_ancestors_pos = INVALID_POS
        # Properties
        self.parent_pos         = np.full(shape=self.shape,                      fill_value=self.default_parent_pos   , dtype=int)
        self.siblings_pos       = np.full(shape=self.shape + (MAX_NB_SIBLINGS,), fill_value=self.default_siblings_pos , dtype=int)
        self.children_pos       = np.full(shape=self.shape + (MAX_NB_CHILDREN,), fill_value=self.default_children_pos , dtype=int)
        self.ancestors_pos      = np.full(shape=self.shape + (self.shape[1], ),  fill_value=self.default_ancestors_pos, dtype=int)

        # Token family relationships: numbers
        # Default values
        self.default_n_siblings  = 0
        self.default_n_children  = 0
        self.default_n_ancestors = 0
        # Properties
        self.n_siblings         = np.full(shape=self.shape,  fill_value=self.default_n_siblings , dtype=int)
        self.n_children         = np.full(shape=self.shape,  fill_value=self.default_n_children , dtype=int)
        self.n_ancestors        = np.full(shape=self.shape,  fill_value=self.default_n_ancestors, dtype=int)

