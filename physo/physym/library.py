import warnings as warnings
import numpy as np

# Internal imports
from physo.physym import token as Tok
from physo.physym import tokenize as tokenize

# Defining these function at upper level to make library pickable
def SUPERPARENT_FUNC():
    raise ValueError("Superparent is a placeholder, it should never be called")
def INVALID_FUNC():
    raise ValueError("Invalid is a placeholder, it should never be called")
def DUMMY_FUNC():
    raise ValueError("Dummy is a placeholder, it should never be called")

class Library:
    """
        Object containing choosable tokens and their properties for a task of symbolic computation
        (properties: non_positional and semi_positional).
        See token.Token.__init__ for full description of token properties.

        Attributes
        ----------
        lib_tokens : numpy.array of token.Token
            List of tokens in the library.
        terminal_units_provided : bool
            Were all terminal tokens (arity = 0) units constraints (choosable and parents only) provided ?, ie is it
            possible to compute units constraints.
        n_library : int
            Number of tokens in the library (including superparent).
        n_choices  : int
            Number of choosable tokens (does not include superparent placeholder).
        superparent : token.Token
            Placeholder token representing the output symbolic function (eg. y in y = 2*x + 1).
        dummy : token.Token
            Placeholder token to complete program trees during program generation.
        invalid : token.Token
            Placeholder for tokens that are not yet generated.
        lib_name_to_idx : dict of {str : int}
            Dictionary containing names and corresponding token index in the library.

        lib_name : numpy.array of str
        lib_choosable_name : numpy.array of str
        lib_sympy_repr : numpy.array of str
        lib_function : numpy.array of objects (callable or None)
        properties : token.VectTokens

        arity                     :  int
        complexity                :  float
        var_type                  :  int
        var_id                    :  int
        behavior_id               :  int
        is_power                  :  bool
        power                     :  float

        is_constraining_phy_units :  bool
        phy_units                 :  numpy.array of shape (UNITS_VECTOR_SIZE,) of float

        Methods
        -------

        reset_library
            Resets token properties vectors based on current library of tokens list (lib_tokens).
        append_custom_tokens
            Adding list of custom Tokens to library.
        append_from_tokenize
            Creates tokens by passing arguments to tokenize.make_tokens and adding them to library.

        Examples
        --------
        add = Tok.TokenOp (name = "add", sympy_repr = "+", arity = 2, function = torch.add)

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
        my_lib = Lib.Library(custom_tokens = [add,], args_make_tokens = args_make_tokens,
                     superparent_units = [1,-1,0], superparent_name = "v")

    """

    def __init__(self, custom_tokens = None, args_make_tokens = None, superparent_units = None, superparent_name = "y"):
        """
        Defines choosable tokens in the library.
        Parameters
        ----------
        args_make_tokens : dict or None
            If not None, arguments are passed to tokenize.make_tokens and tokens are added to the library.
        custom_tokens : list of token.Token or None
            If not None, the tokens are added to the library.
        superparent_units : array_like of float
            Physical units vector of output symbolic function (eg. [1, 0, 0] for a distance assuming a convention such
            as [m, s, kg,...]). None if not using physical units. None by default.
        superparent_name : str
            Customizable name of output symbolic function for display purposes. "y" by default.
        """
        # ------------------------------ SUPERPARENT ------------------------------
        # Should happen 1st so that superparent.name is defined when calling reset_library
        # Units # using retrieve_units for its error raising + padding features
        if superparent_units is None:
            superparent_units_dict = None  # will result in y_units = None => token.units = vector of np.nan
        else:
            superparent_units_dict = {superparent_name: superparent_units}
        y_is_constraining_phy_units, y_units = tokenize.retrieve_units(superparent_units_dict, superparent_name)

        # ------------------------------ SUPERPARENT ------------------------------
        self.superparent = Tok.TokenSpecial(
            name                      = superparent_name,
            sympy_repr                = superparent_name,
            arity                     = 0,
            complexity                = 0.,
            function                  = SUPERPARENT_FUNC,
            is_constraining_phy_units = y_is_constraining_phy_units,
            phy_units                 = y_units,
        )

        # ------------------------------ DUMMY ------------------------------
        # Token
        self.dummy = Tok.TokenSpecial(
            name                      = Tok.DUMMY_TOKEN_NAME,
            sympy_repr                = Tok.DUMMY_TOKEN_NAME,
            arity                     = 0,
            complexity                = 0.,
            function                  = DUMMY_FUNC,
        )

        # ------------------------------ INVALID ------------------------------
        # Token
        self.invalid = Tok.TokenSpecial(
            name                      = Tok.INVALID_TOKEN_NAME,
            sympy_repr                = Tok.INVALID_TOKEN_NAME,
            arity                     = 0,
            complexity                = 0.,
            function                  = INVALID_FUNC,
        )

        # ------------------------------ PLACEHOLDERS ------------------------------
        self.placeholders = [self.superparent, self.dummy, self.invalid]

        # ------------------------------ LIB OF TOKENS ------------------------------
        self.choosable_tokens = []
        self.append_from_tokenize (args_make_tokens = args_make_tokens)
        self.append_custom_tokens (custom_tokens    = custom_tokens)

        # ------------------------------ INPUT VAR ------------------------------
        # Number of input variables
        self.n_input_var        = (self.var_type == Tok.VAR_TYPE_INPUT_VAR).sum()
        # Ids of input variables available
        self.input_var_ids      = self.var_id[self.var_type == Tok.VAR_TYPE_INPUT_VAR]   # (n_input_var,) of int

        # ------------------------------ CLASS FREE CONSTANTS ------------------------------

        # Number of free constants
        self.n_class_free_const = (self.var_type == Tok.VAR_TYPE_CLASS_FREE_CONST).sum()
        # Free constants tokens
        self.class_free_constants_tokens   = self.lib_tokens[self.var_type == Tok.VAR_TYPE_CLASS_FREE_CONST]                                             # (n_class_free_const,) of token.Token
        # Free constants names
        self.class_free_constants_names    = self.lib_name  [self.var_type == Tok.VAR_TYPE_CLASS_FREE_CONST]                                             # (n_class_free_const,) of str
        # Ids of free constants available
        self.class_free_constants_ids      = self.var_id    [self.var_type == Tok.VAR_TYPE_CLASS_FREE_CONST]                                             # (n_class_free_const,) of int
        # Initial values of free constants
        self.class_free_constants_init_val = np.array([token.init_val for token in self.lib_tokens if token.var_type == Tok.VAR_TYPE_CLASS_FREE_CONST])  # (n_class_free_const,) of float

        # ------------------------------ SPE FREE CONSTANTS ------------------------------

        # Number of free constants
        self.n_spe_free_const = (self.var_type == Tok.VAR_TYPE_SPE_FREE_CONST).sum()
        # Free constants tokens
        self.spe_free_constants_tokens   = self.lib_tokens[self.var_type == Tok.VAR_TYPE_SPE_FREE_CONST]                                                              # (n_spe_free_const,) of token.Token
        # Free constants names
        self.spe_free_constants_names    = self.lib_name  [self.var_type == Tok.VAR_TYPE_SPE_FREE_CONST]                                                              # (n_spe_free_const,) of str
        # Ids of free constants available
        self.spe_free_constants_ids      = self.var_id    [self.var_type == Tok.VAR_TYPE_SPE_FREE_CONST]                                                              # (n_spe_free_const,) of int
        # Initial values of free constants
        # May contain mixed shapes (as user can provide a single float or an array_like of floats depending on the
        # token), using object dtype
        self.spe_free_constants_init_val = np.array([token.init_val for token in self.lib_tokens if token.var_type == Tok.VAR_TYPE_SPE_FREE_CONST], dtype=object)  # (n_spe_free_const,) of array_like of floats
        # -> (n_spe_free_const, n_realizations,) of float after check_and_pad_spe_free_const_init_val

        # ------------------------------ FREE CONSTANTS ------------------------------

        self.n_free_const = self.n_class_free_const + self.n_spe_free_const
        self.free_constants_tokens = np.concatenate([self.class_free_constants_tokens, self.spe_free_constants_tokens])

        return None

    def check_and_pad_spe_free_const_init_val (self, n_realizations):
        """
        Asserts that the sizes of free constants init values for each realization are consistent with the number of
        realizations (n_realizations) and makes the necessary padding to ensure a shape of (n_realizations,) (for
        single float initial values).
        Parameters
        ----------
        n_realizations : int
            Number of realizations.
        """
        padded_init_val = []
        for i in range (self.n_spe_free_const):
            init_val = self.spe_free_constants_init_val[i]
            # Padding single float initial values to shape (n_realizations,)
            if init_val.shape == (1,):
                init_val = np.full(shape=(n_realizations,), fill_value=init_val[0])
            # Asserting that the size of the init values is consistent with the number of realizations
            assert init_val.shape == (n_realizations,),"Realization specific free const %s has inconsistent init values shape %s with n_realizations %s" %(self.spe_free_constants_names[i], init_val.shape, n_realizations)
            padded_init_val.append(init_val)

        # Making an empty array of shape (n_spe_free_const, n_realizations,) in case there are no spe free constants
        # to ensure consistency
        if self.n_spe_free_const == 0:
            padded_init_val = np.empty(shape=(self.n_spe_free_const, n_realizations,))

        self.spe_free_constants_init_val = np.array(padded_init_val) # (n_spe_free_const, n_realizations,) of float
        return None

    def reset_library(self):
        self.lib_tokens = np.array(self.choosable_tokens + self.placeholders)
        self.assert_units()
        # Number of tokens in the library
        self.n_library = len(self.lib_tokens)
        self.n_choices = len(self.choosable_tokens)
        # Idx of placeholders
        self.superparent_idx = self.n_choices + 0
        self.dummy_idx       = self.n_choices + 1
        self.invalid_idx     = self.n_choices + 2
        # Token representation
        self.lib_name           = np.array([token.name for token in self.lib_tokens       ]).astype(str)  # str (<MAX_NAME_SIZE) )
        self.lib_choosable_name = np.array([token.name for token in self.choosable_tokens ]).astype(str)  # str (<MAX_NAME_SIZE) )
        self.lib_sympy_repr     = np.array([token.sympy_repr for token in self.lib_tokens ]).astype(str)  # str (<MAX_NAME_SIZE) )
        # Object properties
        self.lib_function   = np.array([token.function   for token in self.lib_tokens])  # object (callable or None)
        # Vectorized properties
        self.properties = Tok.VectTokens(shape = (1, self.n_library,), invalid_token_idx = self.invalid_idx) # not using positional properties
        self.properties.arity                     [0, :] = np.array([token.arity                     for token in self.lib_tokens]).astype(int  )  # int
        self.properties.complexity                [0, :] = np.array([token.complexity                for token in self.lib_tokens]).astype(float)  # float
        self.properties.var_type                  [0, :] = np.array([token.var_type                  for token in self.lib_tokens]).astype(int  )  # int
        self.properties.var_id                    [0, :] = np.array([token.var_id                    for token in self.lib_tokens]).astype(int  )  # int
        self.properties.behavior_id               [0, :] = np.array([token.behavior_id               for token in self.lib_tokens]).astype(int  )  # int
        self.properties.is_power                  [0, :] = np.array([token.is_power                  for token in self.lib_tokens]).astype(bool )  # bool
        self.properties.power                     [0, :] = np.array([token.power                     for token in self.lib_tokens]).astype(float)  # float
        self.properties.is_constraining_phy_units [0, :] = np.array([token.is_constraining_phy_units for token in self.lib_tokens]).astype(bool )  # bool
        self.properties.phy_units                 [0, :] = np.array([token.phy_units                 for token in self.lib_tokens]).astype(float)  # float
        # Giving access to vectorized properties to user without having to use [0, :] at each property access
        self.arity                     = self.properties.arity                     [0, :]
        self.complexity                = self.properties.complexity                [0, :]
        self.var_type                  = self.properties.var_type                  [0, :]
        self.var_id                    = self.properties.var_id                    [0, :]
        self.behavior_id               = self.properties.behavior_id               [0, :]
        self.is_power                  = self.properties.is_power                  [0, :]
        self.power                     = self.properties.power                     [0, :]
        self.is_constraining_phy_units = self.properties.is_constraining_phy_units [0, :]
        self.phy_units                 = self.properties.phy_units                 [0, :]
        # Helper dict
        self.lib_name_to_idx             = {self.lib_name[i] : i                  for i in range (self.n_library)}
        self.lib_choosable_name_to_idx   = {self.lib_name[i] : i                  for i in range (self.n_choices)}
        self.lib_name_to_token           = {self.lib_name[i] : self.lib_tokens[i] for i in range (self.n_library)}

    def append_custom_tokens(self, custom_tokens = None):
        # ----- Handling custom tokens -----
        if custom_tokens is None:
            custom_tokens = []
        self.choosable_tokens += custom_tokens
        self.reset_library()

    def append_from_tokenize(self, args_make_tokens = None):
        # ----- Handling created tokens -----
        if args_make_tokens is None:
            created_tokens = []
        else:
            created_tokens = tokenize.make_tokens(**args_make_tokens).tolist()
        self.choosable_tokens += created_tokens
        self.reset_library()

    def assert_units(self):
        """
        Checks if all terminal tokens (arity = 0) have units constraints ie if units constraints can be computed.
        Tokens in library come from various units assignments processes (from make_tokens : operation definition in
        functions.py, input_var_units dict, constants_units dict ; from custom tokens ; superparent units in
        library.Library.__init__)
        """
        self.terminal_units_provided = True
        # Checking all tokens (except dummy and valid which have arity = 0 and free dim)
        for token in self.choosable_tokens + [self.superparent]:
            if token.arity == 0 and token.is_constraining_phy_units == False:
                # is_constraining_phy_units = False <=> phy_units is Free ie phy_units = vector of NAN
                # (this is ensured via exceptions in token.Token.__init__)
                warnings.warn("The units of token %s were not provided (is_constraining_phy_units=%s ; phy_units=%s), "
                              "unable to compute units constraints."
                              %(token.name, token.is_constraining_phy_units, token.phy_units))
                self.terminal_units_provided = False
        return None

    def get_choosable_prop(self, attr):
        """
        Returns vectorized property of choosable tokens of the library.
        Parameters
        ----------
        attr : str
            Vectorized token property.
        Returns
        -------
        property : numpy.array of shape (self.n_choices,) of type ?
            ? depends on the property
        """
        return getattr(self.properties, attr)[0][:self.n_choices]

    @property
    def free_const_names(self):
        return np.array([tok.__str__() for tok in self.free_constants_tokens])

    @property
    def spe_free_constants_init_val_sizes(self):
        return np.array([c.shape[0] for c in self.spe_free_constants_init_val])

    def __repr__(self):
        return str(self.lib_tokens)

    def __getitem__(self, item):
        return self.lib_tokens[item]