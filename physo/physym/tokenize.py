import warnings as warnings
import numpy as np
import torch as torch

# Internal imports
from physo.physym import token as Tok
from physo.physym import functions as Func
from physo.physym.token import TokenInputVar, TokenFixedConst, TokenClassFreeConst, TokenSpeFreeConst


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
        If dictionary is None or empty, returns token.DEFAULT_COMPLEXITY.
    curr_name : str
        If curr_name is not in units_dict keys, returns token.DEFAULT_COMPLEXITY.
    Returns
    -------
    curr_complexity : float
        Complexity of token.
    """
    curr_complexity = Tok.DEFAULT_COMPLEXITY
    # If complexity dictionary is not None or empty dict
    if (complexity_dict is not None) and (complexity_dict != {}):
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
    init_val_dict : dict of {str : float or array_like of floats} or None
        If dictionary is None or empty, returns token.DEFAULT_FREE_CONST_INIT_VAL.
    curr_name : str
        If curr_name is not in units_dict keys, returns token.DEFAULT_FREE_CONST_INIT_VAL.
    Returns
    -------
    curr_init_val : float
        Initial value of token.
    """
    curr_init_val = Tok.DEFAULT_FREE_CONST_INIT_VAL
    # If init_val dictionary is not None or empty dict
    if (init_val_dict is not None) and (init_val_dict != {}):
        try:
            curr_init_val = init_val_dict[curr_name]
        except KeyError:
            warnings.warn(
                "Initial value of token %s not found in initial value dictionary %s, using initial value = %f" %
                (curr_name, init_val_dict, curr_init_val))
    # Conversion to float(s) if necessary (more flexible in case user passes eg. int, as Token class will need float(s))
    # Single float case
    if np.array(curr_init_val).shape == ():
        curr_init_val = float(curr_init_val)
    # Multiple floats case
    else:
        curr_init_val = np.array(curr_init_val).astype(float)
    return curr_init_val


def retrieve_units(units_dict, curr_name):
    """
    Helper function to safely retrieve units of token named curr_name from a dictionary of units (units_dict).
    Parameters
    ----------
    units_dict : dict of {str : array_like} or None
        If dictionary is None or empty, returned curr_is_constraining_phy_units is False and curr_phy_units is None.
        (Note: creating a token.Token using None in place of units will result in a Token with units = vector of np.nan)
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
    # If units dictionary is not None or empty dict
    if (units_dict is not None) and (units_dict != {}):
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
                # free constants / class free constants (can be used interchangeably)
                free_constants            = None,
                free_constants_init_val   = None,
                free_constants_units      = None,
                free_constants_complexity = None,
                class_free_constants            = None,
                class_free_constants_init_val   = None,
                class_free_constants_units      = None,
                class_free_constants_complexity = None,
                # spe free constants
                spe_free_constants            = None,
                spe_free_constants_init_val   = None,
                spe_free_constants_units      = None,
                spe_free_constants_complexity = None,

                ):
    """
        Makes a list of tokens for a run based on a list of operation names, input variables ids and constants values.
        Parameters
        ----------

        -------- Operations (eg. add, mul, cos, exp) --------
        op_names : list of str or str, optional
            List of names of operations that will be used for a run (eg. ["mul", "add", "neg", "inv", "sin"]), or "all"
            to use all available tokens. By default, op_names = "all".
        use_protected_ops : bool, optional
            If True safe functions defined in functions.OPS_PROTECTED_DICT in place when available (eg. sqrt(abs(x))
            instead of sqrt(x)). False by default.

        -------- Input variables (eg. x0, x1) --------
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

        -------- Fixed constants (eg. pi, 1) --------
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

        -------- Free constants / Class free constants (eg. c0, c1, c2) --------
        free_constants or class_free_constants : set of { str } or None, optional
            Set containing free constant names (eg. 'c0', 'c1', 'c2'). None if no free constants to create.
            None by default.
        free_constants_init_val or class_free_constants_init_val : dict of { str : float } or None, optional
            Dictionary containing free constants names as keys (eg. 'c0', 'c1', 'c2') and corresponding float initial
            values to use during optimization process (eg. 1., 1., 1.). None will result in the usage of
            token.DEFAULT_FREE_CONST_INIT_VAL as initial values. None by default.
        free_constants_units or class_free_constants_units : dict of { str : array_like of float } or None, optional
            Dictionary containing free constants names as keys (eg. 'c0', 'c1', 'c2') and corresponding physical units
            (eg. [0, 0, 0], [1, -1, 0], [0, 0, 1]). With c0 representing a dimensionless number, c1 a velocity and c2 a
            mass assuming a convention such as [m, s, kg,...]). None if not using physical units. None by default.
        free_constants_complexity or class_free_constants_complexity : dict of { str : float } or None, optional
            Dictionary containing free constants names as keys (eg. 'c0', 'c1', 'c2') and corresponding complexities
            (eg. 1., 1., 1.). If None, complexity = token.DEFAULT_COMPLEXITY will be encoded to tokens. None by default.

        -------- Spe free constants (eg. k0, k1, k2) --------
        spe_free_constants : set of { str } or None, optional
            Set containing realization specific free constant names (eg. 'k0', 'k1', 'k2'). None if no free constants to
            create. None by default.
        spe_free_constants_init_val : dict of { str : float } or dict of { str : array_like of shape (n_realizations,) of floats } or None, optional
            Dictionary containing realization specific free constants names as keys (eg. 'k0', 'k1', 'k2') and
            corresponding float initial values to use during optimization process (eg. 1., 1., 1.). Realization
            specific initial values can be used by providing a vector of shape (n_realizations,) for each constant
            in lieu of a single float per constant. None will result in the usage of token.DEFAULT_FREE_CONST_INIT_VAL
            as initial values. None by default.
        spe_free_constants_units : dict of { str : array_like of float } or None, optional
            Dictionary containing realization specific free constants names as keys (eg. 'k0', 'k1', 'k2') and
            corresponding physical units (eg. [0, 0, 0], [1, -1, 0], [0, 0, 1]). With k0 representing a dimensionless
            number, k1 a velocity and k2 a mass assuming a convention such as [m, s, kg,...]). None if not using
            physical units. None by default.
        spe_free_constants_complexity : dict of { str : float } or None, optional
            Dictionary containing realization specific free constants names as keys (eg. 'k0', 'k1', 'k2') and
            corresponding complexities (eg. 1., 1., 1.). If None, complexity = token.DEFAULT_COMPLEXITY will be encoded
            to tokens. None by default.

        Distinction between class free const and spe free const is important in Class SR context only: class free const
        values are shared between all realizations of a single program whereas spe free const values are specific to each
        dataset.

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
    ops_dict = Func.OPS_PROTECTED_DICT if use_protected_ops else Func.OPS_UNPROTECTED_DICT
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
                raise Func.UnknownFunction("%s is unknown, define a custom token function in functions.py or use a "
                                           "function listed in %s"% (name, ops_dict))

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
            tokens_input_var.append(TokenInputVar(
                                          name         = var_name,
                                          sympy_repr   = var_name,
                                          complexity   = complexity,
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
            tokens_constants.append(TokenFixedConst(
                                          name         = const_name,
                                          sympy_repr   = const_name,
                                          complexity   = complexity,
                                          # Fixed const specific
                                          fixed_const  = const_val,
                                          # ---- Physical units : units ----
                                          is_constraining_phy_units = is_constraining_phy_units,
                                          phy_units                 = phy_units,))

    # --------------------- Handling class free constants / free constants args ---------------------
    # Concatenating both args as they refer to the same type of free constants and can be used interchangeably

    # Replacing None by empty sets/dicts in free_constants_x args
    free_constants            = free_constants            if free_constants            is not None else set()
    free_constants_init_val   = free_constants_init_val   if free_constants_init_val   is not None else {}
    free_constants_units      = free_constants_units      if free_constants_units      is not None else {}
    free_constants_complexity = free_constants_complexity if free_constants_complexity is not None else {}
    # Replacing None by empty sets/dicts in class_free_constants_x args
    class_free_constants            = class_free_constants            if class_free_constants            is not None else set()
    class_free_constants_init_val   = class_free_constants_init_val   if class_free_constants_init_val   is not None else {}
    class_free_constants_units      = class_free_constants_units      if class_free_constants_units      is not None else {}
    class_free_constants_complexity = class_free_constants_complexity if class_free_constants_complexity is not None else {}

    # Concatenating
    class_free_constants            .update( free_constants            )
    class_free_constants_init_val   .update( free_constants_init_val   )
    class_free_constants_units      .update( free_constants_units      )
    class_free_constants_complexity .update( free_constants_complexity )

    # --------------------- Handling spe free constants ---------------------

    # Replacing None by empty sets/dicts in spe_free_constants_x args (for symmetrical behavior with class_free_constants_x)
    spe_free_constants            = spe_free_constants            if spe_free_constants            is not None else set()
    spe_free_constants_init_val   = spe_free_constants_init_val   if spe_free_constants_init_val   is not None else {}
    spe_free_constants_units      = spe_free_constants_units      if spe_free_constants_units      is not None else {}
    spe_free_constants_complexity = spe_free_constants_complexity if spe_free_constants_complexity is not None else {}

    # -------------------------------- Handling class free constants --------------------------------

    tokens_class_free_constants = []
    if class_free_constants is not None:
        class_free_constants_sorted = list(sorted(class_free_constants))  # (enumerating on sorted list rather than set)
        # Iterating through free constants
        for i, free_const_name in enumerate(class_free_constants_sorted):
            # ------------- Initial value -------------
            init_val = retrieve_init_val(init_val_dict=class_free_constants_init_val, curr_name=free_const_name)
            # ------------- Units -------------
            is_constraining_phy_units, phy_units = retrieve_units (units_dict=class_free_constants_units, curr_name=free_const_name)
            # ------------- Complexity -------------
            complexity = retrieve_complexity (complexity_dict=class_free_constants_complexity, curr_name=free_const_name)
            # ------------- Token creation -------------
            tokens_class_free_constants.append(TokenClassFreeConst(
                                               name         = free_const_name,
                                               sympy_repr   = free_const_name,
                                               complexity   = complexity,
                                               # Free constant specific
                                               var_id       = i,
                                               init_val     = init_val,
                                               # ---- Physical units : units ----
                                               is_constraining_phy_units = is_constraining_phy_units,
                                               phy_units                 = phy_units,))

    # -------------------------------- Handling spe free constants --------------------------------

    tokens_spe_free_constants = []
    if spe_free_constants is not None:
        spe_free_constants_sorted = list(sorted(spe_free_constants))  # (enumerating on sorted list rather than set)
        # Iterating through free constants
        for i, free_const_name in enumerate(spe_free_constants_sorted):
            # ------------- Initial value -------------
            # NB: Checking init_val shapes consistency is the responsibility of FreeConstantsTable
            init_val = retrieve_init_val(init_val_dict=spe_free_constants_init_val, curr_name=free_const_name)
            # ------------- Units -------------
            is_constraining_phy_units, phy_units = retrieve_units (units_dict=spe_free_constants_units, curr_name=free_const_name)
            # ------------- Complexity -------------
            complexity = retrieve_complexity (complexity_dict=spe_free_constants_complexity, curr_name=free_const_name)
            # ------------- Token creation -------------
            tokens_spe_free_constants.append(TokenSpeFreeConst(
                                               name         = free_const_name,
                                               sympy_repr   = free_const_name,
                                               complexity   = complexity,
                                               # Free constant specific
                                               var_id       = i,
                                               init_val     = init_val,
                                               # ---- Physical units : units ----
                                               is_constraining_phy_units = is_constraining_phy_units,
                                               phy_units                 = phy_units,))

    # -------------------------------- Result --------------------------------
    return np.array(tokens_ops + tokens_constants + tokens_class_free_constants + tokens_spe_free_constants + tokens_input_var)

