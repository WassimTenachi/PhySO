import sympy
import numpy as np
import warnings

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- EQUIVALENCE UTILS  -------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def round_to_sympy_integer (k, limit_err = 0.001,):
    """
    Returns closest integer if it is +/- limit_err.
    Parameters
    ----------
    limit_err : float
        Replaces only if the closest integer is +/- limit_err of the float to be replaced.
    Returns
    -------
    res : sympy expression or float
    Examples
    --------
    round_to_sympy_integer(3.01)
        >> 3.01
    round_to_sympy_integer(3.001)
        >> 3
    """
    new_k, is_rationalized = rationalize (k, limit_err = limit_err, limit_denominator = 1)
    if is_rationalized:
        res = sympy.Integer(new_k)
    else:
        res = k
    return res


def expr_floats_to_pi_fracs (expr, limit_err = 0.001, limit_denominator = 10 ):
    """
    Replaces floats in sympy expression by rational fractions of pi.
    Parameters
    ----------
    expr : sympy expression
        Expression.
    limit_err : float
        Replaces only if the closest rational is +/- limit_err of the float to be replaced.
    limit_denominator : int
        Searches for rationals having a denominator of value up to limit_denominator.
    Returns
    -------
    res_expr : sympy expression
        Expression.
    Examples
    --------
    x = sympy.Symbol('x')
    expr1 = sympy.cos(x + -0.5*3.1415926545)
    expr2 = sympy.sin(x)
    sympy.sympify(expr1-expr2)
        >> -sin(x) + cos(x - 1.57079632725)
    sympy.sympify(expr_floats_to_pi_fracs(expr1-expr2))
        >> 0
    """
    res_expr = expr
    numbers = [s for s in res_expr.atoms(sympy.Number)]
    for n in numbers:
        new_n  = rationalize_to_pi_frac (n, limit_err=limit_err, limit_denominator=limit_denominator)
        res_expr = res_expr.subs(n, new_n)
    return res_expr


def rationalize_to_pi_frac (a, limit_err = 0.001, limit_denominator = 10):
    """
    Rationalizes a float to a rational fraction of pi using sympy.
    Parameters
    ----------
    a : float
        Value to rationalize.
    limit_err : float
        Replaces only if the closest rational is +/- limit_err of the float to be replaced.
    limit_denominator : int
        Searches for rationals having a denominator of value up to limit_denominator.
    Returns
    -------
    res : sympy expression or float
    Examples
    --------
    rationalize_to_pi_frac(0.3333333*3.1415926535)
        >> pi/3
    rationalize_to_pi_frac(0.33*3.1415926535) = 1.036725575655
        >> pi/3
    """
    k = a/np.pi
    new_k, is_rationalized = rationalize (k, limit_err = limit_err, limit_denominator = limit_denominator)
    if is_rationalized:
        res = new_k*sympy.pi
    else:
        res = a
    return res


def rationalize (k, limit_err = 0.001, limit_denominator = 10):
    """
    Rationalizes a float to a rational fraction.
    Parameters
    ----------
    k : float
        Value to rationalize.
    limit_err : float
        Replaces only if the closest rational is +/- limit_err of the float to be replaced.
    limit_denominator : int
        Searches for rationals having a denominator of value up to limit_denominator.
    Returns
    -------
    res, is_rationalized : sympy expression or float, bool
    Examples
    --------
    rationalize(0.333)
        >> 1/3
    rationalize(0.33)
        >> 0.33
    """
    k_rat = sympy.Rational(str(k)).limit_denominator(limit_denominator)
    if abs(k_rat-k)<limit_err:
        is_rationalized = True
        res = k_rat
    else:
        is_rationalized = False
        res = k
    return res, is_rationalized

def replace_sin_by_cos (expr):
    """
    Replaces sin(...) by cos(pi/2 - ...) in a sympy expression.
    Parameters
    ----------
    expr : Sympy Expression
    Returns
    -------
    ex1 : Sympy Expression
    """
    ex1 = expr
    # If sin(...) is encountered, replacing it by cos(pi/2 - ...)
    for a in sympy.preorder_traversal(expr):
        if type(a)==sympy.sin:
            b = sympy.cos(np.pi/2 - a.args[0])
            ex1 = ex1.subs(a, b)
    return ex1

def replace_cos_by_sin (expr):
    """
    Replaces cos(...) by sin(pi/2 - ...) in a sympy expression.
    Parameters
    ----------
    expr : Sympy Expression
    Returns
    -------
    ex1 : Sympy Expression
    """
    ex1 = expr
    # If cos(...) is encountered, replacing it by sin(pi/2 - ...)
    for a in sympy.preorder_traversal(expr):
        if type(a)==sympy.cos:
            b = sympy.sin(np.pi/2 - a.args[0])
            ex1 = ex1.subs(a, b)
    return ex1

def round_floats(expr, round_decimal = 2):
    """
    Rounds the floats in a sympy expression as in SRBench (see https://github.com/cavalab/srbench).
    Parameters
    ----------
    expr : Sympy Expression
    round_decimal : int
        Rounding up to this decimal.
        Use round_decimal = 2 for SRBench-like behavior (as they actually round up to 2 decimals).
    Returns
    -------
    ex2 : Sympy Expression
    """

    # Why not use expr.atoms ?
    # for a in [el for el in expr.atoms()]:

    # Other sympy numbers to convert to floats that evalf does not handle
    ex1 = expr
    sympy_numbers_to_float = [sympy.core.numbers.Exp1,]#[sympy.core.numbers.Pi]
    for a in sympy.preorder_traversal(expr):
        if (type(a) in sympy_numbers_to_float):
            ex1 = ex1.subs(a, round(float(a), round_decimal))

    ex2 = ex1
    for a in sympy.preorder_traversal(ex1):
        if isinstance(a, sympy.Float):
            ex2 = ex2.subs(a, round_to_sympy_integer(a, limit_err=10**(-round_decimal),))
            if abs(a) < 0.0001:
                ex2 = ex2.subs(a, sympy.Integer(0))
            # Should prevent sympy not being able to simplify 1.0*x - x to 0 for example
            elif abs(a - 1.) < 0.0001:
                ex2 = ex2.subs(a, sympy.Integer(1))
            else:
                # A. SRBench postprocessing function uses this (but they actually never use it in the code ?):
                # ex2 = ex2.subs(a, round(a, round_decimal))

                # B. SRBench actually uses this to check if an expr is equivalent (this is visible when checking their
                # positive results on "dataset" feynman_III_8_54)
                # With round_decimal = 3, this rounds up to only 2 decimals, actually.
                # ex2 = ex2.subs(a, sympy.Float(round(a, round_decimal), round_decimal))

                # C. Working on python float for rounding and subs is better as sympy sometimes fails to find and
                # replace
                ex2 = ex2.subs(a, round(float(a), round_decimal))

                # D. Sometimes in sympy, a node can have the value 0.398986816406250 but the exact same node can have
                # a different value such as 0.39898681640625 in the upper node 0.39898681640625*x.
                #ex2 = ex2.xreplace({a: round(a, round_decimal)}) # xreplace is for exact node replacment
    return ex2


def clean_sympy_expr(expr, round_decimal = 2):
    """
    Cleans (rounds floats, simplifies) sympy expression for symbolic comparison purposes as in SRBench
    (see https://github.com/cavalab/srbench).
    Parameters
    ----------
    expr : Sympy Expression
    round_decimal : int
        Rounding up to this decimal.
        Use round_decimal = 2 for SRBench-like behavior (as they actually round up to 2 decimals).
    Returns
    -------
    expr : Sympy Expression
    """
    # Evaluates numeric constants such as sqrt(2*pi)
    expr = expr.evalf()
    # Rounding floats
    expr = round_floats(expr, round_decimal=round_decimal)
    # Simplifying
    expr = sympy.simplify(expr, ratio=1)
    return expr

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- EQUIVALENCE FUNC  -------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def compare_expression (trial_expr,
                        target_expr,
                        handle_trigo            = True,
                        prevent_zero_frac       = True,
                        prevent_inf_equivalence = True,
                        round_decimal = 2,
                        verbose=False):
    """
    Checks if trial_expr is symbolically equivalent to target_expr, following a similar methodology as
    SRBench (see https://github.com/cavalab/srbench).
    I.e, it is deemed equivalent if:
        - the symbolic difference simplifies to 0
        - OR the symbolic difference is a constant
        - OR the symbolic ratio simplifies to a constant
    Parameters
    ----------
    trial_expr : Sympy Expression
        Trial sympy expression with evaluated numeric free constants and assumptions regarding variables
        (positivity etc.) encoded in expression.
    target_expr : Sympy Expression
        Target sympy expression with evaluated numeric free constants and assumptions regarding variables
        (positivity etc.) encoded in expression.
    handle_trigo : bool
        Tries replacing floats by rationalized factors of pi and simplify with that.
    prevent_zero_frac : bool
        If fraction = 0 does not consider expression equivalent.
    prevent_inf_equivalence: bool
        If symbolic error or fraction is infinite does not consider expression equivalent.
    round_decimal : int
        Rounding up to this decimal.
        Use round_decimal = 2 for SRBench-like behavior (as they actually round up to 2 decimals).
    verbose : bool
        Verbose.
    Returns
    -------
    is_equivalent, report : bool, dict
        Is the expression equivalent, A dict containing details about the equivalence SRBench style.
    """

    # Verbose
    if verbose:
        print("  -> Assessing if %s (target) is equivalent to %s (trial)"%(target_expr, trial_expr))

    # Error
    e = ""
    warn_msg = "Could not assess symbolic equivalence of %s with %s." % (target_expr, trial_expr)

    # Utils function to check if expression contains inf
    def contains_no_inf (expr):
        no_inf = not('oo' in str(expr) and prevent_inf_equivalence)
        return no_inf

    # ------ Target expression cleaning ------
    try:
        target_expr = clean_sympy_expr(target_expr, round_decimal=round_decimal)
    except Exception as e:
        target_expr = target_expr

    # ------ Trial expression cleaning ------
    try:
        trial_expr = clean_sympy_expr(trial_expr, round_decimal=round_decimal)
    except Exception as e:
        trial_expr = trial_expr

    # ------ Checking symbolic difference ------

    # Vanilla
    try:
        vanilla_sym_err = clean_sympy_expr(target_expr - trial_expr, round_decimal=round_decimal)
        vanilla_sym_err_is_zero  = str(vanilla_sym_err) == '0'
        vanilla_sym_err_is_const = vanilla_sym_err.is_constant() and contains_no_inf(vanilla_sym_err)
    except Exception as e:
        warnings.warn(warn_msg)
        vanilla_sym_err = ""
        vanilla_sym_err_is_zero  = False
        vanilla_sym_err_is_const = False

    # For trigo cases
    try:
        trigo_sym_err = clean_sympy_expr(
            expr_floats_to_pi_fracs(target_expr - trial_expr, limit_err = 10**(-round_decimal)),
            round_decimal=round_decimal)
        trigo_sym_err_is_zero  = str(trigo_sym_err) == '0'
        trigo_sym_err_is_const = trigo_sym_err.is_constant() and contains_no_inf(trigo_sym_err)
    except Exception as e:
        warnings.warn(warn_msg)
        trigo_sym_err = ""
        trigo_sym_err_is_zero  = False
        trigo_sym_err_is_const = False

    symbolic_error_is_zero     = vanilla_sym_err_is_zero  or (trigo_sym_err_is_zero  and handle_trigo)
    symbolic_error_is_constant = vanilla_sym_err_is_const or (trigo_sym_err_is_const and handle_trigo)

    # ------ Checking symbolic fraction ------

    # Vanilla
    try:
        vanilla_sym_frac = clean_sympy_expr(target_expr / trial_expr, round_decimal=round_decimal)
        vanilla_sym_frac_is_const = vanilla_sym_frac.is_constant() \
                                    and (str(vanilla_sym_frac) != '0' or not prevent_zero_frac) \
                                    and contains_no_inf(vanilla_sym_frac)
    except Exception as e:
        warnings.warn(warn_msg)
        vanilla_sym_frac = ""
        vanilla_sym_frac_is_const = False

    # For trigo cases
    try:
        trigo_sym_frac = clean_sympy_expr(
            expr_floats_to_pi_fracs(target_expr / trial_expr, limit_err = 10**(-round_decimal)),
            round_decimal=round_decimal)
        trigo_sym_frac_is_const = trigo_sym_frac.is_constant() \
                                  and (str(trigo_sym_frac) != '0' or not prevent_zero_frac) \
                                  and contains_no_inf(trigo_sym_frac)
    except Exception as e:
        warnings.warn(warn_msg)
        trigo_sym_frac = ""
        trigo_sym_frac_is_const = False

    symbolic_fraction_is_constant = vanilla_sym_frac_is_const or (trigo_sym_frac_is_const and handle_trigo)

    # ------ Results ------

    # Equivalent if diff simplifies to 0 or if the diff is a constant or if the ratio is a constant
    is_equivalent = symbolic_error_is_zero or symbolic_error_is_constant or symbolic_fraction_is_constant

    if verbose:
        print('   -> Simplified expression :', trial_expr)
        print('   -> Symbolic error        :', vanilla_sym_err)
        print('   -> Symbolic fraction     :', vanilla_sym_frac)
        print('   -> Trigo symbolic error        :', trigo_sym_err)
        print('   -> Trigo symbolic fraction     :', trigo_sym_frac)
        print('   -> Equivalent :', is_equivalent)

    try:
        str_err = str(e)
    except:
        str_err = "Unknown error."

    # SRBench style report
    # Displaying what triggered the equivalence
    display_sym_err  = vanilla_sym_err  if (vanilla_sym_err_is_zero or vanilla_sym_err_is_const) else trigo_sym_err
    display_sym_frac = vanilla_sym_frac if vanilla_sym_frac_is_const                             else trigo_sym_frac
    report = {
        'symbolic_error'                : str(display_sym_err),
        'symbolic_fraction'             : str(display_sym_frac),
        'symbolic_error_is_zero'        : symbolic_error_is_zero,
        'symbolic_error_is_constant'    : symbolic_error_is_constant,
        'symbolic_fraction_is_constant' : symbolic_fraction_is_constant,
        'sympy_exception'               : str_err,
        'symbolic_solution'             : is_equivalent,
        }

    return is_equivalent, report

# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- MISCELLANEOUS  ---------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def expression_size(expr):
    """
    Evaluates complexity as in SRBench
    (see https://github.com/cavalab/srbench).
    Parameters
    ----------
    expr : Sympy Expression
    Returns
    -------
    c : int
    """
    c=0
    for arg in sympy.preorder_traversal(expr):
        c += 1
    return c

def sympy_to_prefix(sympy_expr):
    """
    Converts a sympy expression to prefix notation.
    Parameters
    ----------
    sympy_expr : sympy.core
        Sympy expression
    Returns
    -------
    dict :
        tokens_str : numpy.array of str
            List of tokens in the expression.
        arities : numpy.array of int
            List of arities of the tokens.
        tokens : numpy.array of sympy.core
            List of sympy tokens.
    Usage
    -----
    sympy_expr = sympy.parse_expr("sqrt(x0+2)")
    tokens_str = sympy_to_prefix(sympy_expr)["tokens_str"]
    """
    arities    = []
    tokens     = []
    tokens_str = []
    for symb in sympy.preorder_traversal(sympy_expr):
        # Arity
        arity = len(symb.args)
        # Token analysis
        if arity != 0:
            tok     = symb.func
            tok_str = symb.func.__name__
        if arity == 0:
            # If this is a variable that has no particular value
            if symb.is_Symbol:
                tok     = symb
                tok_str = symb.__str__()
            # If this is a constant
            else:
                tok     = symb
                tok_str = str(symb.evalf())
        arities    .append(arity)
        tokens_str .append(tok_str)
        tokens     .append(tok)
    res = {
        "tokens_str": np.array(tokens_str),
        "arities"   : np.array(arities   ),
        "tokens"    : np.array(tokens    ),
    }
    return res

def sympy_symbol_with_assumptions_from_range(name, low, high):
    """
    Returns a sympy symbol with assumptions from its data range.
    Parameters
    ----------
    name : str
        Name of the variable.
    low : float
        Lowest value taken by the variable.
    high : float
        Highest value taken by the variable.
    Returns
    -------
    sympy.Symbol
    """
    is_positive = low > 0
    is_negative = high < 0
    # If 0 is in interval, do not give assumptions as otherwise sympy will assume 0
    if (not is_positive) and (not is_negative):
        is_positive, is_negative = None, None
    symb = sympy.Symbol(name,  # str
                        # Useful assumptions for simplifying etc
                        real=True,
                        positive=is_positive,
                        negative=is_negative,
                        # If nonzero = False assumes that always = 0 which causes problems
                        # when simplifying
                        # nonzero  = not (low <= 0 and high >= 0),
                        domain=sympy.sets.sets.Interval(float(low), float(high)),
                        )
    return symb
