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


def round_floats(expr, round_decimal = 3):
    """
    Rounds the floats in a sympy expression as in SRBench (see https://github.com/cavalab/srbench).
    Parameters
    ----------
    expr : Sympy Expression
    round_decimal : int
        Rounding up to this decimal.
    Returns
    -------
    ex2 : Sympy Expression
    """
    ex2 = expr
    # Why not use expr.atoms ?
    # Doing it like SRBench
    # todo: clean func
    for a in sympy.preorder_traversal(expr):
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
                ex2 = ex2.subs(a, sympy.Float(round(a, round_decimal), round_decimal))

                # C. Sometimes in sympy, a node can have the value 0.398986816406250 but the exact same node can have
                # a different value such as 0.39898681640625 in the upper node 0.39898681640625*x.
                #ex2 = ex2.xreplace({a: round(a, round_decimal)}) # xreplace is for exact node replacment
    return ex2


def clean_sympy_expr(expr):
    """
    Cleans (rounds floats, simplifies) sympy expression for symbolic comparison purposes as in SRBench
    (see https://github.com/cavalab/srbench).
    Parameters
    ----------
    expr : Sympy Expression
    Returns
    -------
    expr : Sympy Expression
    """
    # Evaluates numeric constants such as sqrt(2*pi)
    expr = expr.evalf()
    # Rounding floats
    expr = round_floats(expr)
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
                        verbose=False):
    """
    Checks if trial_expr is symbolically equivalent to target_expr of this Feynman problem, following a
    similar methodology as SRBench (see https://github.com/cavalab/srbench).
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
        target_expr = clean_sympy_expr(target_expr)
    except Exception as e:
        trial_expr = trial_expr

    # ------ Trial expression cleaning ------
    try:
        trial_expr = clean_sympy_expr(trial_expr)
    except Exception as e:
        trial_expr = trial_expr

    # ------ Checking symbolic difference ------

    # Vanilla
    try:
        vanilla_sym_err = clean_sympy_expr(target_expr - trial_expr)
        vanilla_sym_err_is_zero  = str(vanilla_sym_err) == '0'
        vanilla_sym_err_is_const = vanilla_sym_err.is_constant() and contains_no_inf(vanilla_sym_err)
    except Exception as e:
        warnings.warn(warn_msg)
        vanilla_sym_err = ""
        vanilla_sym_err_is_zero  = False
        vanilla_sym_err_is_const = False

    # For trigo cases
    try:
        trigo_sym_err = clean_sympy_expr(expr_floats_to_pi_fracs(target_expr - trial_expr))
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
        vanilla_sym_frac = clean_sympy_expr(target_expr / trial_expr)
        vanilla_sym_frac_is_const = vanilla_sym_frac.is_constant() \
                                    and (str(vanilla_sym_frac) != '0' or not prevent_zero_frac) \
                                    and contains_no_inf(vanilla_sym_frac)
    except Exception as e:
        warnings.warn(warn_msg)
        vanilla_sym_frac = ""
        vanilla_sym_frac_is_const = False

    # For trigo cases
    try:
        trigo_sym_frac = clean_sympy_expr(expr_floats_to_pi_fracs(target_expr / trial_expr))
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

    # SRBench style report
    report = {
        'symbolic_error'                : str(vanilla_sym_err),
        'symbolic_fraction'             : str(vanilla_sym_frac),
        'symbolic_error_is_zero'        : symbolic_error_is_zero,
        'symbolic_error_is_constant'    : symbolic_error_is_constant,
        'symbolic_fraction_is_constant' : symbolic_fraction_is_constant,
        'sympy_exception'               : str(e),
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

