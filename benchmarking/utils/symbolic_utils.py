import sympy
import numpy as np


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
