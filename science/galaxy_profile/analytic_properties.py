from sympy import symbols, pi, simplify, oo, sqrt, diff, solve, Rational, sympify, Integral, postorder_traversal, lambdify
import sympy
import platform
import numpy as np
import scipy
import pandas as pd
import physo.benchmark.utils.timeout_generic as timeout_generic

# Numerical integrability check parameters
r_max = 1e3  # Maximum radius
r_eps = 1e-10  # Small value to avoid singularity at r=0
r_inf = 1e99 # Large value for numerical limits
num_inf_ratio = 1e3  # Ratio of usual values vs limit value beyond which limit is considered inf
num_Rs = 1. # Value at which profile is expected to have normal value, this is used to compare to value at limit.

# ------- TIMEOUT WRAPPERS -------
PROP_TIMEOUT = 5 # x seconds timeout
# pISO found with PROP_TIMEOUT > 5
ALL_CHECKS_TIMEOUT = 60
@timeout_generic.timeout(PROP_TIMEOUT)
def timed_sympy_integrate(*args, **kwargs):
    return sympy.integrate(*args, **kwargs)

@timeout_generic.timeout(PROP_TIMEOUT)
def timed_sympy_limit(*args, **kwargs):
    return sympy.limit(*args, **kwargs)

@timeout_generic.timeout(PROP_TIMEOUT)
def timed_scipy_integrate_quad(*args, **kwargs):
    return scipy.integrate.quad(*args, **kwargs)


def safe_integrate(integrand, var, symb_lims, check_num_integrability=True, vals_dict=None, num_lims=(r_eps, r_max)):

    # ----- Check numerical integrability -----
    if check_num_integrability:
        assert vals_dict is not None, "vals_dict must be provided for numerical integrability check"
        # Wrap the expression
        all_syms = integrand.free_symbols # Get all symbols in the integrand
        param_syms = [s for s in all_syms if s != var] # Filter out the integration variable
        integrand_fn = lambdify((var, *param_syms), integrand, 'numpy') # Create numerical function
        param_vals = [vals_dict[str(s)] for s in param_syms] # Prepare parameter values in correct order
        # Check if the integral converges numerically
        try:
            integral, error = timed_scipy_integrate_quad(
                lambda x: integrand_fn(x, *param_vals),
                num_lims[0],
                num_lims[1],
                # limit=100, # Increase limit for better convergence
                # max_evals = 10,
            )
            numerical_integral = integral
            is_numerically_integrable = not np.isinf(integral)
        except Exception as e:
            numerical_integral = None
            is_numerically_integrable = False
            if e is timeout_generic.TimeoutError: numerical_integral = "TimeOut"
    else:
        numerical_integral = None
        is_numerically_integrable = None

    # ----- Symbolic integration -----
    try:
        result = timed_sympy_integrate(integrand, (var, symb_lims[0], symb_lims[1]))
        is_symbolically_integrable = True
        # Check if result still contains unevaluated Integral
        for node in postorder_traversal(result):
            if isinstance(node, Integral):
                is_symbolically_integrable = False
        # Check if result is infinite
        if result.has(oo):
            is_symbolically_integrable = False
    except Exception as e:
        result = None
        is_symbolically_integrable = False
        if e is timeout_generic.TimeoutError: result = "TimeOut"


    # ----- Results -----
    results_dict = {
        'numerical_integral': numerical_integral,
        'is_numerically_integrable': is_numerically_integrable,
        'result': result,
        'is_symbolically_integrable': is_symbolically_integrable,
    }
    return result, results_dict

def safe_limit(expr, var, symb_lim, check_num_limit=True, vals_dict=None, num_lim=r_inf, num_inf_ratio=num_inf_ratio):
    """
    Safely compute the limit of an expression as var approaches limit_value.
    """
    # ---- Check numerical limit convergence -----
    if check_num_limit:
        assert vals_dict is not None, "vals_dict must be provided for numerical limit check"
        # Wrap the expression
        all_syms = expr.free_symbols  # Get all symbols in the expr
        param_syms = [s for s in all_syms if s != var]  # Filter out the integration variable
        expr_fn = lambdify((var, *param_syms), expr, 'numpy')  # Create numerical function
        param_vals = [vals_dict[str(s)] for s in param_syms]  # Prepare parameter values in correct order
        # Check if the limit converges numerically
        try:
            # numerical_limit = expr_fn(num_lim, *param_vals)
            # value_at_Rs = expr_fn(num_Rs, *param_vals)
            # is_numerically_conv = (numerical_limit/value_at_Rs) < num_inf_ratio
            numerical_limit = expr_fn(np.inf, *param_vals)
            is_numerically_conv = not np.isinf(numerical_limit)
        except Exception as e:
            numerical_limit = None
            is_numerically_conv = False
            if e is timeout_generic.TimeoutError: numerical_limit = "TimeOut"
    else:
        numerical_limit = None
        is_numerically_conv = None

    # ---- Symbolic limit -----
    try:
        symbolic_limit = timed_sympy_limit(expr, var, symb_lim)
        is_symbolically_conv = True
        # Check if result is infinite
        if symbolic_limit.has(oo):
            is_symbolically_conv = False
    except Exception as e:
        symbolic_limit = None
        is_symbolically_conv = False
        if e is timeout_generic.TimeoutError: symbolic_limit = "TimeOut"

    results_dict = {
        'numerical_limit': numerical_limit,
        'is_numerically_conv': is_numerically_conv,
        'symbolic_limit': symbolic_limit,
        'is_symbolically_conv': is_symbolically_conv,
    }
    return symbolic_limit, results_dict


def profile_from_str(density_profile, free_consts_names=["Rs", "rho0", "Rs_1", "Rs_2"]):
    # Define symbols
    r, rp, G = symbols('r rp G', positive=True)
    symb_dict = {
        'r': r,
        'rp': rp,
        'G': G,
        'pi': pi,
        'Rational': Rational,
    }
    # Make positive constants from free_consts_names
    free_consts = {name: symbols(name, positive=True) for name in free_consts_names}
    symb_dict.update(free_consts)  # Add free constants to the symbol dictionary
    # Convert the string to a SymPy expression
    density_expr = sympify(density_profile, locals=symb_dict)
    return density_expr, symb_dict

def enclosed_mass(density_profile, symb_dict, check_num_integrability=True, vals_dict=None):
    """
    Calculate the enclosed mass M(r) for a given density profile.
    """
    # Symbols setup
    rp = symb_dict['rp']
    r  = symb_dict['r']
    density_profile = density_profile.subs(r, rp) # Replace r by rp in the density profile for integration

    # Integrand: mass contribution from spherical shell at radius rp
    mass_integrand = 4 * pi * rp**2 * density_profile

    # Safe integration
    res = safe_integrate(integrand=mass_integrand, var=rp,
                         symb_lims = (0, r),
                         check_num_integrability  = check_num_integrability,
                         vals_dict = vals_dict,
                        )
    return res

def enclosed_mass_limit(M_r, symb_dict, r_limit=oo, check_num_limit=True, vals_dict=None, num_lim=r_inf):
    """
    Calculate the limit of enclosed mass M(r) as r approaches a specified limit.
    """
    # Symbols setup
    r = symb_dict['r']

    # Calculate the limit
    res = safe_limit(expr=M_r, var=r, symb_lim=r_limit,
                         check_num_limit=check_num_limit, vals_dict=vals_dict,
                         num_lim=num_lim)

    return res

def circular_velocity(M_r, symb_dict):
    """
    Calculate the circular velocity V(r) from the enclosed mass M(r).
    """
    G = symb_dict['G']
    r = symb_dict['r']
    try:
        V_r = simplify(sqrt(G * M_r / r))
    except:
        V_r = None
    return V_r

def gravitational_potential(M_r, symb_dict, use_finite_ref=False, check_num_integrability=True, vals_dict=None):
    """
    Calculate the gravitational potential Φ(r) from the enclosed mass M(r).
    """
    # Symbols setup
    G = symb_dict['G']
    rp = symb_dict['rp']
    r = symb_dict['r']
    if M_r is not None:
        M_rp = M_r.subs(r, rp)  # Substitute r with rp for integration
        # Integrand : dΦ/dr = G * M(r') / r'^2
        integrand = sympy.sympify(G * M_rp / rp**2)
    else:
        integrand = None

    # Limits for integration
    symb_lims = (r, oo) if not use_finite_ref else (0, r)  # Use finite reference or infinity

    # Safe integration
    res = safe_integrate(integrand=integrand, var=rp,
                            symb_lims = symb_lims,
                            check_num_integrability  = check_num_integrability,
                            vals_dict = vals_dict,
                            )
    return res

def radial_velocity_dispersion(density_profile, M_r, symb_dict, check_num_integrability=True, vals_dict=None):
    """
    Calculate the radial velocity dispersion σ_r²(r) using Jeans equation.
    """
    # Symbols setup
    G = symb_dict['G']
    r = symb_dict['r']
    rp = symb_dict['rp']
    if M_r is not None:
        rho_rp = density_profile.subs(r, rp) # Replace r by rp in the density profile for integration
        M_rp   = M_r.subs(r, rp)             # Substitute r with rp for integration
        # Integrand for Jeans equation: σ_r²(r) = (1/ρ(r)) ∫_r^∞ ρ(r') * G M(r') / r'² dr'
        integrand = rho_rp * G * M_rp / rp**2
    else:
        integrand = None

    # Safe integration
    result, results_dict = safe_integrate(integrand=integrand, var=rp,
                                         symb_lims=(r, oo),
                                         check_num_integrability=check_num_integrability,
                                         vals_dict=vals_dict)

    # Divide by the density profile to get σ_r²
    rho_r = density_profile
    if result is not None:
        sigma_r2 = simplify(result / rho_r)
    else:
        sigma_r2 = None

    # Update results with the final calculation
    result = sigma_r2
    results_dict['result'] = result

    return result, results_dict

@timeout_generic.timeout(ALL_CHECKS_TIMEOUT)
def check_analytical_properties(density_profile_str, free_consts_names, num_check=False, free_consts_vals=None, verbose=True):
    """
    Check if the analytical properties are defined correctly.
    Parameters
    ----------
    density_profile_str : str
        String representation of the density profile as a function of the radius r. This is used for symbolic checks.
        Symbolic checks are made using the density profile string only, not free_consts_vals, any non evaluated symbols
        in density_profile_str are kept as is to calculate properties.
    free_consts_names : list
        Declaration of free constants that can appear in density_profile_str so they can be declared as
        with proper assumptions in sympy.
    num_check : bool
        If True, perform numerical checks in addition to symbolic checks.
    free_consts_vals : dict
        Dictionary containing values for free constants like rho0, Rs, etc. This is used for numerical checks only.

    verbose : bool
        If True, print the results of the checks.

    """
    # ---- Handling symbols ----
    if num_check:
        assert free_consts_vals is not None, "free_consts_vals must be provided for numerical checks"
    else:
        free_consts_vals = {}
    dynamical_params_vals = {
        'G': 1.0,  # Example value for G
    }
    vals_dict = {**free_consts_vals, **dynamical_params_vals}  # Combine all values

    # ---- Prepare results DataFrame ----
    results_df_rows = []

    # ---- Handling density profile ----
    density_profile, symb_dict = profile_from_str(density_profile_str, free_consts_names)
    if verbose:
        print("\n--- Density profile ---\n", density_profile)
    results_df_rows.append({
        'Property'       : 'Density',
        'symb_condition' : True,
        'symb_result'    : density_profile,
        'num_condition'  : True,
        'num_result'     : None,
        'logs'           : ''
    })

    # ----- Enclosed mass calculation -----
    M_r, M_r_logs = enclosed_mass(density_profile, symb_dict, check_num_integrability=num_check, vals_dict=vals_dict)
    if verbose:
        print("\n--- Enclosed mass M(r) ---\n", M_r)
    results_df_rows.append({
        'Property'       : 'Enclosed Mass',
        'symb_condition' : M_r_logs['is_symbolically_integrable'],
        'symb_result'    : M_r,
        'num_condition'  : M_r_logs['is_numerically_integrable'],
        'num_result'     : M_r_logs['numerical_integral'],
        'logs'           : M_r_logs,
    })

    # ----- Enclosed mass limit calculation -----
    M_r_limit, M_r_limit_logs = enclosed_mass_limit(M_r, symb_dict, r_limit=oo, check_num_limit=num_check, vals_dict=vals_dict)
    if verbose:
        print("\n--- Enclosed mass limit M(r) as r → ∞ ---\n", M_r_limit)
    results_df_rows.append({
        'Property'       : 'Enclosed Mass Limit',
        'symb_condition' : M_r_limit_logs ['is_symbolically_conv'],
        'symb_result'    : M_r_limit,
        'num_condition'  : M_r_limit_logs ['is_numerically_conv'],
        'num_result'     : M_r_limit_logs ['numerical_limit'],
        'logs'           : M_r_limit_logs,
    })

    # ----- Circular velocity calculation -----
    V_r = circular_velocity(M_r, symb_dict)
    if verbose:
        print("\n--- Circular velocity V(r) ---\n", V_r)
    results_df_rows.append({
        'Property'       : 'Circular Velocity',
        'symb_condition' : M_r_logs['is_symbolically_integrable'],
        'symb_result'    : V_r,
        'num_condition'  : M_r_logs['is_numerically_integrable'],
        'num_result'     : None,
        'logs'           : '',
    })

    # ----- Gravitational potential calculation -----
    Phi_r, Phi_r_logs = gravitational_potential(M_r, symb_dict, use_finite_ref=False, check_num_integrability=num_check, vals_dict=vals_dict)
    if verbose:
        print("\n--- Gravitational potential Φ(r) ---\n", Phi_r)
    results_df_rows.append({
        'Property'       : 'Potential (wo ref. pt.)',
        'symb_condition' : Phi_r_logs['is_symbolically_integrable'],
        'symb_result'    : Phi_r,
        'num_condition'  : Phi_r_logs['is_numerically_integrable'],
        'num_result'     : Phi_r_logs['numerical_integral'],
        'logs'           : Phi_r_logs,
    })

    # ----- Gravitational potential calculation with finite reference point -----
    Phi_finite_ref_r, Phi_finite_ref_r_logs = gravitational_potential(M_r, symb_dict, use_finite_ref=True, check_num_integrability=num_check, vals_dict=vals_dict)
    if verbose:
        print("\n--- Gravitational potential Φ(r) - Φ(0) ---\n", Phi_finite_ref_r)
    results_df_rows.append({
        'Property'       : 'Potential',
        'symb_condition' : Phi_finite_ref_r_logs['is_symbolically_integrable'],
        'symb_result'    : Phi_finite_ref_r,
        'num_condition'  : Phi_finite_ref_r_logs['is_numerically_integrable'],
        'num_result'     : Phi_finite_ref_r_logs['numerical_integral'],
        'logs'           : Phi_finite_ref_r_logs,
    })

    # ----- Radial velocity dispersion calculation -----
    sigma_r2, sigma_r2_logs = radial_velocity_dispersion(density_profile, M_r, symb_dict, check_num_integrability=num_check, vals_dict=vals_dict)
    if verbose:
        print("\n--- Radial velocity dispersion σ_r²(r) ---\n", sigma_r2)
    results_df_rows.append({
        'Property'       : 'Radial Velocity Dispersion',
        'symb_condition' : sigma_r2_logs['is_symbolically_integrable'],
        'symb_result'    : sigma_r2,
        'num_condition'  : sigma_r2_logs['is_numerically_integrable'],
        'num_result'     : sigma_r2_logs['numerical_integral'],
        'logs'           : sigma_r2_logs,
    })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_df_rows)
    if verbose:
        print("\n--- Results DataFrame ---\n", results_df[['Property', "symb_condition", "num_condition"]])
    return results_df
