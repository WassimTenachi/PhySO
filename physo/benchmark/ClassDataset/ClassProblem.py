import pandas as pd
import numpy as np
import pathlib
import sympy
import matplotlib.pyplot as plt

# Internal imports
from physo.benchmark.utils import symbolic_utils as su

# Dataset paths
PARENT_FOLDER = pathlib.Path(__file__).parents[0]
PATH_CLASS_EQS_CSV = PARENT_FOLDER / "ClassEquations.csv"


# ---------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- LOADING CSVs  ---------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

def load_class_equations_csv (filepath_eqs ="ClassEquations.csv"):
    """
    Loads ClassEquations.csv into a pd.DataFrame.
    Parameters
    ----------
    filepath_eqs : str
        Path to ClassEquations.csv.
    Returns
    -------
    eqs_class_df : pd.DataFrame
    """

    eqs_class_df = pd.read_csv(filepath_eqs, sep=";")
    # Max nb of variables columns
    max_n_vars = int(eqs_class_df.columns.to_numpy()[-1].split('_')[0][1:])
    # Number of equations
    n_eqs = len(eqs_class_df)

    # Set types for int columns
    eqs_class_df = eqs_class_df.astype({'Number': int, '# variables': int, '# spe': int})
    # Set types for str columns
    eqs_class_df = eqs_class_df.astype({'v%i_type'%(i):str for i in range(1,max_n_vars+1)})

    # ---- Verifying number of variables for safety ----
    # Checking the number of variables declared in the file for each problem
    # Expected number of variables for each problem
    expected_n_vars = (~eqs_class_df[["v%i_name" % (i) for i in range(1, max_n_vars+1)]].isnull().to_numpy()).sum(axis=1) # (n_eqs,)
    # Declared number of variables for each problem
    n_vars = eqs_class_df["# variables"].to_numpy() + eqs_class_df["# spe"].to_numpy()                                    # (n_eqs,)
    # Is nb of declared variable consistent with variables columns ?
    is_consistent = np.equal(expected_n_vars, n_vars)                                                                     # (n_eqs,)
    assert is_consistent.all(), "Nb. of filled variables columns not consistent with declared nb. of variables for " \
                                "problems:\n %s"%(str(eqs_class_df.loc[~is_consistent]))

    return eqs_class_df


EQS_CLASS_DF = load_class_equations_csv (filepath_eqs = PATH_CLASS_EQS_CSV,)

# Number of equations in dataset
N_EQS = EQS_CLASS_DF.shape[0]

# Size of units vector
CLASS_UNITS_VECTOR_SIZE = 7

# ---------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- UNITS UTILS  ----------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# Gets units from variable name
def get_units (i_eq, i_var = 0, output_var = False):
    """
    Gets units of variable.
    Parameters
    ----------
    i_eq : int
        Equation number in the set of equations.
    i_var : int
        Variable id in its equation line.
    output_var : bool
        If True, returns units of output variable, otherwise returns units of input variable specified by i_var.
    Returns
    -------
    units : numpy.array of shape (CLASS_UNITS_VECTOR_SIZE,) of floats
        Units of variable.
    """

    units = np.zeros(CLASS_UNITS_VECTOR_SIZE)                                               # (CLASS_UNITS_VECTOR_SIZE,)

    if output_var:
        res = EQS_CLASS_DF.iloc[i_eq]["Output_units"]
    else:
        res = EQS_CLASS_DF.iloc[i_eq]["v%i_units"%(i_var)]

    res = res[1:-1]  # Removing brackets
    res = np.fromstring(res, dtype=float, sep=',')
    units[:len(res)] = res  # Replacing units by elements in res

    return units



# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- CLASS PROBLEM  ---------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
CONST_LOCAL_DICT = {"pi" : np.pi}

class ClassProblem:
    """
    Represents a single Class SR benchmark problem.
    (See https://arxiv.org/abs/2312.01816 for details).

    Attributes
    ----------
    i_eq : int
        Equation number in the set of equations.
    eq_name : str
        Equation name in the set of equations (e.g. 'Harmonic Oscillator').
    n_vars : int
        Number of input variables.
    n_spe : int
        Number of realization specific free constants.
    eq_df : pandas.core.series.Series
        Underlying pandas dataframe line of this equation.
    original_var_names : bool
        Using original variable names (e.g. theta, sigma etc.) and original output variable name (e.g. f, E etc.) if
        True, using x0, x1 ... as input variable names k0, k1,... as spe free consts names and y as output variable name
        otherwise.

    y_name_original : str
        Name of output variable as in the Class dataset.
    y_name : str
        Name of output variable.
    y_units : array_like of shape (CLASS_UNITS_VECTOR_SIZE,) of floats
        Units of output variables.

    X_names_original : array_like of shape (n_vars,) of str
        Names of input variables as in the Class dataset.
    X_names : array_like of shape (n_vars,) of str
        Names of input variables.
    X_lows : array_like of shape (n_vars,) of floats
        Lowest values taken by input variables.
    X_highs : array_like of shape (n_vars,) of floats
        Highest values taken by input variables.
    X_units :  array_like of shape (n_vars, CLASS_UNITS_VECTOR_SIZE,) of floats
        Units of input variables.

    K_names_original : array_like of shape (n_spe,) of str
        Names of spe free consts as in the Class dataset.
    K_names : array_like of shape (n_spe,) of str
        Names of spe free consts.
    K_lows : array_like of shape (n_spe,) of floats
        Lowest values taken by spe free consts.
    K_highs : array_like of shape (n_spe,) of floats
        Highest values taken by spe free consts.
    K_units :  array_like of shape (n_spe, CLASS_UNITS_VECTOR_SIZE,) of floats
        Units of spe free consts.

    formula_original : str
        Formula as in the Class dataset.

    X_sympy_symbols : array_like of shape (n_vars,) of sympy.Symbol
        Sympy symbols representing each input variables with assumptions (negative, positive etc.).
    sympy_X_symbols_dict : dict of {str : sympy.Symbol}
        Input variables names to sympy symbols (w assumptions), can be passed to sympy.parsing.sympy_parser.parse_expr
        as local_dict.
    K_sympy_symbols : array_like of shape (n_spe,) of sympy.Symbol
        Sympy symbols representing each spe free consts with assumptions (negative, positive etc.).
    sympy_K_symbols_dict : dict of {str : sympy.Symbol}
        Spe free consts names to sympy symbols (w assumptions), can be passed to sympy.parsing.sympy_parser.parse_expr

    local_dict : dict of {str : sympy.Symbol or float}
        Input variables names and spe free consts to sympy symbols (w assumptions) and constants (eg. pi : np.pi etc.),
        can be passed to sympy.parsing.sympy_parser.parse_expr as local_dict.
    formula_sympy : sympy expression
        Formula in sympy.
    formula_sympy_eval : sympy expression
        Formula in sympy with evaluated fixed constants (eg. pi -> 3.14... etc).
    formula_latex : str
        Formula in latex.

    """

    def __init__(self, i_eq = None, eq_name = None, original_var_names = False):
        """
        Loads a Class problem based on its number in the set or its equation name.
        Parameters
        ----------
        i_eq : int
            Equation number in the set of equations.
        eq_name : str
            Equation name in the set of equations (e.g. 'Harmonic Oscillator').
        original_var_names : bool
            Using original variable names (e.g. theta, sigma etc.) and original output variable name (e.g. f, E etc.) if
            True, using x0, x1 ... as input variable names and y as output variable name otherwise.
        """
        # Selecting equation line in dataframe
        if i_eq is not None:
            self.eq_df  = EQS_CLASS_DF.iloc[i_eq]                                       # pandas.core.series.Series
        elif eq_name is not None:
            self.eq_df = EQS_CLASS_DF[EQS_CLASS_DF ["Name"] == eq_name ].iloc[0]        # pandas.core.series.Series
        else:
            raise ValueError("At least one of equation number (i_eq) or equation name (eq_name) should be specified to select a Class problem.")


        # Equation number
        self.i_eq = i_eq                                                     # int
        # Code name of equation (eg. 'Harmonic Oscillator')
        self.eq_name = self.eq_df["Name"]                                    # str
        # SRBench style name
        self.SRBench_name = "class_%i"%(self.i_eq)                           # str
        # Number of input variables
        self.n_vars = int(self.eq_df["# variables"])                         # int
        # Number of realization specific free constants
        self.n_spe  = int(self.eq_df["# spe"])                               # int

        # --------- Handling input variables vs spe free consts ---------
        # Total number of variables
        n_v     = self.n_vars + self.n_spe                                                      # int
        v_ids   = np.array( [i_var for i_var in range(1, n_v +1)] )                             # (n_v,)
        v_types = self.eq_df[np.array( [ "v%i_type"%(i_var) for i_var in v_ids ] )].to_numpy()  # (n_v,)
        is_var  = v_types == "var"                                                              # (n_v,)
        is_spe  = v_types == "spe"                                                              # (n_v,)


        # Using x0, x1 ... and y names or original names (e.g. theta, sigma, f etc.)
        self.original_var_names = original_var_names                         # bool

        # ----------- y : output variable -----------
        # Name of output variable
        self.y_name_original = self.eq_df["Output_name"]                         # str
        # Name of output variable : y or original name (eg. f, E etc.)
        self.y_name = self.y_name_original if self.original_var_names else 'y'   # str
        # Units of output variables
        self.y_units = get_units(i_eq=self.i_eq, output_var=True)                # (CLASS_UNITS_VECTOR_SIZE,)

         # ----------- X : input variables -----------
        var_ids     = v_ids[is_var]                                                                                        # (n_vars,)
        var_ids_str = np.array( [ "v%i"%(i_var) for i_var in var_ids ]   ).astype(str)                                     # (n_vars,)
        # Names of input variables
        self.X_names_original = np.array( [ self.eq_df[ id + "_name" ] for id in var_ids_str  ]   ).astype(str)            # (n_vars,)
        X_names_xi_style      = np.array( [ "x%i"%(i_var) for i_var in range(self.n_vars)     ]   ).astype(str)            # (n_vars,)
        self.X_names          = self.X_names_original if self.original_var_names else X_names_xi_style                     # (n_vars,)
        # Lowest values taken by input variables
        self.X_lows           = np.array( [ self.eq_df[ id + "_low"  ] for id in var_ids_str ]    ).astype(float)          # (n_vars,)
        # Highest values taken by input variables
        self.X_highs          = np.array( [ self.eq_df[ id + "_high" ] for id in var_ids_str  ]   ).astype(float)          # (n_vars,)
        # Units of input variables
        self.X_units          = np.array( [ get_units(i_eq=self.i_eq, i_var=i_var) for i_var in var_ids ] ).astype(float)  # (n_vars, CLASS_UNITS_VECTOR_SIZE,)

        # Input variables as sympy symbols
        self.X_sympy_symbols = []
        for i in range(self.n_vars):
            self.X_sympy_symbols.append (su.sympy_symbol_with_assumptions_from_range(name = self.X_names[i],
                                                                                     low  = self.X_lows [i],
                                                                                     high = self.X_highs[i],
                                                                                    ))
        # Input variables names to sympy symbols dict
        self.sympy_X_symbols_dict = {self.X_names[i] : self.X_sympy_symbols[i] for i in range(self.n_vars)}                      #  (n_vars,)
        # Dict to use to read original class dataset formula
        # Original names to symbols in usage (i.e. symbols having original names or not)
        # eg. 'theta' -> theta symbol etc. (if original_var_names=True) or 'theta' -> x0 symbol etc. (else)
        self.sympy_original_to_X_symbols_dict = {self.X_names_original[i] : self.X_sympy_symbols[i] for i in range(self.n_vars)} #  (n_vars,)
        # NB: if original_var_names=True, then self.sympy_X_symbols_dict = self.sympy_original_to_X_symbols_dict

         # ----------- K : spe free consts -----------

        spe_ids     = v_ids[is_spe]                                                                                        # (n_spe,)
        spe_ids_str = np.array( [ "v%i"%(i_spe) for i_spe in spe_ids ]   ).astype(str)                                     # (n_spe,)
        # Names of spe free consts
        self.K_names_original = np.array( [ self.eq_df[ id + "_name" ] for id in spe_ids_str  ]   ).astype(str)            # (n_spe,)
        K_names_ki_style      = np.array( [ "k%i"%(i_spe) for i_spe in range(self.n_spe)     ]   ).astype(str)             # (n_spe,)
        self.K_names          = self.K_names_original if self.original_var_names else K_names_ki_style                     # (n_spe,)
        # Lowest values taken by spe free consts
        self.K_lows           = np.array( [ self.eq_df[ id + "_low"  ] for id in spe_ids_str ]    ).astype(float)          # (n_spe,)
        # Highest values taken by spe free consts
        self.K_highs          = np.array( [ self.eq_df[ id + "_high" ] for id in spe_ids_str  ]   ).astype(float)          # (n_spe,)
        # Units of spe free consts
        self.K_units          = np.array( [ get_units(i_eq=self.i_eq, i_var=i_var) for i_var in spe_ids ] ).astype(float)  # (n_spe, CLASS_UNITS_VECTOR_SIZE,)

        # Spe free consts as sympy symbols
        self.K_sympy_symbols = []
        for i in range(self.n_spe):
            self.K_sympy_symbols.append (su.sympy_symbol_with_assumptions_from_range(name = self.K_names[i],
                                                                                     low  = self.K_lows [i],
                                                                                     high = self.K_highs[i],
                                                                                    ))
        # Spe free consts names to sympy symbols dict
        self.sympy_K_symbols_dict = {self.K_names[i] : self.K_sympy_symbols[i] for i in range(self.n_spe)}                      #  (n_spe,)
        # Dict to use to read original class dataset formula
        # Original names to symbols in usage (i.e. symbols having original names or not)
        # eg. 'theta' -> theta symbol etc. (if original_var_names=True) or 'theta' -> k0 symbol etc. (else)
        self.sympy_original_to_K_symbols_dict = {self.K_names_original[i] : self.K_sympy_symbols[i] for i in range(self.n_spe)} #  (n_spe,)
        # NB: if original_var_names=True, then self.sympy_K_symbols_dict = self.sympy_original_to_K_symbols_dict

        # ----------- Formula -----------
        self.formula_original = self.eq_df["Formula"]  # (str)

        self.v_local_dict = {}
        self.v_local_dict.update(self.sympy_original_to_X_symbols_dict)
        self.v_local_dict.update(self.sympy_original_to_K_symbols_dict)


        # Declaring input variables via local_dict to avoid confusion
        # Eg. So sympy knows that we are referring to gamma as a variable and not the function etc.
        # evaluate = False avoids eg. sin(theta) = 0 when theta domain = [0,5] ie. nonzero=False, but no need for this
        # if nonzero assumption is not used
        evaluate = False
        self.formula_sympy   = sympy.parsing.sympy_parser.parse_expr(self.formula_original,
                                                                     local_dict = self.v_local_dict,
                                                                     evaluate   = evaluate)

        # Local dict : dict of input variables (sympy_original_to_X_symbols_dict) and fixed constants (pi -> 3.14.. etc)
        self.local_dict = {}
        self.local_dict.update(self.v_local_dict)
        self.local_dict.update(CONST_LOCAL_DICT)
        self.formula_sympy_eval = sympy.parsing.sympy_parser.parse_expr(self.formula_original,
                                                                     local_dict = self.local_dict,
                                                                     evaluate   = evaluate)
        # Latex formula
        self.formula_latex   = sympy.printing.latex(self.formula_sympy)
        return None


    def target_function(self, X, K):
        """
        Evaluates X with target function, using K values.
        Parameters
        ----------
        X : numpy.array of shape (n_vars, ?,) of floats
            Input variables.
        K : numpy.array of shape (n_spe,) of floats
            Spe free consts.
        Returns
        -------
        y : numpy.array of shape (?,) of floats
        """
        # Getting sympy function
        sympy_symbols = self.X_sympy_symbols + self.K_sympy_symbols
        f = sympy.lambdify(sympy_symbols, self.formula_sympy, "numpy")
        mapping_vals = {}
        # Mapping between variables names and their data value
        mapping_vals.update({self.X_names[i]: X[i] for i in range(self.n_vars)})
        # Mapping between spe free consts names and their data value
        mapping_vals.update({self.K_names[i]: K[i] for i in range(self.n_spe)})
        # Evaluation
        # Forcing float type so if some symbols are not evaluated as floats (eg. if some variables are not declared
        # properly in source file) resulting partly symbolic expressions will not be able to be converted to floats
        # and an error can be raised).
        # This is also useful for detecting issues such as sin(theta) = 0 because theta.is_nonzero = False -> the result
        # is just an int of float
        y = f(**mapping_vals).astype(float)
        return y

    def generate_data_points (self, n_samples = 1_000, n_realizations = 10, return_K = False):
        """
        Generates data points accordingly for this Class problem.
        Parameters
        ----------
        n_samples : int
            Number of samples to draw. By default, 1e3.
        n_realizations : int
            Number of realizations to draw. By default, 10.
        return_K : bool
            If True, returns K values used to generate data as well.
        Returns
        -------
        multi_X : numpy.array of shape (n_realizations, n_vars, n_samples,) of floats,
        multi_y : numpy.array of shape (n_realizations, n_samples,) of floats
        (multi_K : numpy.array of shape (n_realizations, n_spe,) of floats)
        """
        multi_X = []
        multi_y = []
        multi_K = []

        for i_real in range(n_realizations):
            # Random K sample
            K = np.stack([np.random.uniform(self.K_lows[i_var], self.K_highs[i_var], ) for i_var in range(self.n_spe)])           # (n_spe,)
            # Random X sample
            X = np.stack([np.random.uniform(self.X_lows[i_var], self.X_highs[i_var], n_samples) for i_var in range(self.n_vars)]) # (n_vars, n_samples)
            # Evaluating formula
            y = self.target_function(X=X, K=K)                                                                                    # (n_samples,)

            multi_X.append(X)
            multi_y.append(y)
            multi_K.append(K)

        multi_X = np.stack(multi_X)                                                                                     # (n_realizations, n_vars, n_samples)
        multi_y = np.stack(multi_y)                                                                                     # (n_realizations, n_samples)
        multi_K = np.stack(multi_K)                                                                                     # (n_realizations, n_spe)

        if return_K:
            return multi_X, multi_y, multi_K
        else:
            return multi_X, multi_y

    def get_sympy(self, K_vals=None):
        """
        Gets sympy expression of the formula evaluated with spe free consts.
        Parameters
        ----------
        K_vals : numpy.array of shape (?, n_spe,) of floats or None
            Values to evaluate spe free consts with, if None, uses random values and returns only one realization.
        Returns
        -------
        sympy_expr : np.array of shape (?, n_spe,) of Sympy Expression
        """
        if K_vals is None:
            K  = np.stack([np.random.uniform(self.K_lows[i_var], self.K_highs[i_var], ) for i_var in range(self.n_spe)])        # (n_spe,)
            K_vals = np.stack([K,])                                                                                             # (?, n_spe) = (1, n_spe)

        sympy_expr = []
        for K in K_vals:
            K_dict = {self.K_sympy_symbols[i]: K[i] for i in range(self.n_spe)}
            expr   = self.formula_sympy.subs(K_dict)
            sympy_expr.append(expr)
        sympy_expr = np.array(sympy_expr)                                                                                        # (?,)

        return sympy_expr

    def show_sample(self, n_samples = 10_000, n_realizations = 10, do_show = True, save_path = None):
        multi_X, multi_y = self.generate_data_points(n_samples=n_samples, n_realizations=n_realizations)

        n_dim = multi_X.shape[1]
        fig, ax = plt.subplots(n_dim, 1, figsize=(15, n_dim * 6))
        fig.suptitle(self.formula_original)
        for i in range(n_dim):
            curr_ax = ax if n_dim == 1 else ax[i]
            curr_ax.set_xlabel("%s : %s" % (self.X_names[i], self.X_units[i]))
            curr_ax.set_ylabel("%s : %s" % (self.y_name    , self.y_units))
            for i_real in range(n_realizations):
                curr_ax.plot(multi_X[i_real,i], multi_y[i_real], '.', markersize=1.)
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200)
        if do_show:
            plt.show()

    def get_prefix_expression (self):
        """
        Gets the prefix expression of the formula.
        Returns
        -------
        dict :
            tokens_str : numpy.array of str
                List of tokens in the expression.
            arities : numpy.array of int
                List of arities of the tokens.
            tokens : numpy.array of sympy.core
                List of sympy tokens.
        """
        return su.sympy_to_prefix(self.formula_sympy)

    def __str__(self):
        return "ClassProblem : %s : %s\n%s"%(self.i_eq, self.eq_name, str(self.formula_sympy))

    def __repr__(self):
        return str(self)
