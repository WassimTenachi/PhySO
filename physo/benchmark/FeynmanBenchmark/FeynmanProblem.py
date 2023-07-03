import pandas as pd
import numpy as np
import os
import pathlib
import sympy

# Dataset paths
PARENT_FOLER = pathlib.Path(__file__).parents[0]
PATH_FEYNMAN_EQS_CSV = PARENT_FOLER / "FeynmanEquations.csv"
PATH_UNITS_CSV       = PARENT_FOLER / "units.csv"


# ---------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- LOADING CSVs  ---------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

def load_feynman_equations_csv (filepath = "FeynmanEquations.csv"):
    """
    Loads FeynmanEquations.csv into a clean pd.DataFrame.
    Source file can be found here: https://space.mit.edu/home/tegmark/aifeynman.html
    Parameters
    ----------
    filepath : str
        Path to FeynmanEquations.csv.
    Returns
    -------
    eqs_feynman_df : pd.DataFrame
    """
    eqs_feynman_df = pd.read_csv(filepath, sep=",")
    # drop last row(s) of NaNs
    eqs_feynman_df = eqs_feynman_df[~eqs_feynman_df[eqs_feynman_df.columns[0]].isnull()]
    return eqs_feynman_df

def load_feynman_units_csv (filepath = "units.csv"):
    """
    Loads units.csv into a clean pd.DataFrame.
    Source file can be found here: https://space.mit.edu/home/tegmark/aifeynman.html
    Parameters
    ----------
    filepath : str
        Path to units.csv.
    Returns
    -------
    units_df : pd.DataFrame
    """
    units_df = pd.read_csv(filepath, sep=",")
    # drop last row(s) of NaNs
    units_df = units_df[~units_df[units_df.columns[0]].isnull()]
    # drop last column as it contains nothing
    units_df = units_df.iloc[:, :-1]
    return units_df

path_feynman_eqs_csv = os.path.join("physo", "benchmark", "FeynmanBenchmark", "FeynmanEquations.csv")
path_units_csv       = os.path.join("physo", "benchmark", "FeynmanBenchmark", "units.csv")

EQS_FEYNMAN_DF = load_feynman_equations_csv(PATH_FEYNMAN_EQS_CSV)
UNITS_DF       = load_feynman_units_csv(PATH_UNITS_CSV)

# Size of units vector
FEYN_UNITS_VECTOR_SIZE = UNITS_DF.shape[1] - 2

# Number of equations in dataset
N_EQS = EQS_FEYNMAN_DF.shape[0]

# ---------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- UNITS UTILS  ----------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# Gets units from variable name
def get_units (var_name):
    """
    Gets units of variable var_name. Example: get_units("kb")
    Parameters
    ----------
    var_name : str
    Returns
    -------
    units : numpy.array of shape (FEYN_UNITS_VECTOR_SIZE,) of floats
        Units of variable.
    """
    units = UNITS_DF[UNITS_DF["Variable"] == var_name].to_numpy()[0][2:].astype(float)
    return units

# ---------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- EQUATIONS UTILS  -------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


# How to replace str symbols of functions appearing in the Feynman benchmark by their numpy equivalent
DICT_FOR_FEYNMAN_FORMULA_FUNC_TO_NP = {
    "exp"    : "np.exp"    ,
    "sqrt"   : "np.sqrt"   ,
    "pi"     : "np.pi"     ,
    "cos"    : "np.cos"    ,
    "sin"    : "np.sin"    ,
    "tan"    : "np.tan"    ,
    "tanh"   : "np.tanh"   ,
    "ln"     : "np.log"    ,
    "arcsin" : "np.arcsin" ,
}


def replace_symbols_in_formula(formula,
                               dict_for_feynman_formula_var_names
                               ):
    """
    Replaces symbols in a Feynman equation formula by numpy functions for functions and input variables accordingly
    with dict_for_feynman_formula_var_names;
    Parameters
    ----------
    formula : str
        Raw initial formula.
    dict_for_feynman_formula_var_names : dict of (str: str)
        Which symbol to replace by which other for input variables.
    Returns
    -------
    formula : str
        Formula with corrected symbols ready for execution via evaluate(formula).
    """
    dict_fml = DICT_FOR_FEYNMAN_FORMULA_FUNC_TO_NP
    for symbol in dict_fml.keys():
        new_symbol = dict_fml[symbol]
        formula = formula.replace(symbol, new_symbol)
    dict_fml = dict_for_feynman_formula_var_names
    for symbol in dict_fml.keys():
        new_symbol = dict_fml[symbol]
        formula = formula.replace(symbol, new_symbol)
    return formula


# ---------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- FEYNMAN PROBLEM  --------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

class FeynmanProblem:
    def __init__(self, i_eq):
        """
        Loads a Feynman problem based on its number in the set.
        Parameters
        ----------
        i_eq : int
            Equation number in the set of equations.
        """

        # Equation line in dataframe
        self.eq_df  = EQS_FEYNMAN_DF.iloc[i_eq]
        # Code name of equation (eg. 'I.6.2a')
        self.eq_name = self.eq_df["Filename"]   # str
        # Number of input variables
        self.n_vars = int(self.eq_df["# variables"])    # int

        # ----------- y : output variable -----------
        # Name of output variable
        self.y_name   = self.eq_df["Output"]            # str
        # Units of output variables
        self.y_units = get_units(self.y_name)           # (FEYN_UNITS_VECTOR_SIZE,)

        # ----------- X : input variables -----------
        # Utils id of input variables v1, v2 etc. in .csv
        var_names_str = np.array( [ "v%i"%(i_var) for i_var in range(1, self.n_vars+1) ]                             ).astype(str)            # (n_vars,)
        # Names of input variables
        self.X_names = np.array( [ self.eq_df[ var_names_str[i_var] + "_name" ] for i_var in range(self.n_vars)  ]   ).astype(str)            # (n_vars,)
        # Lowest values taken by input variables
        self.X_lows  = np.array( [ self.eq_df[ var_names_str[i_var] + "_low"  ] for i_var in range(self.n_vars) ]    ).astype(float)          # (n_vars,)
        # Highest values taken by input variables
        self.X_highs = np.array( [ self.eq_df[ var_names_str[i_var] + "_high" ] for i_var in range(self.n_vars)  ]   ).astype(float)          # (n_vars,)
        # Units of input variables
        self.X_units = np.array( [ get_units(self.eq_df[ var_names_str[i_var] + "_name" ]) for i_var in range(self.n_vars) ] ).astype(float)  # (n_vars, FEYN_UNITS_VECTOR_SIZE,)

        # ----------- Formula -----------
        self.formula_display = self.eq_df["Formula"]
        self.formula_sympy   = sympy.parsing.sympy_parser.parse_expr(self.formula_display)
        dict_for_feynman_formula_var_names = {self.X_names[i_var]:"X[%i]"%(i_var) for i_var in range(self.n_vars)}
        self.formula = replace_symbols_in_formula(self.formula_display, dict_for_feynman_formula_var_names)

        return None

    def target_function(self, X):
        """
        Evaluates X with target function.
        Parameters
        ----------
        X : numpy.array of shape (n_vars, ?,) of floats
        Returns
        -------
        y : numpy.array of shape (?,) of floats
        """
        y = eval(self.formula)
        return y

    def generate_data_points (self, n_samples = int(1e6)):
        """
        Generates data points accordingly for this Feynman problem.
        Parameters
        ----------
        n_samples : int
            Number of samples to draw. By default, 1e6 as this is the number of data points for each problem in the
            files in https://space.mit.edu/home/tegmark/aifeynman.html
        Returns
        -------
        X, y : numpy.array of shape (n_vars, ?,) of floats, numpy.array of shape (?,) of floats
        """
        X = np.stack([np.random.uniform(self.X_lows[i_var], self.X_highs[i_var], n_samples) for i_var in range(self.n_vars)])
        y = self.target_function(X)
        return X,y

