import pandas as pd
import numpy as np
import os

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

EQS_FEYNMAN_DF = load_feynman_equations_csv(path_feynman_eqs_csv)
UNITS_DF       = load_feynman_units_csv(path_units_csv)

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
    def __int__(self, i_eq):
        """
        Loads a Feynman problem based on its number in the set.
        Parameters
        ----------
        i_eq : int
            Equation number in the set of equations.
        """
        # Equation line in dataframe
        self.eq_df = EQS_FEYNMAN_DF.iloc[i_eq]

        # Number of input variables
        self.n_vars = int(self.eq_df["# variables"])  # (,)

        # Name of output variable
        self.y_name = self.eq_df["Output"]       # (,)
        # Units of output variables
        out_units = get_units(self.y_name)  # (max_units,)


        # Id of input variables v1_name, v2_name etc.
        var_names_str = np.array( [ "v%i"%(i_var) + "_name" for i_var in range(1, n_vars+1) ]          ).astype(str)   # (n_vars,)
        # Names of input variables
        var_names = np.array( [ eq_df[ var_names_str[i] ] for i_var in range(1, n_vars+1) ]            ).astype(str)   # (n_vars,)
        # Lowest values taken by input variables
        var_lows  = np.array( [ eq_df[ var_names_str[i]  ] for i_var in range(1, n_vars+1) ]           ).astype(float) # (n_vars,)
        # Highest values taken by input variables
        var_highs = np.array( [ eq_df[ var_names_str[i] ] for i_var in range(1, n_vars+1) ]            ).astype(float) # (n_vars,)
        # Units of input variables
        var_units = np.array( [ get_units(eq_df[ var_names_str[i] ]) for i_var in range(1, n_vars+1) ] ).astype(float) # (n_vars, max_units,)

dict_for_feynman_formula_var_names = {var_names[i_var]:"X[%i]"%(i_var) for i_var in range(n_vars)}

formula = replace_symbols_in_formula(eq_df["Formula"], dict_for_feynman_formula_var_names)

X = np.stack([np.random.uniform(var_lows[i_var], var_highs[i_var], 10) for i_var in range(n_vars)])
