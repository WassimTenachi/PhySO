import pandas as pd
import numpy as np
import os
import pathlib
import sympy
import matplotlib.pyplot as plt

# Dataset paths
PARENT_FOLER = pathlib.Path(__file__).parents[0]
PATH_FEYNMAN_EQS_CSV = PARENT_FOLER / "FeynmanEquations.csv"
PATH_UNITS_CSV       = PARENT_FOLER / "units.csv"


# ---------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- LOADING CSVs  ---------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

def load_feynman_equations_csv (filepath = "FeynmanEquations.csv"):
    """
    Loads FeynmanEquations.csv into a clean pd.DataFrame (corrects typos).
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
    # Set types for int columns
    eqs_feynman_df = eqs_feynman_df.astype({'Number': int, '# variables': int})

    # ---- Correcting typos in the file ----
    # Equation II.37.1 takes 3 arguments not 6
    eqs_feynman_df.loc[eqs_feynman_df["Filename"] == "II.37.1",   "# variables"] = 3
    # Equation I.18.12 takes 3 arguments not 2
    eqs_feynman_df.loc[eqs_feynman_df["Filename"] == "I.18.12",   "# variables"] = 3
    # Equation I.18.14 takes 4 arguments not 3
    eqs_feynman_df.loc[eqs_feynman_df["Filename"] == "I.18.14",   "# variables"] = 4
    # Equation III.10.19 takes 4 arguments not 3
    eqs_feynman_df.loc[eqs_feynman_df["Filename"] == "III.10.19", "# variables"] = 4
    # Equation I.38.12 takes 4 arguments not 3
    eqs_feynman_df.loc[eqs_feynman_df["Filename"] == "I.38.12",   "# variables"] = 4
    # Equation III.19.51 takes 5 arguments not 4
    eqs_feynman_df.loc[eqs_feynman_df["Filename"] == "III.19.51", "# variables"] = 5

    # ---- Verifying number of variables for safety ----
    # Checking the number of variables declared in the file for each problem
    # Expected number of variables for each problem
    expected_n_vars = (~eqs_feynman_df[["v%i_name" % (i) for i in range(1, 11)]].isnull().to_numpy()).sum(axis=1)       # (N_EQS,)
    # Declared number of variables for each problem
    n_vars = eqs_feynman_df["# variables"].to_numpy()                                                                   # (N_EQS,)
    # Is nb of declared variable consistent with variables columns ?
    is_consistent = np.equal(expected_n_vars, n_vars)                                                                   # (N_EQS,)
    assert is_consistent.all(), "Nb. of filled variables columns not consistent with declared nb. of variables for " \
                                "problems:\n %s"%(str(eqs_feynman_df.loc[~is_consistent]))
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
    assert not pd.isnull(var_name), "Can not get the units of %s as it is a null."%(var_name)
    try:
        units = UNITS_DF[UNITS_DF["Variable"] == var_name].to_numpy()[0][2:].astype(float)
    except:
        raise IndexError("Could not load units of %s"%(var_name))
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
    dict_fml = dict_for_feynman_formula_var_names
    for symbol in dict_fml.keys():
        new_symbol = dict_fml[symbol]
        formula = formula.replace(symbol, new_symbol)
    dict_fml = DICT_FOR_FEYNMAN_FORMULA_FUNC_TO_NP
    for symbol in dict_fml.keys():
        new_symbol = dict_fml[symbol]
        formula = formula.replace(symbol, new_symbol)
    return formula


# ---------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- FEYNMAN PROBLEM  --------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

class FeynmanProblem:
    """
    Represents a single Feynman benchmark problem.
    (See https://arxiv.org/abs/1905.11481 and https://space.mit.edu/home/tegmark/aifeynman.html for details).
    Attributes
    ----------
    i_eq : int
        Equation number in the set of equations (e.g. 1 to 100).
    eq_name : str
        Equation name in the set of equations (e.g. I.6.2a).
    n_vars : int
        Number of input variables.

    y_name : str
        Name of output variable.
    y_units : array_like of shape (FEYN_UNITS_VECTOR_SIZE,) of floats
        Units of output variables.

    X_names : array_like of shape (n_vars,) of str
        Names of input variables.
    X_lows : array_like of shape (n_vars,) of floats
        Lowest values taken by input variables.
    X_highs : array_like of shape (n_vars,) of floats
        Highest values taken by input variables.
    X_units :  array_like of shape (n_vars, FEYN_UNITS_VECTOR_SIZE,) of floats
        Units of input variables.

    formula_display : str
        Formula as in the Feynman dataset.
    X_sympy_symbols : array_like of shape (n_vars,) of sympy.Symbol
        Sympy symbols representing each input variables.
    formula_sympy : sympy expression
        Formula in sympy
    """

    def __init__(self, i_eq = None, eq_name = None):
        """
        Loads a Feynman problem based on its number in the set or its equation name
        Parameters
        ----------
        i_eq : int
            Equation number in the set of equations (e.g. 1 to 100).
        eq_name : str
            Equation name in the set of equations (e.g. I.6.2a).
        """
        # Select equation line in dataframe
        if i_eq is not None:
            self.eq_df  = EQS_FEYNMAN_DF[EQS_FEYNMAN_DF ["Number"]  == i_eq    ].iloc[0] # pandas.core.frame.DataFrame
        elif eq_name is not None:
            self.eq_df = EQS_FEYNMAN_DF[EQS_FEYNMAN_DF ["Filename"] == eq_name ].iloc[0] # pandas.core.frame.DataFrame
        else:
            raise ValueError("At least one of equation number (i_eq) or equation name (eq_name) should be specified to select a Feynman problem.")

        # Equation number (eg. 1 to 100)
        self.i_eq = self.eq_df["Number"]                # int
        # Code name of equation (eg. 'I.6.2a')
        self.eq_name = self.eq_df["Filename"]           # str
        # Number of input variables
        self.n_vars = int(self.eq_df["# variables"])    # int

        # ----------- y : output variable -----------
        # Name of output variable
        self.y_name   = self.eq_df["Output"]            # str
        # Units of output variables
        self.y_units = get_units(self.y_name)           # (FEYN_UNITS_VECTOR_SIZE,)

        # ----------- X : input variables -----------
        # Utils id of input variables v1, v2 etc. in .csv
        var_ids_str = np.array( [ "v%i"%(i_var) for i_var in range(1, self.n_vars+1) ]   ).astype(str)            # (n_vars,)
        # Names of input variables
        self.X_names = np.array( [ self.eq_df[ id + "_name" ] for id in var_ids_str  ]   ).astype(str)            # (n_vars,)
        # Lowest values taken by input variables
        self.X_lows  = np.array( [ self.eq_df[ id + "_low"  ] for id in var_ids_str ]    ).astype(float)          # (n_vars,)
        # Highest values taken by input variables
        self.X_highs = np.array( [ self.eq_df[ id + "_high" ] for id in var_ids_str  ]   ).astype(float)          # (n_vars,)
        # Units of input variables
        self.X_units = np.array( [ get_units(self.eq_df[ id + "_name" ]) for id in var_ids_str ] ).astype(float)  # (n_vars, FEYN_UNITS_VECTOR_SIZE,)

        # ----------- Formula : eval utils -----------
        self.formula_display = self.eq_df["Formula"] # (str)
        dict_for_feynman_formula_var_names = {self.X_names[i_var]:"X[%i]"%(i_var) for i_var in range(self.n_vars)}
        self.formula = replace_symbols_in_formula(self.formula_display, dict_for_feynman_formula_var_names)

        # ----------- Formula : sympy utils -----------
        # Input variables as sympy symbols
        self.X_sympy_symbols = []
        for i in range(self.n_vars):
            self.X_sympy_symbols.append(sympy.Symbol(self.X_names[i],                                                   #  (n_vars,)
                                                     # Useful assumptions for simplifying etc
                                                     real     = True,
                                                     positive = self.X_lows  [i] > 0,
                                                     negative = self.X_highs [i] < 0,
                                                     # If nonzero = False assumes that always = 0 which causes problems
                                                     # when simplifying
                                                     # nonzero  = not (self.X_lows[i] <= 0 and self.X_highs[i] >= 0),
                                                     domain   = sympy.sets.sets.Interval(self.X_lows[i], self.X_highs[i]),
                                                     ))
        # Input variables names to sympy symbols dict
        sympy_X_symbols_dict = {self.X_names[i] : self.X_sympy_symbols[i] for i in range(self.n_vars)}                  #  (n_vars,)
        # Declaring input variables via local_dict to avoid confusion
        # Eg. So sympy knows that we are referring to gamma as a variable and not the function etc.
        local_dict = sympy_X_symbols_dict
        # Avoids eg. sin(theta) = 0 when theta domain = [0,5] ie. nonzero=False
        evaluate = False
        self.formula_sympy   = sympy.parsing.sympy_parser.parse_expr(self.formula_display,
                                                                     local_dict = local_dict,
                                                                     evaluate   = evaluate)

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
        #todo: clean
        #y = eval(self.formula)

        # Getting sympy function
        f = sympy.lambdify(self.X_sympy_symbols, self.formula_sympy, "numpy")
        # Mapping between variables names and their data value
        mapping_var_name_to_X = {self.X_names[i]: X[i] for i in range(self.n_vars)}
        # Evaluation
        # Forcing float type so if some symbols are not evaluated as floats (eg. if some variables are not declared
        # properly in source file) resulting partly symbolic expressions will not be able to be converted to floats
        # and an error can be raised).
        # This is also useful for detecting issues such as sin(theta) = 0 because theta.is_nonzero = False -> the result
        # is just an int of float
        y = f(**mapping_var_name_to_X).astype(float)

        return y

    def generate_data_points (self, n_samples = 1_000_000):
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

    def show_sample(self, n_samples = 100):
        X_array, y_array = self.generate_data_points(n_samples = n_samples)
        n_dim = X_array.shape[0]
        fig, ax = plt.subplots(n_dim, 1, figsize=(10, 5))
        for i in range(n_dim):
            curr_ax = ax if n_dim == 1 else ax[i]
            curr_ax.plot(X_array[i], y_array, 'k.', )
            curr_ax.set_xlabel("X[%i]" % (i))
            curr_ax.set_ylabel("y")
        plt.show()

    def __str__(self):
        return "FeynmanProblem : eq %s\n%s"%(self.eq_name, str(self.formula_sympy))

    def __repr__(self):
        return str(self)