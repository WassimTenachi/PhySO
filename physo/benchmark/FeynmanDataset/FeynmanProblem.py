import pandas as pd
import numpy as np
import pathlib
import sympy
import matplotlib.pyplot as plt

# Internal imports
from physo.benchmark.utils import symbolic_utils as su

# Dataset paths
PARENT_FOLDER = pathlib.Path(__file__).parents[0]
PATH_FEYNMAN_EQS_CSV       = PARENT_FOLDER / "FeynmanEquations.csv"
PATH_FEYNMAN_EQS_BONUS_CSV = PARENT_FOLDER / "BonusEquations.csv"
PATH_UNITS_CSV             = PARENT_FOLDER / "units.csv"


# ---------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- LOADING CSVs  ---------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

def load_feynman_bulk_equations_csv (filepath_eqs ="FeynmanEquations.csv"):
    """
    Loads FeynmanEquations.csv into a clean pd.DataFrame (corrects typos).
    Source file can be found here: https://space.mit.edu/home/tegmark/aifeynman.html
    Parameters
    ----------
    filepath_eqs : str
        Path to FeynmanEquations.csv.
    Returns
    -------
    eqs_feynman_df : pd.DataFrame
    """
    eqs_feynman_df = pd.read_csv(filepath_eqs, sep=",")
    # drop last row(s) of NaNs
    eqs_feynman_df = eqs_feynman_df[~eqs_feynman_df[eqs_feynman_df.columns[0]].isnull()]
    # Set types for int columns
    eqs_feynman_df = eqs_feynman_df.astype({'Number': int, '# variables': int})
    # Number of equations
    n_eqs = len(eqs_feynman_df)

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
    expected_n_vars = (~eqs_feynman_df[["v%i_name" % (i) for i in range(1, 11)]].isnull().to_numpy()).sum(axis=1)       # (n_eqs,)
    # Declared number of variables for each problem
    n_vars = eqs_feynman_df["# variables"].to_numpy()                                                                   # (n_eqs,)
    # Is nb of declared variable consistent with variables columns ?
    is_consistent = np.equal(expected_n_vars, n_vars)                                                                   # (n_eqs,)
    assert is_consistent.all(), "Nb. of filled variables columns not consistent with declared nb. of variables for " \
                                "problems:\n %s"%(str(eqs_feynman_df.loc[~is_consistent]))


    # ---- Making bulk and bonus datasets consistent ----

    # Input variable related columns names: 'v1_name', 'v1_low', 'v1_high', 'v2_name' etc.
    variables_columns_names = np.array([['v%i_name'%(i), 'v%i_low'%(i), 'v%i_high'%(i)] for i in range (1,11)]).flatten()
    # Essential equations related columns names: 'Output', 'Formula', '# variables', 'v1_name', 'v1_low', etc.
    essential_columns_names = ['Output', 'Formula', '# variables'] + variables_columns_names.tolist()

    # Adding columns
    # Adding set columns indicating from which file these equations come from (bulk file or bonus file)
    eqs_feynman_df["Set"] = "bulk"
    # Adding equation names as a column (I.6.2a etc.)
    eqs_feynman_df["Name"] = eqs_feynman_df["Filename"]

    # Columns to keep: 'Filename', 'Name', 'Set', 'Number', 'Output', 'Formula', '# variables', 'v1_name', 'v1_low',etc.
    columns_to_keep_names = ['Filename', 'Name', 'Set', 'Number'] + essential_columns_names
    # Selecting
    eqs_feynman_df = eqs_feynman_df[columns_to_keep_names]

    return eqs_feynman_df


def load_feynman_bonus_equations_csv (filepath_eqs_bonus = "BonusEquations.csv"):
    """
    Loads BonusEquations.csv into a clean pd.DataFrame (corrects typos).
    Source file can be found here: https://space.mit.edu/home/tegmark/aifeynman.html
    Parameters
    ----------
    filepath_eqs_bonus : str
        Path to BonusEquations.csv.
    Returns
    -------
    eqs_feynman_df : pd.DataFrame
    """
    eqs_feynman_df = pd.read_csv(filepath_eqs_bonus, sep=",")
    # drop last row(s) of NaNs
    eqs_feynman_df = eqs_feynman_df[~eqs_feynman_df[eqs_feynman_df.columns[0]].isnull()]
    # Set types for int columns
    eqs_feynman_df = eqs_feynman_df.astype({'Number': int, '# variables': int})
    # Number of equations
    n_eqs = len(eqs_feynman_df)

    # ---- Correcting typos in the file ----
    # Equation test_12 takes 5 arguments not 4
    eqs_feynman_df.loc[eqs_feynman_df["Filename"] == "test_12",   "# variables"] = 5
    # Equation test_13 takes 5 arguments not 4
    eqs_feynman_df.loc[eqs_feynman_df["Filename"] == "test_13",   "# variables"] = 5
    # Equation test_18 takes 5 arguments not 4
    eqs_feynman_df.loc[eqs_feynman_df["Filename"] == "test_18",   "# variables"] = 5
    # Equation test_19 takes 6 arguments not 5
    eqs_feynman_df.loc[eqs_feynman_df["Filename"] == "test_19",   "# variables"] = 6

    # ---- Verifying number of variables for safety ----
    # Checking the number of variables declared in the file for each problem
    # Expected number of variables for each problem
    expected_n_vars = (~eqs_feynman_df[["v%i_name" % (i) for i in range(1, 11)]].isnull().to_numpy()).sum(axis=1)       # (n_eqs,)
    # Declared number of variables for each problem
    n_vars = eqs_feynman_df["# variables"].to_numpy()                                                                   # (n_eqs,)
    # Is nb of declared variable consistent with variables columns ?
    is_consistent = np.equal(expected_n_vars, n_vars)                                                                   # (n_eqs,)
    assert is_consistent.all(), "Nb. of filled variables columns not consistent with declared nb. of variables for " \
                                "problems:\n %s"%(str(eqs_feynman_df.loc[~is_consistent]))

    # ---- Making bulk and bonus datasets consistent ----

    # Input variable related columns names: 'v1_name', 'v1_low', 'v1_high', 'v2_name' etc.
    variables_columns_names = np.array([['v%i_name'%(i), 'v%i_low'%(i), 'v%i_high'%(i)] for i in range (1,11)]).flatten()
    # Essential equations related columns names: 'Output', 'Formula', '# variables', 'v1_name', 'v1_low', etc.
    essential_columns_names = ['Output', 'Formula', '# variables'] + variables_columns_names.tolist()

    # Adding columns
    # Adding set columns indicating from which file these equations come from (bulk file or bonus file)
    eqs_feynman_df["Set"] = "bonus"

    # Columns to keep: 'Filename', 'Name', 'Set', 'Number', 'Output', 'Formula', '# variables', 'v1_name', 'v1_low',etc.
    columns_to_keep_names = ['Filename', 'Name', 'Set', 'Number'] + essential_columns_names
    # Selecting
    eqs_feynman_df = eqs_feynman_df[columns_to_keep_names]

    return eqs_feynman_df


def load_feynman_all_equations_csv (filepath_eqs ="FeynmanEquations.csv", filepath_eqs_bonus = "BonusEquations.csv"):
    """
    Loads FeynmanEquations.csv and BonusEquations.csv into a clean pd.DataFrame (corrects typos).
    Source files can be found here: https://space.mit.edu/home/tegmark/aifeynman.html
    Parameters
    ----------
    filepath_eqs : str
        Path to FeynmanEquations.csv.
    filepath_eqs_bonus : str
        Path to BonusEquations.csv.
    Returns
    -------
    eqs_feynman_df : pd.DataFrame
    """
    bulk_eqs_feynman_df  = load_feynman_bulk_equations_csv  (filepath_eqs       = filepath_eqs       )
    bonus_eqs_feynman_df = load_feynman_bonus_equations_csv (filepath_eqs_bonus = filepath_eqs_bonus )

    eqs_feynman_df = pd.concat((bulk_eqs_feynman_df, bonus_eqs_feynman_df),
                                # True so to get index going from 0 to 119 instead of 0 to 99 and then 0 to 19
                               ignore_index=True)
    return eqs_feynman_df


def load_feynman_units_csv (filepath = "units.csv"):
    """
    Loads units.csv into a clean pd.DataFrame and corrects known errors.
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

    # ---- Correcting errors in the file ----
    # Variable mu_drift should have units s.kg-1 not s-1.kg
    # This variable only appears in eq. I.43.16 : velocity = mu_drift.force = mu_drift.(q.E/d)
    # Hence m.s-1 = (s.kg-1).(m.s-2.kg) i.e. [mu_drift] = s.kg-1
    # (see https://www.feynmanlectures.caltech.edu/I_43.html)
    # NB: charge [q] = A.s = (m2.s-3.kg.V-1).s = m2.s-2.kg.V-1 so this is fine.
    units_df.loc[units_df["Variable"] == "mu_drift", ["s", "kg"]] = [1, -1]
    return units_df

EQS_FEYNMAN_DF = load_feynman_all_equations_csv (filepath_eqs       = PATH_FEYNMAN_EQS_CSV,
                                                 filepath_eqs_bonus = PATH_FEYNMAN_EQS_BONUS_CSV,
                                                 )
UNITS_DF       = load_feynman_units_csv     (filepath           = PATH_UNITS_CSV)

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
# ------------------------------------------------- FEYNMAN PROBLEM  --------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
CONST_LOCAL_DICT = {"pi" : np.pi}

class FeynmanProblem:
    """
    Represents a single Feynman benchmark problem.
    (See https://arxiv.org/abs/1905.11481 and https://space.mit.edu/home/tegmark/aifeynman.html for details).
    Attributes
    ----------
    i_eq : int
        Equation number in the set of equations (e.g. 0 to 99 for bulk eqs and 100 to 119 for bonus eqs).
    eq_name : str
        Equation name in the set of equations (e.g. I.6.2a).
    n_vars : int
        Number of input variables.
    eq_df : pandas.core.series.Series
        Underlying pandas dataframe line of this equation.
    original_var_names : bool
        Using original variable names (e.g. theta, sigma etc.) and original output variable name (e.g. f, E etc.) if
        True, using x0, x1 ... as input variable names and y as output variable name otherwise.

    y_name_original : str
        Name of output variable as in the Feynman dataset.
    y_name : str
        Name of output variable.
    y_units : array_like of shape (FEYN_UNITS_VECTOR_SIZE,) of floats
        Units of output variables.

    X_names_original : array_like of shape (n_vars,) of str
        Names of input variables as in the Feynman dataset.
    X_names : array_like of shape (n_vars,) of str
        Names of input variables.
    X_lows : array_like of shape (n_vars,) of floats
        Lowest values taken by input variables.
    X_highs : array_like of shape (n_vars,) of floats
        Highest values taken by input variables.
    X_units :  array_like of shape (n_vars, FEYN_UNITS_VECTOR_SIZE,) of floats
        Units of input variables.

    formula_original : str
        Formula as in the Feynman dataset.
    X_sympy_symbols : array_like of shape (n_vars,) of sympy.Symbol
        Sympy symbols representing each input variables with assumptions (negative, positive etc.).
    sympy_X_symbols_dict : dict of {str : sympy.Symbol}
        Input variables names to sympy symbols (w assumptions), can be passed to sympy.parsing.sympy_parser.parse_expr
        as local_dict.
    local_dict : dict of {str : sympy.Symbol or float}
        Input variables names to sympy symbols (w assumptions) and constants (eg. pi : np.pi etc.), can be passed to
        sympy.parsing.sympy_parser.parse_expr as local_dict.
    formula_sympy : sympy expression
        Formula in sympy.
    formula_sympy_eval : sympy expression
        Formula in sympy with evaluated fixed constants (eg. pi -> 3.14... etc).
    formula_latex : str
        Formula in latex.
    """

    def __init__(self, i_eq = None, eq_name = None, original_var_names = False):
        """
        Loads a Feynman problem based on its number in the set or its equation name
        Parameters
        ----------
        i_eq : int
            Equation number in the whole set of equations (0 to 99 for bulk eqs and 100 to 119 for bonus eqs).
        eq_name : str
            Equation name in the set of equations (e.g. I.6.2a).
        original_var_names : bool
            Using original variable names (e.g. theta, sigma etc.) and original output variable name (e.g. f, E etc.) if
            True, using x0, x1 ... as input variable names and y as output variable name otherwise.
        """
        # Selecting equation line in dataframe
        if i_eq is not None:
            self.eq_df  = EQS_FEYNMAN_DF.iloc[i_eq]                                     # pandas.core.series.Series
        elif eq_name is not None:
            self.eq_df = EQS_FEYNMAN_DF[EQS_FEYNMAN_DF ["Name"] == eq_name ].iloc[0]    # pandas.core.series.Series
        else:
            raise ValueError("At least one of equation number (i_eq) or equation name (eq_name) should be specified to select a Feynman problem.")

        # Equation number (0 to 99 for bulk eqs and 100 to 119 for bonus eqs)
        self.i_eq = i_eq                                                     # int
        # Equation number in individual datasets (1 to 100 for bulk eqs and 1 to 20 for bonus eqs)
        self.i_eq_feyn = self.eq_df["Number"]                                # int
        # Code name of equation (eg. 'I.6.2a')
        self.eq_name = self.eq_df["Name"]                                    # str
        # Filename column in the Feynman dataset
        self.eq_filename = self.eq_df["Filename"]                            # str
        # SRBench style name
        self.SRBench_name = "feynman_" + self.eq_filename.replace('.', '_')  # str
        # Number of input variables
        self.n_vars = int(self.eq_df["# variables"])                         # int
        # Using x0, x1 ... and y names or original names (e.g. theta, sigma, f etc.)
        self.original_var_names = original_var_names                         # bool

        # ----------- y : output variable -----------
        # Name of output variable
        self.y_name_original = self.eq_df["Output"]                              # str
        # Name of output variable : y or original name (eg. f, E etc.)
        self.y_name = self.y_name_original if self.original_var_names else 'y'   # str
        # Units of output variables
        self.y_units = get_units(self.y_name)                                    # (FEYN_UNITS_VECTOR_SIZE,)

        # ----------- X : input variables -----------
        # Utils id of input variables v1, v2 etc. in .csv
        var_ids_str = np.array( [ "v%i"%(i_var) for i_var in range(1, self.n_vars+1) ]   ).astype(str)                     # (n_vars,)
        # Names of input variables
        self.X_names_original = np.array( [ self.eq_df[ id + "_name" ] for id in var_ids_str  ]   ).astype(str)            # (n_vars,)
        X_names_xi_style      = np.array( [ "x%i"%(i_var) for i_var in range(self.n_vars)     ]   ).astype(str)            # (n_vars,)
        self.X_names          = self.X_names_original if self.original_var_names else X_names_xi_style                     # (n_vars,)
        # Lowest values taken by input variables
        self.X_lows           = np.array( [ self.eq_df[ id + "_low"  ] for id in var_ids_str ]    ).astype(float)          # (n_vars,)
        # Highest values taken by input variables
        self.X_highs          = np.array( [ self.eq_df[ id + "_high" ] for id in var_ids_str  ]   ).astype(float)          # (n_vars,)
        # Units of input variables
        self.X_units          = np.array( [ get_units(self.eq_df[ id + "_name" ]) for id in var_ids_str ] ).astype(float)  # (n_vars, FEYN_UNITS_VECTOR_SIZE,)

        # ----------- Formula -----------
        self.formula_original = self.eq_df["Formula"] # (str)

        # Input variables as sympy symbols
        self.X_sympy_symbols = []
        for i in range(self.n_vars):
            is_positive = self.X_lows  [i] > 0
            is_negative = self.X_highs [i] < 0
            # If 0 is in interval, do not give assumptions as otherwise sympy will assume 0
            if (not is_positive) and (not is_negative):
                is_positive, is_negative = None, None
            self.X_sympy_symbols.append(sympy.Symbol(self.X_names[i],                                                   #  (n_vars,)
                                                     # Useful assumptions for simplifying etc
                                                     real     = True,
                                                     positive = is_positive,
                                                     negative = is_negative,
                                                     # If nonzero = False assumes that always = 0 which causes problems
                                                     # when simplifying
                                                     # nonzero  = not (self.X_lows[i] <= 0 and self.X_highs[i] >= 0),
                                                     domain   = sympy.sets.sets.Interval(self.X_lows[i], self.X_highs[i]),
                                                     ))

        # Input variables names to sympy symbols dict
        self.sympy_X_symbols_dict = {self.X_names[i] : self.X_sympy_symbols[i] for i in range(self.n_vars)}                     #  (n_vars,)
        # Dict to use to read original feynman dataset formula
        # Original names to symbols in usage (i.e. symbols having original names or not)
        # eg. 'theta' -> theta symbol etc. (if original_var_names=True) or 'theta' -> x0 symbol etc. (else)
        self.sympy_original_to_X_symbols_dict = {self.X_names_original[i] : self.X_sympy_symbols[i] for i in range(self.n_vars)} #  (n_vars,)
        # NB: if original_var_names=True, then self.sympy_X_symbols_dict = self.sympy_original_to_X_symbols_dict

        # Declaring input variables via local_dict to avoid confusion
        # Eg. So sympy knows that we are referring to gamma as a variable and not the function etc.
        # evaluate = False avoids eg. sin(theta) = 0 when theta domain = [0,5] ie. nonzero=False, but no need for this
        # if nonzero assumption is not used
        evaluate = False
        self.formula_sympy   = sympy.parsing.sympy_parser.parse_expr(self.formula_original,
                                                                     local_dict = self.sympy_original_to_X_symbols_dict,
                                                                     evaluate   = evaluate)
        # Local dict : dict of input variables (sympy_original_to_X_symbols_dict) and fixed constants (pi -> 3.14.. etc)
        self.local_dict = {}
        self.local_dict.update(self.sympy_original_to_X_symbols_dict)
        self.local_dict.update(CONST_LOCAL_DICT)
        self.formula_sympy_eval = sympy.parsing.sympy_parser.parse_expr(self.formula_original,
                                                                     local_dict = self.local_dict,
                                                                     evaluate   = evaluate)
        # Latex formula
        self.formula_latex   = sympy.printing.latex(self.formula_sympy)
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
            Number of samples to draw. By default, 1e6  as this is the number of data points for each problem in the
            files in https://space.mit.edu/home/tegmark/aifeynman.html
            Note that SRBench https://arxiv.org/abs/2107.14351 uses 1e5.
        Returns
        -------
        X, y : numpy.array of shape (n_vars, ?,) of floats, numpy.array of shape (?,) of floats
        """
        X = np.stack([np.random.uniform(self.X_lows[i_var], self.X_highs[i_var], n_samples) for i_var in range(self.n_vars)])
        y = self.target_function(X)
        return X,y

    def show_sample(self, n_samples = 100, do_show = True, save_path = None):
        X_array, y_array = self.generate_data_points(n_samples = n_samples)
        n_dim = X_array.shape[0]
        fig, ax = plt.subplots(n_dim, 1, figsize=(10, n_dim * 4))
        fig.suptitle(self.formula_original)
        for i in range(n_dim):
            curr_ax = ax if n_dim == 1 else ax[i]
            curr_ax.plot(X_array[i], y_array, 'k.', markersize=1.)
            curr_ax.set_xlabel("%s : %s" % (self.X_names[i], self.X_units[i]))
            curr_ax.set_ylabel("%s : %s" % (self.y_name    , self.y_units))
        if save_path is not None:
            fig.savefig(save_path)
        if do_show:
            plt.show()

    def compare_expression (self, trial_expr,
                            handle_trigo            = True,
                            prevent_zero_frac       = True,
                            prevent_inf_equivalence = True,
                            verbose=False):
        """
        Checks if trial_expr is symbolically equivalent to the target expression of this Feynman problem, following a
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

        # Cleaning target expression like SRBench
        target_expr = su.clean_sympy_expr(self.formula_sympy_eval)

        is_equivalent, report = su.compare_expression (trial_expr  = trial_expr,
                                                       target_expr = target_expr,
                                                       handle_trigo            = handle_trigo,
                                                       prevent_zero_frac       = prevent_zero_frac,
                                                       prevent_inf_equivalence = prevent_inf_equivalence,
                                                       verbose                 = verbose,)

        return is_equivalent, report

    def trial_function (self, trial_expr, X):
        """
        Evaluates X on a trial expression mapping X to input variables names in sympy.
        Parameters
        ----------
        trial_expr : Sympy Expression
            Trial sympy expression with evaluated numeric free constants and assumptions regarding variables
            (positivity etc.) encoded in expression.
        X : numpy.array of shape (n_vars, ?,) of floats
        Returns
        -------
        y : numpy.array of shape (?,) of floats
        """
        # Getting sympy function
        f = sympy.lambdify(self.X_sympy_symbols, trial_expr, "numpy")
        # Mapping between variables names and their data value
        mapping_var_name_to_X = {self.X_names[i]: X[i] for i in range(self.n_vars)}
        # Evaluation
        # Forcing float type so if some symbols are not evaluated as floats (eg. if some variables are not declared
        # properly in source file) resulting partly symbolic expressions will not be able to be converted to floats
        # and an error can be raised).
        # This is also useful for detecting issues such as sin(theta) = 0 because theta.is_nonzero = False -> the result
        # is just an int of float
        y = f(**mapping_var_name_to_X)
        # forcing float type only if result is not already a single float (can happen if expression is a constant)
        if not isinstance(y, float):
            y = y.astype(float)
        return y

    def __str__(self):
        return "FeynmanProblem : %s : %s\n%s"%(self.eq_filename, self.eq_name, str(self.formula_sympy))

    def __repr__(self):
        return str(self)