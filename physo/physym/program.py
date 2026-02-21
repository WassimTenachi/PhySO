import warnings as warnings
import numpy as np
import copy as copy  # for Cursor
import sympy as sympy
import pickle
import shutil

# For tree image (optional)
import matplotlib.pyplot as plt
import io

# Internal imports
from physo.physym import token as Tok
from physo.physym import execute as Exec
from physo.physym import free_const

# Latex usage flag
FLAG_USE_LATEX_RENDERING = True

def latex_display():
    is_available = True
    issues       = []

    # Check shutil
    if shutil.which('latex') is None:
        is_available = False
        msg = "shutil.which('latex') returned None"
        issues.append(msg)

    # Check flag
    if not FLAG_USE_LATEX_RENDERING:
        is_available = False
        msg = "physo.FLAG_USE_LATEX_RENDERING is set to False"
        issues.append(msg)

    # Try to use latex
    if is_available:
        try:
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
        except Exception as e:
            is_available = False
            msg = "plt.rc('text', usetex=True) failed with error: %s" % str(e)
            issues.append(msg)

    # If not available, warn the user
    else:
        msg = "Latex display is not available. Issues: %s" % ", ".join(issues)
        warnings.warn(msg)

    return is_available

# Using latex for display if available
using_tex = latex_display()

# Font size
plt.rc('font', size=16)


# Pickable default identity wrapper
def DEFAULT_WRAPPER (func, X):
        return func(X)

# Load pickled program
def load_program_pkl(fpath):
    """
    Loads program from pickle file.
    """
    with open(fpath, 'rb') as f:
        prog = pickle.load(f)
    return prog

class Cursor:
    """
    Helper class for single-token navigation in tree of programs in VectPrograms.
    Represents the position of a single token in a program in a batch of programs.
    For user-exploration, program testing and debugging.
    Attributes
    ----------
    programs : vect_programs.VectPrograms
        Batch of programs to explore.
    prog_idx : int
        Initial position of cursor in batch dim (= index of program in batch).
    pos : int
        Initial position of cursor in time dim (= index of token in program).
    Methods
    -------
    coords () -> numpy.array of shape (2, 1) of int
        Returns current coordinates in batch (batch dim, time dim) compatible with VectPrograms methods.
    set_pos (new_pos : int) -> program.Cursor
        Sets position of cursor in time dim (= index of token in program) and returns cursor.
    child   (i_child   : int) -> program.Cursor
        Returns a cursor pointing to child number i_child of current token. Raises error if there is no child.
    sibling (i_sibling : int) -> program.Cursor
        Returns a cursor pointing to sibling number i_sibling of current token. Raises error if there is no sibling .
        cursor.
    parent () -> program.Cursor
        Returns a cursor pointing to parent of current token. Raises error if there is no parent.
    """
    def __init__(self, programs, prog_idx=0, pos=0):
        """
        See class documentation.
        Parameters
        ----------
        programs : vect_programs.VectPrograms
        prog_idx : int
        pos : int
        """
        self.programs = programs
        self.prog_idx = prog_idx
        self.pos      = pos

    @property
    def coords(self):
        """
        See class documentation.
        Returns
        -------
        coords : numpy.array of shape (2, 1) of int
        """
        return np.array([[self.prog_idx], [self.pos]])

    @property
    def token(self):
        """
        Returns token object at coords pointed by cursor.
        Returns
        -------
        token : token.Token
        """
        return self.programs.get_token(self.coords)[0]

    def token_prop (self, attr):
        """
        Returns attr attribute in VectPrograms of the token at coords pointed by cursor.
        Returns
        -------
        token_prop : ?
            ? depends on the property.
        """
        return getattr(self.programs.tokens, attr)[tuple(self.coords)][0]

    def set_pos(self, new_pos = 0):
        """
        See class documentation.
        Parameters
        ----------
        new_pos : int
        Returns
        -------
        self : program.Cursor
        """
        self.pos = new_pos
        return self

    def child(self, i_child = 0):
        """
        See class documentation.
        Parameters
        ----------
        i_child : int
        Returns
        -------
        self : program.Cursor
        """
        has_relative     = self.programs.tokens.has_children_mask[tuple(self.coords)][0]
        if not has_relative:
            err_msg = "Unable to navigate to child, Token %s at pos = %i (program %i) has no child." % (
            str(self), self.pos, self.prog_idx)
            raise IndexError(err_msg)
        pos_children    = self.programs.get_children(tuple(self.coords))[1:, 0]
        child = copy.deepcopy(self)
        child.pos        = pos_children[i_child]
        return child

    @property
    def sibling(self, i_sibling = 0):
        """
        See class documentation.
        Parameters
        ----------
        i_sibling : int
        Returns
        -------
        self : program.Cursor
        """
        has_relative = self.programs.tokens.has_siblings_mask[tuple(self.coords)][0]
        if not has_relative:
            err_msg = "Unable to navigate to sibling, Token %s at pos = %i (program %i) has no sibling." % (
                str(self), self.pos, self.prog_idx)
            raise IndexError(err_msg)
        pos_siblings = self.programs.get_siblings(tuple(self.coords))[1:, 0]
        sibling = copy.deepcopy(self)
        sibling.pos     = pos_siblings[i_sibling]
        return sibling

    @property
    def parent(self,):
        """
        See class documentation.
        Returns
        -------
        self : program.Cursor
        """
        has_relative = self.programs.tokens.has_parent_mask[tuple(self.coords)][0]
        if not has_relative:
            err_msg = "Unable to navigate to parent, Token %s at pos = %i (program %i) has no parent." % (
                str(self), self.pos, self.prog_idx)
            raise IndexError(err_msg)
        pos_parent = self.programs.get_parent(tuple(self.coords))[1, 0]
        parent = copy.deepcopy(self)
        parent.pos   = pos_parent
        return parent

    def __repr__(self):
        return self.programs.lib_names[self.programs.tokens.idx[tuple(self.coords)]][0]


class Program:
    """
    Interface class representing a single program.
    Attributes
    ----------
    tokens : array_like of token.Token
        Tokens making up program.
    size : int
        Size of program.
    library : library.Library
        Library of tokens that could appear in Program.
    is_physical : bool or None
        Is program physical (units-wize) ?
    free_consts : free_const.FreeConstantsTable
        Free constants register for this program (having batch_size = 1).
        Ie. shape = (1, n_class_free_const,), (1, n_spe_free_const, n_realizations,)
    candidate_wrapper : callable
        Wrapper to apply to candidate program's output, candidate_wrapper taking func, X as arguments where func is
        a candidate program callable (taking X as arg). By default = None, no wrapper is applied (identity).
    n_realizations : int
        Number of realizations for this program, ie. number of datasets this program has to fit.
        Dataset specific free constants will have different values different for each realization.
        If free_consts is given, n_realizations is taken from it and this argument is ignored.
    has_free_consts : bool
        Is there at least one free constant token appearing in the program ?
        If None, it is determined from tokens.
    """
    def __init__(self, tokens, library, free_consts = None, is_physical = None, candidate_wrapper = None, n_realizations=1, has_free_consts = None):
        """
        Parameters
        ----------
        See attributes help for details.
        """
        # Asserting that tokens make up a full tree representation, no more, no less
        total_arity = np.sum([tok.arity for tok in tokens])
        assert len(tokens)-total_arity==1, "Tokens making up Program must consist in a full tree representation " \
                                           "(length - total arities = 1), no more, no less"
        self.tokens       = tokens
        self.size         = len(tokens)
        self.library      = library
        self.is_physical  = is_physical

        if candidate_wrapper is None:
            candidate_wrapper = DEFAULT_WRAPPER
        self.candidate_wrapper = candidate_wrapper

        # ----- free const related -----
        # Determining if program contains free constants tokens
        if has_free_consts is None:
            is_token_free_const = np.array([t.is_class_free_const or t.is_spe_free_const for t in self.tokens]) # (n_tokens,)
            has_free_consts = np.any(is_token_free_const)  # True if at least one token is a free constant
        self.has_free_consts = has_free_consts

        # If no free constants table is given, let's create one and warn the user
        if free_consts is None:
            warnings.warn("No free constants table was given when initializing prog %s, a default one will be created with initial values."%(str(self)))
            free_consts = free_const.FreeConstantsTable(batch_size = 1, library = library, n_realizations = n_realizations)

        self.free_consts    = free_consts             # (1, n_class_free_const,), (1, n_spe_free_const, n_realizations,)
        # Taking n_realizations from free_consts so if free_consts is given, n_realizations is taken from it.
        self.n_realizations = free_consts.n_realizations

    def execute_wo_wrapper(self, X, i_realization = 0, n_samples_per_dataset = None):
        """
        Executes program on X.
        Parameters
        ----------
        X : torch.tensor of shape (n_dim, ?,) of float
            Values of the input variables of the problem with n_dim = nb of input variables, ? = number of samples.
        i_realization : int, optional
            Index of realization to use for dataset specific free constants (0 by default).
        n_samples_per_dataset : array_like of shape (n_realizations,) of int or None, optional
            Overrides i_realization if given. If given assumes that X contains multiple datasets with samples of each
            dataset following each other and each portion of X corresponding to a dataset should be treated with its
            corresponding dataset specific free constants values. n_samples_per_dataset is the number of samples for
            each dataset. Eg. [90, 100, 110] for 3 datasets, this will assume that the first 90 samples of X are for
            the first dataset, the next 100 for the second and the last 110 for the third.
        Returns
        -------
        y : torch.tensor of shape (?,) of float
            Result of computation.
        """
        # If n_samples_per_dataset is given, we need to flatten the free constants to match the number of samples
        # No need to flatten if there are no spe free constants -> this could be faster
        # One would probably not pass n_samples_per_dataset if there are no spe free constants but physo.SR will
        # as SR problems are treated as Class SR problems of one realization.
        if self.free_consts.n_spe_free_const == 0:
            class_vals = self.free_consts.class_values [0]                              # (n_class_free_const,)
            spe_vals   = None
        elif n_samples_per_dataset is not None :
            class_const_flatten, spe_const_flatten = self.free_consts.flatten_like_data(n_samples_per_dataset=n_samples_per_dataset)
            # class_const_flatten                                                       # (1, n_class_free_const, ?)
            # spe_const_flatten                                                         # (1, n_spe_free_const,   ?)
            class_vals = class_const_flatten [0]                                        # (n_class_free_const, ?)
            spe_vals   = spe_const_flatten   [0]                                        # (n_spe_free_const,   ?)
        else:
            # self.free_consts.class_values                                             # (1, n_class_free_const,)
            # self.free_consts.spe_values                                               # (1, n_spe_free_const, n_realizations,)
            class_vals = self.free_consts.class_values[0]                               # (n_class_free_const,)
            spe_vals   = self.free_consts.spe_values  [0,:,i_realization]               # (n_spe_free_const,)

        y = Exec.ExecuteProgram(input_var_data         = X,
                                program_tokens         = self.tokens,
                                class_free_consts_vals = class_vals,
                                spe_free_consts_vals   = spe_vals,
                                )
        return y

    def execute(self, X, i_realization = 0, n_samples_per_dataset = None):
        """
        Executes program on X.
        Parameters
        ----------
        X : torch.tensor of shape (n_dim, ?,) of float
            Values of the input variables of the problem with n_dim = nb of input variables, ? = number of samples.
        i_realization : int, optional
            Index of realization to use for dataset specific free constants (0 by default).
        n_samples_per_dataset : array_like of shape (n_realizations,) of int or None, optional
            Overrides i_realization if given. If given assumes that X contains multiple datasets with samples of each
            dataset following each other and each portion of X corresponding to a dataset should be treated with its
            corresponding dataset specific free constants values. n_samples_per_dataset is the number of samples for
            each dataset. Eg. [90, 100, 110] for 3 datasets, this will assume that the first 90 samples of X are for
            the first dataset, the next 100 for the second and the last 110 for the third.
        Returns
        -------
        y : torch.tensor of shape (?,) of float
            Result of computation.
        """
        y = self.candidate_wrapper(lambda X: self.execute_wo_wrapper(X=X, i_realization=i_realization, n_samples_per_dataset=n_samples_per_dataset), X)
        return y

    def optimize_constants(self, X, y_target, y_weights = 1., i_realization = 0, n_samples_per_dataset = None, args_opti = None, freeze_class_free_consts = False):
        """
        Optimizes free constants of program.
        If there are no free constant tokens in the program, does nothing and returns empty history.
        Parameters
        ----------
        X : torch.tensor of shape (n_dim, ?,) of float
            Values of the input variables of the problem with n_dim = nb of input variables, ? = number of samples.
        y_target : torch.tensor of shape (?,) of float
            Values of target output, ? = number of samples.
        y_weights : torch.tensor of shape (?,) of float, optional
            Weights for each data point.
        i_realization : int, optional
            Index of realization to use for dataset specific free constants (0 by default).
        n_samples_per_dataset : array_like of shape (n_realizations,) of int or None, optional
            Overrides i_realization if given. If given assumes that X/y_target contain multiple datasets with samples of
            each dataset following each other and each portion of X/y_target corresponding to a dataset should be treated
            with their corresponding dataset specific free constants values. n_samples_per_dataset is the number of
            samples for each dataset. Eg. [90, 100, 110] for 3 datasets, this will assume that the first 90 samples of X
            are for the first dataset, the next 100 for the second and the last 110 for the third.
        args_opti : dict or None, optional
            Arguments to pass to free_const.optimize_free_const. By default, free_const.DEFAULT_OPTI_ARGS
            arguments are used.
        freeze_class_free_consts : bool, optional
            If True, class free constants are not optimized.
        """
        if args_opti is None:
            args_opti = free_const.DEFAULT_OPTI_ARGS
        func_params = lambda params: self.__call__(X, i_realization=i_realization, n_samples_per_dataset=n_samples_per_dataset)

        if self.has_free_consts:
            if freeze_class_free_consts:
                history = free_const.optimize_free_const (  func      = func_params,
                                                            params    = [self.free_consts.spe_values],
                                                            y_target  = y_target,
                                                            y_weights = y_weights,
                                                            **args_opti)
            else:
                history = free_const.optimize_free_const (  func      = func_params,
                                                            params    = [self.free_consts.class_values, self.free_consts.spe_values],
                                                            y_target  = y_target,
                                                            y_weights = y_weights,
                                                            **args_opti)

        else:
            # If there are no free constants, we do not optimize anything
            history = []

        # Logging optimization process
        self.free_consts.is_opti    [0] = True
        self.free_consts.opti_steps [0] = len(history)  # Number of iterations it took to optimize the constants

        return history

    def save(self, fpath):
        """
        Saves program as a pickle file.
        """
        # Detach const data
        self.detach()
        # Save
        with open(fpath, 'wb') as f:
            pickle.dump(self, f)
        return None
    def detach(self):
        """
        Detaches program's free constants.
        """
        # Detach const data
        self.free_consts.detach()
        return self

    def make_skeleton (self):
        """
        Strips program to its bare minimum light pickable version for eg. parallel execution purposes.
        """
        # Exporting without library so it is lighter to pickle
        self.library             = None
        self.free_consts.library = None
        return None

    def __call__(self, X, i_realization = 0, n_samples_per_dataset=None):
        """
        Executes program on X. See Program.execute for details.
        """
        return self.execute(X = X, i_realization = i_realization, n_samples_per_dataset = n_samples_per_dataset)

    def __getitem__(self, key):
        """
        Returns token at position = key.
        """
        return self.tokens[key]

    def __repr__(self):
        return str(self.tokens)

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- REPRESENTATION : INFIX RELATED -----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def get_infix_str (self):
        """
        Computes infix str representation of a program.
        (which is the usual way to note symbolic function: +34 (in polish notation) = 3+4 (in infix notation))
        Returns
        -------
        program_str : str
        """
        program_str = Exec.ComputeInfixNotation(self.tokens)
        return program_str

    def get_sympy_local_dicts (self, replace_nan_with = 1.):
        """
        Returns a list of local dicts for each realization of the program to replace free constants by their values in
        sympy symbolic representation of the program.
        Parameters
        ----------
        replace_nan_with : float, optional
            Value to replace NaNs with in free constants values.
        Returns
        -------
        sympy_local_dicts : list of dict
        """
        sympy_local_dicts = []                                                      # (n_realizations,)
        for i_real in range(self.n_realizations):
            local_dict = {}
            class_local_dict = {str(self.library.class_free_constants_names[cid]): float(np.nan_to_num(
                self.free_consts.class_values[0][cid],
                nan=replace_nan_with))
                for cid in self.library.class_free_constants_ids}

            spe_local_dict = {str(self.library.spe_free_constants_names[cid]): float(np.nan_to_num(
                self.free_consts.spe_values[0][cid, i_real],
                nan=replace_nan_with))
                for cid in self.library.spe_free_constants_ids}
            local_dict.update(class_local_dict)
            local_dict.update(spe_local_dict)
            sympy_local_dicts.append(local_dict)
        return sympy_local_dicts                                                    # (n_realizations,)

    def get_infix_sympy (self, do_simplify = False, evaluate_consts = False, replace_nan_with = 1.):
        """
        Returns sympy symbolic representation of a program.
        Parameters
        ----------
        do_simplify : bool, optional
            If True performs a symbolic simplification of program.
        evaluate_consts : bool, optional
            If True replaces free constants by their values in the sympy symbolic representation of the program.
        replace_nan_with : float, optional
            Value to replace NaNs with in free constants values.
        Returns
        -------
        program_sympy : sympy.core or array of shape (n_realizations,) of sympy.core
            Sympy symbolic function. It is possible to run program_sympy.evalf(subs={'x': 2.4}) where 'x' is a variable
            appearing in the program to evaluate the function with x = 2.4.
            Returns an array of sympy.core if evaluate_consts is True (one for each realization as spe consts have
            different values for each realization).
        """
        program_str = self.get_infix_str()
        program_sympy = sympy.parsing.sympy_parser.parse_expr(program_str, evaluate=False)
        if do_simplify:
            program_sympy = sympy.simplify(program_sympy, rational=True) # 2.0 -> 2
        if evaluate_consts:
            sympy_local_dicts = self.get_sympy_local_dicts(replace_nan_with=replace_nan_with)                         # (n_realizations,)
            program_sympy = [ program_sympy.subs(sympy_local_dicts[i_real]) for i_real in range(self.n_realizations)] # (n_realizations,)
            if do_simplify:
                program_sympy = [ sympy.simplify(program_sympy[i_real], rational=True) for i_real in range(self.n_realizations)]
            program_sympy = np.array(program_sympy)
        return program_sympy

    def get_infix_pretty (self, do_simplify = False):
        """
        Returns a printable ASCII sympy.pretty representation of a program.
        Parameters
        ----------
        do_simplify : bool
            If True performs a symbolic simplification of program.
        Returns
        -------
        program_pretty_str : str
        """
        program_sympy = self.get_infix_sympy(do_simplify = do_simplify)
        program_pretty_str = sympy.pretty (program_sympy)
        return program_pretty_str


    def get_infix_latex (self,replace_dummy_symbol = True, new_dummy_symbol = "?", do_simplify = True):
        """
        Returns an str latex representation of a program.
        Parameters
        ----------
        replace_dummy_symbol : bool
            If True, dummy symbol is replaced by new_dummy_symbol.
        new_dummy_symbol : str or None
            Replaces dummy symbol if replace_dummy_symbol is True.
        do_simplify : bool
            If True performs a symbolic simplification of program.
        Returns
        -------
        program_latex_str : str
        """
        program_sympy = self.get_infix_sympy(do_simplify=do_simplify)
        program_latex_str = sympy.latex (program_sympy)
        if replace_dummy_symbol:
            program_latex_str = program_latex_str.replace(Tok.DUMMY_TOKEN_NAME, new_dummy_symbol)
        return program_latex_str


    def get_infix_fig (self,
                       replace_dummy_symbol = True,
                       new_dummy_symbol = "?",
                       do_simplify = True,
                       show_superparent_at_beginning = True,
                       text_size = 16,
                       text_pos  = (0.0, 0.5),
                       figsize   = (10, 2),
                       ):
        """
        Returns pyplot (figure, axis) containing analytic symbolic function program.
        Parameters
        ----------
        replace_dummy_symbol : bool
            If True, dummy symbol is replaced by new_dummy_symbol.
        new_dummy_symbol : str or None
            Replaces dummy symbol if replace_dummy_symbol is True.
        do_simplify : bool
            If True performs a symbolic simplification of program.
        show_superparent_at_beginning : bool
            If True, shows superparent in Figure like "y = ..." instead of just "..."
        text_size : int
            Size of text in figure.
        text_pos : (float, float)
            Position of text in figure.
        figsize : (int, int)
            Shape of figure.
        Returns
        -------
        fig, ax : matplotlib.pyplot.Figure, matplotlib.pyplot.AxesSubplot
        """
        # Latex str of symbolic function
        latex_str = self.get_infix_latex(replace_dummy_symbol = replace_dummy_symbol,
                                         new_dummy_symbol = new_dummy_symbol,
                                         do_simplify = do_simplify)
        # Adding "superparent =" before program to make it pretty
        if show_superparent_at_beginning:
            latex_str = self.library.superparent.name + ' =' + latex_str

        # Prettier fig with:
        #   plt.rc('text', usetex=True)
        #   plt.rc('font', family='serif')

        # Enables new_dummy_symbol = "\square":
        # plt.rc('text.latex', preamble=r'\usepackage{amssymb} \usepackage{xcolor}')
        if new_dummy_symbol == r"\square":
            msg = "Use of \\square as new_dummy_symbol is not supported by matplotlib alone. " \
                  "Use plt.rc('text.latex', preamble=r'\\usepackage{amssymb} \\usepackage{xcolor}') to enable it."
            warnings.warn(msg)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.axis('off')
        ax.text(text_pos[0], text_pos[1], f'${latex_str}$', size = text_size)
        return fig, ax


    def get_infix_image(self,
                        replace_dummy_symbol = True,
                        new_dummy_symbol = "?",
                        do_simplify = True,
                        text_size    = 16,
                        text_pos     = (0.0, 0.5),
                        figsize      = (8, 2),
                        dpi          = 512,
                        fpath        = None,
                        ):
        """
        Returns image containing analytic symbolic function program.
        Parameters
        ----------
        replace_dummy_symbol : bool
            If True, dummy symbol is replaced by new_dummy_symbol.
        new_dummy_symbol : str or None
            Replaces dummy symbol if replace_dummy_symbol is True.
        do_simplify : bool
            If True performs a symbolic simplification of program.
        text_size : int
            Size of text in figure.
        text_pos : (float, float)
            Position of text in figure.
        figsize : (int, int)
            Shape of figure.
        dpi : int
            Pixel density for raster image.
        fpath : str or None
            Path where to save image. Default = None, not saved.
        Returns
        -------
        image : PIL.Image.Image
        """

        try:
            import PIL as PIL
        except:
            print("Unable to import PIL (which is needed to make image data). "
                  "Please install it via 'pip install pillow >= 9.0.1'.")

        # Getting fig, ax
        fig, ax = self.get_infix_fig (
                            replace_dummy_symbol = replace_dummy_symbol,
                            new_dummy_symbol = new_dummy_symbol,
                            do_simplify = do_simplify,
                            text_size = text_size,
                            text_pos  = text_pos,
                            figsize   = figsize,
                            )

        # Exporting image to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi)
        plt.close()

        # Buffer -> img
        white = (255, 255, 255, 255)
        img = PIL.Image.open(buf)
        bg = PIL.Image.new(img.mode, img.size, white)
        diff = PIL.ImageChops.difference(img, bg)
        diff = PIL.ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        img = img.crop(bbox)

        # Saving if fpath is given
        if fpath is not None:
            fig.savefig(fpath, dpi=dpi)

        return img

    def show_infix(self,
                   replace_dummy_symbol = True,
                   new_dummy_symbol = "?",
                   do_simplify = False,
                   text_size=24,
                   text_pos=(0.0, 0.5),
                   figsize=(10, 1),
                   ):
        """
        Shows pyplot (figure, axis) containing analytic symbolic function program.
        Parameters
        ----------
        replace_dummy_symbol : bool
            If True, dummy symbol is replaced by new_dummy_symbol.
        new_dummy_symbol : str or None
            Replaces dummy symbol if replace_dummy_symbol is True.
        do_simplify : bool
            If True performs a symbolic simplification of program.
        text_size : int
            Size of text in figure.
        text_pos : (float, float)
            Position of text in figure.
        figsize : (int, int)
            Shape of figure.
        """
        # Getting fig, ax
        fig, ax = self.get_infix_fig (
                            replace_dummy_symbol = replace_dummy_symbol,
                            new_dummy_symbol = new_dummy_symbol,
                            do_simplify = do_simplify,
                            text_size = text_size,
                            text_pos  = text_pos,
                            figsize   = figsize,
                            )
        # Show
        plt.show()
        return None