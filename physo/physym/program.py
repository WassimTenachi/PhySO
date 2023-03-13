import warnings as warnings
import numpy as np
import copy as copy  # for Cursor
import sympy as sympy

# For tree image (optional)
import os
import matplotlib.pyplot as plt
import shutil
import io

# For tree image (optional)
try:
    import pygraphviz as pgv
    import dot2tex as dot2tex
    from pdflatex import PDFLaTeX
    import pdf2image
    from PIL import Image, ImageChops
except:
    warnings.warn("Can not import display packages.")

# Internal imports
from physo.physym import token as Tok
from physo.physym import execute as Exec
from physo.physym import dimensional_analysis as phy
from physo.physym import free_const


class Cursor:
    """
    Helper class for single-token navigation in tree of programs in VectPrograms.
    Represents the position of a single token in a program in a batch of programs.
    For user-exploration, program testing and debugging.
    Attributes
    ----------
    programs : program.VectPrograms
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
        programs : program.VectPrograms
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
    free_const_values : array_like of float or None
        Values of free constants for each constant in the library.
    candidate_wrapper : callable
        Wrapper to apply to candidate program's output, candidate_wrapper taking func, X as arguments where func is
        a candidate program callable (taking X as arg). By default = None, no wrapper is applied (identity).
    """
    def __init__(self, tokens, library, is_physical=None, free_const_values=None, candidate_wrapper = lambda func, X : func(X)):
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
            candidate_wrapper = lambda f,x: f(x)
        self.candidate_wrapper = candidate_wrapper

        # free const related
        self.free_const_values = free_const_values

    def execute_wo_wrapper(self, X):
        """
        Executes program on X.
        Parameters
        ----------
        X : torch.tensor of shape (n_dim, ?,) of float
            Values of the input variables of the problem with n_dim = nb of input variables.
        Returns
        -------
        y : torch.tensor of shape (?,) of float
            Result of computation.
        """
        y = Exec.ExecuteProgram(input_var_data     = X,
                                 free_const_values = self.free_const_values,
                                 program_tokens    = self.tokens)
        return y

    def execute(self, X):
        """
        Executes program on X.
        Parameters
        ----------
        X : torch.tensor of shape (n_dim, ?,) of float
            Values of the input variables of the problem with n_dim = nb of input variables.
        Returns
        -------
        y : torch.tensor of shape (?,) of float
            Result of computation.
        """
        y = self.candidate_wrapper(lambda X: self.execute_wo_wrapper(X), X)
        return y

    def optimize_constants(self, X, y_target, args_opti = None):
        """
        Optimizes free constants of program.
        Parameters
        ----------
        X : torch.tensor of shape (n_dim, ?,) of float
            Values of the input variables of the problem with n_dim = nb of input variables.
        y_target : torch.tensor of shape (?,) of float
            Values of target output.
        args_opti : dict or None, optional
            Arguments to pass to free_const.optimize_free_const. By default, free_const.DEFAULT_OPTI_ARGS
            arguments are used.
        """
        if args_opti is None:
            args_opti = free_const.DEFAULT_OPTI_ARGS
        func_params = lambda params: self.__call__(X)

        history = free_const.optimize_free_const ( func     = func_params,
                                                       params   = self.free_const_values,
                                                       y_target = y_target,
                                                       **args_opti)
        return history

    def __call__(self, X):
        """
        Executes program on X.
        """
        return self.execute(X=X)

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

    def get_infix_sympy (self, do_simplify = False):
        """
        Returns sympy symbolic representation of a program.
        Parameters
        ----------
        do_simplify : bool
            If True performs a symbolic simplification of program.
        Returns
        -------
        program_sympy : sympy.core
            Sympy symbolic function. It is possible to run program_sympy.evalf(subs={'x': 2.4}) where 'x' is a variable
            appearing in the program to evaluate the function with x = 2.4.
        """
        program_str = self.get_infix_str()
        program_sympy = sympy.parsing.sympy_parser.parse_expr(program_str, evaluate=False)
        if do_simplify:
            program_sympy = sympy.simplify(program_sympy, rational=True) # 2.0 -> 2
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
        # Fig
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        # enables new_dummy_symbol = "\square"
        plt.rc('text.latex', preamble=r'\usepackage{amssymb} \usepackage{xcolor}')
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
        img = Image.open(buf)
        bg = Image.new(img.mode, img.size, white)
        diff = ImageChops.difference(img, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
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

class VectPrograms:
    """
    Represents a batch of symbolic programs (jit-able class).
    Attributes
    ----------
    batch_size : int
        Number of programs in batch.
    max_time_step : int
        Max number of tokens programs can contain.
    shape : (int, int)
        Shape of batch (batch_size, max_time_step,).

    library : library.Library
        Library of tokens that can appear in programs.
    n_choices : int
        Number of choosable tokens.
    n_library : int
        Number of tokens in the library (including placeholders).
    superparent_idx : int
        Index of superparent placeholder in the library.
    dummy_idx : int
        Index of dummy placeholder in the library.
    invalid_idx : int
        Index of invalid void placeholder in the library.
    lib_names : numpy.array of shape (n_library,) of str
        Names of tokens in the library.
    lib_vect : token.VectTokens of shape (1, n_library,)
        Vectorized tokens of library.

    curr_step : int
        Current time step (ie. index of last token added).
    safe_max_time_step : int
        Number of tokens that can safely be contained in programs if there are chosen completely randomly. (ie. without
        making sure there final lengths will be <= max_time_step).

    n_lengths     : numpy.array of shape (batch_size,) of int
        Lengths of programs (not counting placeholders) (!= curr_step as length stops growing after program is finished).
    n_dummies     : numpy.array of shape (batch_size,) of int
        Number of dummy placeholders added to complete program trees.
    total_arities : numpy.array of shape (batch_size,) of int
        Total arities of programs.
    is_complete   : numpy.array of shape (batch_size,) of bool
        Are programs complete bool.
    n_dummies_history : numpy.array of shape (batch_size, max_time_step,) of int
        Number of dummies of programs at each time step.

    is_physical                 : numpy.array of shape (batch_size,) of bool
        Are programs physical units-wise.
    units_analysis_cases : numpy.array of shape (batch_size, max_time_step,) of int
        Dimensional analysis assignment case code. Units requirement was performed on token if > -1.

    tokens : token.VectTokens of shape (batch_size, max_time_step,)
        Vectorized tokens of contained in batch (including idx in library ie. nature of tokens).

    free_consts : free_const.FreeConstantsTable
        Free constant register.
    candidate_wrapper : callable
        Wrapper to apply to candidate program's output, candidate_wrapper taking func, X as arguments where func is
        a candidate program callable (taking X as arg). By default = None, no wrapper is applied (identity).
    """
    def __init__(self, batch_size, max_time_step, library, candidate_wrapper=None):
        """
        Parameters
        ----------
        batch_size : int
            Number of programs in batch.
        max_time_step : int
            Max number of tokens programs can contain.
        library : library.Library
            Library of tokens that can appear in programs.
        candidate_wrapper : callable or None, optional
            Wrapper to apply to candidate program's output, candidate_wrapper taking func, X as arguments where func is
            a candidate program callable (taking X as arg). By default = None, no wrapper is applied (identity).
        """
        # Assertions
        assert isinstance(batch_size,    int) and batch_size    > 0, "batch_size    must be a >0 int."
        assert isinstance(max_time_step, int) and max_time_step > 0, "max_time_step must be a >0 int."

        # Number of candidate programs
        self.batch_size = batch_size                                        # int
        self.shape      = (batch_size, max_time_step)                       # (int, int)

        # ---------------------------- LIBRARY ----------------------------
        self.library         = library
        self.n_choices       = library.n_choices                            # int
        self.n_library       = library.n_library                            # int
        # Placeholder tokens
        self.superparent_idx = library.superparent_idx                      # int
        self.dummy_idx       = library.dummy_idx                            # int
        self.invalid_idx     = library.invalid_idx                          # int
        # For display purposes (remove when jit-ing ?)
        self.lib_names       = library.lib_name                             # (n_library,) of str (<MAX_NAME_SIZE)
        # Token properties useful for step-by-step operations
        # (only keeping jit-able vectors in VectTokens)
        self.lib_vect        = library.properties                           # VectTokens

        # ---------------------------- STEP COUNTER ----------------------------
        # initializing current step to 0
        self.curr_step      = 0                                             # int
        self.max_time_step = max_time_step                                  # int

        # Number of tokens that can be appended safely ie. even
        # in the worst case scenario which generates the maximum number
        # of dummies  where only tokens of arity = Tok.MAX_ARITY are
        # appended to a single program, if this limit is respected,
        # max_time_step will not be out-bounded by dummies.
        self.safe_max_time_step = np.floor(                                 # int
            (self.max_time_step - 1)/Tok.MAX_ARITY).astype(int)

        # ---------------------------- PROGRAM MANAGEMENT ---------------------------- -> batch dim
        # Individual lengths of programs <= max_time_step,
        # tokens after n_lengths are dummies or do not have meaning
        self.n_lengths     = np.zeros(self.batch_size, dtype=int)           # (batch_size,) of int
        # Number of tokens needed to finish the program
        # = number of dummies = number of loose ends
        self.n_dummies     = np.zeros(self.batch_size, dtype=int)           # (batch_size,) of int
        # Sum of arities over time dim
        self.total_arities = np.zeros(self.batch_size, dtype=int)           # (batch_size,) of int
        # Is the program complete, if n_dummies passes through 0
        # at one point, this remembers that the program is complete
        self.is_complete   = np.full(self.batch_size, False)                # (batch_size,) of bool

        # ---------------------------- TOKEN MANAGEMENT ---------------------------- -> time dim
        # Number of dummy at any point
        self.n_dummies_history = np.zeros(shape = self.shape, dtype=int)                      # (batch_size, max_time_step,) of int
        # Token main properties
        self.tokens = Tok.VectTokens(shape =self.shape, invalid_token_idx = self.invalid_idx) # (batch_size, max_time_step,)

        # ---------------------------- UNITS RELATED MANAGEMENT ----------------------------

        # mask : is program physically correct units-wise ? (program management) -> batch dim
        self.is_physical = np.full(shape = self.batch_size, fill_value = True, dtype = bool)  # (batch_size,) of bool

        # mask : dimensional analysis assignment case code (Token management) -> time dim
        self.units_analysis_cases = np.full(shape = self.shape, fill_value = phy.UNITS_ANALYSIS_NOT_PERFORMED_CASE_CODE, dtype = int)  # (batch_size, max_time_step,) of bool

        # ---------------------------- INIT 0TH DUMMY ----------------------------

        self.total_arities = self.compute_sum_arities(step=self.curr_step)    # (batch_size,) of int
        self.n_dummies     = self.total_arities - self.n_lengths              # (batch_size,) of int

        # Coords of initial dummies
        coords_initial_dummies = self.coords_of_step(step=self.curr_step)     # (2, batch_size,) of int

        # Affect 0th token = dummy
        self.set_non_positional_from_idx (
            coords_dest    = coords_initial_dummies,                          # (2, batch_size,) of int
            tokens_idx_src = np.full(self.batch_size, self.dummy_idx)         # (batch_size,) of int
                                            )
        # Affect 0th token units = that of superparent
        self.set_static_units_from_idx (
            coords_dest    = coords_initial_dummies,                          # (2, batch_size,) of int
            tokens_idx_src = np.full(self.batch_size, self.superparent_idx)   # (batch_size,) of int
                                          )
        # Affect 0th token depth = 0
        self.tokens.depth[tuple(coords_initial_dummies)] = np.full(self.batch_size, 0)

        # Affect 0th token' ancestors record
        self.register_ancestor (coords_dest = coords_initial_dummies)

        # ---------------------------- FREE CONSTANTS REGISTER ----------------------------
        self.free_consts = free_const.FreeConstantsTable(batch_size = self.batch_size, library =self.library)

        # ---------------------------- EXECUTION RELATED ----------------------------
        # Wrapper to apply to candidate programs when executing
        if candidate_wrapper is None:
            candidate_wrapper = lambda func, X : func(X)
        self.candidate_wrapper = candidate_wrapper

        return None

    def lib (self, attr):
        """
        Gives access to vectorized properties of tokens in library without having to use [0, :] (as batch_size = 1 in
        vectorized properties of library).
        Parameters
        ----------
        attr : str
            Attribute of library vectorized properties to access.
        Returns
        -------
        numpy.array of shape (n_library,) of ?
            Vectorized property of library. (? depends on attr, eg. ? = int for arity, ? = float for complexity etc.)
        """
        # Giving access to vectorized properties to user without having to use [0, :] at each property access
        return getattr(self.lib_vect, attr)[0]

    def append (self, new_tokens_idx, forbid_inconsistent_units = False):
        """
        Appends new tokens to batch.
        New tokens appended to already complete programs (ie. out of tree tokens) are ignored.
        Note that units requirements and update is not done in append (as it is computationally costly and unnecessary
        if units are not used). Use the assign_required_units method on all previous steps + this one before appending
        to compute units. This is done by prior.PhysicalUnitsPrior.
        Parameters
        ----------
        new_tokens_idx : numpy.array of shape (batch_size,) of int
            Index of tokens to append in the library.
        forbid_inconsistent_units : bool
            If True, forbids (by raising an error) appending of new tokens having constraining units inconsistent
            with current units constraints. Consistency of new tokens' physical units vs the current programs is
            checked. To work properly the assign_required_units method should have been called on all previous steps +
            this one before appending.
        """
        # Will be modified afterward
        new_tokens_idx = np.copy(new_tokens_idx)

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------- ASSERTIONS -----------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # Basic assertions

        # --------------------- Assert that time step is not exceeded ---------------------
        if self.curr_step == self.max_time_step:
            raise IndexError("Can not append to batch programs as it is already full over time dim, max_time_step "
                "= %i was reached." % (self.max_time_step))

        # ------------------------------ Check new_tokens_idx ------------------------------
        # Type
        assert type(new_tokens_idx) == np.ndarray, "Arg new_tokens_idx must be a numpy array of dtype = int"
        assert (new_tokens_idx.dtype == int or new_tokens_idx.dtype==np.dtype("int64")), "Arg new_tokens_idx must be a numpy array of dtype = int"

        # Shape
        assert new_tokens_idx.shape == (self.batch_size,), "Arg new_tokens_idx must have shape = (batch_size,) = (%i,)" \
                                                           % self.batch_size

        # Min / Max
        assert new_tokens_idx.min() >= 0, "Min value of new_tokens_idx must be >= 0."
        assert new_tokens_idx.max() < self.n_choices, "Max value of new_tokens_idx must be < %i" % self.n_choices

        # ------------------ Assert enough space for new tokens' dummies ------------------
        # Raise error if number of dummies needed to handle new tokens exceeds max_time_step

        # Space necessary in time dim to finish the program if new tokens were appended
        n_space_necessary = self.lib("arity")[new_tokens_idx] + self.n_dummies + self.n_lengths  # (batch_size,) of int

        # Is it impossible to append new token to program : mask
        mask_impossible_to_append = n_space_necessary > self.max_time_step                       # (batch_size,) of bool

        if mask_impossible_to_append.any():
            error_msg = "Can not append tokens :\n%s\n which have arities :\n%s\n to programs :\n%s\n as the number " \
                        "of tokens that would be required to finish the program would then be of " \
                        "(n_dummies + new token arity) =\n%s\n which would exceed max_time_step = %i" \
                        % (self.lib_names[new_tokens_idx][mask_impossible_to_append],
                           self.lib("arity")[new_tokens_idx][mask_impossible_to_append],
                           self.lib_names[self.tokens.idx][mask_impossible_to_append],
                           n_space_necessary[mask_impossible_to_append],
                           self.max_time_step,)
            raise IndexError(error_msg)

        # --------------------------------------------------------------------------------------------------------------
        # ----------------------------- REPLACING NEW TOKENS OF COMPLETED PROGRAMS BY VOID -----------------------------
        # --------------------------------------------------------------------------------------------------------------
        # Completed programs do not need new tokens, replacing them by void invalid tokens.

        # Number of complete programs
        self.n_complete = self.is_complete.sum()                                                # int
        # Replacing new tokens trying to be added to complete programs by void
        new_tokens_idx[self.is_complete] = np.full(self.n_complete, self.invalid_idx, int)      # (n_complete,) of int

        # --------------------------------------------------------------------------------------------------------------
        # ------------------------------------ ASSERTIONS : UNITS CONSISTENCY CHECK ------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # Check if new tokens are inconsistent with the current batch units-wise

        # Legacy units (from dummies)
        units_from_dummies = self.tokens.phy_units [:, self.curr_step]                                  # (batch_size, UNITS_VECTOR_SIZE,) of float
        # Units from new tokens
        units_from_new_tokens = self.lib("phy_units") [new_tokens_idx]                                  # (batch_size, UNITS_VECTOR_SIZE,) of float

        # Do new tokens contain constraining units: mask
        mask_is_constraining_new_tokens = self.lib("is_constraining_phy_units") [new_tokens_idx]        # (batch_size,) of bool
        # Do legacy dummies contain constraining units: mask
        mask_is_constraining_dummies    = self.tokens.is_constraining_phy_units[:, self.curr_step]      # (batch_size,) of bool

        # Are new tokens units equal to legacy units ie dummy units they are replacing: mask
        mask_units_unequal = np.logical_not((units_from_dummies == units_from_new_tokens).all(axis=1))  # (batch_size,) of bool

        # Are new token inconsistent with current units constraints: mask
        # (ie. new token has different units AND has constraints AND legacy dummy has constraints) (avoids NAN != NAN)
        mask_inconsistency = np.logical_and.reduce((                 # (batch_size,) of bool
                                    mask_units_unequal,
                                    mask_is_constraining_dummies,
                                    mask_is_constraining_new_tokens,
                                                   ))

        # mask: was units analysis performed for this step if it was necessary to decide if new tokens' units are
        # consistent (not checked if complete programs already as there is no need to run units analysis for those).
        mask_necessary_units_analysis_performed = np.logical_or.reduce((
                    ~(self.units_analysis_cases[:, self.curr_step,] == phy.UNITS_ANALYSIS_NOT_PERFORMED_CASE_CODE),
                     self.is_complete
                                                            ))

        # Registering unphysical programs
        # Program is physical if it was already considered as physical AND new tokens have consistent units AND units
        # requirement was computed for this step (otherwise tokens could seem to have consistent units just because
        # units requirements were not performed thus making any token look acceptable).
        self.is_physical = self.is_physical & (~mask_inconsistency) & mask_necessary_units_analysis_performed

        # Raise error
        if mask_inconsistency.any() and forbid_inconsistent_units:
            error_msg = "Can not append new tokens as %i of them (tokens: %s) have inconsistent units regarding " \
                        "current units constraints.\nThese tokens respectively have units:\n %s \nBut according to " \
                        "current constraints they should have units:\n %s" \
                        % (mask_inconsistency.sum(), self.lib_names[new_tokens_idx[mask_inconsistency]],
                           units_from_new_tokens[mask_inconsistency], units_from_dummies[mask_inconsistency])
            raise phy.PhyUnitsError(error_msg)

        # --------------------------------------------------------------------------------------------------------------
        # -------------------------------------------- APPENDING NEW TOKENS --------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # Replacing dummy by new token (but keeping positional + units info)

        # Is the token a new token: mask
        mask_new_tokens = self.tokens.pos == self.curr_step

        # --- NON_POSITIONAL INFO ---
        coords_new_tokens = self.coords_of_step(step=self.curr_step)       # (2, batch_size,) of int
        self.set_non_positional_from_idx (
            coords_dest    = coords_new_tokens,                            # (2, batch_size,) of int
            tokens_idx_src = new_tokens_idx                                # (batch_size,) of int
                                            )

        # --- POSITIONAL INFO ---
        # Inheriting positional info from legacy dummy: OK

        # --- UNITS INFO ---
        # Number of units constraining new tokens
        n_constraining_new_tokens = mask_is_constraining_new_tokens.sum()  # int
        # Adding unit info for new tokens which have units constraints (those that don't will inherit
        # legacy constraints). No units conflicts as it was checked in UNITS CONSISTENCY CHECK.
        # If consistency check error raise is ignored, conflicts should be resolved by giving priority
        # to new tokens' units.
        self.set_units(
            coords_dest = self.coords_of_step(step =self.curr_step)[:, mask_is_constraining_new_tokens], # (2, n_constraining_new_tokens,) of int
            new_is_constraining_phy_units = np.full(n_constraining_new_tokens, True),                    # (n_constraining_new_tokens,) of bool
            new_phy_units = units_from_new_tokens[mask_is_constraining_new_tokens],                      # (n_constraining_new_tokens, Tok.UNITS_VECTOR_SIZE) of float
                                )

        # --------------------------------------------------------------------------------------------------------------
        # ----------------------------------- UPDATING PROGRAM MANAGEMENT VARIABLES ------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # -> batch dim

        # Number of dummies that will be left after new tokens replace
        # 1st dummy (counting all dummies minus dummy being replaced)
        n_legacy_dummies = self.n_dummies - np.ones(self.batch_size, dtype=int)    # (batch_size,) of int
        # Complete programs do not need dummies
        n_legacy_dummies [self.is_complete] = 0                                    # (self.is_complete.sum(),) of int

        # Update program lengths for those that are still incomplete
        self.n_lengths [self.is_complete == False] += 1                            # (self.is_complete.sum(),) of int
        # Update of arities over time dim with new tokens
        self.total_arities += self.tokens.arity[:, self.curr_step]                 # (batch_size,) of int
        # Number of dummy placeholders
        self.n_dummies     = self.total_arities - self.n_lengths                   # (batch_size,) of int
        # Complete programs do not need dummies
        self.n_dummies [self.is_complete] = 0                                      # (self.is_complete.sum(),) of int
        # Dummies history
        self.n_dummies_history[:, self.curr_step] = self.n_dummies                 # (batch_size,) of int

        # Update time step
        self.curr_step += 1

        # Update complete status of programs which were previously incomplete AND n_dummies=0
        self.is_complete[np.logical_and(self.is_complete == False, self.n_dummies == 0)] = True  # (bool_array.sum(),) of int

        # Number of dummies necessary to complete program now that new tokens were added
        n_new_dummies = self.n_dummies - n_legacy_dummies
        # Complete programs do not need dummies
        n_new_dummies [self.is_complete] = 0                                # (self.is_complete.sum(),) of int

        # --------------------------------------------------------------------------------------------------------------
        # ------------------------------------------ SHIFTING LEGACY DUMMIES -------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Shifting legacy dummies (except the one replaced by token) to make room for new dummies

        # -------- BEFORE SHIFT COORDS --------
        # Is token a legacy token (before shift): mask
        mask_legacy_tokens_noshift = np.logical_and.reduce((                              # (batch_size, max_time_step,) of bool
            self.tokens.pos >= self.curr_step,                                            # (batch_size, max_time_step,) of bool
            self.tokens.pos < self.tile_batch_vect(n_legacy_dummies) + self.curr_step,    # (batch_size, max_time_step,) of bool
        ))
        # Coords of legacy tokens
        n_legacy_dummies_total, coords_legacy_tokens_noshift = self.mask_to_coords(mask_legacy_tokens_noshift)  # int, (2, n_legacy_dummies_total) of int

        # -------- SHIFTED COORDS --------
        # Amount by which legacy tokens should be shifted in time dim (array over batch dim)
        # Ie. amount of space to leave for new dummies that will be added between new tokens and legacy dummies
        legacy_shift = n_new_dummies[                                                    # (n_legacy_dummies_total,) of int
            coords_legacy_tokens_noshift[0]  # coords of legacy tokens in batch dim      # (n_legacy_dummies_total,) of int
        ]
        # Legacy tokens shift
        coords_legacy_tokens_shifted = np.stack((                                        # (2, n_legacy_dummies_total,) of int
            coords_legacy_tokens_noshift[0],                  # batch dim coord -> no change
            coords_legacy_tokens_noshift[1] + legacy_shift,   # time dim coord  -> shifted
            ), axis=0)
        # Mask corresponding to legacy tokens after shift
        mask_legacy_tokens_shifted = self.coords_to_mask(coords_legacy_tokens_shifted)  # (batch_size, max_time_step,) of bool

        # -------- PERFORMING SHIFT --------
        # Performing shift if necessary
        # (ie. if there are legacy tokens)
        if n_legacy_dummies_total > 0 :
            coords_src  = coords_legacy_tokens_noshift
            coords_dest = coords_legacy_tokens_shifted
            # COPYING INFO
            # NON_POSITIONAL PROPERTIES
            self.move_dummies(coords_src = coords_src, coords_dest = coords_dest)

        # --------------------------------------------------------------------------------------------------------------
        # ------------------------------------------- COMPLETING WITH DUMMIES ------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # Adding dummies to complete incomplete programs

        # Is token coord an emplacement for a new dummy: mask
        # ie. where pos >= step AND pos < (n_new_dummies + step)
        mask_new_dummies = np.logical_and.reduce((                                        # (batch_size, max_time_step,) of bool
            self.tokens.pos >= self.curr_step,                                            # (batch_size, max_time_step,) of bool
            self.tokens.pos < self.tile_batch_vect(n_new_dummies) + self.curr_step,       # (batch_size, max_time_step,) of bool
        ))

        # Coords of new dummies to create
        n_new_dummies_total, coords_new_dummies  = self.mask_to_coords(mask_new_dummies)  # int, (2, n_new_dummies_total) of int

        # Internal assertion
        # Checking that new dummies will not be added where there already are legacy dummies (after they were shifted)
        assert np.logical_and(mask_new_dummies, mask_legacy_tokens_shifted, ).any() == False, "Internal error, can " \
            "not create new dummies at the location of legacy dummies"

        # --------------------------------------------------------------------------------------------------------------
        # -------------------------------- COMPLETING WITH DUMMIES : NON_POSITIONAL INFO -------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # Adding new dummies' non_positional info

        self.set_non_positional_from_idx (
            coords_dest    = coords_new_dummies,                            # (2, n_new_dummies_total,) of int
            tokens_idx_src = np.full(n_new_dummies_total, self.dummy_idx)   # (n_new_dummies_total,) of int
                                            )

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------- COMPLETING WITH DUMMIES : POSITIONAL INFO ----------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Number of new dummies along batch_dim
        n_new_dummies = mask_new_dummies.sum(axis=1)                        # (batch_size,) of int

        # -------- PARENT INFO --------
        # PARENT of new dummies : parent = new_tokens
        # pos of parent = previous step
        self.set_parent  (coords_dest = coords_new_dummies, has_mask = True , pos_val = self.curr_step - 1,)

        # -------- CHILDREN INFO --------
        # CHILDREN of new dummies : No children
        self.set_children(coords_dest = coords_new_dummies, has_mask = False, pos_val = Tok.INVALID_POS, nb = 0,)

        # -------- SIBLINGS INFO --------
        # SIBLINGS of new dummies

        # -> No siblings where n_new_dummies <= 1
        mask_new_dummies_wo_siblings = np.logical_and(                      # (batch_size, max_time_step,) of bool
            mask_new_dummies,                                               # (batch_size, max_time_step,) of bool
            self.tile_batch_vect(n_new_dummies) <= 1,                       # (batch_size, max_time_step,) of bool
                                                        )
        n_new_dummies_wo_sibling, coords_new_dummies_wo_sibling = self.mask_to_coords(       # int, (2, n_new_dummies_wo_sibling) of int
            mask = mask_new_dummies_wo_siblings)
        self.set_siblings(coords_dest = coords_new_dummies_wo_sibling, has_mask = False, pos_val = Tok.INVALID_POS, nb = 0)

        # -> Siblings where n_new_dummies = 2
        # 0th sibling
        mask_new_dummies_w_siblings_0 = np.logical_and.reduce((             # (batch_size, max_time_step,) of bool
            mask_new_dummies,                                               # (batch_size, max_time_step,) of bool
            self.tile_batch_vect(n_new_dummies) == 2,                       # (batch_size, max_time_step,) of bool
            self.tokens.pos == (self.curr_step + 0),                        # (batch_size, max_time_step,) of bool
                                                        ))
        n_new_dummies_w_sibling_0, coords_new_dummies_w_siblings_0 = self.mask_to_coords(mask_new_dummies_w_siblings_0)
        # 1st sibling
        mask_new_dummies_w_siblings_1 = np.logical_and.reduce((             # (batch_size, max_time_step,) of bool
            mask_new_dummies,                                               # (batch_size, max_time_step,) of bool
            self.tile_batch_vect(n_new_dummies) == 2,                       # (batch_size, max_time_step,) of bool
            self.tokens.pos == (self.curr_step + 1),                        # (batch_size, max_time_step,) of bool
                                                        ))
        n_new_dummies_w_sibling_1, coords_new_dummies_w_siblings_1 = self.mask_to_coords(mask_new_dummies_w_siblings_1)
        # (n_new_dummies_w_sibling_0 = n_new_dummies_w_sibling_1)
        # Setting sibling relationships : masks
        self.tokens.has_siblings_mask[tuple(coords_new_dummies_w_siblings_0)] = True # (n_new_dummies_w_sibling_0,) of bool
        self.tokens.has_siblings_mask[tuple(coords_new_dummies_w_siblings_1)] = True # (n_new_dummies_w_sibling_1,) of bool
        # Setting sibling relationships : pos
        self.tokens.siblings_pos[tuple(coords_new_dummies_w_siblings_0)] =\
            self.tokens.pos[tuple(coords_new_dummies_w_siblings_1)][:, np.newaxis]  # sibling 0's sibling is sibling 1 # (n_new_dummies_w_sibling_0,) of int
        self.tokens.siblings_pos[tuple(coords_new_dummies_w_siblings_1)] =\
            self.tokens.pos[tuple(coords_new_dummies_w_siblings_0)][:, np.newaxis]  # sibling 1's sibling is sibling 0 # (n_new_dummies_w_sibling_1,) of int
        # Setting sibling relationships : nb of dummies
        self.tokens.n_siblings[tuple(coords_new_dummies_w_siblings_0)] = 1
        self.tokens.n_siblings[tuple(coords_new_dummies_w_siblings_1)] = 1

        # -------- DEPTH INFO --------
        # DEPTH of new dummies = new_tokens depth + 1
        # Coords of parent are the same as new dummies in batch dim
        # with time dim = previous step (=step of new tokens)
        coords_new_dummies_parents = np.stack((                                               # (2, n_new_dummies_total,) of int
            coords_new_dummies[0],                             # batch dim coord
            np.full(n_new_dummies_total, self.curr_step - 1),  # time dim coord
            # = self.tokens.pos[coords_new_dummies[0], self.curr_step - 1]
            ), axis=0)
        # Computing depth
        self.tokens.depth[tuple(coords_new_dummies)] = self.tokens.depth[tuple(coords_new_dummies_parents)] + 1

        # -------- ANCESTORS INFO --------
        # Copy ancestor register from parents
        self.tokens.ancestors_pos[tuple(coords_new_dummies)] = self.tokens.ancestors_pos[tuple(coords_new_dummies_parents)]
        self.register_ancestor(coords_new_dummies)

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------- COMPLETING WITH DUMMIES : INFORMING PARENT ---------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # Informing parent of new dummies (ie. new tokens) that they have a child

        # Is token part of an incomplete program : mask
        mask_incomplete = np.logical_not(self.tile_batch_vect(self.is_complete))

        # -------- HANDLING LONELY CHILDREN --------
        # New tokens having only one dummy child : mask
        # Ie. is a new token AND has arity == 1 AND program is not complete
        mask_new_tokens_w_lonely_child = np.logical_and.reduce((                    # (batch_size, max_time_step,) of bool
            mask_new_tokens,
            self.tokens.arity == 1,
            mask_incomplete,
        ))
        # n_new_tokens_w_lonely_child = mask_new_tokens_w_lonely_child.sum()
        n_new_tokens_w_lonely_child, coords_new_tokens_w_lonely_child = self.mask_to_coords(mask_new_tokens_w_lonely_child)     # int, (2, n_new_tokens_w_lonely_child,) of int
        # Positions of lonely children in time dim
        pos_lonely_children = np.stack((                                           # (n_new_tokens_w_lonely_child, 1,) of int
            coords_new_dummies_wo_sibling[1],                             # position of child 0 in time dim
            np.full(n_new_tokens_w_lonely_child, Tok.INVALID_POS, int)),  # no 2nd child
            axis=1,)
        # Setting children
        self.set_children(coords_dest = tuple(coords_new_tokens_w_lonely_child),  # (2, n_new_tokens_w_lonely_child,) of int
                          has_mask    = True,                                     # (n_new_tokens_w_lonely_child,) of bool
                          pos_val     = pos_lonely_children,                      # (n_new_tokens_w_lonely_child, 1,) of int
                          nb          = 1,                                        # (n_new_tokens_w_lonely_child,) of int
                          )

        # -------- HANDLING DOUBLE CHILDREN --------
        # New tokens having two children : mask
        # Ie. is a new token AND has arity == 2 AND program is not complete
        mask_new_tokens_w_two_children = np.logical_and.reduce((                  # (batch_size, max_time_step,) of bool
            mask_new_tokens,
            self.tokens.arity == 2,
            mask_incomplete,
        ))
        # n_new_tokens_w_two_children = mask_new_tokens_w_two_children.sum()
        n_new_tokens_w_two_children, coords_new_tokens_w_two_children = self.mask_to_coords(mask_new_tokens_w_two_children)  # int, (2, n_new_tokens_w_two_children,) of int
        # Positions of double children in time dim
        pos_double_children = np.stack((                                     # (n_new_tokens_w_two_children, 2,) of int
            coords_new_dummies_w_siblings_0[1],   # position of child 0 in time dim
            coords_new_dummies_w_siblings_1[1]),  # position of child 1 in time dim
            axis=1)
        # Setting children
        self.set_children(coords_dest = tuple(coords_new_tokens_w_two_children),  # (2, n_new_tokens_w_two_children,) of int
                          has_mask    = True,                                     # (n_new_tokens_w_two_children,) of bool
                          pos_val     = pos_double_children,                      # (n_new_tokens_w_two_children, 2,) of int
                          nb          = 2,                                        # (n_new_tokens_w_two_children,) of int
                          )
        # --------------------------------------------------------------------------------------------------------------
        # ------------------------------------ COMPLETING WITH DUMMIES : UNITS INFO ------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # Update units requirements of all free dummies in case there is new information available and update units
        # requirement of dummy representing next token to guess.
        # This responsibility is transferred to the user of append who can use the assign_required_units method.
        return None

    def set_programs (self, tokens_idx, forbid_inconsistent_units = False):
        """
        Sets all programs in batch by appending tokens_idx step by step.
        Parameters
        ----------
        tokens_idx : numpy.array of shape (batch_size, int <= max_time_step) of int
            Index of tokens making up programs in the library.
            If programs have different shapes, tokens_idx must be a matrix of shape (batch_size, int <= max_time_step).
            Tokens out of tree will be ignored.
        forbid_inconsistent_units : bool
            Passed to append method.
        """
        # Assertions will be handled by append.
        max_size = tokens_idx.shape[1]
        for i in range(max_size):
            self.append(tokens_idx[:, i], forbid_inconsistent_units = forbid_inconsistent_units)

    # -----------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------- UNITS -----------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------

    def assign_required_units (self, step = None, ignore_unphysical = True,):
        """
        Runs required units assignment routine (assign_required_units) on programs at step.
        Parameters
        ----------
        step : int or None
            Required units assignment routine is run on tokens at step. By default, step = current step.
        ignore_unphysical : bool
            Should routine be run on already unphysical programs (units-wise) ?
        """
        # Step
        if step is None:
            step = self.curr_step

        # Assertion
        assert step < self.max_time_step, "Can not work on tokens at step = %i as this step is out of range " \
                                          "(max_time_step = %i)"%(step, self.max_time_step)

        # mask : should run be performed according to ignore_unphysical arg ?
        do_unphysical = np.full(self.batch_size, fill_value = not ignore_unphysical, dtype = bool)

        # mask : should assign_required_units be run on program ?
        # Do run only on incomplete programs AND on (physical programs only OR when do_unphysical is True)
        mask_do_run = (~self.is_complete) & (self.is_physical | do_unphysical)

        # Coords
        coords = self.coords_of_step(step=step) [:, mask_do_run]

        # Perform assignment
        phy.assign_required_units(programs=self, coords=coords)

        return None

    # -----------------------------------------------------------------------------------------------------------------
    # --------------------------------------------- UTILS : MISCELLANEOUS ---------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------

    def mask_to_coords (self, mask):
        """
        Helper function returning coordinates where mask is True.
        Parameters
        ----------
        mask : numpy.array of shape (batch_size, max_time_step) of bool
            Mask.
        Returns
        -------
        mask_sum, coordinates : int, numpy.array of shape (2, mask_sum) of int
            Number of coordinates where mask is True, coordinates where mask is True.
        """
        # Showing that memory space can accurately be allocated before-hand
        mask_sum = mask.sum()                                                       # int
        coords = np.zeros(shape = (2, mask_sum,), dtype=int)                        # (2, mask_sum,) of int
        # Coordinates
        coords [:, :] = np.stack((                                                  # (2, mask_sum,) of int
            self.tokens.pos_batch[mask],  # batch dim coord
            self.tokens.pos[mask],        # time dim coord
            ), axis=0)
        return mask_sum, coords

    def coords_to_mask (self, coords):
        """
        Helper function returning mask of batch shape (batch_size, max_time_step,) containing True at coords.
        Parameters
        ----------
        coords : numpy.array of shape (2, ?) of int
            Coordinates where mask should be True
        Returns
        -------
        mask : numpy.array of shape (batch_size, max_time_step,) of bool
            Matrix of shape = batch shape.
        """
        mask = np.full(shape = self.shape, fill_value = False, dtype = bool)
        mask[tuple(coords)] = True
        return mask

    def tile_batch_vect (self, batch_vect):
        """
        Tiles a vector along batch dim to a 2D matrix (batch dim x time dim) with copies along time dim.
        Parameters
        ----------
        batch_vect : numpy.array of shape (batch_size,) of type ?
            Vector to tile.
        Returns
        -------
        numpy.array of shape (batch_size, max_time_step,) of type ?
            Matrix of shape = batch shape.
        """
        return np.tile(batch_vect, (self.max_time_step, 1)).transpose()  # (batch_size, max_time_step,) of ?

    def coords_of_step (self, step):
        """
        Helper method returning the tuple of coordinates corresponding to a step.
        Parameters
        ----------
        step : int
            Step.
        Returns
        -------
        coords : numpy.array of shape (2, batch_size,) of int
            Coordinates, 0th array in batch dim and 1th array in time dim.
        """
        coords = np.stack((                   # (2, batch_size,) of int
            self.tokens.pos_batch [:, step],  # batch dim coord
            self.tokens.pos       [:, step],  # time dim coord
            ), axis=0)
        return coords

    # -----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- UTILS : TOKEN MANAGEMENT -------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------

    # -------- GET POSITIONAL INFO : FAMILY RELATIVES POS --------

    def get_parent (self, coords):
        """
        Get parent's coordinates of tokens at coords.
        Parameters
        ----------
        coords : numpy.array of shape (2, ?) of int
            Coords of tokens, 0th array in batch dim and 1th array in time dim.
        Returns
        -------
        parent_coords : numpy.array of shape (2, ?) of int
            Coords of parents, 0th array in batch dim and 1th array in time dim.
        """
        parent_pos = self.tokens.parent_pos[tuple(coords)]  # (?,) of int

        parent_coords = np.stack((                                 # (2, ?,) of int
            coords[0],   # same batch coords           # batch dim coord
            parent_pos,                                # time dim coord
            ), axis=0)
        return parent_coords

    def get_siblings (self, coords):
        """
        Get siblings' coordinates of tokens at coords.
        Parameters
        ----------
        coords : numpy.array of shape (2, ?) of int
            Coords of tokens, 0th array in batch dim and 1th array in time dim.
        Returns
        -------
        siblings_coords : numpy.array of shape (1 + Tok.MAX_NB_SIBLINGS, ?) of int
            Coords of siblings, 0th array in batch dim and 1th array in time dim (1st sibling),
            2nd array in time dim (2nd sibling)...
            Use siblings_coords[:,[0,1]] to access 1st sibling, siblings_coords[:,[0,2]] to access 2nd sibling...
        """
        siblings_pos = self.tokens.siblings_pos[tuple(coords)]  # (?, Tok.MAX_NB_SIBLINGS,) of int

        siblings_coords = np.concatenate((                                       # (1 + Tok.MAX_NB_SIBLINGS, ?) of int
            coords[0][:, np.newaxis],   # same batch coords   # batch dim coord  # (?, 1) of int
            siblings_pos,                                     # time dim coord   # (?, Tok.MAX_NB_SIBLINGS) of int
            ), axis=1).transpose()
        return siblings_coords

    def get_children (self, coords):
        """
        Get children's coordinates of tokens at coords.
        Parameters
        ----------
        coords : numpy.array of shape (2, ?) of int
            Coords of tokens, 0th array in batch dim and 1th array in time dim.
        Returns
        -------
        children_coords : numpy.array of shape (1 + Tok.MAX_NB_CHILDREN, ?) of int
            Coords of children, 0th array in batch dim and 1th array in time dim (1st child),
            2th array in time dim (2nd child)...
            Use children_coords[:,[0,1]] to access 1st child, children_coords[:,[0,2]] for 2nd child etc.
        """
        children_pos = self.tokens.children_pos[tuple(coords)]  # (?, Tok.MAX_NB_CHILDREN,) of int

        children_coords = np.concatenate((                                       # (1 + Tok.MAX_NB_CHILDREN, ?) of int
            coords[0][:, np.newaxis],   # same batch coords   # batch dim coord  # (?, 1) of int
            children_pos,                                     # time dim coord   # (?, Tok.MAX_NB_CHILDREN) of int
            ), axis=1).transpose()
        return children_coords

    def get_ancestors (self, coords):
        """
        Get ancestors' coordinates of tokens at coords.
        Parameters
        ----------
        coords : numpy.array of shape (2, ?) of int
            Coords of tokens, 0th array in batch dim and 1th array in time dim.
        Returns
        -------
        ancestors_coords : numpy.array of shape (1 + max_time_step, ?) of int
            Coords of ancestors, 0th array in batch dim, 1th array in time dim (1st ancestor),
            2th array in time dim (2nd ancestor)...
            Use max_time_step[:,[0,1]] to access 1st ancestor, max_time_step[:,[0,2]] for 2nd ancestor etc.
        """
        ancestors_pos = self.tokens.ancestors_pos[tuple(coords)]  # (?, max_time_step,) of int

        ancestors_coords = np.concatenate((                                      # (1 + max_time_step, ?) of int
            coords[0][:, np.newaxis],   # same batch coords   # batch dim coord  # (?, 1) of int
            ancestors_pos,                                    # time dim coord   # (?, max_time_step) of int
            ), axis=1).transpose()
        return ancestors_coords

    # -------- GET POSITIONAL INFO : FAMILY RELATIVES IDX --------
    # Used for symbolic Regression RNN state update

    def get_parent_idx(self, coords, no_parent_idx_filler=None):
        """
        Get parents idx of tokens at coords.
        Parameters
        ----------
        coords : numpy.array of shape (2, ?,) of int
            Coords of tokens, 0th array in batch dim and 1th array in time dim.
        no_parent_idx_filler : int
            Fill value to return where tokens have no parent.
        Returns
        -------
        parents_idx : numpy.array of shape (?,) of int
            Idx of parents.
        """
        # For this function, target = parent

        # What to put as placeholder where tokens at coords do not have a target relationship
        if no_parent_idx_filler is None:
            no_target = self.library.invalid_idx
        else:
            no_target = no_parent_idx_filler

        # coords of target related to tokens at coords
        coords_target = self.get_parent(coords)
        # mask : do tokens at coords have a target relationship
        have_target = self.tokens.has_parent_mask[tuple(coords)]

        # initializing result
        target_idx = np.full(coords.shape[1], no_target, dtype=int)                        # (?,) of int
        # handling tokens at coords having a target relationship
        # (accessing idx at coords_target)
        # ?1 = have_target.sum().astype(int)
        target_idx[have_target] = self.tokens.idx[tuple(coords_target[:, have_target])]    # (?1,) of int
        # handling tokens at coords NOT having a target relationship
        # (filling with no_target placeholder)
        # ?2 = np.logical_not(have_target).sum().astype(int) = ? - ?1
        target_idx[np.logical_not(have_target)] = no_target                                # (?2,) of int

        # Returning result
        return target_idx

    def get_sibling_idx(self, coords, no_sibling_idx_filler=None):
        """
        Get siblings idx of tokens at coords.
        Parameters
        ----------
        coords : numpy.array of shape (2, ?,) of int
            Coords of tokens, 0th array in batch dim and 1th array in time dim.
        no_sibling_idx_filler : int
            Fill value to return where tokens have no sibling.
        Returns
        -------
        siblings_idx : numpy.array of shape (?,) of int
            Idx of siblings.
        """
        # For this function, target = sibling

        # What to put as placeholder where tokens at coords do not have a target relationship
        if no_sibling_idx_filler is None:
            no_target = self.library.invalid_idx
        else:
            no_target = no_sibling_idx_filler

        # coords of target related to tokens at coords
        coords_target = self.get_siblings(coords)
        # mask : do tokens at coords have a target relationship
        have_target = self.tokens.has_siblings_mask[tuple(coords)]

        # initializing result
        target_idx = np.full(coords.shape[1], no_target, dtype=int)                        # (?,) of int
        # handling tokens at coords having a target relationship
        # (accessing idx at coords_target)
        # ?1 = have_target.sum().astype(int)
        target_idx[have_target] = self.tokens.idx[tuple(coords_target[:, have_target])]    # (?1,) of int
        # handling tokens at coords NOT having a target relationship
        # (filling with no_target placeholder)
        # ?2 = np.logical_not(have_target).sum().astype(int) = ? - ?1
        target_idx[np.logical_not(have_target)] = no_target                                # (?2,) of int

        # Returning result
        return target_idx

    def get_ancestors_idx(self, coords, no_ancestor_idx_filler=None):
        """
        Get ancestors idx of tokens at step.
        Parameters
        ----------
        coords : numpy.array of shape (2, ?,) of int
            Coords of tokens, 0th array in batch dim and 1th array in time dim.
        no_ancestor_idx_filler : int
            Fill value to return where tokens have no ancestors and for padding family tree lines.
        Returns
        -------
        ancestors_idx : numpy.array of shape (?, max_n_ancestors = max_time_step) of int
            Idx of all ancestors for each token of batch. For token having has_ancestors_mask = False, a vector filled
            with no_ancestor_idx_filler is given. Family lines of size < max_time_step contain invalid token which is
            replaced with no_ancestor_idx_filler.
        """
        # For this function, relative target = ancestor

        # What to put as placeholder where tokens at coords do not have a target relationship
        if no_ancestor_idx_filler is None:
            no_target = self.library.invalid_idx
        else:
            no_target = no_ancestor_idx_filler

        # Number of token to handle here: n_tokens = ?
        n_tokens = coords.shape[1]

        # --- COORDS ---
        # coords of target related to tokens at coords
        coords_target = self.get_ancestors(coords)                                                                       # (max_n_relatives+1, ?)
        # Number of targets for each token (shape[0] - 1 because Oth dim stores batch_dim coords and others store time
        # dim coords). For ancestors max_n_relatives = max_time_step
        max_n_relatives = coords_target.shape[0]-1

        # --- MASK ---
        # Linear tile along target / relatives dim (eg. family line dim for target = ancestors)
        relatives_tile = np.tile(np.arange(max_n_relatives), reps = (n_tokens,1))                                        # (?, max_n_relatives)
        # mask : is target valid ie. it is related to a token having a target relationship (n_tokens dim) AND it is not
        # just a padding value but a real target (relatives dim)
        mask_valid_target = np.tile(self.tokens.n_ancestors[tuple(coords)], reps=(max_n_relatives,1)).transpose() > relatives_tile    # (?, max_n_relatives)

        # --- COORDS FOR SLICING ---
        # Coords of targets that are suitable for numpy slicing. By default slice in (0,0) -> default value will be
        # kept along relatives dim (ie. (batch_pos, 0)) for no_target areas and result of slicing will be replaced at
        # the last step by no_target using mask. Here we use pos = 0 but any pos that exist and is not out of range
        # would work.
        coords_target_slicing = np.zeros(shape=(n_tokens, max_n_relatives, 2), dtype=int)                                # (?, max_n_relatives, 2)
        # 0th dim is the prog position in the batch (pos_batch dim), it is the same for all along relatives dim
        coords_target_slicing[:, :, 0] = np.tile(coords_target[0, :], reps=(max_n_relatives, 1)).transpose()             # (?, max_n_relatives)
        # 1th dim is the pos of the target along the time dim coords_target[1:, :] to get rid of batch_pos dim contained
        # in 1st array which we already used at to determine the 0th dim
        # Only performing assignation where mask_valid_target is True to avoid getting INVALID_POS (using default 0
        # value pos for these non-valid areas)
        coords_target_slicing[:, :, 1][mask_valid_target] = coords_target[1:, :].transpose()[mask_valid_target]          # (?, max_n_relatives)
        coords_target_slicing = np.moveaxis(coords_target_slicing, source=2, destination=0)                              # (2, ?, max_n_relatives)

        # --- IDX ---
        # Getting idx
        target_idx = self.tokens.idx[tuple(coords_target_slicing)]                                                       # (?, max_n_relatives)
        # Replacing non-valid areas' temporary values with filler given in arg
        target_idx[np.logical_not(mask_valid_target)] = no_target                                                        # (?, max_n_relatives)

        return target_idx

    def get_parent_idx_of_step(self, step=None, no_parent_idx_filler=None):
        """
        Get parents idx of tokens at step.
        Parameters
        ----------
        step : int
            Step of token from which parent idx should be returned.
            By default, step = current step
        no_parent_idx_filler : int
            Fill value to return where tokens have no parent.
        Returns
        -------
        parents_idx : numpy.array of shape (batch_size,) of int
            Idx of parents.
        """
        # For this function, target = parent
        if step is None:
            step = self.curr_step
        # Assertion
        assert step < self.max_time_step, "Can not get information on token at step = %i as this step is out of range " \
                                          "(max_time_step = %i)"%(step, self.max_time_step)
        coords = self.coords_of_step(step=step)
        target_idx = self.get_parent_idx(coords, no_parent_idx_filler=no_parent_idx_filler)
        return target_idx

    def get_sibling_idx_of_step(self, step=None, no_sibling_idx_filler=None):
        """
        Get siblings idx of tokens at step.
        Parameters
        ----------
        step : int
            Step of token from which sibling idx should be returned.
            By default, step = current step
        no_sibling_idx_filler : int
            Fill value to return where tokens have no sibling.
        Returns
        -------
        siblings_idx : numpy.array of shape (batch_size,) of int
            Idx of siblings.
        """
        # For this function, target = sibling
        if step is None:
            step = self.curr_step
        # Assertion
        assert step < self.max_time_step, "Can not get information on token at step = %i as this step is out of range " \
                                          "(max_time_step = %i)"%(step, self.max_time_step)
        coords = self.coords_of_step(step=step)
        target_idx = self.get_sibling_idx(coords, no_sibling_idx_filler=no_sibling_idx_filler)
        return target_idx

    def get_ancestors_idx_of_step(self, step=None, no_ancestor_idx_filler=None):
        """
        Get ancestors idx of tokens at step.
        Parameters
        ----------
        step : int
            Step of token from which ancestors idx should be returned.
            By default, step = current step
        no_ancestor_idx_filler : int
            Fill value to return where tokens have no ancestors and for padding family tree lines.
        Returns
        -------
        ancestors_idx : numpy.array of shape (batch_size, max_n_ancestors = max_time_step) of int
            Idx of all ancestors for each token of batch. For token having has_ancestors_mask = False, a vector filled
            with no_ancestor_idx_filler is given. Family lines of size < max_time_step contain invalid token which is
            replaced with no_ancestor_idx_filler.
        """
        # For this function, target = ancestor
        if step is None:
            step = self.curr_step
        # Assertion
        assert step < self.max_time_step, "Can not get information on token at step = %i as this step is out of range " \
                                          "(max_time_step = %i)"%(step, self.max_time_step)
        coords = self.coords_of_step(step=step)
        target_idx = self.get_ancestors_idx(coords, no_ancestor_idx_filler=no_ancestor_idx_filler)
        return target_idx

    def count_tokens_idx (self, tokens_idx):
        """
        Creates a library size vector containing count of token idx (along ?1 dim) in tokens_idx for each line of
        tokens_idx. Eg: [ [2, 2, 1, 2, 5], [1, 1, 1, 0, 4] ] -> [0,1,3,0,0,1] [1,3,0,0,1,0] assuming n_library = 5.
        Parameters
        ----------
        tokens_idx : numpy.array of shape (?, ?1) of int
            Sample of tokens vectors.
        Returns
        -------
        counts : numpy.array of shape (?, n_library)
            Counts along ?1 dim.
        """
        eye = np.eye(self.n_library)
        counts = eye[tokens_idx].sum(axis=1)
        return counts

    # -------- GET INFO : FAMILY RELATIVES' PROPERTIES --------

    def get_property_of_relative(self, coords, relative, attr):
        """
        Returns the attribute (eg. phy_units, arity etc.) of the [relative] of tokens at coords. Fills with default
        value of attribute in VectTokens where tokens at coords do not have [relative].
        Parameters
        ----------
        coords : numpy.array of shape (2, ?,) of int
            Coords of tokens, 0th array in batch dim and 1th array in time dim.
        relative : str
            Supported relative args : "parent", "siblings".
        attr : str
            Attribute of VectTokens.
        Returns
        -------
        is_meaningful, attribute : numpy.array of shape (?,) of bool, numpy.array of shape (?, (+ depending on attr))
        The mask is_meaningful contains False where the attribute has no meaning and just contains a filler value
         (where there is no [relative]).
        """
        # Assertion
        supported_relationships = ["parent", "siblings"]
        assert relative in supported_relationships,"%s not supported, use one of %s"%(relative, supported_relationships)

        # Number of tokens considered here = ?
        n_tokens = coords.shape[1]

        # mask : does the relative exist ie. will the attribute we are retrieving have a meaning or will it just be
        # a filler
        is_meaningful = self.tokens.__getattribute__("has_%s_mask" % relative) [tuple(coords)]                  # (?,)
        # Number of tokens having relatives that do exist
        n_meaningful = is_meaningful.sum()

        # Initializing result
        attribute_filler_value  = self.tokens.__getattribute__("default_%s" % attr)
        extra_shape             = self.tokens.__getattribute__("%s" % attr).shape[2:]
        attribute_shape         = (n_tokens,) + extra_shape
        attribute_type          = self.tokens.__getattribute__("%s" % attr).dtype
        attribute = np.full(shape=attribute_shape, fill_value=attribute_filler_value, dtype=attribute_type)     # (?, (+ depending on attr))                                                                                    # (?,)

        # Coords of relatives
        coords_relative = self.__getattribute__("get_%s" % relative) (coords)                                   # (2, ?)
        # Coords relatives that do exist only
        coords_existing_relatives = coords_relative[:, is_meaningful]                                           # (2, n_meaningful)

        # Saving relatives' property in result arrays when it has a meaning otherwise leaving filler value
        attribute[is_meaningful] = self.tokens.__getattribute__("%s" % attr)[tuple(coords_existing_relatives)] # (n_meaningful,)

        return is_meaningful, attribute

    # -------- SET POSITIONAL INFO --------

    def set_parent(self, coords_dest, has_mask, pos_val):
        """
        Sets parent properties of tokens of coordinates coords_dest with new values given in args.
        Parameters
        ----------
        coords_dest : numpy.array of shape (2, ?) of int
            Coords where to set property, 0th array in batch dim and 1th array in time dim.
        has_mask : numpy.array of shape (?,) of bool
            New value to set for has_property_mask.
        pos_val : numpy.array of shape (?,) of int
            New value to set.
        """
        self.tokens.has_parent_mask [coords_dest[0], coords_dest[1]] = has_mask     # (?,) of bool
        self.tokens.parent_pos      [coords_dest[0], coords_dest[1]] = pos_val      # (?,) of int

    def set_children(self, coords_dest, has_mask, pos_val, nb):
        """
        Sets children properties of tokens of coordinates coords_dest with new values given in args.
        Parameters
        ----------
        coords_dest : numpy.array of shape (2, ?) of int
            Coords where to set property, 0th array in batch dim and 1th array in time dim.
        has_mask : numpy.array of shape (?,) of bool
            New value to set for has_property_mask.
        pos_val : numpy.array of shape (?, Tok.MAX_NB_CHILDREN) of int
            New value to set.
        nb : numpy.array of shape (?,) of int
            Number of relatives.
        """
        self.tokens.has_children_mask [coords_dest[0], coords_dest[1]] = has_mask     # (?,) of bool
        self.tokens.children_pos      [coords_dest[0], coords_dest[1]] = pos_val      # (?, Tok.MAX_NB_CHILDREN) of int
        self.tokens.n_children        [coords_dest[0], coords_dest[1]] = nb           # (?,) of int

    def set_siblings(self, coords_dest, has_mask, pos_val, nb):
        """
        Sets siblings properties of tokens of coordinates coords_dest with new values given in args.
        Parameters
        ----------
        coords_dest : numpy.array of shape (2, ?) of int
            Coords where to set property, 0th array in batch dim and 1th array in time dim.
        has_mask : numpy.array of shape (?,) of bool
            New value to set for has_property_mask.
        pos_val : numpy.array of shape (?, Tok.MAX_NB_CHILDREN) of int
            New value to set.
        nb : numpy.array of shape (?,) of int
            Number of relatives.
        """
        self.tokens.has_siblings_mask [coords_dest[0], coords_dest[1]] = has_mask     # (?,) of bool
        self.tokens.siblings_pos      [coords_dest[0], coords_dest[1]] = pos_val      # (?, Tok.MAX_NB_SIBLINGS) of int
        self.tokens.n_siblings        [coords_dest[0], coords_dest[1]] = nb           # (?,) of int

    def set_ancestors(self, coords_dest, has_mask, pos_val, nb):
        """
        Sets ancestors properties of tokens of coordinates coords_dest with new values given in args.
        Parameters
        ----------
        coords_dest : numpy.array of shape (2, ?) of int
            Coords where to set property, 0th array in batch dim and 1th array in time dim.
        has_mask : numpy.array of shape (?,) of bool
            New value to set for has_property_mask.
        pos_val : numpy.array of shape (?, max_time_step) of int
            New value to set.
        nb : numpy.array of shape (?,) of int
            Number of ancestors counting the token itself as its own ancestor.
        """
        self.tokens.has_ancestors_mask [coords_dest[0], coords_dest[1]] = has_mask     # (?,) of bool
        self.tokens.ancestors_pos      [coords_dest[0], coords_dest[1]] = pos_val      # (?, max_time_step) of int
        self.tokens.n_ancestors        [coords_dest[0], coords_dest[1]] = nb           # (?,) of int

    def register_ancestor(self, coords_dest, ):
        """
        Registers tokens located at coords_dest in their own ancestor records (as a token counts as its own ancestor)
        and updates the number of ancestors. Depths must be up-to-date for this function to perform correctly.
        Parameters
        ----------
        coords_dest : numpy.array of shape (2, ?) of int
            Coords of tokens which's ancestors records should be updated, 0th array in batch dim and 1th array in time dim.
        """
        # ? = number of tokens which need their ancestor to be updated
        n_tokens = coords_dest.shape[1]

        # Records of ancestors positions for token at coords_dest (ie vectors of size ? of family lines)
        records_ancestors_pos = self.tokens.ancestors_pos[tuple(coords_dest)]  # (?, max_time_step) of int

        # Coords of locations in records_ancestors_pos where the new ancestors should be placed. Since we are
        # registering token as their own ancestors, this is performed at their own depth in the family line.
        coords_new_ancestors = np.stack((  # (2, ?,) of int
            np.arange(n_tokens),
            # token dim coord (always = [1,2,3..] because records_ancestors_pos is already the subset of interest)
            self.tokens.depth[tuple(coords_dest)],  # ancestor line dim coord ie own depth of tokens to affect
        ), axis=0)

        # Registering tokens as their own ancestors ie. adding own token positions (time dim) in their records
        records_ancestors_pos[tuple(coords_new_ancestors)] = coords_dest[1]
        self.tokens.ancestors_pos[tuple(coords_dest)] = records_ancestors_pos

        # Update number of ancestors
        self.tokens.n_ancestors[tuple(coords_dest)] = self.tokens.depth[tuple(coords_dest)] + 1

        # Update number of ancestors
        self.tokens.has_ancestors_mask[tuple(coords_dest)] = True

        return None

    def set_units (self, coords_dest, new_is_constraining_phy_units, new_phy_units):
        """
        Sets units properties of tokens of coordinates coords_dest with new values given in args.
        Parameters
        ----------
        coords_dest : numpy.array of shape (2, ?) of int
            Coords where to set new tokens, 0th array in batch dim and 1th array in time dim.
        new_is_constraining_phy_units : numpy.array of shape (?,) of bool
            New value to set for is_constraining_phy_units.
        new_phy_units : numpy.array of shape (?, Tok.UNITS_VECTOR_SIZE) of float
            New value to set for phy_units.
        """
        self.tokens.is_constraining_phy_units [coords_dest[0], coords_dest[1]] = new_is_constraining_phy_units
        self.tokens.phy_units                 [coords_dest[0], coords_dest[1]] = new_phy_units
        return None

    # -------- SET NON_POSITIONAL INFO --------

    def set_static_units_from_idx (self, coords_dest, tokens_idx_src):
        """
        Sets units properties of tokens at coords_dest from units of tokens tokens_idx_src as given in the library.
        Parameters
        ----------
        coords_dest : numpy.array of shape (2, ?) of int
            Coords where to set new tokens, 0th array in batch dim and 1th array in time dim.
        tokens_idx_src : numpy.array of shape (batch_size,)
            Index of tokens from which units should be taken.
        """
        self.set_units(coords_dest                    = coords_dest,
                       new_is_constraining_phy_units  = self.lib("is_constraining_phy_units")[tokens_idx_src],
                       new_phy_units                  = self.lib("phy_units")[tokens_idx_src, :]
                       )
        return None

    def set_non_positional_from_idx (self, coords_dest, tokens_idx_src):
        """
        Sets non_positional properties and index of new tokens as given in the library.
        Parameters
        ----------
        coords_dest : numpy.array of shape (2, ?) of int
            Coords where to set new tokens, 0th array in batch dim and 1th array in time dim.
        tokens_idx_src : numpy.array of shape (batch_size,)
            Index of new tokens.
        """
        # ------------------------ non_positional ------------------------
        # Index
        self.tokens.idx   [coords_dest[0], coords_dest[1]] = tokens_idx_src
        # ------------------------ non_positional ------------------------
        # ---- Token representation
        # ( name                    :  str (<MAX_NAME_SIZE) )
        # ( sympy_repr              :  str (<MAX_NAME_SIZE) )
        # ---- Token main properties
        self.tokens.arity        [coords_dest[0], coords_dest[1]] = self.lib("arity")        [tokens_idx_src]
        self.tokens.complexity   [coords_dest[0], coords_dest[1]] = self.lib("complexity")   [tokens_idx_src]
        self.tokens.var_type     [coords_dest[0], coords_dest[1]] = self.lib("var_type")     [tokens_idx_src]
        self.tokens.var_id       [coords_dest[0], coords_dest[1]] = self.lib("var_id")       [tokens_idx_src]
        # ( function                :  callable or None )
        # ---- Physical units : behavior id
        self.tokens.behavior_id  [coords_dest[0], coords_dest[1]] = self.lib("behavior_id")  [tokens_idx_src]
        # ---- Physical units : power
        self.tokens.is_power     [coords_dest[0], coords_dest[1]] = self.lib("is_power")     [tokens_idx_src]
        self.tokens.power        [coords_dest[0], coords_dest[1]] = self.lib("power")        [tokens_idx_src]
        return None

    # -------- MISCELLANEOUS--------

    def move_dummies (self, coords_src, coords_dest, do_update_relationships_pos = True, do_fill_with_void = True):
        """
        Moves dummies from coords_src to coords_dest.
        ! This function only works for dummies placed as placeholders that are completing programs. !
             -> Does not work for moved tokens that have sibling relationships to each others
                and does not update dummies' children nor their ancestors. This function works for
                dummies completing programs as they do not have children, do not have parent dummies
                nor sibling dummies.
        Parameters
        ----------
        coords_src : numpy.array of shape (2, ?) of int
            Coords of tokens to move, 0th array in batch dim and 1th array in time dim.
        coords_dest : numpy.array of shape (2, ?) of int
            Coords where to move tokens, 0th array in batch dim and 1th array in time dim.
        do_update_relationships_pos : bool
            If True, updates position of moved tokens in other tokens representing family relationships
            (parent, siblings,).
            ! Does not update siblings relationships between moved tokens and does not update children. !
        do_fill_with_void : bool
            If True, fills coords_src with void of invalid tokens after move where there is no overlap
            between src and dest.
        """

        # ----------------------------------------------------------------
        # ------------------------ COPYING TOKENS ------------------------
        # ----------------------------------------------------------------

        # ------------ Index in library ------------
        self.tokens.idx                       [tuple(coords_dest)] = self.tokens.idx                       [tuple(coords_src)]
        # ------------ non_positional properties ------------
        self.tokens.arity                     [tuple(coords_dest)] = self.tokens.arity                     [tuple(coords_src)]
        self.tokens.complexity                [tuple(coords_dest)] = self.tokens.complexity                [tuple(coords_src)]
        self.tokens.var_type                  [tuple(coords_dest)] = self.tokens.var_type                  [tuple(coords_src)]
        self.tokens.var_id                    [tuple(coords_dest)] = self.tokens.var_id                    [tuple(coords_src)]
        self.tokens.behavior_id               [tuple(coords_dest)] = self.tokens.behavior_id               [tuple(coords_src)]
        self.tokens.is_power                  [tuple(coords_dest)] = self.tokens.is_power                  [tuple(coords_src)]
        self.tokens.power                     [tuple(coords_dest)] = self.tokens.power                     [tuple(coords_src)]

        # ------------ semi_positional properties ------------
        self.tokens.is_constraining_phy_units [tuple(coords_dest)] = self.tokens.is_constraining_phy_units [tuple(coords_src)]
        self.tokens.phy_units                 [tuple(coords_dest)] = self.tokens.phy_units                 [tuple(coords_src)]

        # ------------ Positional properties ------------
        # Must not be copied
        #self.tokens.pos                       [tuple(coords_dest)] = self.tokens.pos                       [tuple(coords_src)]
        #self.tokens.pos_batch                 [tuple(coords_dest)] = self.tokens.pos_batch                 [tuple(coords_src)]
        # ---- Depth ----
        self.tokens.depth                     [tuple(coords_dest)] = self.tokens.depth                     [tuple(coords_src)]
        # ---- Family relationships ----
        # Token family relationships: family mask
        self.tokens.has_parent_mask           [tuple(coords_dest)] = self.tokens.has_parent_mask           [tuple(coords_src)]
        self.tokens.has_siblings_mask         [tuple(coords_dest)] = self.tokens.has_siblings_mask         [tuple(coords_src)]
        self.tokens.has_children_mask         [tuple(coords_dest)] = self.tokens.has_children_mask         [tuple(coords_src)]
        self.tokens.has_ancestors_mask        [tuple(coords_dest)] = self.tokens.has_ancestors_mask        [tuple(coords_src)]
        # Token family relationships: pos
        self.tokens.parent_pos                [tuple(coords_dest)] = self.tokens.parent_pos                [tuple(coords_src)]
        self.tokens.siblings_pos              [tuple(coords_dest)] = self.tokens.siblings_pos              [tuple(coords_src)]
        self.tokens.children_pos              [tuple(coords_dest)] = self.tokens.children_pos              [tuple(coords_src)]
        self.tokens.ancestors_pos             [tuple(coords_dest)] = self.tokens.ancestors_pos             [tuple(coords_src)]
        # Token family relationships: numbers
        self.tokens.n_siblings                [tuple(coords_dest)] = self.tokens.n_siblings                [tuple(coords_src)]
        self.tokens.n_children                [tuple(coords_dest)] = self.tokens.n_children                [tuple(coords_src)]
        self.tokens.n_ancestors               [tuple(coords_dest)] = self.tokens.n_ancestors               [tuple(coords_src)]

        # ----------------------------------------------------------------
        # -------------------- UPDATING RELATIONSHIPS --------------------
        # ----------------------------------------------------------------

        def update_relationships_pos_of_moved_tokens (coords_b_move, coords_a_move):
            """
            Updates position of dummy tokens that were moved in other tokens corresponding to family relationships
            (parent, siblings).
            ! This function only works for dummies placed as placeholders that are completing programs. !
                 -> Does not work for moved tokens that have sibling relationships to each others
                    and does not update dummies' children nor their ancestors. This function works for
                    dummies completing programs as they do not have children, do not have parent dummies
                    nor sibling dummies.
            Parameters
            ----------
            coords_b_move : numpy.array of shape (2, ?) of int
                Coords of tokens before move, 0th array in batch dim and 1th array in time dim.
            coords_a_move : numpy.array of shape (2, ?) of int
                Coords of tokens after move, 0th array in batch dim and 1th array in time dim.
            """

            # ------------ Updating ancestors pos ------------
            # (as the dummy token counts as its own ancestor, and it was just moved)
            self.register_ancestor(coords_dest = coords_a_move)

            # ------------ Informing parent ------------

            # Do moved tokens have parents : mask
            mask_has_parent = self.tokens.has_parent_mask[tuple(coords_a_move)]                         # (?,) of bool

            # Coords of parent when available
            coords_valid_parent = np.stack((                                                            # (2, ?0,) of int
                # ?0 = mask_has_parent.sum() < ?
                coords_a_move[0],  # batch dim coord -> same as tokens     # (?,) of int
                self.tokens.parent_pos[tuple(coords_a_move)],   # time dim coord  -> parent             # (?,) of int
                ), axis=0
            )[:, mask_has_parent]  # where has_parent_mask is True (ie. pos are valid)

            # pos of children of parent of moved token
            pos_parent_children = self.tokens.children_pos[tuple(coords_valid_parent)]                  # (?0, Tok.MAX_NB_CHILDREN)

            # Pos (before move) of moved tokens which have parents, tiled to match Tok.MAX_NB_CHILDREN
            # must use coords_b_move
            pos_tokens_w_parent_b_move_tiled = np.tile(                                                 # (2, ?0,) of int
                coords_b_move[1],  # 'batch' of pos (before move) of moved tokens (in time dim)         # (?,) of int
                  (Tok.MAX_NB_CHILDREN, 1,)
                ).transpose()[mask_has_parent, :]

            # Pos (after move) of moved tokens which have parents
            pos_tokens_w_parent_a_move = coords_a_move[1][mask_has_parent]                              # (?0,) of int

            # Update the right pos child (ie. the one where child pos = pos of token before move)
            # Update with new position (after move) where it is relevant (ie.
            pos_parent_children[
                pos_parent_children == pos_tokens_w_parent_b_move_tiled                                 # (?0,) of int (if size > ?0 => double child)
                                ] = pos_tokens_w_parent_a_move                                          # (?0,) of int

            # Update matrix of properties
            self.tokens.children_pos[tuple(coords_valid_parent)] = pos_parent_children

            # ------------ Informing sibling ------------

            # Do moved tokens have siblings : mask
            mask_has_sibling = self.tokens.has_siblings_mask[tuple(coords_a_move)]                       # (?,) of bool

            # Position of valid siblings
            # ?1 = mask_has_sibling.sum() < ? # where mask_has_sibling is True (ie. pos are valid)
            pos_valid_sibling = self.tokens.siblings_pos[tuple(coords_a_move)][mask_has_sibling, :]      # (?1, Tok.MAX_NB_SIBLINGS) of int

            # Coords of siblings when available
            coords_valid_sibling = np.stack((                                                            # (2, ?1,) of int
                coords_a_move[0][mask_has_sibling],              # batch dim coord -> same as tokens     # (?1,) of int
                pos_valid_sibling[:, 0],                         # time dim coord  -> siblings           # (?1,) of int
                ), axis=0
            )

            # Position of sibling's sibling
            pos_sibling_sibling = self.tokens.siblings_pos[tuple(coords_valid_sibling)]                  # (?1, Tok.MAX_NB_SIBLINGS) of int

            # Updating sibling's sibling pos to token pos (after move)
            pos_sibling_sibling = coords_a_move[1][mask_has_sibling][:, np.newaxis]                      # (?1, Tok.MAX_NB_SIBLINGS) of int

            # Update matrix of properties
            self.tokens.siblings_pos[tuple(coords_valid_sibling)] = pos_sibling_sibling                  # (?1, Tok.MAX_NB_SIBLINGS) of int

            return None

        if do_update_relationships_pos:
            update_relationships_pos_of_moved_tokens(coords_b_move = coords_src, coords_a_move = coords_dest)

        # ----------------------------------------------------------------
        # -------------------- UPDATING RELATIONSHIPS --------------------
        # ----------------------------------------------------------------

        if do_fill_with_void :

            # Is token a src  coordinates : mask
            mask_src  = self.coords_to_mask (coords = coords_src )                  # (batch_size, max_time_step,) of bool
            # Is token a dest coordinates : mask
            mask_dest = self.coords_to_mask (coords = coords_dest)                  # (batch_size, max_time_step,) of bool

            # Where to fill with void : mask
            # Ie. fill src coordinates if they are not also dest coordinates
            mask_fill_void = np.logical_and(mask_src, np.logical_not(mask_dest))    # (batch_size, max_time_step,) of bool
            # Coords where to fill with void
            n_fill_void, coords_fill_void = self.mask_to_coords(mask_fill_void)     # int, (2, n_fill_void) of int

            # Perform fill
            self.fill_with_void(coords_dest = coords_fill_void)

        return None

    def fill_with_void(self, coords_dest):
        """
        Helper function that fills coords_dest with void invalid token.
        Parameters
        ----------
        coords_dest : numpy.array of shape (2, ?) of int
            Coords where to fill with void, 0th array in batch dim and 1th array in time dim.
        """

        # ------------ Index + non_positional properties ------------
        self.set_non_positional_from_idx (coords_dest = coords_dest, tokens_idx_src = self.invalid_idx)

        # ------------ semi_positional properties ------------
        self.set_static_units_from_idx   (coords_dest = coords_dest, tokens_idx_src = self.invalid_idx)

        # ------------ Positional properties ------------
        # Must not be modified
        # self.tokens.pos
        # self.tokens.pos_batch
        # ---- Depth ----
        self.tokens.depth                     [tuple(coords_dest)] = Tok.INVALID_DEPTH  # (?,) of int
        # ---- Family relationships ----
        # Token family relationships: family mask
        self.tokens.has_parent_mask           [tuple(coords_dest)] = False              # (?,) of bool
        self.tokens.has_siblings_mask         [tuple(coords_dest)] = False              # (?,) of bool
        self.tokens.has_children_mask         [tuple(coords_dest)] = False              # (?,) of bool
        self.tokens.has_ancestors_mask        [tuple(coords_dest)] = False              # (?,) of bool
        # Token family relationships: pos
        self.tokens.parent_pos                [tuple(coords_dest)] = Tok.INVALID_POS    # (?,) of int
        self.tokens.siblings_pos              [tuple(coords_dest)] = Tok.INVALID_POS    # (?,) of int
        self.tokens.children_pos              [tuple(coords_dest)] = Tok.INVALID_POS    # (?,) of int
        self.tokens.ancestors_pos             [tuple(coords_dest)] = Tok.INVALID_POS    # (?,) of int
        # Token family relationships: numbers
        self.tokens.n_siblings                [tuple(coords_dest)] = 0                  # (?,) of int
        self.tokens.n_children                [tuple(coords_dest)] = 0                  # (?,) of int
        self.tokens.n_ancestors               [tuple(coords_dest)] = 0                  # (?,) of int

        return None

    # -----------------------------------------------------------------------------------------------------------------
    # ------------------------------------------- UTILS : PROGRAM MANAGEMENT ------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------

    def compute_sum_arities(self, step=0):
        """
        Computes total arities of programs along time dim from step = 0 to step = step.
        Parameters
        ----------
        step : int
            Step where to end summation.
        Returns
        -------
        sum : numpy.array of shape (batch_size,) of int
            Sum along time dim of size (batch_size,).
        """
        # Initializing to 1 as superparent has 1 child (initial loose end)
        # Arities of real tokens only
        return np.ones(self.batch_size, dtype=int) + self.tokens.arity[:, :step].sum(axis=1)  # sum along time dim

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------ UTILS : INTERFACE -----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def n_completed(self):
        """
        Lengths of programs when completed by dummies.
        Returns
        -------
        completed_lengths : numpy.array of shape (batch_size,) of int
        """
        # Computes length when completed by dummies.
        return self.n_lengths + self.n_dummies  # (batch_size,) of int

    @property
    def n_dangling(self):
        """
        Vocabulary used in symbolic regression research.
        Returns
        -------
        n_dummies : numpy.array of shape (batch_size,) of int
        """
        return self.n_dummies

    @property
    def n_complexity(self):
        """
        Complexities of programs.
        Returns
        -------
        complexity : numpy.array of shape (batch_size,) of int
        """
        # Summing over time dim
        return self.tokens.complexity.sum(axis=1) # (batch_size,) of int

    @property
    def n_free_const_occurrences(self):
        """
        Number of occurrences of free const in programs
        Returns
        -------
        occurrences : numpy.array of shape (batch_size,) of int
        """
        return (self.tokens.var_type == 2).sum(axis=1) # (batch_size,) of int

    def get_token(self, coords):
        """
        Returns token objects at coords.
        Parameters
        ----------
        coords : numpy.array of shape (2, ?) of int
            Coords of tokens to return, 0th array in batch dim and 1th array in time dim.
        Returns
        -------
        tokens : numpy.array of shape (?,) of token.Token
        """
        return self.library.lib_tokens[self.tokens.idx[tuple(coords)]]

    def as_tokens(self):
        """
        Returns all tokens contained in VectPrograms.
        Returns
        -------
        tokens : numpy.array of shape (batch_size,max_time_step,) of token.Token
        """
        return self.library.lib_tokens[self.tokens.idx]

    def get_prog_tokens(self, prog_idx=0):
        """
        Returns a list of tokens representing a single program of idx = prog_idx.
        Discards void tokens beyond program length.
        Parameters
        ----------
        prog_idx : int
            Index of program in batch.
        Returns
        -------
        tokens : numpy.array of token.Token
            Tokens making up program.
        """
        length = self.n_completed        [prog_idx]
        # Keeping only tokens before actual length of program
        # taking dummies via n_completed rather than n_lengths so tree is complete
        idx    = self.tokens.idx         [prog_idx, 0:length]
        tokens = self.library.lib_tokens [idx]
        return tokens

    def get_prog(self, prog_idx=0):
        """
        Returns a Program object of program of idx = prog_idx in batch.
        Discards void tokens beyond program length.
        Parameters
        ----------
        prog_idx : int
            Index of program in batch.
        Returns
        -------
        program : program.Program
            Program making up symbolic function.
        """
        tokens = self.get_prog_tokens(prog_idx=prog_idx)
        is_physical              = self.is_physical        [prog_idx]
        free_const_values        = self.free_consts.values [prog_idx]
        prog = Program(tokens            = tokens,
                       library           = self.library,
                       is_physical       = is_physical,
                       free_const_values = free_const_values,
                       candidate_wrapper = self.candidate_wrapper,
                       )
        return prog

    def get_programs_array (self):
        """
        Returns all programs in this vector of programs as a numpy array of Program objects.
        Discards void tokens beyond program length.
        Returns
        -------
        program : numpy.array of program.Program of shape (batch_size,)
            Array of programs representing symbolic functions.
        """
        progs = np.array([self.get_prog(i) for i in range(self.batch_size)])
        return progs

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- REPRESENTATION : INFIX RELATED -----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def get_infix_str (self, prog_idx = 0):
        """
        Computes infix str representation of a program.
        (which is the usual way to note symbolic function: +34 (in polish notation) = 3+4 (in infix notation))
        Parameters
        ----------
        prog_idx : int
            Index of program in VectPrograms.
        Returns
        -------
        program_str : str
        """
        prog = self.get_prog(prog_idx = prog_idx)
        res  = prog.get_infix_str()
        return res

    def get_infix_sympy (self, prog_idx = 0, do_simplify = True):
        """
        Returns sympy symbolic representation of a program.
        Parameters
        ----------
        prog_idx : int
            Index of program in VectPrograms.
        do_simplify : bool
            If True performs a symbolic simplification of program.
        Returns
        -------
        program_sympy : sympy.core
            Sympy symbolic function. It is possible to run program_sympy.evalf(subs={'x': 2.4}) where 'x' is a variable
            appearing in the program to evaluate the function with x = 2.4.
        """
        prog = self.get_prog(prog_idx = prog_idx)
        res  = prog.get_infix_sympy(do_simplify=do_simplify)
        return res

    def get_infix_pretty (self, prog_idx = 0, do_simplify = True):
        """
        Returns a printable ASCII sympy.pretty representation of a program.
        Parameters
        ----------
        prog_idx : int
            Index of program in VectPrograms.
        do_simplify : bool
            If True performs a symbolic simplification of program.
        Returns
        -------
        program_pretty_str : str
        """
        prog = self.get_prog(prog_idx = prog_idx)
        res  = prog.get_infix_pretty(do_simplify=do_simplify)
        return res

    def get_infix_latex (self, prog_idx = 0, replace_dummy_symbol = True, new_dummy_symbol = "?", do_simplify = True):
        """
        Returns an str latex representation of a program.
        Parameters
        ----------
        prog_idx : int
            Index of program in VectPrograms.
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
        prog = self.get_prog(prog_idx = prog_idx)
        res  = prog.get_infix_latex(replace_dummy_symbol = replace_dummy_symbol,
                                    new_dummy_symbol = new_dummy_symbol,
                                    do_simplify = do_simplify,)
        return res

    def get_infix_fig (self,
                       prog_idx = 0,
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
        prog_idx : int
            Index of program in VectPrograms.
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
        prog = self.get_prog(prog_idx=prog_idx)
        res = prog.get_infix_fig(
                       replace_dummy_symbol = replace_dummy_symbol,
                       new_dummy_symbol = new_dummy_symbol,
                       do_simplify = do_simplify,
                       show_superparent_at_beginning = show_superparent_at_beginning,
                       text_size = text_size,
                       text_pos  = text_pos,
                       figsize   = figsize,
                        )
        return res

    def get_infix_image(self,
                        prog_idx = 0,
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
        prog_idx : int
            Index of program in VectPrograms.
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
        prog = self.get_prog(prog_idx=prog_idx)
        res = prog.get_infix_image(
                        replace_dummy_symbol = replace_dummy_symbol,
                        new_dummy_symbol = new_dummy_symbol,
                        do_simplify = do_simplify,
                        text_size    = text_size,
                        text_pos     = text_pos,
                        figsize      = figsize,
                        dpi          = dpi,
                        fpath        = fpath,
                        )
        return res

    def show_infix(self,
                   prog_idx=0,
                   replace_dummy_symbol = True,
                   new_dummy_symbol = "?",
                   do_simplify = True,
                   text_size=16,
                   text_pos=(0.0, 0.5),
                   figsize=(10, 2),
                   ):
        """
        Shows pyplot (figure, axis) containing analytic symbolic function program.
        Parameters
        ----------
        prog_idx : int
            Index of program in VectPrograms.
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
        prog = self.get_prog(prog_idx=prog_idx)
        res = prog.show_infix(
                   replace_dummy_symbol = replace_dummy_symbol,
                   new_dummy_symbol = new_dummy_symbol,
                   do_simplify = do_simplify,
                   text_size=text_size,
                   text_pos=text_pos,
                   figsize=figsize,
                        )
        return res

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------- REPRESENTATION : TREE REPR -------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def get_tree_graph (self, prog_idx = 0,
        n_dim_units       = 3,
        shape             = "circle",
        constraint_color  = "blue",
        vanilla_color     = "black",
        dummy_color       = "red",
        special_color     = "red",
        special_color_pos = None,
        edge_color        = "grey",
        optimize_for_tex  = False,
        phy_units_display = None,
                 ):
        """
        Returns a graph representation of a program tree encoding parent, children relationships, physical units
        and position.
        Parameters
        ----------
        prog_idx : int
             Index of program in VectPrograms.
        shape : str
            Shape of nodes in graph (passed to pygraphviz.AGraph.add_node)
        n_dim_units : int
            Does not display units beyond n_dim_units <= Tok.UNITS_VECTOR_SIZE.
        constraint_color : str
            Color for tokens having constraining static physical units.
        vanilla_color : str
            Default color for tokens.
        dummy_color : str
            Color for dummy placeholder tokens.
        special_color : str
             Color for special token.
        special_color_pos : int
            Position of special token.
        edge_color : str
            Color of edges of graph
        optimize_for_tex : bool
            Should the graph be optimized for latex export via dot2tex (prettier).
        phy_units_display : list of str
            List of names of units to display units as power of those names. Only works with optimize_for_tex = True.
            Must have len(phy_units_display) >= n_dim_units <= Tok.UNITS_VECTOR_SIZE.
            Default value = None => will use: ["L", "T", "M", "I", "\\theta", "N", "J"] (SI dimensions).
        Returns
        -------
        graph : pygraphviz.AGraph
        """

        # Initializing graph repr
        G = pgv.AGraph(directed=True)

        # Useful documentation
        # node attributes: http://www.graphviz.org/doc/info/attrs.html
        # dot2tex (relevant for label of nodes): https://dot2tex.readthedocs.io/en/latest/usage_guide.html#labels

        # Utils function for units display
        def format_unit_str   (units):
            units = units[0:n_dim_units]
            units_str = ""
            for i,unit in enumerate(units):
                u_val = str(unit)
                u_val = u_val.rstrip('0').rstrip('.') if '.' in u_val else u_val
                units_str += u_val + ", "
            units_str = "(" + units_str[:-1] + ")"
            return units_str
        # Utils function for units display : via tex
        def format_unit_str_for_tex (units):
            units = units[0:n_dim_units]
            units_str = ""
            for i,unit in enumerate(units):
                u_val = str(unit)
                u_val = u_val.rstrip('0').rstrip('.') if '.' in u_val else u_val
                units_str += phy_units_display[i] + "^{" + u_val + "}, "
            units_str =  units_str[:-2]
            return units_str

        # Units display config
        if phy_units_display is None:
            phy_units_display = ["L", "T", "M", "I", "\\theta", "N", "J"]  # works in tex mode only
        if optimize_for_tex:
            format_unit_func = format_unit_str_for_tex
        else:
            format_unit_func = format_unit_str

        # Utils function to infer color of node
        def color_from_cursor (cursor):
            if cursor.pos == special_color_pos:
                color = special_color  # special color token
            elif cursor.token_prop("idx") == self.library.dummy_idx:
                color = dummy_color  # dummy token color
            elif cursor.token_prop("is_constraining_phy_units"):
                # is constraining due to position in program, not nature of token
                color = constraint_color  # positional constraining units
            elif cursor.token.is_constraining_phy_units:
                color = constraint_color  # constraining units token
            else:
                color = vanilla_color  # vanilla token color
            return color

        # Utils function to add a node representing a token
        def add_token_node    (name, pos, units, color):
            units_str = format_unit_func (units)
            pos_str   = str(pos).zfill(len(str(self.max_time_step)))
            # For tex display
            if optimize_for_tex:
                # Must use \\ to get \ in tex via dot2tex (see dot2tex doc)
                label_name    = "\\mathbf{[%s]}" % name
                label_pos_str = "-%s-" % pos_str
                label_units_str = units_str
                label = "$\\begin{array}{c} %s \\\ \\scriptscriptstyle{%s} \\\ \\scriptscriptstyle{%s} \\end{array}$" \
                        % (label_name, label_pos_str, label_units_str)
                # It is important to use argument texlbl instead of label for latex export down the line
                # See dot2tex doc
                G.add_node(pos,
                           texlbl = label,
                           fontcolor=color,
                           color = color,
                           shape = shape,
                           fontsize = 16,)
            # For normal display
            else:
                label = "%s: %s \n%s " % (pos_str, name, units_str)
                G.add_node(pos, label=label, fontcolor=color, color=color, fontsize=10)
            return None

        # Utils function to add a node representing a token from a Cursor object
        def node_from_cursor  (cursor):
            units = cursor.token_prop("phy_units")
            if not cursor.token_prop("is_constraining_phy_units"):
                units = np.full (units.shape, '-') # Fill with - instead of nan if there are no units in token
            add_token_node (name  = cursor.token.name,
                            pos   = cursor.pos,
                            # Getting units constraints from VectPrograms (not static units constraints in tokens)
                            units = units,
                            color = color_from_cursor (cursor),)
            return None

        # Initializing cursor pointing to first token
        cursor = Cursor(self, prog_idx = prog_idx, pos = 0)
        # Handling superparent
        superparent     = self.library.superparent
        superparent_pos = "--"
        add_token_node (name  = superparent.name,
                        pos   = superparent_pos,
                        units = superparent.phy_units,
                        color = constraint_color)
        # Handling first token
        node_from_cursor (cursor)
        # First edge
        G.add_edge(superparent_pos, cursor.pos, color=edge_color)
        # Iterating through tokens
        for pos in range (1, self.n_completed[prog_idx]):  # start at pos = 1
            # Update cursor pos
            cursor.set_pos(pos)
            # Handling pos-th token
            node_from_cursor(cursor)
            # Edge: pos-th token <-> its parent
            G.add_edge(cursor.parent.pos, cursor.pos, color=edge_color)
        return G

    def get_tree_latex (self, prog_idx = 0, fpath = None, **args_get_tree_graph):
        """
        Returns a latex code of the tree program.
        Parameters
        ----------
        prog_idx : int
             Index of program in VectPrograms.
        fpath : str or None
            Path where to save latex code. By default = None, nothing is saved.
        args_get_tree_graph : dict
            Additional argument for customizing tree plot, passed to get_tree_graph.
        Returns
        -------
        tree_latex : str
        """
        # Useful doc
        # https://stackoverflow.com/questions/35830447/python-graphs-latex-math-rendering-of-node-labels
        # https://dot2tex.readthedocs.io/en/latest/usage_guide.html#command-line-options

        # Generate graph
        G = self.get_tree_graph(prog_idx = prog_idx, optimize_for_tex = True, **args_get_tree_graph)
        G.layout(prog='dot')
        # Generating latex code via dot2tex
        tree_latex = dot2tex.dot2tex(G.to_string(),
                                     format  = 'pgf',  # works best for this application
                                     texmode = "raw",  # avoids deletion of latex special characters
                                     autosize = True,  # node auto sizing
                                     crop     = True,)
        # Save
        if fpath is not None:
            with open(fpath, 'w') as f:
                f.write(tree_latex)
        return tree_latex

    def get_tree_image (self, prog_idx = 0, fpath = None, **args_get_tree_graph):
        """
        Returns an image of the tree program (less pretty than get_tree_image_via_tex).
        Parameters
        ----------
        prog_idx : int
             Index of program in VectPrograms.
        fpath : str or None
            Path where to save image. By default, = None, nothing is saved.
        args_get_tree_graph : dict
            Additional argument for customizing tree plot, passed to get_tree_graph.
        Returns
        -------
        image : numpy.array
        """
        # Even if we don't want to save image (fpath = None), we have to save it in temp file, load it and then delete
        # the temp file as AGraph does not support direct image generation

        # Should we save result
        if fpath is None:
            do_save = False
        else:
            do_save = True

        # If result should not be saved, set up a temporary file path + folder
        if not do_save:
            # Name for temp file and folder
            prog_idx_str = str(prog_idx).zfill(len(str(self.batch_size)))
            temp_name = "temp_%s" %prog_idx_str
            # Folder
            if temp_name not in os.listdir():
                os.mkdir(temp_name)
            # File
            fpath = os.path.join(temp_name, temp_name+".png")

        G = self.get_tree_graph (prog_idx = prog_idx, **args_get_tree_graph)
        # save an image file
        G.layout(prog='dot')  # use dot
        G.draw(fpath)

        # load image from file
        img_np = plt.imread(fpath)[:,:,0:3]
        img_np_int = (img_np*255).astype('uint8')
        img = Image.fromarray(img_np_int.astype('uint8'), 'RGB')

        # Deleting temp file and folder if we don't want to save image
        if not do_save:
            shutil.rmtree(temp_name)

        return img

    def get_tree_image_via_tex (self, prog_idx = 0, fname = None, dpi = 300, **args_get_tree_graph):
        """
        Returns an image of the tree program going through latex (prettier than get_tree_image by leveraging features
        of latex that are not available in AGraph.draw).
        Exports AGraph -> .tex (via dot2tex) -> .pdf (via PDFLaTeX) -> image (via pdf2image)
        Parameters
        ----------
        prog_idx : int
             Index of program in VectPrograms.
        fname : str or None
            Name of files (vectorial .pdf and raster .png) to save. By default, = None, nothing is saved.
        dpi : int
            Pixels density.
        args_get_tree_graph : dict
            Additional argument for customizing tree plot, passed to get_tree_graph.
        Returns
        -------
        image : PIL.Image.Image
        """

        # Useful doc:
        # https://stackoverflow.com/questions/64841849/how-to-convert-latex-to-image-in-python

        # ---- Save or not ----
        if fname is None:
            do_save = False
        else:
            do_save = True

        # ---- Temp folder ----
        # Name for temp file and folder
        prog_idx_str = str(prog_idx).zfill(len(str(self.batch_size)))
        temp_name = "temp_%s" %prog_idx_str
        if temp_name not in os.listdir():
            os.mkdir(temp_name)
        # Temp .tex file
        fpath_tex = os.path.join(temp_name, temp_name+".tex")

        # ---- fpath ----
        if do_save:
            fname_pdf = fname
        else:
            fname_pdf = temp_name  # PDFLaTeX adds ".pdf"

        # ---- Create tex ----
        self.get_tree_latex(prog_idx = prog_idx, fpath = fpath_tex, **args_get_tree_graph)

        # ---- Create pdf from tex ----
        pdfl = PDFLaTeX.from_texfile(fpath_tex) # PDFLaTeX adds ".pdf"
        pdfl.set_pdf_filename(fname_pdf)
        pdf, log, completed_process = pdfl.create_pdf(keep_pdf_file=True)

        # ---- pdf to image ----
        img = pdf2image.convert_from_path(fname_pdf + ".pdf", dpi=dpi)[0]

        # ---- Delete temp folder ----
        shutil.rmtree(temp_name)
        if not do_save:
            os.remove(fname_pdf + ".pdf") # If no save then delete .pdf
        else:
            img.save(fname + ".png") # If save keep .pdf + save image .png
        return img

    def show_tree (self, prog_idx = 0, via_tex = False, figsize = (30,30), dpi = 300, **args_get_tree_graph):
        """
        Shows pyplot (figure, axis) containing tree of program.
        Parameters
        ----------
        prog_idx : int
             Index of program in VectPrograms.
        via_tex : bool
            If True uses get_tree_image_via_tex (prettier), else uses get_tree_image.
        figsize : tuple of int
            Size of figure, passed to pyplot.
        dpi : int
            Pixels density (only possible to adjust dpi via tex).
        args_get_tree_graph : dict
            Additional argument for customizing tree plot, passed to get_tree_graph.
        """
        # Creating image
        if via_tex:
            img = self.get_tree_image_via_tex (prog_idx = prog_idx, fname = None, dpi = dpi, **args_get_tree_graph)
        else:
            img = self.get_tree_image (prog_idx = prog_idx, **args_get_tree_graph)
        # Figure
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.set_title("Program %s (step = %i)" % (str(prog_idx).zfill(len(str(self.batch_size))), self.curr_step))
        ax.axis('off')
        ax.imshow(img)
        plt.show()
        return None

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------- REPRESENTATION : PRINT ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def idx_as_names (self, idx):
        return self.lib_names[idx]

    def full_status (self):
        def print_prop_vect(prop_vect):
            for i in range(self.batch_size):
                print("%s : "%(prop_vect[i]), self.idx_as_names(self.tokens.idx[i]))
        def print_prop_matrix(prop_matrix):
            for i in range(self.batch_size):
                print("%s : "%(prop_matrix[i]), self.idx_as_names(self.tokens.idx[i]))
        def print_prop_units(prop_matrix):
            for i in range(self.batch_size):
                str_units = ""
                for step in range (self.max_time_step):
                    unit = str(prop_matrix[i, step][0:3])
                    str_units += "%s, "%(unit)
                print("%s : " % (str_units), self.idx_as_names(self.tokens.idx[i]))
        print("\n")
        print("-----------------------------------")
        print("---------- VectPrograms ----------")
        print("-----------------------------------")
        print("shape = (batch_size : %i, max_time_step : %i)" % (self.batch_size, self.max_time_step))
        print("curr_step = %i"%(self.curr_step))
        print("----------------- VECT PROPERTIES -----------------")
        print("---- is_complete ----")
        print_prop_vect(self.is_complete)
        print("---- n_lengths ----")
        print_prop_vect(self.n_lengths)
        print("---- n_dummies ----")
        print_prop_vect(self.n_dummies)
        print("---- total_arities ----")
        print_prop_vect(self.total_arities)
        print("----------------- MATRIX PROPERTIES -----------------")
        print("---- arity ----")
        print_prop_matrix(self.tokens.arity)
        print("---- complexity ----")
        print_prop_matrix(self.tokens.complexity)
        print("---- var_type ----")
        print_prop_matrix(self.tokens.var_type)
        print("---- var_id ----")
        print_prop_matrix(self.tokens.var_id)
        print("---- behavior_id ----")
        print_prop_matrix(self.tokens.behavior_id)
        print("---- is_power ----")
        print_prop_matrix(self.tokens.is_power)
        print("---- power ----")
        print_prop_matrix(self.tokens.power)
        print("---- is_constraining_phy_units ----")
        print_prop_matrix(self.tokens.is_constraining_phy_units)
        print("---- phy_units ----")
        print_prop_units(self.tokens.phy_units)
        print("---- pos ----")
        print_prop_matrix(self.tokens.pos)
        print("---- pos_batch ----")
        print_prop_matrix(self.tokens.pos_batch)
        print("---- depth ----")
        print_prop_matrix(self.tokens.depth)
        print("---- has_parent_mask ----")
        print_prop_matrix(self.tokens.has_parent_mask)
        print("---- parent_pos ----")
        print_prop_matrix(self.tokens.parent_pos)
        print("---- has_siblings_mask ----")
        print_prop_matrix(self.tokens.has_siblings_mask)
        print("---- siblings_pos 0 ----")
        print_prop_matrix(self.tokens.siblings_pos[:,:,0])
        print("---- n_siblings ----")
        print_prop_matrix(self.tokens.n_siblings)
        print("---- has_children_mask ----")
        print_prop_matrix(self.tokens.has_children_mask)
        print("---- children_pos 0 ----")
        print_prop_matrix(self.tokens.children_pos[:,:,0])
        print("---- children_pos 1 ----")
        print_prop_matrix(self.tokens.children_pos[:,:,1])
        print("---- n_children ----")
        print_prop_matrix(self.tokens.n_children)

    def status(self):
        return self.idx_as_names(self.tokens.idx)

    def __repr__(self):
        return str(self.status())

