import numpy as np
import torch

# Internal imports
from physo.physym import token
from physo.physym import program
from physo.physym import library
from physo.physym import prior
from physo.physym import dataset
from physo.physym import reward
from physo.physym import execute

# Embedding output in SR interface
INTERFACE_UNITS_AVAILABLE   = 1.
INTERFACE_UNITS_UNAVAILABLE = 0.
INTERFACE_UNITS_UNAVAILABLE_FILLER = lambda shape: np.random.uniform(size=shape, low=-4, high=4)

class Batch:
    """
    Batch containing symbolic function programs with interfaces for symbolic regression.
    Input  :
        ----- per step -----
        - new tokens can be appended by their int idx in the library of choosable tokens.
    Output :
        ----- per step -----
        - prior values for choice of next token.
        - environment of next token to guess parent/sibling one hots etc.
        ----- per epoch -----
        - reward values of programs
        - physicality of programs
        - lengths of programs
    """
    def __init__(self,
                library_args,
                priors_config,
                X,
                y_target,
                rewards_computer,
                batch_size,
                max_time_step,
                free_const_opti_args = None,
                candidate_wrapper    = None,
                observe_units        = True,
                ):
        """
        Parameters
        ----------
        library_args: dict
            Arguments passed to library.__init__
        priors_config : list of couples (str : dict)
            List of priors. List containing couples with prior name as first item in couple (see prior.PRIORS_DICT for list
            of available priors) and additional arguments (besides library and programs) to be passed to priors as second
            item of couple, leave None for priors that do not require arguments.
        X : torch.tensor of shape (n_dim, ?,) of float
            Values of the input variables of the problem with n_dim = nb of input variables.
        y_target : torch.tensor of shape (?,) of float
            Values of the target symbolic function on input variables contained in X_target.
        rewards_computer : callable
            Function taking programs (program.VectPrograms), X (torch.tensor of shape (n_dim,?,) of float), y_target
            (torch.tensor of shape (?,) of float) as key arguments and returning reward for each program (array_like
            of float).
        batch_size : int
            Number of programs in batch.
        max_time_step : int
            Max number of tokens programs can contain.
        free_const_opti_args : dict or None, optional
            Arguments to pass to free_const.optimize_free_const for free constants optimization. By default,
            free_const.DEFAULT_OPTI_ARGS arguments are used.
        candidate_wrapper : callable or None, optional
            Wrapper to apply to candidate program's output, candidate_wrapper taking func, X as arguments where func is
            a candidate program callable (taking X as arg). By default = None, no wrapper is applied (identity).
        observe_units : bool, optional
            Should units be included in "in situ" observation vector (True) or should this information be zeroed out
            (False).
        """

        # Batch
        self.batch_size    = batch_size
        self.max_time_step = max_time_step
        # Library
        self.library  = library.Library(**library_args)

        # Programs
        self.programs = program.VectPrograms(batch_size        = self.batch_size,
                                             max_time_step     = self.max_time_step,
                                             library           = self.library,
                                             candidate_wrapper = candidate_wrapper)
        # Prior
        self.prior   = prior.make_PriorCollection(programs      = self.programs,
                                                  library       = self.library,
                                                  priors_config = priors_config,)
        # Dataset
        self.dataset = dataset.Dataset(
            library = self.library,
            X = X,
            y_target = y_target,)

        # Reward func
        self.rewards_computer = rewards_computer

        # Sending free const table to same device as dataset
        self.programs.free_consts.values = self.programs.free_consts.values.to(self.dataset.detected_device)

        # Free constants optimizer args
        self.free_const_opti_args = free_const_opti_args

        # Observations
        self.observe_units = observe_units

    # ---------------------------- INTERFACE FOR SYMBOLIC REGRESSION ----------------------------

    def get_sibling_one_hot (self, step = None):
        """
        Get siblings one hot of tokens at step. 0 one hot vectors for dummies.
        Parameters
        ----------
        step : int
            Step of token from which sibling one hot should be returned.
            By default, step = current step
        Returns
        -------
        one_hot : numpy.array of shape (batch_size, n_choices) of int
            One hot.
        """
        if step is None:
            step = self.programs.curr_step
        # Idx of siblings
        siblings_idx      = self.programs.get_sibling_idx_of_step(step = step)      # (batch_size,)
        # Do tokens have siblings : mask
        has_siblings_mask = np.logical_and(                                         # (batch_size,)
            self.programs.tokens.has_siblings_mask[:, step],
            siblings_idx < self.programs.library.n_choices) # gets rid of dummies tokens which are valid siblings
        # Initialize one hot result
        one_hot = np.zeros((self.batch_size, self.library.n_choices))               # (batch_size, n_choices)
        # Affecting only valid siblings and leaving zero vectors where no siblings
        one_hot[has_siblings_mask, :] = np.eye(self.library.n_choices)[siblings_idx[has_siblings_mask]]
        return one_hot

    def get_parent_one_hot (self, step = None):
        """
        Get parents one hot of tokens at step.
        Parameters
        ----------
        step : int
            Step of token from which parent one hot should be returned.
            By default, step = current step
        Returns
        -------
        one_hot : numpy.array of shape (batch_size, n_choices) of int
            One hot.
        """
        if step is None:
            step = self.programs.curr_step
        # Idx of parents
        parents_idx      = self.programs.get_parent_idx_of_step(step = step)         # (batch_size,)
        # Do tokens have parents : mask
        has_parents_mask = self.programs.tokens.has_parent_mask[:, step]             # (batch_size,)
        # Initialize one hot result
        one_hot = np.zeros((self.batch_size, self.library.n_choices))                 # (batch_size, n_choices)
        # Affecting only valid parents and leaving zero vectors where no parents
        one_hot[has_parents_mask, :] = np.eye(self.library.n_choices)[parents_idx[has_parents_mask]]
        return one_hot

    def get_previous_tokens_one_hot(self):
        """
        Get previous step tokens as one hot.
        Returns
        -------
        one_hot : numpy.array of shape (batch_size, n_choices) of int
            One hot.
        """
        # Return 0 if 0th step
        if self.programs.curr_step == 0:
            one_hot = np.zeros((self.batch_size, self.library.n_choices))
        else:
            # Idx of tokens at previous step
            tokens_idx = self.programs.tokens.idx[:, self.programs.curr_step - 1]  # (batch_size,)
            # Are these tokens outside of library (void tokens)
            valid_mask = self.programs.tokens.idx[:,
                         self.programs.curr_step - 1] < self.library.n_choices     # (batch_size,)
            # Initialize one hot result
            one_hot = np.zeros((self.batch_size, self.library.n_choices))          # (batch_size, n_choices)
            # Affecting only valid tokens and leaving zero vectors where previous vector has no meaning
            one_hot[valid_mask, :] = np.eye(self.library.n_choices)[tokens_idx[valid_mask]]

        return one_hot

    def get_sibling_units_obs (self, step = None):
        """
        Get (required) units of sibling of tokens at step. Filling using INTERFACE_UNITS_UNAVAILABLE_FILLER where units
        are not available. Adding a vector in addition to the units indicating if units are available or not (equal to
        INTERFACE_UNITS_AVAILABLE where units are available and equal to INTERFACE_UNITS_UNAVAILABLE where there are no
        units infos available).
        Parameters
        ----------
        step : int
            Step of token which's sibling's (required) units be returned.
            By default, step = current step.
        Returns
        -------
        units_obs : numpy.array of shape (batch_size, token.UNITS_VECTOR_SIZE + 1) of float
            Units and info availability mask.
        """
        if step is None:
            step = self.programs.curr_step

        # Coords
        coords = self.programs.coords_of_step(step)                                                     # (2, batch_size)

        # Initialize result with filler (unavailable units everywhere)
        units_obs = np.zeros((self.batch_size, token.UNITS_VECTOR_SIZE + 1 ), dtype=float)              # (batch_size, UNITS_VECTOR_SIZE + 1)
        # filling units
        units_obs[:, :-1] = INTERFACE_UNITS_UNAVAILABLE_FILLER(                                         # (batch_size, UNITS_VECTOR_SIZE)
            shape=(self.batch_size, token.UNITS_VECTOR_SIZE))
        # availability mask
        units_obs[:, -1] = INTERFACE_UNITS_UNAVAILABLE                                                  # (batch_size,)

        # Sibling
        has_sibling    = self.programs.tokens.has_siblings_mask[tuple(coords)]                          # (batch_size,)
        n_has_sibling = has_sibling.sum()
        coords_sibling = self.programs.get_siblings(coords)[:, has_sibling]                             # (2, n_has_sibling)

        # Units
        # mask : are units of available siblings available ?
        is_available  = self.programs.tokens.is_constraining_phy_units[tuple(coords_sibling)]           # (n_has_sibling,)
        n_is_available = is_available.sum()
        # Coordinates of available siblings having available units
        coords_sibling_and_units_available = coords_sibling[:, is_available]                            # (2, n_is_available)
        # Units of available siblings having available units
        phy_units = self.programs.tokens.phy_units[tuple(coords_sibling_and_units_available)]           # (n_is_available, UNITS_VECTOR_SIZE)

        # Putting units of available siblings having available units in units_obs
        units_obs[coords_sibling_and_units_available[0], :-1] = phy_units                               # (n_is_available, UNITS_VECTOR_SIZE)
        units_obs[coords_sibling_and_units_available[0],  -1] = INTERFACE_UNITS_AVAILABLE               # (n_is_available,)

        return units_obs

    def get_parent_units_obs (self, step = None):
        """
        Get (required) units of parent of tokens at step. Filling using INTERFACE_UNITS_UNAVAILABLE_FILLER where units
        are not available. Adding a vector in addition to the units indicating if units are available or not (equal to
        INTERFACE_UNITS_AVAILABLE where units are available and equal to INTERFACE_UNITS_UNAVAILABLE where there are no
        units infos available).
        Parameters
        ----------
        step : int
            Step of token which's parent's (required) units be returned.
            By default, step = current step.
        Returns
        -------
        units_obs : numpy.array of shape (batch_size, token.UNITS_VECTOR_SIZE + 1) of float
            Units and info availability mask.
        """
        if step is None:
            step = self.programs.curr_step

        # Coords
        coords = self.programs.coords_of_step(step)                                                     # (2, batch_size)

        # Initialize result with filler (unavailable units everywhere)
        units_obs = np.zeros((self.batch_size, token.UNITS_VECTOR_SIZE + 1 ), dtype=float)              # (batch_size, UNITS_VECTOR_SIZE + 1)
        # filling units
        units_obs[:, :-1] = INTERFACE_UNITS_UNAVAILABLE_FILLER(                                         # (batch_size, UNITS_VECTOR_SIZE)
            shape=(self.batch_size, token.UNITS_VECTOR_SIZE))
        # availability mask
        units_obs[:, -1] = INTERFACE_UNITS_UNAVAILABLE                                                  # (batch_size,)

        # If 0-th step, units are those of superparent
        if step == 0:
            units_obs[:, :-1] = self.library.superparent.phy_units                                      # (batch_size, UNITS_VECTOR_SIZE)
            units_obs[:,  -1] = INTERFACE_UNITS_AVAILABLE                                               # (batch_size,)

        # If 0-th step, this part does nothing as n_is_available = 0 in this case
        # parent
        has_parent    = self.programs.tokens.has_parent_mask[tuple(coords)]                             # (batch_size,)
        n_has_parent  = has_parent.sum()
        coords_parent = self.programs.get_parent(coords)[:, has_parent]                                 # (2, n_has_parent)

        # Units
        # mask : are units of available parents available ?
        is_available  = self.programs.tokens.is_constraining_phy_units[tuple(coords_parent)]           # (n_has_parent,)
        n_is_available = is_available.sum()
        # Coordinates of available parent having available units
        coords_parent_and_units_available = coords_parent[:, is_available]                             # (2, n_is_available)
        # Units of available parents having available units
        phy_units = self.programs.tokens.phy_units[tuple(coords_parent_and_units_available)]           # (n_is_available, UNITS_VECTOR_SIZE)

        # Putting units of available parents having available units in units_obs
        units_obs[coords_parent_and_units_available[0], :-1] = phy_units                               # (n_is_available, UNITS_VECTOR_SIZE)
        units_obs[coords_parent_and_units_available[0],  -1] = INTERFACE_UNITS_AVAILABLE               # (n_is_available,)

        return units_obs

    def get_previous_tokens_units_obs (self, step = None):
        """
        Get (required) units of tokens before step. Filling using INTERFACE_UNITS_UNAVAILABLE_FILLER where units are not
        available. Adding a vector in addition to the units indicating if units are available or not (equal to
        INTERFACE_UNITS_AVAILABLE where units are available and equal to INTERFACE_UNITS_UNAVAILABLE where there are no
        units infos available).
        Parameters
        ----------
        step : int
            Step of token which's previous tokens' (required) units be returned.
            By default, step = current step.
        Returns
        -------
        units_obs : numpy.array of shape (batch_size, token.UNITS_VECTOR_SIZE + 1) of float
            Units and info availability mask.
        """
        if step is None:
            step = self.programs.curr_step

        # Initialize result with filler (unavailable units everywhere)
        units_obs = np.zeros((self.batch_size, token.UNITS_VECTOR_SIZE + 1 ), dtype=float)              # (batch_size, UNITS_VECTOR_SIZE + 1)
        # filling units
        units_obs[:, :-1] = INTERFACE_UNITS_UNAVAILABLE_FILLER(                                         # (batch_size, UNITS_VECTOR_SIZE)
            shape=(self.batch_size, token.UNITS_VECTOR_SIZE))
        # availability mask
        units_obs[:, -1] = INTERFACE_UNITS_UNAVAILABLE                                                  # (batch_size,)

        # If step == 0, leave empty unavailable units filling
        if step > 0:
            units_obs = self.get_tokens_units_obs(step = step - 1)                                      # (batch_size, UNITS_VECTOR_SIZE + 1)

        return units_obs

    def get_tokens_units_obs (self, step = None):
        """
        Get (required) units of tokens at step. Filling using INTERFACE_UNITS_UNAVAILABLE_FILLER where units are not
        available. Adding a vector in addition to the units indicating if units are available or not (equal to
        INTERFACE_UNITS_AVAILABLE where units are available and equal to INTERFACE_UNITS_UNAVAILABLE where there are no
        units infos available).
        Parameters
        ----------
        step : int
            Step of token which's (required) units be returned.
            By default, step = current step.
        Returns
        -------
        units_obs : numpy.array of shape (batch_size, token.UNITS_VECTOR_SIZE + 1) of float
            Units and info availability mask.
        """
        if step is None:
            step = self.programs.curr_step

        # Coords
        coords = self.programs.coords_of_step(step)                                                     # (2, batch_size)

        # Initialize result
        units_obs = np.zeros((self.batch_size, token.UNITS_VECTOR_SIZE + 1 ), dtype=float)              # (batch_size, UNITS_VECTOR_SIZE + 1)

        # mask : is units information available
        is_available  = self.programs.tokens.is_constraining_phy_units[tuple(coords)]                   # (batch_size,)
        n_available   = is_available.sum()
        n_unavailable = self.batch_size - n_available
        # Coords of tokens which's units are available
        coords_available = coords[:, is_available]                                                      # (2, n_available)

        # Result : units (where available)
        units_obs[is_available,  :-1] = self.programs.tokens.phy_units[tuple(coords_available)]         # (n_available,   UNITS_VECTOR_SIZE)
        # Result : filler units (where unavailable)
        units_obs[~is_available, :-1] = INTERFACE_UNITS_UNAVAILABLE_FILLER(                             # (n_unavailable, UNITS_VECTOR_SIZE)
            shape=(n_unavailable, token.UNITS_VECTOR_SIZE))
        # Result : availability mask
        units_obs[is_available , -1] = INTERFACE_UNITS_AVAILABLE                                        # (batch_size,)
        units_obs[~is_available, -1] = INTERFACE_UNITS_UNAVAILABLE                                      # (batch_size,)

        return units_obs

    def get_obs(self):
        """
        Computes observation of current step for symbolic regression task.
        Returns
        -------
        obs : numpy.array of shape (batch_size, 3*n_choices+1,) of float
        """
        # Relatives one-hots
        parent_one_hot   = self.get_parent_one_hot()                         # (batch_size, n_choices,)
        sibling_one_hot  = self.get_sibling_one_hot()                        # (batch_size, n_choices,)
        previous_one_hot = self.get_previous_tokens_one_hot()                # (batch_size, n_choices,)
        # Number of dangling dummies
        n_dangling       = self.programs.n_dangling                          # (batch_size,)
        # Units obs
        do_obs = int(self.observe_units)
        units_obs_current  = do_obs * self.get_tokens_units_obs()            # (batch_size, UNITS_VECTOR_SIZE + 1)
        units_obs_sibling  = do_obs * self.get_sibling_units_obs()           # (batch_size, UNITS_VECTOR_SIZE + 1)
        units_obs_parent   = do_obs * self.get_parent_units_obs()            # (batch_size, UNITS_VECTOR_SIZE + 1)
        units_obs_previous = do_obs * self.get_previous_tokens_units_obs()   # (batch_size, UNITS_VECTOR_SIZE + 1)

        obs = np.concatenate((                                               # (batch_size, obs_size,)
            # Relatives one-hots
            parent_one_hot,
            sibling_one_hot,
            previous_one_hot,
            # Dangling
            n_dangling[:, np.newaxis],
            # Units obs
            units_obs_current,
            units_obs_sibling,
            units_obs_parent,
            units_obs_previous,
            ), axis = 1).astype(np.float32)

        return obs

    @property
    def obs_size(self):
        """
        Size of observation vector.
        Returns
        -------
        obs_size : int
        """
        return (3*self.n_choices) + 1 + 4*(token.UNITS_VECTOR_SIZE+1)

    @property
    def n_choices (self):
        return self.library.n_choices

    def get_rewards (self):
        """
        Computes rewards of programs contained in batch.
        Returns
        -------
        rewards : numpy.array of shape (batch_size,) of float
            Rewards of programs.
        """
        rewards = self.rewards_computer(programs             = self.programs,
                                        X                    = self.dataset.X,
                                        y_target             = self.dataset.y_target,
                                        free_const_opti_args = self.free_const_opti_args)
        return rewards


    def __repr__(self):
        s = ""
        s += "-------------------------- Library -------------------------\n%s\n"%(self.library )
        s += "--------------------------- Prior --------------------------\n%s\n"%(self.prior   )
        s += "-------------------------- Dataset -------------------------\n%s\n"%(self.dataset )
        s += "-------------------------- Programs ------------------------\n%s\n"%(self.programs)
        return s

