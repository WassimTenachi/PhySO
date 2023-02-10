import warnings
import numpy as np
from abc import ABC, abstractmethod

# Internal imports
from physo.physym import token as Tok
from physo.physym import functions as Func

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------- PRIOR CLASS ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


class Prior (ABC):
    """
    Abstract prior.
    """
    def __init__(self, library, programs):
        """
        Parameters
        ----------
        library : library.Library
            Library of choosable tokens.
        programs : program.VectPrograms
            Programs in the batch.
        """
        self.lib       = library
        self.progs     = programs
        self.get_default_mask_prob = lambda : np.ones((self.progs.batch_size, self.lib.n_choices), dtype = float)
        self.reset_mask_prob()

    def reset_mask_prob (self):
        """
        Resets mask of probabilities to one.
        """
        self.mask_prob = self.get_default_mask_prob()

    def __call__(self):
        """
        Returns probabilities of priors for each choosable token in the library.
        Returns
        -------
        mask_probabilities : numpy.array of shape (self.progs.batch_size, self.lib.n_choices) of float
        """
        raise NotImplementedError

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ INDIVIDUAL PRIORS IMPLEMENTATION ------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


class UniformArityPrior (Prior):
    """
    Uniform probability distribution over tokens by their arities.
    This prior encourages tokens with an arity that is under-represented and discourages tokens with an arity that
    is over-represented by normalising token probabilities by the number of tokens having its arity.
    """

    def __init__(self, library, programs):
        """
        Parameters
        ----------
        library : library.Library
        programs : program.VectPrograms
        """
        Prior.__init__(self, library, programs)
        # Number of tokens per arity
        # Sum of tokens having arity = idx on choosable tokens
        count_arities = np.array ([ (self.lib.get_choosable_prop("arity") == arity).sum() for arity in range (Tok.MAX_ARITY + 1) ])
        # Uniform mask over arities ie. inverse of total number of tokens per arity for each choosable token arity
        # Mask for one prog
        individual_mask = 1 / count_arities[self.lib.get_choosable_prop("arity")].astype(float)
        # Mask is the same for every program -> tile
        self.reset_mask_prob()
        self.mask_prob[:,:] = np.tile(individual_mask, (self.progs.batch_size, 1))

    def __call__(self):
        return self.mask_prob

    def __repr__(self):
        return "UniformArityPrior"


class HardLengthPrior (Prior):
    """
    Forces programs to have lengths such that min_length <= lengths <= max_length finished.
    Enforces lengths <= max_length by forbidding non-terminal tokens when choosing non-terminal tokens would mean
    exceeding max length of program.
    Enforces min_length <= lengths by forbidding terminal tokens when choosing a terminal token would mean finishing a
    program before min_length.
    """

    def __init__(self, library, programs, min_length, max_length):
        """
        Parameters
        ----------
        library : library.Library
        programs : program.VectPrograms
        min_length : float
            Minimum length that programs are allowed to have.
        max_length : float
            Maximum length that programs are allowed to have.
        """
        # Assertions
        try: min_length = float(min_length)
        except ValueError: raise TypeError("max_length must be cast-able to a float")
        try: max_length = float(max_length)
        except ValueError: raise TypeError("max_length must be cast-able to a float")
        assert min_length <= programs.max_time_step, "min_length must be such as: min_length <= max_time_step"
        assert max_length <= programs.max_time_step, "max_length must be such as: max_length <= max_time_step"
        assert max_length >= 1,                      "max_length must be such as: max_length >= 1"
        assert min_length <= max_length,             "Must be: min_length <= max_length"

        Prior.__init__(self, library, programs)
        # Is token of the library a terminal token : mask
        terminal_arity = 0
        self.mask_lib_is_terminal = (self.lib.get_choosable_prop("arity") == terminal_arity)
        assert min_length < max_length, "Min length must be such that: min_length < max_length"
        self.min_length = min_length
        self.max_length = max_length
        self.reset_mask_prob()

    def __call__(self):
        # Reset probs
        self.reset_mask_prob()

        # --- MAX ---
        # Would library token exceed max length if chosen in next step : mask
        mask_would_exceed_max = np.add.outer(self.progs.n_completed, self.lib.get_choosable_prop("arity")) > self.max_length
        # Going to reach max length => next token must be terminal => prob for non-terminal must be = 0
        self.mask_prob[mask_would_exceed_max] *= 0 # = 0 for token exceeding max

        # --- MIN ---
        # Progs having only one dummy AND length (including dummies) < min : mask
        # These programs are going to finish at next step if we allow terminal tokens to be chosen.
        mask_going_to_finish_before_min = np.logical_and(self.progs.n_dangling == 1, self.progs.n_completed < self.min_length)
        # Going to be finished with length < min length => next token must be non-terminal => prob for terminal must be = 0
        mask_would_be_inferior_to_min = np.outer(mask_going_to_finish_before_min, self.mask_lib_is_terminal)
        self.mask_prob[mask_would_be_inferior_to_min] *= 0 # = 0 for terminal
        return self.mask_prob

    def __repr__(self):
        return "HardLengthPrior (min_length = %i, max_length = %i)"%(self.min_length, self.max_length)


class SoftLengthPrior (Prior):
    """
    Soft prior that encourages programs to have a length close to length_loc.
    Before loc: scales terminal token probabilities by gaussian where dangling == 1 (ie. programs that might finish
    next step). After loc: scales non-terminal token probabilities by gaussian.
    """

    def __init__(self, library, programs, length_loc, scale):
        """
        Parameters
        ----------
        library : library.Library
        programs : program.VectPrograms
        length_loc : float
            Desired length of programs.
        scale : float
            Scale of gaussian used as  prior.
        """
        # Assertions
        try: length_loc = float(length_loc)
        except ValueError:" length_loc must be cast-able to a float"
        try: scale = float(scale)
        except ValueError:" scale must be cast-able to a float"

        Prior.__init__(self, library, programs)
        # If we want length = 3, gaussian value must be max at step = 2 (ie. when generating token n°3)
        self.length_loc = length_loc
        # => step_loc = length_loc - 1
        self.step_loc = float(self.length_loc) - 1
        self.scale    = float(scale)
        # Value of gaussian at all steps
        steps = np.arange(0, self.progs.max_time_step + 1) # gaussian_vals[step_loc] = gaussian_vals[steps[step_loc]]
        self.gaussian_vals = np.exp(-(steps - self.step_loc) ** 2 / (2 * self.scale))
        # Is token of the library a terminal token : mask
        terminal_arity = 0
        self.mask_lib_is_terminal = (self.lib.get_choosable_prop("arity") == terminal_arity)

    def __call__(self):
        # Reset probs
        self.reset_mask_prob()
        # Progs having only one dummy (going to finish at next step if choosing a terminal token) : mask
        mask_one_dummy_progs = (self.progs.n_dangling == 1)
        # Before loc
        if self.progs.curr_step < self.step_loc:
            # Scale terminal token probs by gaussian where progs have only one dummy
            mask_scale_terminal = np.outer(mask_one_dummy_progs, self.mask_lib_is_terminal)
            self.mask_prob[mask_scale_terminal] *= self.gaussian_vals[self.progs.curr_step]
        # At loc: gaussian value = 1.
        # After loc
        elif self.progs.curr_step > self.step_loc:
            # Scale non-terminal tokens probs by gaussian
            self.mask_prob[:, np.logical_not(self.mask_lib_is_terminal)] *= self.gaussian_vals[self.progs.curr_step]
        return self.mask_prob

    def __repr__(self):
        return "SoftLengthPrior (length_loc = %i, scale = %i)"%(self.length_loc, self.scale)


class RelationshipConstraintPrior (Prior):
    """
    Forces programs to comply with relationships constraints. Enforcing that [targets] cannot be the [relationship] of
    [effectors].  Where targets are choosable tokens for the current batch, effectors are already chosen tokens having
    a [relationship] relationship (descendant, child or sibling) with targets. This constraint between elements of
    effectors list and targets list in a one to one fashion so effectors and targets list should have the same size.
    Eg. effectors = ["sin", "n2", "exp"], relationship = "child", targets = ["cos", "sqrt", "log"] forbids cos from
    being the child of sin, sqrt from being the child of n2 and log from being the child of exp.
    """
    def __init__(self, library, programs, effectors, relationship, targets, max_nb_violations = None):
        """
        Enforcing that [targets] cannot be the [relationship] of [effectors].
        Parameters
        ----------
        library : library.Library
        programs : program.VectPrograms
        effectors : list of str
            List of effector tokens' name.
        relationship : str
            Relationship to forbid between effectors and targets ("descendant", "child" or "sibling").
        targets : list of str
            List of target tokens' name.
        max_nb_violations : None or list of int
            List containing max number of acceptable violations for each constraint relationship in case there are
            multiple relatives having [relationship] with [targets] (eg. multiple ancestors). By default = None, zero
            violations are allowed. Should have the same size as effectors and targets lists. Remark:
            using max_nb_violations with values > 0 on single relative relationship cases (eg. parent) would mean
            applying no constraint whatsoever.
        """
        # -------- ASSERTIONS --------

        # Relationship argument ---
        legal_relationships = ["descendant", "child", "sibling"]
        assert isinstance(relationship, str) and relationship in legal_relationships, "relationship argument should " \
            " either one of ('descendant', 'child', 'sibling')."

        # effectors argument ---
        effectors = np.array(effectors)
        err_msg = "Argument effectors should be a list of strings, not %s."%(effectors)
        assert len(effectors.shape) == 1 and effectors.dtype.char == "U", err_msg
        err_msg = "Some tokens given in argument effectors: %s are not in the library of tokens: %s" \
                  % (effectors, library.lib_name)
        assert np.isin(effectors, library.lib_name).all(), err_msg

        # targets argument ---
        targets = np.array(targets)
        err_msg = "Argument targets should be a list of strings."
        assert len(targets.shape) == 1 and targets.dtype.char == "U", err_msg
        err_msg = "Some tokens given in argument targets: %s are not in the library of tokens: %s" \
                  % (targets, library.lib_name)
        assert np.isin(targets, library.lib_name).all(), err_msg

        # targets and effectors arguments ---
        assert len(effectors) == len(targets), "List of targets and effectors must have the same size as constraints " \
            "will be applied one-to-one (1st token of targets with 1st token of effectors, etc.)"

        # max_nb_violations argument ---
        if max_nb_violations is None:
            max_nb_violations = np.zeros(shape = (len(effectors),), dtype = int)
        max_nb_violations = np.array(max_nb_violations)
        err_msg = "Argument max_nb_violations should be a list of positive integers having the same size as " \
                  "effectors and targets lists."
        assert len(max_nb_violations.shape) == 1 and max_nb_violations.dtype == int, err_msg
        assert (max_nb_violations >= 0).all() == True, err_msg
        assert len(effectors) == len(max_nb_violations), err_msg

        # -------- ARGUMENTS HANDLING --------

        Prior.__init__(self, library, programs)

        # relationship ---
        inverse_relationships = {'descendant' : 'ancestor',
                                    'child'      : 'parent'  ,
                                    'sibling'    : 'sibling' , }
        # Enforcing that [targets] can not be the [targets_role] of [effectors]
        # ie. enforcing that [effectors] can not be the [effectors_role] of [targets]
        self.targets_role   = relationship
        self.effectors_role = inverse_relationships[relationship]

        # targets and effectors ---
        # Working with tokens' idx in the library instead of their name
        self.effectors     = np.array([self.lib.lib_name_to_idx[tok_name] for tok_name in effectors])
        self.targets       = np.array([self.lib.lib_name_to_idx[tok_name] for tok_name in targets  ])
        self.n_constraints = len(self.targets)

        # max_nb_violations argument ---
        self.max_nb_violations = max_nb_violations

        # ----HANDLING RELATIONSHIP VARIATIONS  ----

        # Max number of relatives that can be [effectors_role] of tokens to be chosen.
        max_n_relatives_dict = {'ancestor' : self.progs.max_time_step,
                                'parent'   : 1,
                                'sibling'  : 1}
        self.max_n_relatives = max_n_relatives_dict[self.effectors_role]
        # Method of programs returning [effectors_role] of token at step (and filling non-existing relatives with
        # self.lib.invalid_idx). Adding new axis when necessary for problem symmetry.
        get_relatives_idx_dict = {
            'ancestor' : lambda step: self.progs.get_ancestors_idx_of_step (step, no_ancestor_idx_filler = self.lib.invalid_idx),                 # Returns (batch_size, max_n_relatives)
            'parent'   : lambda step: self.progs.get_parent_idx_of_step    (step, no_parent_idx_filler   = self.lib.invalid_idx)[:, np.newaxis],  # Returns (batch_size, 1)
            'sibling'  : lambda step: self.progs.get_sibling_idx_of_step   (step, no_sibling_idx_filler  = self.lib.invalid_idx)[:, np.newaxis],  # Returns (batch_size, 1)
                              }
        self.get_relatives_idx = get_relatives_idx_dict[self.effectors_role]                            # Returns (batch_size, max_n_relatives)

        # -------- CONSTRAINTS MASK  --------

        # Is relationship allowed between effectors and targets : mask
        # Shape = ( number of possible effectors (including : superparent, dummy and invalid token)
        # vs number of possible targets (ie choosable tokens only) )
        self.mask_constraints = np.ones(shape = (self.lib.n_library, self.lib.n_choices), dtype = float) # (lib.n_library, lib.n_choices)

        # Put 0 weights on forbidden relationships
        self.mask_constraints[(self.effectors, self.targets)] = 0                                        # (n_constraints,)

        # -------- MAX VIOLATIONS COUNT  --------

        # Used for multiple relatives cases (eg. ancestors) but not single relatives cases (eg. parent, sibling)
        # Matrix of relationships (similar shape as mask_constraints) containing max number of violations tolerated
        # for each relationship.
        self.count_max_violations = np.full(shape = (self.lib.n_library, self.lib.n_choices),              # (lib.n_library, lib.n_choices)
                                            fill_value = self.max_n_relatives,
                                            dtype = float)
        self.count_max_violations[(self.effectors, self.targets)] = self.max_nb_violations                 # (n_constraints,)

    def __call__(self):

        # --- Faster code that works only for self.max_n_relatives == 1 cases ---
        # -> If using this code, delete sentence "Remark: using max_nb_violations with values > 0 on single relative
        # relationship cases (eg. parent) would mean applying no constraint whatsoever." in max_nb_violations arg
        # description.

        # if self.max_n_relatives == 1:
        #     # Getting idx in the lib of relatives
        #     relatives_idx = self.get_relatives_idx(step=self.progs.curr_step)  # (batch_size, max_n_relatives)
        #     # Batch of prior mask corresponding to each relative
        #     # For parents / sibling relationships: this is simply the batch of prior mask (as max_n_relatives = 1 for those)
        #     # For ancestor relationships: this is the batch of prior for each ancestor
        #     # (i.e. constraints arising from each ancestor with vectors of ones for non-existing "-" ancestors)
        #     mask_prob_relatives = self.mask_constraints[relatives_idx,
        #                           :]  # (batch_size, max_n_relatives, lib.n_choices)
        #     # Multiplying along the relatives' axis to get the whole constraint for each program
        #     # For parents / sibling relationships: this simply deletes an unnecessary dimension as each prog only has 1
        #     # vector of constraints arising from only one relationship (the parent or the sibling) (max_n_relatives = 1)
        #     # For ancestor relationships: this reduces dimensionality by multiplying the constraints masks
        #     mask_prob = mask_prob_relatives.prod(axis=1)  # (batch_size, lib.n_choices)
        # else: # self.max_n_relatives > 1:

        # Getting idx in the lib of relatives
        relatives_idx = self.get_relatives_idx(step=self.progs.curr_step)  # (batch_size, max_n_relatives)

        # Counts of relatives in a complete lib (including dummy, invalid etc.) size vector for each prog of batch
        counts_relatives = self.progs.count_tokens_idx(relatives_idx)                                                # (batch_size, lib.n_library)
        # Tile of this count along a new n_choices size dimension for each prog in batch
        tile_counts = np.moveaxis(np.tile(counts_relatives,                                                          # (batch_size, lib.n_library, lib.n_choices,)
                                  reps=(self.lib.n_choices, 1, 1)), source=(1,2), destination=(0,1))
        # Tile of general relationship counts constraints along a new batch_size dim
        tile_count_max_violations = np.tile(self.count_max_violations, reps=(self.progs.batch_size, 1, 1))           # (batch_size, lib.n_library, lib.n_choices,)
        # Would max number of violations be respected if choosing token in lib.n_choices dim
        mask_max_violations_respected = (tile_counts <= tile_count_max_violations)                                   # (batch_size, lib.n_library, lib.n_choices,)
        mask_prob = mask_max_violations_respected.prod(axis=1)

        return mask_prob

    def __repr__(self):
        if self.max_n_relatives > 1:
            repr = "RelationshipConstraintPrior (%s can have up to %s %s of type %s)" \
               % (self.lib.lib_name[self.targets], self.max_nb_violations, self.effectors_role+"(s)",
                  self.lib.lib_name[self.effectors])
        else:
            repr = "RelationshipConstraintPrior (%s can not be %s of %s)" \
               % (self.lib.lib_name[self.targets], self.targets_role, self.lib.lib_name[self.effectors])
        return repr

class NoUselessInversePrior(Prior):
    """
    Forbids useless inverse sequences. Enforcing that op can not be the child of op^(-1) and that op^(-1) can not be
    the child of op for all op having an inverse op^(-1) listed in functions.INVERSE_OP_DICT.
    """
    def __init__(self, library, programs,):
        """
        Enforcing functions are not child of their inverse function.
        Parameters
        ----------
        library : library.Library
        programs : program.VectPrograms
        """
        Prior.__init__(self, library, programs)

        # Considering (function, inverse function) couples where both tokens are in library
        effectors = []
        targets   = []
        for func_name, inverse_func_name in Func.INVERSE_OP_DICT.items():
            if func_name in self.lib.lib_name and inverse_func_name in self.lib.lib_name:
                effectors.append (func_name       )
                targets  .append (inverse_func_name )
        self.effectors    = effectors
        self.targets      = targets
        self.relationship = "child"

        # Is this prior active
        self.active = True

        # If no (function, inverse function) detected.
        if len(self.effectors) == 0:
            # warnings.warn("No (func, inverse func) couples detected, no prior from %s" % (self))
            self.active = False
        # Using RelationshipConstraintPrior prior
        # Enforcing that [targets] cannot be the [relationship] of [effectors]
        else:
            self.prior = RelationshipConstraintPrior (library = self.lib, programs = self.progs,
                                          targets      = self.targets,
                                          relationship = self.relationship,
                                          effectors    = self.effectors,)

    def __call__(self):
        if self.active:
            mask_prob = self.prior()              # (batch_size, lib.n_choices)
        else:
            mask_prob = self.get_default_mask_prob()  # (batch_size, lib.n_choices)
        return mask_prob

    def __repr__(self):
        repr = "NoUselessInversePrior (%s can not be %s of %s)" \
               % (self.targets, self.relationship, self.effectors)
        return repr


class NestedFunctions (Prior):
    """
    Regulates nesting for a group of tokens. Enforcing that any token in [functions] can only have up to [max_nesting]
    ancestors listed in [functions].
    """
    def __init__(self, library, programs, functions, max_nesting = 1):
        """
        Enforcing that [functions] can not be nested or only up to max_nesting level.
        Parameters
        ----------
        library : library.Library
        programs : program.VectPrograms
        functions : list of str
            List of tokens' names which's nesting will be forbidden.
        max_nesting : int
            Max level of nesting allowed. By default = 1, no nesting allowed.
        """
        # -------- ASSERTIONS --------

        # functions argument ---
        functions = np.array(functions)
        err_msg = "Argument functions should be a list of strings, not %s."%(functions)
        assert len(functions.shape) == 1 and functions.dtype.char == "U", err_msg
        err_msg = "Some tokens given in argument functions: %s are not in the library of tokens: %s" \
                  % (functions, library.lib_name)
        assert np.isin(functions, library.lib_name).all(), err_msg

        # max_nesting argument ---
        err_msg = "Argument max_nesting should be an int >= 1."
        assert isinstance(max_nesting, int) and max_nesting >= 1, err_msg

        # -------- ARGUMENTS HANDLING --------

        Prior.__init__(self, library, programs)

        # functions ---
        # Working with tokens' idx in the library instead of their name
        self.functions = np.array([self.lib.lib_name_to_idx[tok_name] for tok_name in functions])
        self.n_functions = len(self.functions)

        # max_nesting ---
        self.max_nesting = max_nesting

        # ---- ANCESTOR RELATIONSHIP PARAMETERS ----

        # Max number of ancestors
        self.max_n_ancestors = self.progs.max_time_step

        # Method of programs returning ancestors of token at step (and filling non-existing relatives with
        # self.lib.invalid_idx).
        self.get_ancestors_idx = lambda step: self.progs.get_ancestors_idx_of_step (
            step, no_ancestor_idx_filler = self.lib.invalid_idx)               # Returns (batch_size, max_n_ancestors)

        # ---- PRIOR TEMPLATES ----

        # Vector of prior for one prog : no restrictions for next token choice
        allow_all_prior = np.ones(shape=self.lib.n_choices, dtype=float)                                                 # (lib.n_choices)

        # Vector of prior for one prog : forbidding [functions] for next token choice
        forbid_functions_prior = np.ones(shape=self.lib.n_choices, dtype=float)                                          # (lib.n_choices)
        forbid_functions_prior[self.functions] = 0                                                                       # (n_functions,)

        # Both template of priors in one array so it is ready to be sliced with mask of True, False depending on prog
        self.template_prior = np.array([forbid_functions_prior, allow_all_prior])                                        # (2, lib.n_choices)

    def __call__(self):

        # Getting idx in the lib of ancestors
        ancestors_idx = self.get_ancestors_idx(step=self.progs.curr_step)                                            # (batch_size, max_n_ancestors)

        # Counts of ancestors in a complete lib (including dummy, invalid etc.) size vector for each prog of batch
        counts_ancestors = self.progs.count_tokens_idx(ancestors_idx)                                                # (batch_size, lib.n_library)

        # Number of ancestors that are part of [functions] for each prog in batch
        nesting_level = counts_ancestors[:, self.functions].sum(axis=1)                                              # (batch_size,)

        # mask : is prog allowed to continue with tokens of type [functions]
        mask_allow = nesting_level < self.max_nesting                                                                # (batch_size,)

        # Slicing array containing two cases (continuing with all tokens allowed or forbidding those in [functions])
        mask_prob = self.template_prior[mask_allow.astype(int)]                                                      # (batch_size, lib.n_choices)

        return mask_prob

    def __repr__(self):
        if self.max_nesting == 1:
            repr = "NestedFunctions (tokens = %s, nesting forbidden)" \
               % (self.lib.lib_name[self.functions],)
        else:
            repr = "NestedFunctions (tokens = %s, max nesting level = %i)" \
               % (self.lib.lib_name[self.functions], self.max_nesting,)
        return repr


class NestedTrigonometryPrior(Prior):
    """
    Regulates nesting of trigonometric functions listed in functions.TRIGONOMETRIC_OP. Enforcing that any trigonometric
    function can only have up to [max_nesting] ancestors that also are trigonometric functions.
    """
    def __init__(self, library, programs, max_nesting = 1):
        """
        Enforcing that trigonometric functions can not be nested or only up to max_nesting level.
        Parameters
        ----------
        library : library.Library
        programs : program.VectPrograms
        max_nesting : int
            Max level of nesting allowed. By default = 1, no nesting allowed.
        """
        Prior.__init__(self, library, programs)

        # Considering tokens in library that are declared as trigonometric functions in Func.TRIGONOMETRIC_OP
        trigonometric_functions = []
        for name in self.lib.lib_name:
            if name in Func.TRIGONOMETRIC_OP:
                trigonometric_functions.append(name)
        self.trigonometric_functions = np.array(trigonometric_functions)

        # max_nesting
        self.max_nesting = max_nesting

        # Is this prior active
        self.active = True

        # If no trigonometric functions detected.
        if len(self.trigonometric_functions) == 0:
            warnings.warn("No trigonometric functions detected, no prior from %s" % (self))
            self.active = False
        # Using NestedFunctions prior
        else:
            self.prior = NestedFunctions (library = self.lib, programs = self.progs,
                                          functions   = self.trigonometric_functions,
                                          max_nesting = max_nesting)

    def __call__(self):
        if self.active:
            mask_prob = self.prior()              # (batch_size, lib.n_choices)
        else:
            mask_prob = self.get_default_mask_prob()  # (batch_size, lib.n_choices)
        return mask_prob

    def __repr__(self):
        if self.max_nesting == 1:
            repr = "NestedTrigonometryPrior (tokens = %s, nesting forbidden)" \
               % (self.trigonometric_functions,)
        else:
            repr = "NestedTrigonometryPrior (tokens = %s, max nesting level = %i)" \
               % (self.trigonometric_functions, self.max_nesting,)
        return repr



class OccurrencesPrior (Prior):
    """
    Enforces that [targets] can not appear more than [max] times in programs.
    """
    def __init__(self, library, programs, targets, max):
        """
        Parameters
        ----------
        library : library.Library
        programs : program.VectPrograms
        targets : list of str
            List of tokens' names which's number of occurrences should be constrained.
        max : list of int
            List of maximum occurrences of tokens (must have the same length as targets).
        """
        # --- targets argument ---
        targets = np.array(targets)
        err_msg = "Argument targets should be a list of strings."
        assert len(targets.shape) == 1 and targets.dtype.char == "U", err_msg
        err_msg = "Some tokens given in argument targets: %s are not in the library of tokens: %s" \
                  % (targets, library.lib_name)
        assert np.isin(targets, library.lib_name).all(), err_msg

        # --- max argument ---
        max = np.array(max)
        err_msg = "Argument max should be a list of positive integers having the same size as targets list."
        assert len(max.shape) == 1 and max.dtype == int, err_msg
        assert (max >= 0).all() == True, err_msg
        assert len(max) == len(targets), err_msg
        max = max.astype(int)                                                                                   # (n_constraints,)

        # -------- ARGUMENTS HANDLING --------

        Prior.__init__(self, library, programs)

        self.targets_str   = targets                                                                             # (n_constraints,)
        self.n_constraints = len(self.targets_str)   # n_constraints <= n_choices
        self.targets       = np.array([self.lib.lib_name_to_idx[tok_name] for tok_name in self.targets_str])     # (n_constraints,)

        # Max number of occurrences allowed for each target
        self.max = max                                                                                           # (n_constraints,)

    def __call__(self):
        # Recounting at each step allows for the use of this prior even if it was not used before
        # For each prog in batch, number of occurrences of each target
        counts = np.equal.outer(self.progs.tokens.idx, self.targets,).sum(axis=1)                               # (batch_size, n_constraints,)
        # For each prog in batch, for each target : is target allowed at next step ?
        is_target_allowed = np.less(counts, self.max)                                                           # (batch_size, n_constraints,)
        # mask : for each prog in batch, for each token in choosable tokens, is token allowed
        self.reset_mask_prob()
        self.mask_prob[:, self.targets] = is_target_allowed.astype(float)                                       # (batch_size, n_choices,)
        return self.mask_prob

    def __repr__(self):
        return "OccurrencesPrior (tokens %s can be used %s times max)"%(self.targets_str, self.max)

class PhysicalUnitsPrior(Prior):
    """
    Enforces that next token should be physically consistent units-wise with current program based on current units
    constraints computed live (during program generation). If there is no way get a constraint all tokens are allowed.
    """
    def __init__(self, library, programs, prob_eps = 0.):
        """
        Parameters
        ----------
        library : library.Library
        programs : program.VectPrograms
        prob_eps : float
            Value to return for the prior inplace of zeros (useful for avoiding sampling problems)
        """
        # ------- INITIALIZING -------
        Prior.__init__(self, library, programs)
        # Tolerance when comparing two units vectors (eg. 0.333333334 == 0.333333333)
        self.tol = 1e2*np.finfo(np.float32).eps
        # Value to return for the prior inplace of zeros.
        self.prob_eps = prob_eps

        # ------- LIB_IS_CONSTRAINING -------
        # mask : are tokens in the library constraining units-wise
        self.lib_is_constraining = self.lib.is_constraining_phy_units[:self.lib.n_choices]                              # (n_choices,)
        # mask : are tokens in the library constraining units-wise (expanding in a new batch_size axis)
        self.lib_is_constraining_padded = np.tile(self.lib_is_constraining, reps=(self.progs.batch_size, 1))            # (batch_size, n_choices,)

        # ------- LIB_UNITS -------
        # Units of choosable tokens in the library
        self.lib_units = self.lib.phy_units[:self.lib.n_choices]                                                        # (n_choices, UNITS_VECTOR_SIZE,)
        # Padded units of choosable tokens in the library (expanding in a new batch_size axis)
        self.lib_units_padded = np.tile(self.lib_units, reps=(self.progs.batch_size, 1, 1))                             # (batch_size, n_choices, UNITS_VECTOR_SIZE,)

    def __call__(self):

        # Current step
        curr_step = self.progs.curr_step

        # ------- COMPUTE REQUIRED UNITS -------
        # Updating programs with newest most constraining units constraints
        self.progs.assign_required_units(step=curr_step)

        # ------- IS_PHYSICAL -------
        # mask : is dummy at current step part of a physical program units-wise
        is_physical = self.progs.is_physical                                                                            # (batch_size,)
        # mask : is dummy at current step part of a physical program units-wise (expanding in a new n_choices axis)
        is_physical_padded = np.moveaxis( np.tile(is_physical, reps=(self.lib.n_choices, 1))                            # (batch_size, n_choices,)
                                              , source=0, destination=1)

        # ------- IS_CONSTRAINING -------
        # mask : does dummy at current step contain constraints units-wise
        is_constraining = self.progs.tokens.is_constraining_phy_units[:, curr_step]                                     # (batch_size,)
        # mask : does dummy at current step contain constraints units-wise (expanding in a new n_choices axis)
        is_constraining_padded = np.moveaxis( np.tile(is_constraining, reps=(self.lib.n_choices, 1))                    # (batch_size, n_choices,)
                                              , source=0, destination=1)
        # Number of programs in batch that constraining at this step
        n_constraining  = is_constraining.sum()
        # mask : for each token in batch, for each token in library are both tokens constraining
        mask_prob_is_constraining_info = self.lib_is_constraining_padded & is_constraining_padded                       # (batch_size, n_choices,)

        # Useful as to forbid a choice, the choosable token must be constraining and the current dummy must also be
        # constraining, otherwise the choice should be legal regardless of the units of any of these tokens
        # (non-constraining tokens should contain NaNs units).

        # ------- UNITS -------
        # Units requirements at current step dummies
        units_requirement       = self.progs.tokens.phy_units[:, curr_step, :]                                          # (batch_size, UNITS_VECTOR_SIZE)
        # Padded units requirements of dummies at current step (expanding in a new n_choices axis)
        units_requirement_padded = np.moveaxis(np.tile(units_requirement, reps=(self.lib.n_choices, 1, 1))              # (batch_size, n_choices, UNITS_VECTOR_SIZE)
                                               , source=0, destination=1)
        # mask : for each token in batch, is choosing token in library legal units-wise
        mask_prob_units_legality = (np.abs(units_requirement_padded - self.lib_units_padded) < self.tol).prod(axis=-1)  # (batch_size, n_choices)

        # ------- RESULT -------
        # Token in library should be allowed if there are no units constraints on any side (library, current dummies)
        # OR if the units are consistent OR if the program is unphysical.
        # Ie. all tokens in the library are allowed if there are no constraints on any sides or if the program is
        # unphysical anyway.
        mask_prob = np.logical_or.reduce((                                                                              # (batch_size, n_choices)
            (~ mask_prob_is_constraining_info),
            (~ is_physical_padded),
            mask_prob_units_legality,
                                          )).astype(float)
        mask_prob[mask_prob == 0] = self.prob_eps
        return mask_prob

    def __repr__(self):
        repr = "PhysicalUnitsPrior"
        return repr

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- INDIVIDUAL PRIORS DICT -----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# Priors that don't take additional arguments
PRIORS_WO_ARGS = {
    "UniformArityPrior"     : UniformArityPrior,
    "NoUselessInversePrior" : NoUselessInversePrior,
}

# Priors that take additional arguments
PRIORS_W_ARGS = {
    "HardLengthPrior"             : HardLengthPrior,
    "SoftLengthPrior"             : SoftLengthPrior,
    "RelationshipConstraintPrior" : RelationshipConstraintPrior,
    "NestedFunctions"             : NestedFunctions,
    "NestedTrigonometryPrior"     : NestedTrigonometryPrior,
    "OccurrencesPrior"            : OccurrencesPrior,
    "PhysicalUnitsPrior"          : PhysicalUnitsPrior,
}

# All priors
PRIORS_DICT = {}
PRIORS_DICT.update(PRIORS_WO_ARGS)
PRIORS_DICT.update(PRIORS_W_ARGS)

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- PRIOR COLLECTION --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def make_PriorCollection (library, programs, priors_config,):
    """
    Makes PriorCollection object from arguments.
    Parameters
    ----------
    library : library.Library
        Library of choosable tokens.
    programs : program.VectPrograms
        Programs in the batch.
    priors_config : list of couples (str : dict)
        List of priors. List containing couples with prior name as first item in couple (see prior.PRIORS_DICT for list
        of available priors) and additional arguments (besides library and programs) to be passed to priors as second
        item of couple, leave None for priors that do not require arguments.
    Returns
    -------
    Prior.PriorCollection
    """
    type_err_msg = "priors_config should be a list containing couples with prior name string as first item in couple " \
                   "and additional arguments to be passed to priors dictionary as second item of couple, leave None " \
                   "for priors that do not require arguments."
    # Assertion
    assert isinstance(priors_config, list), type_err_msg
    # PriorCollection
    prior_collection = PriorCollection(library = library, programs = programs)
    # Individual priors
    priors = []
    # Iterating through individual priors
    for config in priors_config:
        # --- TYPE ASSERTIONS ---
        assert len(config) == 2, type_err_msg
        assert isinstance(config[0], str), type_err_msg
        assert isinstance(config[1], dict) or config[1] is None, type_err_msg
        # --- GETTING ITEMS ---
        name, args = config[0], config[1]
        # --- ASSERTIONS ---
        assert name in PRIORS_DICT, "Prior %s is not in the list of available priors :\n %s"%(name, PRIORS_DICT.keys())
        if name in PRIORS_W_ARGS:
            assert args is not None, "Arguments for making prior %s were not given." % (name)
        # --- MAKING PRIOR ---
        # If individual prior has additional args get them
        if name in PRIORS_W_ARGS:
            prior_args = args
        else:
            prior_args = {}
        # Appending individual prior
        prior = PRIORS_DICT[name](library = library, programs = programs, **prior_args)
        priors.append (prior)
    # Setting priors in PriorCollection
    prior_collection.set_priors(priors)
    return prior_collection


class PriorCollection:
    """
    Collection of prior.Prior, returns value of element-wise multiplication of constituent priors.
    """
    def __init__(self, library, programs,):
        """
        Parameters
        ----------
        library : library.Library
            Library of choosable tokens.
        programs : program.VectPrograms
            Programs in the batch.
        """
        self.priors    = []
        self.lib       = library
        self.progs     = programs
        self.init_prob = np.ones( (self.progs.batch_size, self.lib.n_choices), dtype = float)

    def set_priors (self, priors):
        """
        Sets constituent priors.
        Parameters
        ----------
        priors : list of prior.Prior
        """
        for prior in priors:
            self.priors.append(prior)

    def __call__(self):
        """
        Returns probabilities of priors for each choosable token in the library.
        Returns
        -------
        mask_probabilities : numpy.array of shape (self.progs.batch_size, self.lib.n_choices) of float
        """
        res = self.init_prob
        for prior in self.priors:
            res = np.multiply(res, prior())
        return res

    def __repr__(self):
        #repr = np.array([str(prior) for prior in self.priors])
        repr = "PriorCollection:"
        for prior in self.priors:
            repr += "\n- %s"%(prior)
        return str(repr)