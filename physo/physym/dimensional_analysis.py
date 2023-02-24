import time

import numpy as np

# Internal imports
from physo.physym import program
from physo.physym import token as Tok
from physo.physym import functions as Func

# Case-code for when units analysis was not performed.
UNITS_ANALYSIS_NOT_PERFORMED_CASE_CODE = 0

# Physical units inconsistency error
class PhyUnitsError(Exception):
    pass


def assign_required_units_at_step (programs, step = None, from_scratch = False):
    """
    Usage: computes units that will be used to update units of dummy representing next token to guess
    Must be able to work with coords instead of step to work so it can be used on next token to guess dummies and not -
    void tokens.
    Parameters
    ----------
    programs : program.VectPrograms
        Programs on which physical units requirements should be computed.
    step : int or None (optional)
        Step of programs for which units requirement should be computed.
    from_scratch : bool (optional)
        If True, runs units requirement function from step = 0 to step (use this if this function was not ran at all
        steps before step as it should be).
    Returns
    -------
    tokens_cases_record : numpy.array of shape (?,) of int
        Encountered physical units deduction case for each token.
    """
    if step is None:
        # Protection to never go above programs.max_time_step-1
        step = min(programs.curr_step, programs.max_time_step-1)
    assert step < programs.max_time_step, "Step argument = %i is out of range of programs of max_time_step = %i. " \
                                          "Next token's units requirement can not be computed as there is no next " \
                                          "token." % (step, programs.max_time_step)
    if from_scratch:
        for i in range (step):
            coords = programs.coords_of_step(i)                                                       # (2, batch_size,)
            tokens_cases_record = assign_required_units(programs=programs, coords=coords, )           # (batch_size,)
    else:
        coords = programs.coords_of_step(step)                                                        # (2, batch_size,)
        tokens_cases_record = assign_required_units(programs = programs, coords = coords,)            # (batch_size,)
    return tokens_cases_record


def assign_required_units(programs, coords,):
    """
    Computes and assigns physical units requirements to tokens at coords, works with complete or incomplete programs
    (containing dummies).
    In certain cases, a bottom-up physical units computation and assignment is performed on whole subtrees of
    programs as it is necessary to compute the units constraints of a particular token.
    This function is able to provide the most constraining physical units constraints only if it was also ran at all
    positions before the tokens at coords.

    This function fills is_constraining_phy_units a boolean property of tokens indicating if the token has a physical
    unit constraint associated to it and phy_units, a vector of floats containing the physical units associated with the
    token. If it is impossible to compute the physical units constraints of a token at the current step (which can
    happen in incomplete programs), is_constraining_phy_units is put to False and phy_units is filled with NaNs.

    This function handles all possible cases of tokens arrangements situations relevant for live (working with
    unfinished trees) physical units constraints computation. Each case is associated with a unique id code, the id
    code of the cases in which tokens at coords fell into are returned for analyses purposes.

    Using this function on tokens of the same program simultaneously was not tested.

    Parameters
    ----------
    programs : program.VectPrograms
        Programs on which physical units requirements should be computed.
    coords : numpy.array of shape (2, ?) of int
        Coords of tokens in programs for which physical units requirements should be computed and assigned, 0-th array
        in batch dim and 1-th array in time dim.
    Returns
    -------
    tokens_cases_record : numpy.array of shape (?,) of int
        Encountered physical units deduction case for each token.
    """
    # -------------------- ASSERTIONS --------------------

    assert (coords[1, :] < programs.max_time_step).all(), "Some tokens have a position that is out of range of " \
                                                          "programs (max_time_step = %i)." % (programs.max_time_step)

    # -------------------- COORDS --------------------

    # mask : is token in VectPrograms part of the selection considered here
    mask_tokens = programs.coords_to_mask(coords)                                                         # (batch_size, max_time_step,)
    # mask : is token in VectPrograms already constraining ?
    mask_tokens_already_constraining = programs.tokens.is_constraining_phy_units                          # (batch_size, max_time_step,)
    # Number of tokens considered here
    n_tokens = coords.shape[1] # = mask_tokens.sum()

    # Positions
    batch_pos = coords[0, :] # Position in batch dim                                                      # (n_tokens,)
    pos       = coords[1, :] # Position in time sequence                                                  # (n_tokens,)

    # -------------------- INITIALIZING RESULTS --------------------

    # Array containing detected cases codes of each token to process. By default = no_case_code
    no_case_code = UNITS_ANALYSIS_NOT_PERFORMED_CASE_CODE
    tokens_cases_record = np.full(shape = n_tokens, fill_value = no_case_code, dtype = int)               # (n_tokens,)

    # mask : do we have constraints regarding the physical units of the token ?
    # By default = False (no constraint)
    is_constraining = np.full(shape = n_tokens, fill_value = False, dtype = bool)                         # (n_tokens,)

    # Required units for each token in batch. By default = np.NAN (no constraint)
    phy_units = np.full(shape = (n_tokens, Tok.UNITS_VECTOR_SIZE) , fill_value = np.NAN, dtype = float)   # (n_tokens, UNITS_VECTOR_SIZE)

    # -------------------- UTILS --------------------
    # Dummy token's idx in the library
    dummy_idx = programs.library.dummy_idx
    # Current token's idx in the library
    curr_token_idx                  = programs.tokens.idx  [tuple(coords)]                                 # (n_tokens,)
    # Current token's arity
    curr_token_arity                = programs.tokens.arity[tuple(coords)]                                 # (n_tokens,)
    # Current token is already constraining ?
    curr_token_already_constraining = programs.tokens.is_constraining_phy_units[tuple(coords)]             # (n_tokens,)
    n_tokens_already_constraining = curr_token_already_constraining.sum()

    # Utils function to apply constraints at mask_case based on inferred_phy_units of shape (n_tokens,) as masking is
    # handled in the function
    def apply_constraints (mask_case, const_is_constraining, inferred_phy_units):
        is_constraining [mask_case]    = const_is_constraining                                            # (n_in_case,)
        phy_units       [mask_case, :] = inferred_phy_units       [mask_case]                             # (n_in_case, UNITS_VECTOR_SIZE)

    # Utils function to apply constant constraints at mask_case
    def apply_const_constraints (mask_case, const_is_constraining, const_phy_units):
        is_constraining [mask_case]    = const_is_constraining                                            # (n_in_case,)
        phy_units       [mask_case, :] = const_phy_units                                                  # (n_in_case, UNITS_VECTOR_SIZE)

    # Utils function to apply no constraints at mask_case
    def apply_no_constraints (mask_case):
        apply_const_constraints(mask_case, const_is_constraining = False,
                                           const_phy_units = np.NAN)

    # -------------------- GETTING PROPERTIES OF PARENT --------------------
    # mask : does token have parent (ie. are any of the properties meaningful ie. they do not consist of filler values)
    has_parent = programs.tokens.has_parent_mask[tuple(coords)]
    # parent idx in library
    parent_idx = programs.get_parent_idx(coords)

    def get_parent_info():
        # mask : does parent have units constraints
        _, parent_is_constraining = programs.get_property_of_relative (coords, relative = "parent",           # (n_tokens,)
                                                                       attr = "is_constraining_phy_units")
        # physical units of the parent
        _, parent_phy_units       = programs.get_property_of_relative (coords, relative = "parent",           # (n_tokens,)
                                                                       attr = "phy_units")
        # behavior id of the parent
        _, parent_behavior_id     = programs.get_property_of_relative (coords, relative = "parent",           # (n_tokens,)
                                                                       attr = "behavior_id")
        # power of the parent
        _, parent_power           = programs.get_property_of_relative (coords, relative = "parent",           # (n_tokens,)
                                                                       attr = "power")
        # children pos of the parent
        _, parent_children_pos    = programs.get_property_of_relative (coords, relative = "parent",           # (n_tokens,)
                                                                       attr = "children_pos")

        # Return
        return parent_is_constraining, parent_phy_units, parent_behavior_id, parent_power, parent_children_pos

    parent_is_constraining, parent_phy_units, parent_behavior_id, parent_power, parent_children_pos = get_parent_info()

    # -------------------- GETTING PROPERTIES OF SIBLING --------------------
    # mask : does token have sibling (ie. are any of the properties meaningful ie. they do not consist of filler values)
    has_sibling = programs.tokens.has_siblings_mask[tuple(coords)]
    # sibling idx in library
    sibling_idx = programs.get_sibling_idx(coords)

    def get_sibling_info():
        # mask : does sibling have units constraints
        _, sibling_is_constraining = programs.get_property_of_relative (coords, relative = "siblings",           # (n_tokens,)
                                                                       attr = "is_constraining_phy_units")
        # physical units of the sibling
        _, sibling_phy_units       = programs.get_property_of_relative (coords, relative = "siblings",           # (n_tokens,)
                                                                       attr = "phy_units")
        # behavior id of the sibling
        _, sibling_behavior_id     = programs.get_property_of_relative (coords, relative = "siblings",           # (n_tokens,)
                                                                       attr = "behavior_id")
        # Is it a dummy ?
        sibling_is_dummy = (sibling_idx == dummy_idx)                                                            # (n_tokens,)
        # Return
        return sibling_is_constraining, sibling_phy_units, sibling_behavior_id, sibling_is_dummy

    sibling_is_constraining, sibling_phy_units, sibling_behavior_id, sibling_is_dummy = get_sibling_info()

    # -------------------- ASSERTION --------------------
    # Can not work on void filler tokens as they are not part of tree
    mask_invalid_tokens = (curr_token_idx == programs.invalid_idx)
    assert mask_invalid_tokens.sum() == 0, "Detected invalid void filler token(s) at pos = %s, batch_pos = %s"%(
                                            pos, batch_pos)
    # Tokens at pos > 0 not having parents -> invalid trees
    mask_invalid_trees = (pos > 0) & (has_parent == False)
    assert mask_invalid_trees.sum() == 0, "Detected token(s) at pos > 0 have no parents, programs at batch_pos %s " \
                                          "have invalid trees"%(batch_pos)

    # -------------------- BEHAVIOR IDS --------------------
    # Getting behavior ids
    bh_default               = Func.UNIT_BEHAVIORS_DICT["DEFAULT_BEHAVIOR"        ]
    bh_binary_additive       = Func.UNIT_BEHAVIORS_DICT["BINARY_ADDITIVE_OP"      ]
    bh_binary_multiplicative = Func.UNIT_BEHAVIORS_DICT["BINARY_MULTIPLICATIVE_OP"]
    bh_multiplication_op     = Func.UNIT_BEHAVIORS_DICT["MULTIPLICATION_OP"       ]
    bh_division_op           = Func.UNIT_BEHAVIORS_DICT["DIVISION_OP"             ]
    bh_unary_power           = Func.UNIT_BEHAVIORS_DICT["UNARY_POWER_OP"          ]
    bh_unary_additive        = Func.UNIT_BEHAVIORS_DICT["UNARY_ADDITIVE_OP"       ]
    bh_unary_dimensionless   = Func.UNIT_BEHAVIORS_DICT["UNARY_DIMENSIONLESS_OP"  ]

    # -------------------- CASES METHODOLOGY --------------------
    # General remarks regarding mask_case:
    # Logical usage: numpy' logical operators (&, ~ etc.) have higher precedence than logical operators (<,>,==)
    # Alternatively, one can use np.logical_and.reduce() to process multiple arrays through a single function
    # Vectorized elif behavior analog: mask_case must always contain a "& (tokens_cases_record == no_case_code)" to
    # only process tokens that were not processed before because tokens may fall into several cases and the first case
    # it fell into must take priority.
    # When conditions on properties of the parent are used, "& has_parent" must be included to avoid filler values
    # (same for sibling) (same for sibling).

    # -------------------- CASE 1 -------------------- #DONE
    # If the parent is an additive token AND we know the sibling AND we know the sibling's physical units, the token's
    # physical units must be that of the sibling regardless of the parent's units (even if it is free)
    case_code = 1

    # mask : does token fall into this case ?
    mask_case = np.logical_and.reduce((                                                                   # (n_tokens,)
        (tokens_cases_record == no_case_code),
        has_parent,
        bh_binary_additive.is_id(parent_behavior_id),
        has_sibling,
        ~ sibling_is_dummy,
        sibling_is_constraining == True,
                                     ))
    # Number of tokens falling into this case
    n_in_case = mask_case.sum()
    # Declaring tokens that fell into this case in
    tokens_cases_record [mask_case] = case_code                                                           # (n_in_case,)

    # Handling is_constraining and phy_units
    apply_constraints(mask_case             = mask_case,
                      const_is_constraining = True,
                      inferred_phy_units    = sibling_phy_units,)

    # -------------------- CASE 20 -------------------- #DONE
    # Elif the parent is an additive token AND parent dim is free (eg: nested in left part of multiplicative token) AND
    # it is the right child AND sibling dim is free then we can compute the sibling's physical units bottom up which
    # is the required units of the current token at right part (right token = 1th child of parent)
    case_code = 20

    # mask : does token fall into this case ?
    mask_case = np.logical_and.reduce((                                                                   # (n_tokens,)
        (tokens_cases_record == no_case_code),
        has_parent,
        bh_binary_additive.is_id(parent_behavior_id),
        parent_is_constraining == False,
        has_sibling,
        parent_children_pos[:, 1] == pos,
        sibling_is_constraining == False,
                                     ))
    # Number of tokens falling into this case
    n_in_case = mask_case.sum()
    # Declaring tokens that fell into this case in
    tokens_cases_record [mask_case] = case_code                                                           # (n_in_case,)

    # Perform dimensional analysis of sibling and its subtree if not done already
    # mask : tokens for which we do not know the physical units of the sibling ie. for which we have to perform bottom
    # up dimensional analysis
    mask_case_da_todo = mask_case & (sibling_is_constraining == False)                                     # (n_tokens,)
    # Determining subtrees on which to perform bottom up dimensional analysis's (start and ends)
    # Start of subtree is the sibling of the current token
    coords_start = programs.get_siblings(coords)       [:, mask_case_da_todo]
    # End of subtree is the token just before the current token (pos - 1)
    coords_end   = np.stack((batch_pos, pos-1), axis=0)[:, mask_case_da_todo]
    # Perform dimensional analysis
    assign_units_bottom_up (programs = programs, coords_start = coords_start,  coords_end = coords_end)
    # Refresh info after this bottom-up assignment
    parent_is_constraining, parent_phy_units, parent_behavior_id, parent_power, parent_children_pos = get_parent_info()
    sibling_is_constraining, sibling_phy_units, sibling_behavior_id, sibling_is_dummy               = get_sibling_info()

    apply_constraints(mask_case             = mask_case,
                      const_is_constraining = True,
                      inferred_phy_units    = sibling_phy_units,)

    # -------------------- CASE 21 -------------------- #DONE
    # Elif the parent's dim is free (no constraints), then the child's dim is also free (no constraints)
    # Unless: 1. the parent is an additive token, and we know everything about the sibling (already handled in case 1
    # above)
    # Unless: 2. the parent is an additive token of free dim (eg. descendant of left part of multiplicative token)
    # and we are dealing with right part, left part dim is complete and can thus be computed (already handled in case
    # 20 above)
    # Unless: 3. the parent is a multiplicative token and the sibling is not a dummy. We use "& has_sibling" in
    # "unless 3" mask line so unary parents can fall into this case as "unless 3" does not concern them:
    # if sibling_is_dummy is True, a fortiori has_sibling is True => sibling_is_dummy = (sibling_is_dummy & has_sibling)
    # if sibling_is_dummy is False then regardless of has_sibling => sibling_is_dummy = (sibling_is_dummy & has_sibling)
    case_code = 21

    # mask : does token fall into this case ?
    mask_case = np.logical_and.reduce ((                                                                   # (n_tokens,)
        (tokens_cases_record == no_case_code),
        (parent_is_constraining == False) & has_parent,
        ~ ( (bh_binary_multiplicative.is_id(parent_behavior_id)) & (~ (sibling_is_dummy & has_sibling) ) ),  # Unless 3
    ))
    # Number of tokens falling into this case
    n_in_case = mask_case.sum()
    # Declaring tokens that fell into this case in
    tokens_cases_record [mask_case] = case_code                                                           # (n_in_case,)

    # Handling is_constraining and phy_units
    apply_no_constraints(mask_case)

    # -------------------- CASE 3 -------------------- #DONE
    # Elif tokens are at pos = 0 -> the required units are those of the superparent
    case_code = 3

    # mask : does token fall into this case ?
    mask_case = np.logical_and.reduce((                                                                   # (n_tokens,)
        (tokens_cases_record == no_case_code),
        (pos == 0)
                                     ))
    # Number of tokens falling into this case
    n_in_case = mask_case.sum()
    # Declaring tokens that fell into this case in 
    tokens_cases_record [mask_case] = case_code                                                           # (n_in_case,)

    # Handling is_constraining and phy_units
    apply_const_constraints(mask_case,
        const_is_constraining = programs.library.superparent.is_constraining_phy_units,
        const_phy_units       = programs.library.superparent.phy_units)

    # -------------------- CASE 4 -------------------- #DONE
    # Elif parent is an additive token (binary or unary), then its child must have same physical units
    case_code = 4

    # mask : does token fall into this case ?
    mask_case = np.logical_and.reduce((                                                                   # (n_tokens,)
        (tokens_cases_record == no_case_code),
        has_parent,
        (( bh_binary_additive.is_id(parent_behavior_id) ) | ( bh_unary_additive.is_id(parent_behavior_id) )),
                                     ))
    # Number of tokens falling into this case
    n_in_case = mask_case.sum()
    # Declaring tokens that fell into this case in
    tokens_cases_record [mask_case] = case_code                                                           # (n_in_case,)

    # Handling is_constraining and phy_units
    apply_constraints(mask_case             = mask_case,
                      const_is_constraining = True,
                      inferred_phy_units    = parent_phy_units,)

    # -------------------- CASE 5 -------------------- #DONE
    # Elif parent is a power token. parent = (token)**n_power => dim(parent) = n_power*dim(token)
    case_code = 5

    # mask : does token fall into this case ?
    mask_case = np.logical_and.reduce((                                                                   # (n_tokens,)
        (tokens_cases_record == no_case_code),
        has_parent,
        bh_unary_power.is_id(parent_behavior_id),
                                     ))
    # Number of tokens falling into this case
    n_in_case = mask_case.sum()
    # Declaring tokens that fell into this case in
    tokens_cases_record [mask_case] = case_code                                                           # (n_in_case,)

    # Tiling power vector along a new units dim to have the same shape as phy_units vector so phy_units can be divided
    # by power of parent vector
    tiled_power = np.tile(parent_power, reps=(Tok.UNITS_VECTOR_SIZE,1)).transpose()                       # (n_tokens, UNITS_VECTOR_SIZE)
    # Handling is_constraining and phy_units
    apply_constraints(mask_case             = mask_case,
                      const_is_constraining = True,
                      inferred_phy_units    = parent_phy_units/tiled_power,)

    # -------------------- CASE 6 -------------------- #DONE
    # Elif parent is a dimensionless token.
    case_code = 6

    # mask : does token fall into this case ?
    mask_case = np.logical_and.reduce((                                                                   # (n_tokens,)
        (tokens_cases_record == no_case_code),
        has_parent,
        bh_unary_dimensionless.is_id(parent_behavior_id),
                                     ))
    # Number of tokens falling into this case
    n_in_case = mask_case.sum()
    # Declaring tokens that fell into this case in
    tokens_cases_record [mask_case] = case_code                                                          # (n_in_case,)

    apply_const_constraints(mask_case             = mask_case,
                            const_is_constraining = True,
                            const_phy_units       = 0,
                            )

    # -------------------- CASE 7 -------------------- #DONE
    # Parent is a multiplicative token
    mask_general_case = np.logical_and.reduce((                                                            # (n_tokens,)
        has_parent,
        ( bh_binary_multiplicative.is_id(parent_behavior_id) ),
                                             ))

    # -------------------- CASE 70 -------------------- #DONE
    # Elif we do not have access to the sibling yet (when dealing with the left part of the tree 1st arg (polish
    # notation) of a binary multiplicative token) then we can not determine the dim (we must leave it as a free
    # parameter). This also applies to when we are in the left or the right part of a subtree but both tokens are
    # dummies (incomplete tree).
    case_code = 70

    # mask : does token fall into this case ?
    mask_case = np.logical_and.reduce((                                                                   # (n_tokens,)
        (tokens_cases_record == no_case_code),
        mask_general_case,
        (sibling_is_dummy & has_sibling),
        (curr_token_idx == dummy_idx),
                                      ))
    # Number of tokens falling into this case
    n_in_case = mask_case.sum()
    # Declaring tokens that fell into this case in
    tokens_cases_record [mask_case] = case_code                                                           # (n_in_case,)

    # Handling is_constraining and phy_units
    apply_no_constraints(mask_case)

    # -------------------- CASE 71 -------------------- #DONE
    # Elif we are in left part of subtree then sibling is unknown therefore units of current token is unknown.
    # The best way to know if we are dealing with left part is to check if this is the 0th child of the parent. Checking
    # 'if sibling is dummy' is not good enough as this prevents us from using the method on a complete tree full of
    # free tokens. Useful when doing units determination from_scratch.
    case_code = 71

    # mask : does token fall into this case ?
    mask_case = np.logical_and.reduce((                                                                   # (n_tokens,)
        (tokens_cases_record == no_case_code),
        mask_general_case,
        has_sibling,
        parent_children_pos[:, 0] == pos,
                                      ))
    # Number of tokens falling into this case
    n_in_case = mask_case.sum()
    # Declaring tokens that fell into this case in
    tokens_cases_record [mask_case] = case_code                                                           # (n_in_case,)

    # Handling is_constraining and phy_units
    apply_no_constraints(mask_case)

    # -------------------- CASE 72 -------------------- #DONE
    # Elif we don't have units constraints on the parent (ie. if there are nested multiplicative tokens and we are in
    # left part) there is no way to compute a physical units constraint yet.
    case_code = 72

    # mask : does token fall into this case ?
    mask_case = np.logical_and.reduce((                                                                   # (n_tokens,)
        (tokens_cases_record == no_case_code),
        mask_general_case,
        (parent_is_constraining == False) & has_parent,
                                      ))
    # Number of tokens falling into this case
    n_in_case = mask_case.sum()
    # Declaring tokens that fell into this case in
    tokens_cases_record [mask_case] = case_code                                                           # (n_in_case,)

    # Handling is_constraining and phy_units
    apply_no_constraints(mask_case)

    # -------------------- CASE 73 -------------------- #DONE
    # Else: this should be applied to ALL tokens left in general_case mask (parent is a multiplicative token) that were
    # not already handled in the above subcases (70, 71, 72). If we already have access to the sibling (when dealing
    # with the right part of the tree ie 2nd arg of a binary multiplicative token) then we can deduce the physical
    # units. If we already know the sibling then the sibling is the 1st arg of the parent (ie left part of tree).
    case_code = 73

    # mask : does token fall into this case ?
    mask_case = np.logical_and.reduce((                                                                   # (n_tokens,)
        (tokens_cases_record == no_case_code),
        mask_general_case,
                                      ))
    # Number of tokens falling into this case
    n_in_case = mask_case.sum()
    # Declaring tokens that fell into this case in
    tokens_cases_record [mask_case] = case_code                                                           # (n_in_case,)

    # Perform dimensional analysis of sibling and its subtree if not done already
    # mask : tokens for which we do not know the physical units of the sibling ie. for which we have to perform bottom
    # up dimensional analysis
    mask_case_da_todo = mask_case & (sibling_is_constraining == False)                                     # (n_tokens,)
    # Determining subtrees on which to perform bottom up dimensional analysis's start and ends
    # Start of subtree is the sibling of the current token
    coords_start = programs.get_siblings(coords)       [:, mask_case_da_todo]
    # End of subtree is the token just before the current token (pos - 1)
    coords_end   = np.stack((batch_pos, pos-1), axis=0)[:, mask_case_da_todo]
    # Perform dimensional analysis
    assign_units_bottom_up (programs = programs, coords_start = coords_start,  coords_end = coords_end)
    # Refresh info after this bottom-up assignment
    parent_is_constraining, parent_phy_units, parent_behavior_id, parent_power, parent_children_pos = get_parent_info()
    sibling_is_constraining, sibling_phy_units, sibling_behavior_id, sibling_is_dummy               = get_sibling_info()

    # Handling multiplication tokens parents
    # parent = sibling x token => dim(token) = dim(parent) - dim(sibling)
    mask_case_mul = mask_case & bh_multiplication_op.is_id(parent_behavior_id)                             # (n_tokens,)
    apply_constraints(mask_case             = mask_case_mul,
                      const_is_constraining = True,
                      inferred_phy_units    = (parent_phy_units - sibling_phy_units),)

    # Handling division tokens parents
    # parent = sibling/token => dim(token) = dim(sibling) - dim(parent)
    mask_case_div = mask_case & bh_division_op      .is_id(parent_behavior_id)                             # (n_tokens,)
    apply_constraints(mask_case             = mask_case_div,
                      const_is_constraining = True,
                      inferred_phy_units    = (sibling_phy_units - parent_phy_units),)

    # -------------------- CASE : NO CASE --------------------
    # Check that all tokens were handled (ie. that they all fell into one of the above cases)
    # Even if it is sometimes impossible to compute required units, all cases (including those) should be covered by
    # this function, if a token does not fall into any of the above cases, raise an error
    # mask : does token not fall into any of the above cases ?
    mask_no_case = (tokens_cases_record == no_case_code)                                                  # (n_tokens,)
    n_no_case = mask_no_case.sum()
    assert n_no_case == 0, "Failure to process units requirements of %i token. Positions in batch = %s ; positions in" \
                            " sequence = %s (no corresponding case found)." % (
                            n_no_case, batch_pos[mask_no_case], pos[mask_no_case])

    # -------------------- ASSIGNMENT --------------------
    # Only assign units to tokens for which we don't already have units information
    # (otherwise we could erase valuable info)
    # (In normal workflow this is not necessary as this function is used for dummies which's units are overwritten in
    # case they are replaced by terminal tokens but this can cause problem when computing all constraints from scratch.)
    mask_assign = mask_tokens & (mask_tokens_already_constraining == False)
    programs.tokens.is_constraining_phy_units [mask_assign] = is_constraining [curr_token_already_constraining == False] # (n_tokens_already_constraining,)
    programs.tokens.phy_units                 [mask_assign] = phy_units       [curr_token_already_constraining == False] # (n_tokens_already_constraining, UNITS_VECTOR_SIZE)

    # print("pos %s"% coords[1,:])
    # print("is_constraining_phy_units", programs.tokens.is_constraining_phy_units[mask_tokens][0])
    # print("phy_units", programs.tokens.phy_units[mask_tokens][0])
    # print("tokens_cases_record", tokens_cases_record)

    # -------------------- REGISTERING UNITS ANALYSIS RUN --------------------
    # Registering that required units assignment was performed on tokens at coords with the encountered codes.
    programs.units_analysis_cases[tuple(coords)] = tokens_cases_record                                    # (n_tokens,)

    return tokens_cases_record


def assign_units_bottom_up (programs, coords_start, coords_end):
    """
    Performs a bottom up (in the tree representation of programs) dimensional analysis and assigns units along the way
    for multiple subtrees.
    Parameters
    ----------
    programs : program.VectPrograms
        Programs on which bottom up units assignment should be performed.
    coords_start : numpy.array of shape (2, ?) of int
        Coords of starts of subtrees, 0th array in batch dim and 1th array in time dim.
    coords_end : numpy.array of shape (2, ?) of int
        Coords of ends of subtrees, 0th array in batch dim and 1th array in time dim.
    Returns
    -------
    """
    # Assertions
    assert coords_start.shape[1] == coords_end.shape[1], "%i subtrees starts but %i subtrees ends were given." % (
        coords_start.shape[1], coords_end.shape[1])
    assert np.array_equal(coords_start[0], coords_end[0]), "Start and end of subtrees should be located on the same " \
                                                           "program."
    # Arguments
    n_subtrees = coords_start.shape[1]
    batch_pos = coords_start[0]
    start_pos = coords_start[1]
    end_pos   = coords_end  [1]

    # Iterating through subtrees
    # todo: vectorize bottom-up units assignment process
    #t0 = time.perf_counter()
    for k in range (n_subtrees):
        # Starting at the end of subtree and parsing in reverse polish notation
        start  = end_pos   [k]
        # End of parsing is start of subtree
        end    = start_pos [k]
        prog_i = batch_pos [k]
        # Error messages
        error_msg_unknown_dim          = "Unknown physical units token encountered in bottom up units assignment " \
                                         "process."
        error_msg_dimensionless_child  = "Non-dimensionless token encountered as child of dimensionless op (eg cos, " \
                                         "exp, log etc) in bottom up units assignment process."
        error_msg_dimensionless_token  = "Dimensionless token having non-zero physical units encountered in bottom " \
                                         "up units assignment process."
        error_msg_additive_discrepancy = "Two children of binary_additive_op (eg: addition, subtraction) having " \
                                         "different physical units encountered in bottom up units assignment process."
        error_msg_incomplete_tree      = "Regular bottom up dimensional analysis can not be performed on incomplete " \
                                         "tree (containing terminal tokens with unknown physical units: eg. dummies)"

        # Utils function to assign units
        def assign_units(pos, phy_units, is_constraining):
            programs.tokens.phy_units                 [prog_i, pos] = phy_units
            programs.tokens.is_constraining_phy_units [prog_i, pos] = is_constraining
            return None

        # Parser
        def parser(i):
            # Stop condition (getting outside of subtree)
            if i == end-1:
                return True
            # Current parsing
            idx             = programs.tokens.idx                       [prog_i, i]
            arity           = programs.tokens.arity                     [prog_i, i]
            behavior_id     = programs.tokens.behavior_id               [prog_i, i]
            phy_units       = programs.tokens.phy_units                 [prog_i, i]
            is_constraining = programs.tokens.is_constraining_phy_units [prog_i, i]
            # Check that subtree is complete (no dummies should be encountered during parsing)
            assert idx != programs.dummy_idx, error_msg_incomplete_tree
            # Arity = 0 ---
            if arity == 0:
                pass
            # Arity = 1 ---
            elif arity == 1:
                # Position of the lonely child of the token (arity = 1)
                child_pos             = programs.tokens.children_pos              [prog_i, i][0]
                child_phy_units       = programs.tokens.phy_units                 [prog_i, child_pos]
                child_is_constraining = programs.tokens.is_constraining_phy_units [prog_i, child_pos]
                # Making sure that the child of unary tokens are not free
                assert child_is_constraining == True, error_msg_unknown_dim
                # If token is an unary power op -> apply power to units
                if  Func.UNIT_BEHAVIORS_DICT["UNARY_POWER_OP"].is_id(behavior_id):
                    n_power = programs.tokens.power    [prog_i, i]
                    assign_units (pos = i, phy_units = n_power * child_phy_units, is_constraining = True)
                # Elif token is an unary additive op -> copy-paste units from child
                elif Func.UNIT_BEHAVIORS_DICT["UNARY_ADDITIVE_OP"].is_id(behavior_id):
                    assign_units (pos = i, phy_units = child_phy_units, is_constraining = True)
                # Elif token is an unary dimensionless op -> nothing to do but making sure that child token is
                # dimensionless (as it should be) just in case and that current token is dimensionless
                elif Func.UNIT_BEHAVIORS_DICT["UNARY_DIMENSIONLESS_OP"].is_id(behavior_id):
                    assert np.array_equal(child_phy_units, np.zeros(Tok.UNITS_VECTOR_SIZE)) and \
                           child_is_constraining == True, error_msg_dimensionless_child
                    assert np.array_equal(phy_units, np.zeros(Tok.UNITS_VECTOR_SIZE)) and \
                           is_constraining == True, error_msg_dimensionless_token
            # Arity = 2 ---
            elif arity == 2:
                # Children positions
                child0_pos             = programs.tokens.children_pos              [prog_i, i][0]
                child1_pos             = programs.tokens.children_pos              [prog_i, i][1]
                # Child 0 units
                child0_phy_units       = programs.tokens.phy_units                 [prog_i, child0_pos]
                child0_is_constraining = programs.tokens.is_constraining_phy_units [prog_i, child0_pos]
                # Child 1 units
                child1_phy_units       = programs.tokens.phy_units                 [prog_i, child1_pos]
                child1_is_constraining = programs.tokens.is_constraining_phy_units [prog_i, child1_pos]
                # Assertion: making sure that children of binary tokens are not free
                assert (child0_is_constraining == True) and (child1_is_constraining == True), error_msg_unknown_dim
                # If token is an additive token -> units are those of any children (as they should be the same
                # among them) but making sure that children of additive binary tokens have the same units for safety.
                if Func.UNIT_BEHAVIORS_DICT["BINARY_ADDITIVE_OP"].is_id(behavior_id):
                    assert np.array_equal(child1_phy_units, child0_phy_units), error_msg_additive_discrepancy
                    assign_units (pos = i, phy_units = child0_phy_units, is_constraining = True)
                # Elif token is a multiplicative token
                elif Func.UNIT_BEHAVIORS_DICT["BINARY_MULTIPLICATIVE_OP"].is_id(behavior_id):
                    # token = child0 * child1 => units(token) = child0_phy_units + child1_phy_units
                    if Func.UNIT_BEHAVIORS_DICT["MULTIPLICATION_OP"].is_id(behavior_id):
                        assign_units (pos = i, phy_units = child0_phy_units + child1_phy_units, is_constraining = True)
                    # token = child0 / child1 => units(token) = child0_phy_units - child1_phy_units
                    elif Func.UNIT_BEHAVIORS_DICT["DIVISION_OP"].is_id(behavior_id):
                        assign_units (pos = i, phy_units = child0_phy_units - child1_phy_units, is_constraining = True)
            # print("------")
            # print("Parsing i = ",i)
            # print("name      = ", programs.lib_names[programs.tokens.idx[prog_i, i]])
            # print("units     = ", programs.tokens.phy_units[prog_i, i])
            # Parsing previous token afterwards
            parser(i - 1)
        parser(start)
    #t1 = time.perf_counter()
    #print("assign_units_bottom_up %f ms"%((t1-t0)*1e3))
    return None

