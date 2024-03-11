from physo.physym import token as Tok

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ SINGLE EXECUTION ------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

def ExecuteProgram (input_var_data, program_tokens, class_free_consts_vals=None, spe_free_consts_vals=None):
    """
    Executes a symbolic function program.
    Parameters
    ----------
    input_var_data : torch.tensor of shape (n_dim, ?,) of float
        Values of the input variables of the problem with n_dim = nb of input variables.
    program_tokens : list of token.Token
        Symbolic function program in reverse Polish notation order.
    class_free_consts_vals : torch.tensor of shape (n_class_free_const,) or of shape (n_class_free_const, ?) of float or None
        Values of class free constants to use for program execution. Works with either a single value for each constant
        (shape (n_class_free_const,)) or a value for each constant and each data point (shape (n_class_free_const, ?)).
        class_free_consts_vals must be given if program_tokens contains one or more class free constants.
    spe_free_consts_vals : torch.tensor of shape (n_spe_free_const,) or of shape (n_spe_free_const, ?) of float or None
        Values of spe free constants to use for program execution. Works with either a single value for each constant
        (shape (n_spe_free_const,)) or a value for each constant and each data point (shape (n_spe_free_const, ?)).
        spe_free_consts_vals must be given if program_tokens contains one or more spe free constants.
    Returns
    -------
    y : torch.tensor of shape (?,) of float
        Result of computation.
    """

    # Size
    (n_dim, data_size,) = input_var_data.shape

    # Number of tokens in the program
    n_tokens = len(program_tokens)

    # Current stack of computed results
    curr_stack = []

    # De-stacking program (iterating from last token to first)
    start = n_tokens - 1
    for i in range (start, -1, -1):
        token = program_tokens[i]
        # Terminal token
        if token.arity == 0:
            # Function type token
            if token.var_type == Tok.VAR_TYPE_OP:
                #curr_stack.append(token.function())
                raise ValueError("Function of arity = 0 encountered. Use var_type = %i for fixed constants."%(Tok.VAR_TYPE_FIXED_CONST))
            # Input variable (eg. x0, x1 etc.)
            elif token.var_type == Tok.VAR_TYPE_INPUT_VAR:
                curr_stack.append(input_var_data[token.var_id])
            # Class free constant variable (eg. c0, c1 etc.)
            elif token.var_type == Tok.VAR_TYPE_CLASS_FREE_CONST:
                if class_free_consts_vals is not None:
                    curr_stack.append(class_free_consts_vals[token.var_id])
                else:
                    raise ValueError("Class free constant encountered in program evaluation but class free constant values "
                                     "were not given.")
            # Spe free constant variable (eg. k0, k1 etc.)
            elif token.var_type == Tok.VAR_TYPE_SPE_FREE_CONST:
                if spe_free_consts_vals is not None:
                    curr_stack.append(spe_free_consts_vals[token.var_id])
                else:
                    raise ValueError("Spe free constant encountered in program evaluation but spe free constant values "
                                     "were not given.")
            # Fixed constant (eg. pi, 1 etc.)
            elif token.var_type == Tok.VAR_TYPE_FIXED_CONST:
                curr_stack.append(token.fixed_const)
            else:
                raise NotImplementedError("Token of unknown var_type encountered in ExecuteProgram.")
        # Non-terminal token
        elif token.arity > 0:
            # Last pending elements are those needed for next computation (in reverse order)
            args = curr_stack[-token.arity:][::-1]
            res = token.function(*args)
            # Removing those pending elements as they were used
            curr_stack = curr_stack[:-token.arity]
            # Appending last result to stack
            curr_stack.append(res)
    y = curr_stack[0]
    return y

def ComputeInfixNotation (program_tokens):
    """
    Computes infix str representation of a program.
    (which is the usual way to note symbolic function: +34 (in polish notation) = 3+4 (in infix notation))
    Parameters
    ----------
    program_tokens : list of token.Token
        List of tokens making up the program.
    Returns
    -------
    program_str : str
    """
    # Number of tokens in the program
    n_tokens = len(program_tokens)

    # Current stack of computed results
    curr_stack = []

    # De-stacking program (iterating from last token to first)
    start = n_tokens - 1
    for i in range (start, -1, -1):
        token = program_tokens[i]
        # Last pending elements are those needed for next computation (in reverse order)
        args = curr_stack[-token.arity:][::-1]
        if token.arity == 0:
            res = token.sympy_repr
        elif token.arity == 1:
            if token.is_power is True:
                pow = '{:g}'.format(token.power)  # without trailing zeros
                res = "((%s)**(%s))" % (args[0], pow)
            else:
                res = "%s(%s)" % (token.sympy_repr, args[0])
        elif token.arity == 2:
            res = "(%s%s%s)" % (args[0], token.sympy_repr, args[1])
        elif token.arity > 2 :
            args_str = ""
            for arg in args: args_str+="%s,"%arg
            args_str = args_str[:-1] # deleting last ","
            res = "%s(%s)" % (token.sympy_repr, args_str)
        if token.arity > 0:
            # Removing those pending elements as they were used
            curr_stack = curr_stack[:-token.arity]
        # Appending last result to stack
        curr_stack.append(res)
    return curr_stack[0]
