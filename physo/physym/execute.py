import warnings

import numpy as np
import torch as torch

def ExecuteProgram (input_var_data, program_tokens, free_const_values=None):
    """
    Executes a symbolic function program.
    Parameters
    ----------
    X : torch.tensor of shape (n_dim, ?,) of float
        Values of the input variables of the problem with n_dim = nb of input variables.
    program_tokens : list of token.Token
        Symbolic function program in reverse Polish notation order.
    free_const_values : torch.tensor of shape (n_free_const,) of float or None
        Current values of free constants with for program made of program_tokens n_free_const = nb of choosable free
        constants (library.n_free_constants). free_const_values must be given if program_tokens contains one or more
         free const tokens.
    Returns
    -------
    y : torch.tensor of shape (?,) of float
        Result of computation.
    """

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
            # Fixed constant (eg. pi, 1 etc.)
            if token.var_type == 0:
                curr_stack.append(token.function())
            # Input variable (eg. x0, x1 etc.)
            elif token.var_type == 1:
                curr_stack.append(input_var_data[token.var_id])
            # Free constant variable (eg. c0, c1 etc.)
            elif token.var_type == 2:
                if free_const_values is not None:
                    # curr_stack.append(torch.abs(free_const_values[token.var_id])) # Making free const positive values only
                    curr_stack.append(free_const_values[token.var_id])
                else:
                    raise ValueError("Free constant encountered in program evaluation but free constant values were "
                                     "not given.")
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

