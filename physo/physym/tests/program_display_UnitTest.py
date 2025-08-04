import unittest
import warnings

import matplotlib.pyplot as plt
import numpy as np
import time as time
import platform

# Internal imports
from physo.physym import library as Lib
from physo.physym import vect_programs as VProg


def make_lib():
    # LIBRARY CONFIG
    args_make_tokens = {
                    # operations
                    "op_names"             : "all",  # or ["mul", "neg", "inv", "sin"]
                    "use_protected_ops"    : False,
                    # input variables
                    "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                    "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                    "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                    # constants
                    "constants"            : {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                    "constants_units"      : {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] },
                    "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                    # free constants
                    "free_constants"            : {"c0"             , "c1"               , "c2"             },
                    "free_constants_init_val"   : {"c0" : 1.        , "c1"  : 10.        , "c2" : 1.        },
                    "free_constants_units"      : {"c0" : [0, 0, 0] , "c1"  : [1, -1, 0] , "c2" : [0, 0, 1] },
                    "free_constants_complexity" : {"c0" : 0.        , "c1"  : 0.         , "c2" : 1.        },
                           }

    my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                         superparent_units = [1, -2, 1], superparent_name = "y")
    return my_lib


class DisplayTest(unittest.TestCase):

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------- PROGRAM REPRESENTATION ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def test_infix_repr(self):
        # LIBRARY CONFIG
        args_make_tokens = {
                        # operations
                        "op_names"             : "all",  # or ["mul", "neg", "inv", "sin"]
                        "use_protected_ops"    : True,
                        # input variables
                        "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                        "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                        "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                        # constants
                        "constants"            : {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       , "const1" : 1         },
                        "constants_units"      : {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] , "const1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        , "const1" : 1.        },
                            }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [1, -2, 1], superparent_name = "y")

        # TEST PROGRAM
        test_program_str = ["mul", "mul", "M", "n2", "c", "sub", "inv", "sqrt", "sub", "const1", "div", "n2", "v", "n2",
                            "c", "cos", "div", "sub", "const1", "div", "v", "c", "div", "v", "c"]
        test_program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str])
        test_program_length = len(test_program_str)
        test_program_idx = test_program_idx[np.newaxis, :]

        # BATCH
        my_programs = VProg.VectPrograms(batch_size=1, max_time_step=test_program_length, library=my_lib, n_realizations=1)
        my_programs.set_programs(test_program_idx)

        # TEST get_pretty
        expected_pretty_old_sympy = '                                     2        \n     2    ⎛c⋅const₁    ⎞          M⋅c         \n- M⋅c ⋅cos⎜──────── - 1⎟ + ───────────────────\n          ⎝   v        ⎠         _____________\n                                ╱           2 \n                               ╱           v  \n                              ╱   const₁ - ── \n                             ╱              2 \n                           ╲╱              c  '
        expected_pretty_new_sympy = '                                      2       \n     2    ⎛c⋅const₁    ⎞           M⋅c        \n- M⋅c ⋅cos⎜──────── - 1⎟ + ───────────────────\n          ⎝   v        ⎠         _____________\n                                ╱           2 \n                               ╱           v  \n                              ╱   const₁ - ── \n                             ╱              2 \n                           ╲╱              c  '
        result_pretty = my_programs.get_infix_pretty(prog_idx=0)
        # Compare without spaces as they can vary depending on sympy version
        expected_pretty_no_spaces = expected_pretty_new_sympy.replace(" ", "")
        result_pretty_no_spaces   = result_pretty.replace(" ", "")
        works_bool = expected_pretty_no_spaces == result_pretty_no_spaces
        self.assertTrue(works_bool)

        # TEST get_latex
        expected_latex = '- M c^{2} \\cos{\\left(\\frac{c const_{1}}{v} - 1 \\right)} + \\frac{M c^{2}}{\\sqrt{const_{1} - \\frac{v^{2}}{c^{2}}}}'
        result_latex = my_programs.get_infix_latex(prog_idx=0)
        works_bool = expected_latex == result_latex
        self.assertTrue(works_bool)

        # TEST get_sympy
        t0 = time.perf_counter()
        N = int(1e4)
        for _ in range (N):
            my_programs.get_prog(0).get_infix_sympy()
        t1 = time.perf_counter()
        print("get_infix_sympy time = %.3f ms"%((t1-t0)*1e3/N))

        # TEST get_infix_str
        t0 = time.perf_counter()
        N = int(1e4)
        for _ in range (N):
            my_programs.get_prog(0).get_infix_str()
        t1 = time.perf_counter()
        print("get_infix_str time = %.3f ms"%((t1-t0)*1e3/N))

        # TEST
        try:
            t0 = time.perf_counter()
            img = my_programs.get_infix_image(prog_idx=0,)
            # my_programs.show_infix(prog_idx=0,)
            t1 = time.perf_counter()
            print("\nget_infix_image time = %.3f s" % (t1 - t0))
        except:
            print("Infix generation failed : get_infix_image (Acceptable failure)")

        return None

    def test_infix_repr_with_pow(self):
        # LIBRARY CONFIG
        args_make_tokens = {
                        # operations
                        "op_names"             : "all",  # or ["mul", "neg", "inv", "sin"]
                        "use_protected_ops"    : True,
                        # input variables
                        "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                        "input_var_units"      : {"x" : [0, 0, 0] , "v" : [0, 0, 0]  , "t" : [0, 0, 0] },
                        "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                        # constants
                        "constants"            : {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       , "const1" : 1         },
                        "constants_units"      : {"pi" : [0, 0, 0] , "c" : [0, 0, 0] , "M" : [0, 0, 0] , "const1" : [0, 0, 0] },
                        "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        , "const1" : 1.        },
                            }
        my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                             superparent_units = [0, 0, 0], superparent_name = "y")

        # TEST PROGRAM
        test_program_str = ["mul", "x", "pow", "t", "c",]
        test_program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str])
        test_program_length = len(test_program_str)
        test_program_idx = test_program_idx[np.newaxis, :]

        # BATCH
        my_programs = VProg.VectPrograms(batch_size=1, max_time_step=test_program_length, library=my_lib, n_realizations=1)
        my_programs.set_programs(test_program_idx)

        # TEST get_pretty
        expected_pretty_sympy = ' c  \nt ⋅x'
        result_pretty = my_programs.get_infix_pretty(prog_idx=0)
        # Compare without spaces as they can vary depending on sympy version
        expected_pretty_no_spaces = expected_pretty_sympy.replace(" ", "")
        result_pretty_no_spaces   = result_pretty.replace(" ", "")
        works_bool = expected_pretty_no_spaces == result_pretty_no_spaces
        self.assertTrue(works_bool)

        # TEST get_latex
        expected_latex = 't^{c} x'
        result_latex = my_programs.get_infix_latex(prog_idx=0)
        works_bool = expected_latex == result_latex
        self.assertTrue(works_bool)

        # TEST get_sympy
        try:
            my_programs.get_prog(0).get_infix_sympy()
        except:
            self.fail("get_infix_sympy failed")

        # TEST get_infix_str
        try:
            my_programs.get_prog(0).get_infix_str()
        except:
            self.fail("get_infix_str failed")


        return None

    def test_tree_rpr(self):

        if platform.system() == "Windows":
            print("Not testing tree representation features on Windows as this generally causes problems and is only "
                  "useful for physo developers.")
        else:
            # LIBRARY CONFIG
            args_make_tokens = {
                            # operations
                            "op_names"             : "all",  # or ["mul", "neg", "inv", "sin"]
                            "use_protected_ops"    : True,
                            # input variables
                            "input_var_ids"        : {"x" : 0         , "v" : 1          , "t" : 2,        },
                            "input_var_units"      : {"x" : [1, 0, 0] , "v" : [1, -1, 0] , "t" : [0, 1, 0] },
                            "input_var_complexity" : {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                            # constants
                            "constants"            : {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       , "const1" : 1         },
                            "constants_units"      : {"pi" : [0, 0, 0] , "c" : [1, -1, 0], "M" : [0, 0, 1] , "const1" : [0, 0, 0] },
                            "constants_complexity" : {"pi" : 0.        , "c" : 0.        , "M" : 1.        , "const1" : 1.        },
                                }
            my_lib = Lib.Library(args_make_tokens = args_make_tokens,
                                 superparent_units = [1, -2, 1], superparent_name = "y")

            # TEST PROGRAM WO DUMMIES
            test_program_str = ["mul", "mul", "M", "n2", "c", "sub", "inv", "sqrt", "sub", "const1", "div", "n2", "v", "n2",
                                "c", "cos", "div", "sub", "const1", "div", "v", "c", "div", "v", "c"]
            test_program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str])
            test_program_length = len(test_program_str)
            test_program_idx = test_program_idx[np.newaxis, :]
            my_programs_wo_dummies = VProg.VectPrograms(batch_size=1, max_time_step=test_program_length, library=my_lib, n_realizations=1)
            my_programs_wo_dummies.set_programs(test_program_idx)

            # TEST PROGRAM W DUMMIES
            test_program_str = ["mul", "mul", "M", "n2", "c", "sub", "inv", "sqrt", "sub", "const1", "div", "n2", "v", "n2",
                                "c", "cos", "div", "sub", "const1", "div", "v", "c", "div", "v",]
            test_program_idx = np.array([my_lib.lib_name_to_idx[tok_str] for tok_str in test_program_str])
            test_program_length = len(test_program_str) + 1
            test_program_idx = test_program_idx[np.newaxis, :]
            my_programs_w_dummies = VProg.VectPrograms(batch_size=1, max_time_step=test_program_length, library=my_lib, n_realizations=1)
            my_programs_w_dummies.set_programs(test_program_idx)

            # TEST
            for my_programs in [my_programs_wo_dummies,my_programs_w_dummies]:
                # get_tree_latex
                try:
                    t0 = time.perf_counter()
                    tree_latex = my_programs.get_tree_latex(prog_idx=0,)
                    t1 = time.perf_counter()
                    print("\nget_tree_latex time = %.3f s"%(t1-t0))
                except:
                    print("Tree generation failed : get_tree_latex (Acceptable failure)")
                # get_tree_image
                try:
                    t0 = time.perf_counter()
                    img = my_programs.get_tree_image(prog_idx=0)
                    t1 = time.perf_counter()
                    print("\nget_tree_image time = %.3f s"%(t1-t0))
                except:
                    print("Tree generation failed : get_tree_image (Acceptable failure)")
                # get_tree_image_via_tex
                try:
                    t0 = time.perf_counter()
                    img = my_programs.get_tree_image_via_tex(prog_idx=0, fname="tree_repr_test")
                    t1 = time.perf_counter()
                    print("\nget_tree_image_via_tex time = %.3f s"%(t1-t0))
                except:
                    print("Tree generation failed : get_tree_image_via_tex (Acceptable failure)")
                # show_tree (via_tex=True)
                try:
                    t0 = time.perf_counter()
                    my_programs.show_tree(prog_idx=0, via_tex=True)
                    plt.close()
                    t1 = time.perf_counter()
                    print("\nshow_tree (via_tex=True) time = %.3f s"%(t1-t0))
                except:
                    print("Tree generation failed : show_tree (via_tex=True) (Acceptable failure)")
                # show_tree (via_tex=False)
                try:
                    t0 = time.perf_counter()
                    my_programs.show_tree(prog_idx=0, via_tex=False)
                    plt.close()
                    t1 = time.perf_counter()
                    print("\nshow_tree (via_tex=False) time = %.3f s"%(t1-t0))
                except:
                    print("Tree generation failed : show_tree (via_tex=False) (Acceptable failure)")
        return None

if __name__ == '__main__':
    unittest.main(verbosity=2)
