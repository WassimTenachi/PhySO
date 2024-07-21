#!/usr/bin/env python
# coding: utf-8

# # $\Phi$-SO demo : Class SR quick start

# ## Class SR definition

# Class Symbolic Regression:
# Automatically finding a single analytical functional form that accurately fits multiple datasets - each governed by its own (possibly) unique set of fitting parameters.
# This hierarchical framework leverages the common constraint that all the members of a single class of physical phenomena follow a common governing law.

# ![class_sr_framework.png](attachment:3b74f0d7-8a06-4663-bf0d-e442c0f40aff.png)

# ## Package import

# In[1]:


# External packages
import numpy as np
import matplotlib.pyplot as plt
import torch

# In[2]:


# Internal code import
import physo
import physo.learn.monitoring as monitoring

if __name__ == '__main__':


    # ## Fixing seed

    # In[3]:


    # Seed
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)


    # ## Dataset

    # In[4]:


    # Making toy synthetic data
    multi_X = []
    multi_y = []

    # Realization 0
    x0 = np.random.uniform(-10, 10, 256)
    x1 = np.random.uniform(-10, 10, 256)
    X = np.stack((x0, x1), axis=0)
    y = 1.123*x0 + 1.123*x1 + 10.123
    multi_X.append(X)
    multi_y.append(y)

    # Realization 1
    x0 = np.random.uniform(-11, 11, 500)
    x1 = np.random.uniform(-11, 11, 500)
    X = np.stack((x0, x1), axis=0)
    y = 2*1.123*x0 + 1.123*x1 + 10.123
    multi_X.append(X)
    multi_y.append(y)

    # Realization 2
    x0 = np.random.uniform(-12, 12, 256)
    x1 = np.random.uniform(-12, 12, 256)
    X = np.stack((x0, x1), axis=0)
    y = 1.123*x0 + 1.123*x1 + 0.5*10.123
    multi_X.append(X)
    multi_y.append(y)

    # Realization 3
    x0 = np.random.uniform(-13, 13, 256)
    x1 = np.random.uniform(-13, 13, 256)
    X = np.stack((x0, x1), axis=0)
    y = 0.5*1.123*x0 + 1.123*x1 + 2*10.123
    multi_X.append(X)
    multi_y.append(y)


    # Where `multi_X` and `multi_y` are lists containing $n_{reals}$ realizations (here $n_{reals}=4$). `multi_X` should contain $n_{reals}$ elements consisting of input variables values ($X$) (one per realization) and `multi_y` should contain $n_{reals}$ corresponding target values ($y$) (one per realization).
    # Elements $X$, $y$ should respectively be of shape $(n_{dim}, ?,)$ and $(?,)$ where $n_{dim}$ is the number of input variables (here $n_{dim}=2$ as there are 2 input variables: $\{x_0, x_1\}$) and `?` is the number of data points (this number must be consistent across each $\{X, y\}$ pair but can depend on the realization).
    #
    # It should be noted that free constants search starts around 1. by default. Therefore when using default hyperparameters, normalizing the data around an order of magnitude of 1 is strongly recommended.

    # Dataset plot

    # In[5]:


    n_reals = len(multi_X)
    for i in range(n_reals):
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        fig.suptitle("Realization {}".format(i))
        ax[0].plot(multi_X[i][0], multi_y[i], 'k.')
        ax[0].set_xlabel("x0")
        ax[0].set_ylabel("y")
        ax[1].plot(multi_X[i][1], multi_y[i], 'k.')
        ax[1].set_xlabel("x1")
        ax[1].set_ylabel("y")
        plt.show()


    # ## Running Class SR task

    # ### Available configurations

    # It should be noted that SR capabilities of `physo` are heavily dependent on hyperparameters, it is therefore recommended to tune hyperparameters to your own specific problem for doing science.
    # Summary of currently available hyperparameters presets configurations:
    #
    # |  Config    |            Recommended usecases                           |    Speed    |   Effectiveness   |                           Notes                                |
    # |:----------:|:---------------------------------------------------------:|:-----------:|:-----------------:|:--------------------------------------------------------------:|
    # | `config0b` | Demos                                                     |     +++     |          +        | Light and fast config.                                         |
    # | `config1b` | SR with DA $^*$ ;  Class SR with DA $^*$                    |       +     |        +++        | Config used for Feynman Benchmark and MW streams Benchmark.    |
    # | `config2b` | SR ; Class SR                                             |      ++     |         ++        | Config used for Class Benchmark.                               |
    #
    # $^*$ DA = Dimensional Analysis
    #
    # By default, `config0b` is used, however it is recommended to follow the upper recommendations for doing science.
    #
    # __Class SR side notes__:
    #
    # 1. It is recommended to always use the `b` variant (ie. `configxb`) of configurations for Class SR as these variants specify a larger number of steps for the free constants optimizations that is performed during expression candidate evaluation (this is necessary because all free constants specific to each realization are optimized at the same time which typically makes it necessary to have a larger number of optimization steps).
    #
    # 2. Due to the typically higher number of free constants values to tune, Class SR is typically much more expansive computationally than regular SR, lower batch size configurations are therefore recommended.
    #
    # __DA side note__:
    # 1. During the first tens of iterations, the neural network is typically still learning the rules of dimensional analysis, resulting in most candidates being discarded and not learned on, effectively resulting in a much smaller batch size (typically 10x smaller), thus making the evaluation process much less computationally expensive. It is therefore recommended to compensate this behavior by using a higher batch size configuration which helps provide the neural network sufficient learning information.
    #

    # ### Logging config

    # In[6]:


    save_path_training_curves = 'demo_curves.png'
    save_path_log             = 'demo.log'

    run_logger     = lambda : monitoring.RunLogger(save_path = save_path_log,
                                                    do_save = True)

    run_visualiser = lambda : monitoring.RunVisualiser (epoch_refresh_rate = 1,
                                               save_path = save_path_training_curves,
                                               do_show   = False,
                                               do_prints = True,
                                               do_save   = True, )


    # __Class SR run__ :
    #
    # Automatically finding a single analytical functional form that accurately fits multiple datasets - each governed by its own (possibly) unique set of fitting parameters.
    # This hierarchical framework leverages the common constraint that all the members of a single class of physical phenomena follow a common governing law.
    #
    # Ie. recovering a single analytical function $f$ that best fits $y_{i_{real}} = f_{i_{real}}(x_0, ..., x_{n_{dim}})$ for each realization $i_{real} \leq n_{reals}$ of a phenomena given $\{\{x_0, ..., x_{n_{dim}}\}_{i_{real}}\}_{i_{real} \leq n_{reals}}$ (`multi_X`) data and $\{y_{i_{real}}\}_{i_{real} \leq n_{reals}}$ (`multi_y`) data.
    # Realization-specific free constants ie. free constants taking different values depending on the realization ($\{k_0, k_1, ...\}_{i_{real} \leq n_{reals}}$) are designated as `spe_free_consts` and free constants common to the whole class ($\{c_0, c_1, ...\}$) are designated as `class_free_consts`.

    # ### Run

    # In[ ]:


    # Running SR task
    expression, logs = physo.ClassSR(multi_X, multi_y,
                                # Giving names of variables (for display purposes)
                                X_names = [ "x0"       , "x1"        ],
                                # Associated physical units (ignore or pass zeroes if irrelevant)
                                X_units = [ [0, 0, 0] , [0, 0, 0]    ],
                                # Giving name of root variable (for display purposes)
                                y_name  = "y",
                                y_units = [0, 0, 0],
                                # Fixed constants
                                fixed_consts       = [ 1.      ],
                                fixed_consts_units = [ [0, 0, 0] ],
                                # Whole class free constants
                                class_free_consts_names = [ "c0"      ,],
                                class_free_consts_units = [ [0, 0, 0] ,],
                                # Realization specific free constants
                                spe_free_consts_names = [ "k0"      , "k1"        ],
                                spe_free_consts_units = [ [0, 0, 0] , [0, 0, 0]   ],
                                # Run config
                                run_config = physo.config.config0b.config0b,
                                # Symbolic operations that can be used to make f
                                op_names = ["add", "sub", "mul", "div"],
                                get_run_logger     = run_logger,
                                get_run_visualiser = run_visualiser,
                                # Parallel mode (only available when running from python scripts, not notebooks)
                                parallel_mode = True,
                                # Number of iterations
                                epochs = 10,
    )



    # ## Inspecting best expression found

    # ### Getting best expression

    # The best expression found (in accuracy) is returned in `expression`:

    # In[ ]:


    best_expr = expression
    print(best_expr.get_infix_pretty())


    # It can also be loaded later on from log files:

    # In[ ]:


    import physo
    from physo.benchmark.utils import symbolic_utils as su
    import sympy

    # Loading pareto front expressions
    pareto_expressions = physo.read_pareto_pkl("demo_curves_pareto.pkl")
    # Most accurate expression is the last in the Pareto front:
    best_expr = pareto_expressions[-1]
    print(best_expr.get_infix_pretty())


    # ### Display

    # The expression can be converted into...
    # A sympy expression:

    # In[ ]:


    best_expr.get_infix_sympy()


    # A sympy expression (with evaluated free constants values):

    # In[ ]:


    # In Class SR this will return an array of expression each containing constants values specific to each realization:
    best_expr.get_infix_sympy(evaluate_consts=True)


    # A latex string:

    # In[ ]:


    best_expr.get_infix_latex()


    # A latex string (with evaluated free constants values):

    # In[ ]:


    sympy.latex(best_expr.get_infix_sympy(evaluate_consts=True))


    # ### Getting free constant values

    # Class free constants

    # In[ ]:


    best_expr.free_consts


    # In[ ]:


    best_expr.free_consts.class_values


    # Realization-specific free constants

    # In[ ]:


    best_expr.free_consts.spe_values


    # ### Checking exact symbolic recovery

    # In[ ]:


    # To sympy
    best_expr = best_expr.get_infix_sympy(evaluate_consts=True)

    best_expr = best_expr[0] # Considering the expression with its constants set for realization no 0 (Class SR)

    # Printing best expression simplified and with rounded constants
    print("best_expr : ", su.clean_sympy_expr(best_expr, round_decimal = 4))

    # Target expression was:
    target_expr = sympy.parse_expr("1.123*x0 + 1.123*x1 + 10.123")
    print("target_expr : ", su.clean_sympy_expr(target_expr, round_decimal = 4))

    # Check equivalence
    print("\nChecking equivalence:")
    is_equivalent, log = su.compare_expression(
                            trial_expr  = best_expr,
                            target_expr = target_expr,
                            handle_trigo            = True,
                            prevent_zero_frac       = True,
                            prevent_inf_equivalence = True,
                            verbose                 = True,
    )
    print("Is equivalent:", is_equivalent)

