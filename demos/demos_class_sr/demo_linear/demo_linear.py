#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import torch

# Internal code import
import physo
import physo.learn.monitoring as monitoring


# ### Fixing seed

# In[2]:


# Seed
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)


# ### Dataset

# In[3]:


# Dataset
multi_X = []
multi_y = []


# Object 0
x0 = np.random.uniform(-10, 10, 256)
x1 = np.random.uniform(-10, 10, 256)
X = np.stack((x0, x1), axis=0)
y = 1.123*x0 + 1.123*x1 + 10.123
multi_X.append(X)
multi_y.append(y)

# Object 1
x0 = np.random.uniform(-11, 11, 500)
x1 = np.random.uniform(-11, 11, 500)
X = np.stack((x0, x1), axis=0)
y = 2*1.123*x0 + 1.123*x1 + 10.123
multi_X.append(X)
multi_y.append(y)

# Object 2
x0 = np.random.uniform(-12, 12, 256)
x1 = np.random.uniform(-12, 12, 256)
X = np.stack((x0, x1), axis=0)
y = 1.123*x0 + 1.123*x1 + 0.5*10.123
multi_X.append(X)
multi_y.append(y)

# Object 3
x0 = np.random.uniform(-13, 13, 256)
x1 = np.random.uniform(-13, 13, 256)
X = np.stack((x0, x1), axis=0)
y = 0.5*1.123*x0 + 1.123*x1 + 2*10.123
multi_X.append(X)
multi_y.append(y)


# In[ ]:





# Dataset plot

# In[4]:


n_objects = len(multi_X)
for i in range(n_objects):
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    fig.suptitle("Object {}".format(i))
    ax[0].scatter(multi_X[i][0], multi_y[i])
    ax[0].set_xlabel("x0")
    ax[0].set_ylabel("y")
    ax[1].scatter(multi_X[i][1], multi_y[i])
    ax[1].set_xlabel("x1")
    ax[1].set_ylabel("y")
    plt.show()


# ### Log config

# In[5]:


save_path_training_curves = 'demo_curves.png'
save_path_log             = 'demo.log'

run_logger     = lambda : monitoring.RunLogger(save_path = save_path_log,
                                                do_save = True)

run_visualiser = lambda : monitoring.RunVisualiser (epoch_refresh_rate = 1,
                                           save_path = save_path_training_curves,
                                           do_show   = False,
                                           do_prints = True,
                                           do_save   = True, )


# ### Run

# In[6]:


# Running SR task
expression, logs = physo.ClassSR(multi_X, multi_y,
                            # Giving names of variables (for display purposes)
                            X_names = [ "x0"       , "x1"        ],
                            # Giving units of input variables
                            X_units = [ [0, 0, 0] , [0, 0, 0] ],
                            # Giving name of root variable (for display purposes)
                            y_name  = "y",
                            # Giving units of the root variable
                            y_units = [0, 0, 0],
                            # Fixed constants
                            fixed_consts       = [ 1.      ],
                            # Units of fixed constants
                            fixed_consts_units = [ [0, 0, 0] ],
                            # Whole class free constants
                            class_free_consts_names = [ "c0"      ,],
                            class_free_consts_units = [ [0, 0, 0] ,],
                            # Dataset specific free constants
                            spe_free_consts_names = [ "k0"      , "k1"        ],
                            spe_free_consts_units = [ [0, 0, 0] , [0, 0, 0]   ],
                            # Run config
                            run_config = physo.config.config0b.config0b,
                            # FOR TESTING
                            op_names = ["add", "sub", "mul", "div"],
                            get_run_logger     = run_logger,
                            get_run_visualiser = run_visualiser,
                            parallel_mode = False,
                            epochs = 10,
)



# In[ ]:





# In[ ]:





# In[10]:


# Inspecting pareto front expressions
pareto_front_complexities, pareto_front_expressions, pareto_front_r, pareto_front_rmse = logs.get_pareto_front()


# In[11]:


expression.get_infix_sympy()


# ### Loading best expression and checking exact symbolic recovery

# In[13]:


import physo
from physo.benchmark.utils import symbolic_utils as su
import sympy

# Loading best expression from log file
pareto_expressions = physo.read_pareto_pkl("demo_curves_pareto.pkl")
best_expr = pareto_expressions[-1]

# To sympy
best_expr = best_expr.get_infix_sympy(evaluate_consts=True)
# Considering the expression with its constants set for realization no 0
best_expr = best_expr[0]

# Printing best expression simplified and with rounded constants
print("best_expr : ", su.clean_sympy_expr(best_expr, round_decimal = 3))

# Target expression was:
target_expr = sympy.parse_expr("1.123*x0 + 1.123*x1 + 10.123")
print("target_expr : ", su.clean_sympy_expr(target_expr, round_decimal = 3))

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


# In[ ]:





# In[ ]:





# In[ ]:




