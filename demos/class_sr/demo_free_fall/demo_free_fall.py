#!/usr/bin/env python
# coding: utf-8

# # $\Phi$-SO demo : Free fall

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


    NOISE_LEVEL = 0.0
    data_size = 100

    multi_X = []
    multi_y = []

    low = 0
    up  = 1

    # Object 1
    t  = np.random.uniform(low, up, data_size)
    g  = 9.81
    v0 = 7.34
    h0 = 1.23
    y  = -0.5*g*t**2 + v0*t + h0
    X = np.stack((t,), axis=0)
    y_rms = ((y ** 2).mean()) ** 0.5
    epsilon = NOISE_LEVEL * np.random.normal(0, y_rms, len(y))
    y = y + epsilon
    multi_X.append(X)
    multi_y.append(y)

    # Object 2
    t  = np.random.uniform(low, up, data_size)
    g  = 9.81
    v0 = -1.17
    h0 = 6.48
    y  = -0.5*g*t**2 + v0*t + h0
    X = np.stack((t,), axis=0)
    y_rms = ((y ** 2).mean()) ** 0.5
    epsilon = NOISE_LEVEL * np.random.normal(0, y_rms, len(y))
    y = y + epsilon
    multi_X.append(X)
    multi_y.append(y)


    # Object 3
    t  = np.random.uniform(low, up, data_size)
    g  = 9.81
    v0 = 5.74
    h0 = -2.13
    y  = -0.5*g*t**2 + v0*t + h0
    X = np.stack((t,), axis=0)
    y_rms = ((y ** 2).mean()) ** 0.5
    epsilon = NOISE_LEVEL * np.random.normal(0, y_rms, len(y))
    y = y + epsilon
    multi_X.append(X)
    multi_y.append(y)

    # Object 4
    t  = np.random.uniform(low, up, data_size)
    g  = 9.81
    v0 = 2.12
    h0 = 1.42
    y  = -0.5*g*t**2 + v0*t + h0
    X = np.stack((t,), axis=0)
    y_rms = ((y ** 2).mean()) ** 0.5
    epsilon = NOISE_LEVEL * np.random.normal(0, y_rms, len(y))
    y = y + epsilon
    multi_X.append(X)
    multi_y.append(y)


    # Dataset plot

    # In[5]:


    n_objects = len(multi_X)

    cmap = plt.cm.get_cmap('inferno', n_objects)
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    for i in range(n_objects):
        ax.scatter(multi_X[i][0], multi_y[i], c=cmap(i), label="Object %i"%(i))
    ax.legend()
    fig.savefig("data.png")
    plt.show()


    # ## Running Class SR task

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


    # ### Run

    # In[ ]:


    # Running SR task
    expression, logs = physo.ClassSR(multi_X, multi_y,
                                         # Giving names of variables (for display purposes)
                                         X_names = [ "t" ,       ],
                                         # Giving units of input variables
                                         X_units = [ [0, 0, 0] , ],
                                         # Giving name of root variable (for display purposes)
                                         y_name  = "y",
                                         # Giving units of the root variable
                                         y_units = [0, 0, 0],
                                         # Fixed constants
                                         fixed_consts       = [ 1.      ],
                                         # Units of fixed constants
                                         fixed_consts_units = [ [0, 0, 0] ],
                                         # Free constants names (for display purposes)
                                         class_free_consts_names = [ "c0"       ],
                                         # Units of free constants
                                         class_free_consts_units = [ [0, 0, 0]  ],
                                         # Free constants names (for display purposes)
                                         spe_free_consts_names = [ "k0"       , "k1"       , "k2"       ],
                                         # Units of free constants
                                         spe_free_consts_units = [ [0, 0, 0]  , [0, 0, 0]  , [0, 0, 0]  ],
                                         # Run config
                                         run_config = physo.config.config0b.config0b,

                                         op_names = ["mul", "add", "sub", "div", "inv", "n2", "sqrt", "neg", "exp", "log", "sin", "cos"],
                                         get_run_logger     = run_logger,
                                         get_run_visualiser = run_visualiser,

                                         parallel_mode = True,
                                         # n_cpus = 8,
                                         epochs = int(1e9),
                                         )


    # ## Inspecting pareto front expressions

    # In[ ]:


    pareto_front_complexities, pareto_front_expressions, pareto_front_r, pareto_front_rmse = logs.get_pareto_front()


    # In[ ]:


    expression.get_infix_sympy()

