#!/usr/bin/env python
# coding: utf-8

# # $\Phi$-SO demo : Damped harmonic oscillator

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


    data_size = int(1e3)


    # In[5]:


    # Data points
    t = np.random.uniform(np.pi, 10*np.pi, data_size)
    X = np.stack((t,), axis=0)
    f      = 0.784
    alpha0 = 1./9.89
    phi    = 0.997
    y = np.exp(-t*alpha0)*np.cos(f*t + phi)


    # Dataset plot

    # In[6]:


    n_dim = X.shape[0]
    fig, ax = plt.subplots(n_dim, 1, figsize=(10,5))
    for i in range (n_dim):
        curr_ax = ax if n_dim==1 else ax[i]
        curr_ax.plot(X[i], y, 'k.',)
        curr_ax.set_xlabel("X[%i]"%(i))
        curr_ax.set_ylabel("y")
    plt.show()


    # ## Running SR task

    # ### Logging config

    # In[7]:


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

    # In[8]:


    #physo.physym.batch_execute.SHOW_PROGRESS_BAR = True


    # In[9]:


    run_config = physo.config.config1.config1


    # In[ ]:


    # Running SR task
    expression, logs = physo.SR(X, y,
                                # Giving names of variables (for display purposes)
                                X_names = [ "t"       ],
                                # Associated physical units (ignore or pass zeroes if irrelevant)
                                X_units = [ [0, 0, 1] ],
                                # Giving name of root variable (for display purposes)
                                y_name  = "y",
                                y_units = [0, 0, 0],
                                # Fixed constants
                                fixed_consts       = [ 1.      ],
                                fixed_consts_units = [ [0,0,0] ],
                                # Free constants names (for display purposes)
                                free_consts_names = [ "f"        , "alpha0"   , "phi"     ],
                                free_consts_units = [ [0, 0, -1] , [0, 0, -1] , [0, 0, 0] ],
                                # Symbolic operations that can be used to make f
                                op_names = ["mul", "add", "sub", "div", "inv", "n2", "sqrt", "neg", "exp", "log", "sin", "cos"],
                                get_run_logger     = run_logger,
                                get_run_visualiser = run_visualiser,
                                # Run config
                                run_config = run_config,
                                # Parallel mode (only available when running from python scripts, not notebooks)
                                parallel_mode = True,
                                #n_cpus = 8,
                                # Number of iterations
                                epochs = int(1e99)

    )


    # ## Inspecting best expression found

    # In[ ]:


    pareto_front_complexities, pareto_front_programs, pareto_front_r, pareto_front_rmse = run_logger.get_pareto_front()


    # In[ ]:


    for prog in pareto_front_programs:
        prog.show_infix(do_simplify=True)
        free_consts = prog.free_consts.class_values[0].detach().cpu().numpy()
        for i in range (len(free_consts)):
            print("%s = %f"%(prog.library.free_const_names[i], free_consts[i]))

