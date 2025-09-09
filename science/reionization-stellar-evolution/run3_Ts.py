import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

# Internal code import
import physo
import physo.learn.monitoring as monitoring

if __name__ == '__main__':

    run_name = 3

    # Seed
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Dataset
    PATH_DATA = "data/4096_curves.csv"
    data = pd.read_csv(PATH_DATA)
    data = data[data["z"] < 22] # Limit to z<22
    # Subsample data to 10k
    data = data.sample(n=10000, random_state=seed).reset_index(drop=True)
    # Randomize data
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    shorter_names = {
        "z": "z",
        "F_STAR10": "Fs",
        "F_ESC10": "Fesc",
        "ALPHA_STAR": "As",
        "ALPHA_ESC": "Aesc",
        "M_TURN": "Mt",
        "L_X": "LX",
        "t_STAR": "taus",
        "xHI": "xHI",
        "Ts": "Ts",
        "Tb": "Tb"
    }

    # Weights
    # Weight to the data as a function of z
    # 1 value everywhere with a gaussian centered at z=20 with width 2
    y_weights_func = lambda z : 1 + np.exp(-0.5*((z-20)/0.8)**2)
    fig, ax = plt.subplots(1,1)
    z_plot = np.linspace(data["z"].min(), data["z"].max(), 1000)
    ax.plot(z_plot, y_weights_func(z_plot), 'r-')
    ax.set_xlabel("z")
    ax.set_ylabel("y_weights")
    fig.savefig("%s_Ts_weights.png"%(run_name))
    plt.show()

    var_names = ["z", "F_STAR10", "F_ESC10", "ALPHA_STAR", "ALPHA_ESC", "M_TURN", "L_X", "t_STAR"]
    X = data[var_names].values.T
    y = data["Ts"].values

    X_names = [shorter_names[name] for name in var_names]
    y_name = shorter_names["Ts"]
    free_consts_names = ["c%i" % i for i in range(6)]

    # Dataset plot
    n_dim = X.shape[0]
    fig, ax = plt.subplots(n_dim, 1, figsize=(10,30))
    for i in range (n_dim):
        curr_ax = ax if n_dim==1 else ax[i]
        curr_ax.plot(X[i], y, 'k.',)
        curr_ax.set_xlabel("X[%i]"%(i))
        curr_ax.set_ylabel("y")
    plt.show()

    # Logging config

    save_path_training_curves = '%s_Ts_curves.png'%(run_name)
    save_path_log             = '%s_Ts.log'%(run_name)

    run_logger     = lambda : monitoring.RunLogger(save_path = save_path_log,
                                                    do_save = True)

    run_visualiser = lambda : monitoring.RunVisualiser (epoch_refresh_rate = 5,
                                               save_path = save_path_training_curves,
                                               do_show   = False,
                                               do_prints = True,
                                               do_save   = True, )


    # Run
    physo.physym.batch_execute.SHOW_PROGRESS_BAR = True
    run_config = physo.config.config3.config3

    physo.config.utils.soft_length_plot(run_config, do_show=True, save_path="%s_Ts_soft_length_prior.png"%(run_name))

    # Running SR task
    expression, logs = physo.SR(X, y,
                                X_names = X_names,
                                # Giving name of root variable (for display purposes)
                                y_name  = y_name,
                                # Fixed constants
                                fixed_consts       = [ 1.      ],
                                # Free constants names (for display purposes)
                                free_consts_names = free_consts_names,
                                # Symbolic operations that can be used to make f
                                op_names = ["mul", "add", "sub", "div", "inv", "n2", "n3", "sqrt", "neg", "exp", "log","cos", "tan"],
                                get_run_logger     = run_logger,
                                get_run_visualiser = run_visualiser,
                                # Run config
                                run_config = run_config,
                                # Parallel mode (only available when running from python scripts, not notebooks)
                                parallel_mode = False,
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

