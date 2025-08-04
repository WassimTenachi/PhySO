import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

# Internal code import
import physo
import physo.learn.monitoring as monitoring

if __name__ == '__main__':

    # Seed
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Dataset
    PATH_DATA = "data/10k_curves_physical.csv"
    data = pd.read_csv(PATH_DATA)
    # Subsample data to 50k
    data = data.sample(n=50000, random_state=42).reset_index(drop=True)
    # Randomize data
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    shorter_names = {
        "z": "z",
        "F_STAR10": "f_*",
        "F_ESC10": "f_esc",
        "ALPHA_STAR": "α_*",
        "ALPHA_ESC": "α_esc",
        "M_TURN": "M_t",
        "L_X": "L_X",
        "t_STAR": "τ_*",
        "xHI": "x_HI",
        "Ts": "T_s",
        "Tb": "T_b"
    }
    var_names = ["z", "F_STAR10", "F_ESC10", "ALPHA_STAR", "ALPHA_ESC", "M_TURN", "L_X", "t_STAR"]
    X = data[var_names].values.T
    y = data["Ts"].values

    X_names = [shorter_names[name] for name in var_names]
    y_name = shorter_names["Ts"]
    free_consts_names = ["c%i" % i for i in range(10)]

    # Dataset plot
    n_dim = X.shape[0]
    fig, ax = plt.subplots(n_dim, 1, figsize=(10,5))
    for i in range (n_dim):
        curr_ax = ax if n_dim==1 else ax[i]
        curr_ax.plot(X[i], y, 'k.',)
        curr_ax.set_xlabel("X[%i]"%(i))
        curr_ax.set_ylabel("y")
    plt.show()

    # Logging config

    save_path_training_curves = 'demo_curves.png'
    save_path_log             = 'demo.log'

    run_logger     = lambda : monitoring.RunLogger(save_path = save_path_log,
                                                    do_save = True)

    run_visualiser = lambda : monitoring.RunVisualiser (epoch_refresh_rate = 1,
                                               save_path = save_path_training_curves,
                                               do_show   = False,
                                               do_prints = True,
                                               do_save   = True, )


    # Run
    #physo.physym.batch_execute.SHOW_PROGRESS_BAR = True
    run_config = physo.config.config1.config1
    run_config["learning_config"]["batch_size"] = 20

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
                                op_names = ["mul", "add", "sub", "div", "inv", "n2", "sqrt", "neg", "exp", "log",],
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

