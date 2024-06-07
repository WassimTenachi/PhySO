#!/usr/bin/env python
# coding: utf-8

# # $\Phi$-SO demo : Milky Way potential from stellar streams

# In[ ]:


#%matplotlib widget
# External packages
import torch
import numpy as np
import pandas as pd
import os
import argparse
# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
# Internal code import
import physo
import physo.learn.monitoring as monitoring


# Parallel config :
# Parallel mode may cause issues due to the number of samples, non-parallel mode is recommended
# Single core with so many samples will actually use up to 10 cores via pytorch parallelization along sample dim
PARALLEL_MODE_DEFAULT = False
N_CPUS_DEFAULT        = 1

# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------
parser = argparse.ArgumentParser(description     = "Runs the Milky Way potential from stellar streams class SR problem.",
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-t", "--trial", default = 0,
                    help = "Trial number (sets seed accordingly).")
parser.add_argument("-n", "--noise", default = 0.,
                    help = "Noise level fraction.")
parser.add_argument("-f", "--frac_real", default = 1.,
                    help = "Fraction of realizations to use (rounded up). Use eg. 1e-6 which will be rounded up to "
                           "use only one realization.")
parser.add_argument("-p", "--parallel_mode", default = PARALLEL_MODE_DEFAULT,
                    help = "Should parallel mode be used.")
parser.add_argument("-ncpus", "--ncpus", default = N_CPUS_DEFAULT,
                    help = "Nb. of CPUs to use")
config = vars(parser.parse_args())

# Trial number
N_TRIAL = int(config["trial"])
# Noise level
NOISE_LEVEL = float(config["noise"])
# Fraction of realizations to use
FRAC_REALIZATION = float(config["frac_real"])
# Parallel config
PARALLEL_MODE = bool(config["parallel_mode"])
N_CPUS        = int(config["ncpus"])

print("Run config :")
print("Trial number : %i"%(N_TRIAL))
print("Noise level  : %f"%(NOISE_LEVEL))
print("Fraction of realizations to use : %f"%(FRAC_REALIZATION))
print("Parallel mode : %s"%(PARALLEL_MODE))
print("Nb. of CPUs   : %i"%(N_CPUS))

# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------


# ## Run params

# In[ ]:

# In[ ]:


# Paths
RUN_NAME       = "StreamsSR_%i_%i_%f_%f"%(0, N_TRIAL, NOISE_LEVEL, FRAC_REALIZATION)
PATH_DATA      = "%s_data.csv"%(RUN_NAME) # PATH WHERE TO SAVE RUN DATA BACKUP
PATH_DATA_PLOT = "%s_data.png"%(RUN_NAME) # PATH WHERE TO SAVE RUN DATA BACKUP PLOT

# Defining source data abs path before changing directory
PATH_SOURCE_DATA = os.path.join(os.path.abspath(''), 'streams.csv',)

# Making a directory for this run and running in it
if not os.path.exists(RUN_NAME):
    os.makedirs(RUN_NAME)
os.chdir(os.path.join(os.path.abspath(''), RUN_NAME,))


# ## Fixing seed

# In[ ]:


# Seed
seed = 10+N_TRIAL
np.random.seed(seed)
torch.manual_seed(seed)


# ## Dataset

# ### Loading data

# In[ ]:


df = pd.read_csv(PATH_SOURCE_DATA)
df["r"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
df["v"] = np.sqrt(df["vx"]**2 + df["vy"]**2 + df["vz"]**2)
    
stream_ids = np.unique(df["sID"].to_numpy())              # (n_streams,)
stream_dfs = [df[df["sID"] == sID] for sID in stream_ids] # (n_streams,)
n_streams = len(stream_dfs)


# #### Data inspection

# In[ ]:

mpl.rcParams.update(mpl.rcParamsDefault) # Avoids latex missing on HPC
fig, ax = plt.subplots(1,1, figsize=(10,8))

curr_ax = ax

cmap = plt.cm.get_cmap('viridis', n_streams)
for i, df in enumerate(stream_dfs): 
    E_kin = 0.5*df["v"]**2
    curr_ax.scatter(df["r"], E_kin, marker='.', s=1., c=cmap(i),)
    

y_ticks = np.array([0.0, 0.2, 0.4, 0.6, 0.8,])*1e5
curr_ax.set_yticks(ticks=y_ticks, labels=y_ticks/1e5)
curr_ax.set_xticks(ticks=[0,25,50,75,100,125])

curr_ax.set_xlim(0., 130.)
curr_ax.set_ylim(0., 0.9*1e5)

#curr_ax.set_xlabel(r"${\rm r}$ [${\rm kpc}$]")
#curr_ax.set_ylabel(r"${\rm E}_{\rm kin}$ [$\times 10^5\ {\rm km}^{2}.{\rm s}^{-2}$]")
fig.suptitle("n_realizations : %i"%(n_streams))
fig.savefig("streams.png")

#plt.show()


# ### Data formatting

# #### Normalizing

# In[ ]:


# Dataset
multi_X = []
multi_y = []
for i, df in enumerate(stream_dfs):  
    r = (df["r"]/20.0 ).to_numpy()
    v = (df["v"]/200.0).to_numpy()
    E_kin = 0.5 * v**2
    X = np.stack((r, ), axis=0)
    y = E_kin
    multi_X.append(X)
    multi_y.append(y)


# #### Sub-sampling realizations

# In[ ]:


n_realizations = int(np.ceil(len(multi_X)*FRAC_REALIZATION))
print("n_realizations", n_realizations)
assert n_realizations > 0, "No realization to use, please check FRAC_REALIZATION value."

# Using only a fraction of the realizations available (random selection)
idxs = np.random.choice(len(multi_X), n_realizations, replace=False)
multi_X = [multi_X[i] for i in idxs]
multi_y = [multi_y[i] for i in idxs]


# #### Adding noise

# In[ ]:


for i in range(len(multi_X)):
    y = multi_y[i]
    y_rms      = ((y ** 2).mean()) ** 0.5
    epsilon    = NOISE_LEVEL * np.random.normal(0, y_rms, len(y))
    multi_y[i] = y + epsilon


# #### Data inspection

# In[ ]:

mpl.rcParams.update(mpl.rcParamsDefault) # Avoids latex missing on HPC
fig, ax = plt.subplots(1,1, figsize=(10,8))

curr_ax = ax

cmap = plt.cm.get_cmap('viridis', len(multi_X))
for i in range(len(multi_X)): 
    curr_ax.scatter(multi_X[i][0], multi_y[i], marker='.', s=1., c=cmap(i),)

curr_ax.set_xlim(0., 130./20.0)
curr_ax.set_ylim(0., 0.9*1e5/(200.0**2))

#curr_ax.set_xlabel(r"${\rm r}$ [$\times 20\ {\rm kpc}$]")
#curr_ax.set_ylabel(r"${\rm E}_{\rm kin}$ [$\times 4.10^9\ {\rm km}^{2}.{\rm s}^{-2}$]")
fig.suptitle("n_realizations : %i"%(len(multi_X)))
fig.savefig(PATH_DATA_PLOT)

#plt.show()


# #### Backuping run data

# In[ ]:


# Save run data (x0, x1, ..., y)
col_names = ["i_real",] + ["x%i"%(i) for i in range(multi_X[0].shape[0])] + ["y"]
backup_df = pd.DataFrame(columns = col_names)
for i in range(len(multi_X)):
    X = multi_X[i]
    y = multi_y[i]
    df = pd.DataFrame(data = np.concatenate(([np.full_like(y,i).astype(int)], X, [y,]), axis=0).T, columns = col_names)
    backup_df = pd.concat([backup_df, df], axis=0)

backup_df.to_csv(PATH_DATA, sep=";", index=False)


# ## Logging config

# In[ ]:


save_path_training_curves = 'run_curves.png'
save_path_log = 'run.log'

run_logger     = lambda: monitoring.RunLogger(save_path = save_path_log,
                                              do_save   = True)

run_visualiser = lambda: monitoring.RunVisualiser(epoch_refresh_rate = 1,
                                                  save_path = save_path_training_curves,
                                                  do_show   = False,
                                                  do_prints = True,
                                                  do_save   = True, )


# ## Run config

# In[ ]:


run_config = physo.config.config1b.config1b

PARALLEL_MODE     = PARALLEL_MODE
N_CPUS            = N_CPUS
MAX_N_EVALUATIONS = int(2.5*1e5) + 1
# Allowed to search in an infinitely large search space, research will be stopped by MAX_N_EVALUATIONS
N_EPOCHS          = int(1e99) 


# Uncomment this to cheat and enforce the correct solution

# In[ ]:


# target_prog_str = ["add", "E_t", "mul", "A", "mul", "div", "R", "r", "log", "add", "1.0", "div", "r", "R"]
# cheater_prior_config = ('SymbolicPrior', {'expression': target_prog_str})
# run_config["priors_config"].append(cheater_prior_config)


# ## Run

# In[ ]:


# Running SR task
expression, logs = physo.ClassSR(multi_X, multi_y,
                        # Giving names of variables (for display purposes)
                        X_names = [ "r"        ],
                        # Giving units of input variables
                        X_units = [ [1, 0, 0] ],
                        # Giving name of root variable (for display purposes)
                        y_name  = "y",
                        # Giving units of the root variable
                        y_units = [2,-2, 0],
                        # Fixed constants
                        fixed_consts       = [ 1.      ],
                        fixed_consts_units = [ [0, 0, 0] ],
                        # Whole class free constants
                        class_free_consts_names = [ "R"       , "A"        , "c"     ],
                        class_free_consts_units = [ [1, 0, 0] , [2,-2, 0]  , [0,0,0] ],
                        # Dataset specific free constants
                        spe_free_consts_names = [ "E_t"     , ],
                        spe_free_consts_units = [ [2,-2, 0] , ],
                        # Run config
                        run_config = run_config,
                        op_names = ["add", "sub", "mul", "div", "inv", "n2", "sqrt", "neg", "log", "exp"],
                        get_run_logger     = run_logger,
                        get_run_visualiser = run_visualiser,
                        parallel_mode     = PARALLEL_MODE,
                        n_cpus            = N_CPUS,
                        max_n_evaluations = MAX_N_EVALUATIONS,
                        epochs            = N_EPOCHS,
)


# ## Results

# In[ ]:


# Inspecting pareto front expressions
pareto_front_complexities, pareto_front_expressions, pareto_front_r, pareto_front_rmse = logs.get_pareto_front()

expression.get_infix_sympy()

