import physo
import torch
import numpy as np

"""
SR performances are highly dependent on hyperparameters.
For doing science, it is recommended to tune hyperparameters to the specific problem to get best performances.
"""

"""
Configuration config1 was tuned on a very few astrophysical cases for the paper: arXiv:2303.03192.
Better configurations will be added in the future.
"""

# Maximum length of expressions
MAX_LENGTH = 35

# ---------- REWARD CONFIG ----------
reward_config = {
                 "reward_function"     : physo.physym.reward.SquashedNRMSE,
                 "zero_out_unphysical" : True,
                 "zero_out_duplicates" : False,
                 "keep_lowest_complexity_duplicate" : False,
                 # "parallel_mode" : True,
                 # "n_cpus"        : None,
                }

# ---------- LEARNING CONFIG ----------
# Number of trial expressions to try at each epoch
BATCH_SIZE = int(1e4)
# Function returning the torch optimizer given a model
GET_OPTIMIZER = lambda model : torch.optim.Adam(
                                    model.parameters(),
                                    lr=0.0025,
                                                )
# Learning config
learning_config = {
    # Batch related
    'batch_size'       : BATCH_SIZE,
    'max_time_step'    : MAX_LENGTH,
    'n_epochs'         : int(1e9),
    # Loss related
    'gamma_decay'      : 0.7,
    'entropy_weight'   : 0.005,
    # Reward related
    'risk_factor'      : 0.05,
    'rewards_computer' : physo.physym.reward.make_RewardsComputer (**reward_config),
    # Optimizer
    'get_optimizer'    : GET_OPTIMIZER,
    'observe_units'    : True,
}

# ---------- FREE CONSTANT OPTIMIZATION CONFIG ----------
free_const_opti_args = {
            'loss'   : "MSE",
            'method' : 'LBFGS',
            'method_args': {
                        'n_steps' : 20,
                        'tol'     : 1e-8,
                        'lbfgs_func_args' : {
                            'max_iter'       : 4,
                            'line_search_fn' : "strong_wolfe",
                                             },
                            },
        }

# ---------- PRIORS CONFIG ----------
priors_config  = [
                #("UniformArityPrior", None),
                # LENGTH RELATED
                ("HardLengthPrior"  , {"min_length": 4, "max_length": MAX_LENGTH, }),
                ("SoftLengthPrior"  , {"length_loc": 8, "scale": 5, }),
                # RELATIONSHIPS RELATED
                ("NoUselessInversePrior"  , None),
                ("PhysicalUnitsPrior", {"prob_eps": np.finfo(np.float32).eps}), # PHYSICALITY
                ("NestedFunctions", {"functions":["exp",], "max_nesting" : 1}),
                ("NestedFunctions", {"functions":["log",], "max_nesting" : 1}),
                ("NestedTrigonometryPrior", {"max_nesting" : 1}),
                #("OccurrencesPrior", {"targets" : ["1",], "max" : [3,] }),
                 ]

# ---------- RNN CELL CONFIG ----------
cell_config = {
    "hidden_size" : 128,
    "n_layers"    : 1,
    "is_lobotomized" : False,
}

# ---------- RUN CONFIG ----------
config1 = {
    "learning_config"      : learning_config,
    "reward_config"        : reward_config,
    "free_const_opti_args" : free_const_opti_args,
    "priors_config"        : priors_config,
    "cell_config"          : cell_config,
}
