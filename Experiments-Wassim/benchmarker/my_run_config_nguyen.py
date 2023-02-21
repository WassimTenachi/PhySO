import physo
import torch

# Reward
reward_config = {
                 "reward_function"     : physo.physym.reward.SquashedNRMSE, 
                 "zero_out_unphysical" : False,                             # PHYSICALITY
                 "zero_out_duplicates" : False,
                 "keep_lowest_complexity_duplicate" : False,
                }

# Learning config
MAX_TRIAL_EXPRESSIONS = 2*1e6
BATCH_SIZE = 500
MAX_LENGTH = 30
GET_OPTIMIZER = lambda model : torch.optim.Adam(
                                    model.parameters(),                
                                    lr=0.0025, #0.001, #0.0050, #0.0005, #1,  #lr=0.0025
                                                )
learning_config = {
    # Batch related
    'batch_size'      : BATCH_SIZE,
    'max_time_step'   : MAX_LENGTH,
    'n_epochs'        : int(MAX_TRIAL_EXPRESSIONS/BATCH_SIZE),
    # Loss related
    'gamma_decay'     : 0.7,
    'entropy_weight'  : 0.03,
    # Reward related
    'risk_factor'     : 0.02,
    'rewards_computer' : physo.physym.reward.make_RewardsComputer (**reward_config),
    # Optimizer
    'get_optimizer'   : GET_OPTIMIZER,
    'observe_units'   : True,
}




# Free constant optimizer config
free_const_opti_args = {
            'loss'   : "MSE",
            'method' : 'LBFGS',
            'method_args': {
                        'n_steps' : 30,
                        'tol'     : 1e-3,
                        'lbfgs_func_args' : {
                            'max_iter'       : 2,
                            'line_search_fn' : "strong_wolfe",
                                             },
                            },
        }
# Priors
priors_config  = [
                #("UniformArityPrior", None),
                # LENGTH RELATED
                ("HardLengthPrior"  , {"min_length": 1, "max_length": MAX_LENGTH, }),
                ("SoftLengthPrior"  , {"length_loc": 10, "scale": 5, }),
                # RELATIONSHIPS RELATED
                ("NoUselessInversePrior"  , None),
                #("PhysicalUnitsPrior", {"prob_eps": np.finfo(np.float32).eps}),
                ("NestedTrigonometryPrior", {"max_nesting" : 1}),
                #("NestedFunctions", {"functions":["exp",], "max_nesting" : 1}),
                #("NestedFunctions", {"functions":["log",], "max_nesting" : 1}),
                 ]

# Cell
cell_config = {
    "hidden_size" : 128,
    "n_layers"    : 1,
}