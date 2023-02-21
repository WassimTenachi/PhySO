import physo
import torch
import numpy as np

from run_config import *

# Using random number generator instead of RNN
cell_config.update({"is_lobotomized": True})

# Not using units constraints
reward_config.update({"zero_out_unphysical": False})
priors_config = [prior for prior in priors_config if prior[0] != "PhysicalUnitsPrior"]

# Not observing units
run_config["learning_config"].update({"observe_units" : False})
