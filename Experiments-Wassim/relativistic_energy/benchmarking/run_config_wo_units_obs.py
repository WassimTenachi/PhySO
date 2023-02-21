import physo
import torch
import numpy as np

from run_config import *

# Not observing units
run_config["learning_config"].update({"observe_units" : False})
