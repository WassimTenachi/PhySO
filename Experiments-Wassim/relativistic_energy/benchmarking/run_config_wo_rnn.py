import physo
import torch
import numpy as np

from run_config import *

# Using random number generator instead of RNN
cell_config.update({"is_lobotomized": True})