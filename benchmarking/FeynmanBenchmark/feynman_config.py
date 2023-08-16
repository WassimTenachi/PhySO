import numpy as np

import physo.benchmark.FeynmanDataset.FeynmanProblem as Feyn
import physo

# Nb of trials per problem
N_TRIALS = 5
# Equations that are excluded in SRBench (see section Feynman datasets of https://arxiv.org/abs/2107.14351)
EXCLUDED_IN_SRBENCH_EQS_FILENAMES = ['I.26.2', 'I.30.5', 'II.11.17', 'test_10']
# Using original variable names (eg. theta, sigma etc.), not x0, x1 etc.
ORIGINAL_VAR_NAMES = True

# ----- HYPERPARAMS : CONSTANTS -----
# Since physical constants (G, c etc.) are treated as input variables taking a range of values, two dimensionless
# free constants + a fixed constant (1.) should be enough for most cases
# Even 1 dimensionless free constant + 2 fixed constants (1. and pi) could be enough
dimensionless_units = np.zeros(Feyn.FEYN_UNITS_VECTOR_SIZE)
FIXED_CONSTS        = [1.]
FIXED_CONSTS_UNITS  = [dimensionless_units]
FREE_CONSTS_NAMES   = ["c1", "c2"]
FREE_CONSTS_UNITS   = [dimensionless_units, dimensionless_units]

# ----- HYPERPARAMS : OPERATORS -----
OP_NAMES = ["mul", "add", "sub", "div", "inv", "n2", "sqrt", "neg", "exp", "log", "sin", "cos"]

# ----- HYPERPARAMS : DATA -----
# SRBench (https://arxiv.org/abs/2107.14351) uses 100k data points for Feynman (subsec A.4)
# But they say they downsample to 10k in subsec A.5
# N_SAMPLES = int(1e5)
# Faster with fewer samples (this should not hinder performances too much).
N_SAMPLES = int(1e4)

# Nb of samples for testing results
N_SAMPLES_TEST = int(1e5)

# ----- HYPERPARAMS : CONFIG -----
CONFIG = physo.config.config1.config1

# ----- HYPERPARAMS : MAX NUMBER OF EVALUATIONS -----
# 1M evaluation maximum allowed in SRBench https://arxiv.org/abs/2107.14351
MAX_N_EVALUATIONS = int(1e6) + 1
# Allowed to search in an infinitely large search space, research will be stopped by MAX_N_EVALUATIONS
N_EPOCHS = int(1e99)
#int(MAX_N_EVALUATIONS/CONFIG["learning_config"]["batch_size"])