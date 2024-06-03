import numpy as np

import physo.benchmark.ClassDataset.ClassProblem as ClPb
import physo

# Nb of trials per problem
N_TRIALS = 5
# Noise levels
NOISE_LEVELS   = [0.000, 0.001, 0.010, 0.100,]
# Nb of realizations
N_REALIZATIONS = [1, 10]

# Using original variable names (eg. theta, sigma etc.), not x0, x1 etc.
ORIGINAL_VAR_NAMES = True

# Equations to exclude from the benchmark (names)
EXCLUDED_EQS = []

# ----- HYPERPARAMS : CONSTANTS -----
DIMENSIONLESS_RUN = True

dimensionless_units = np.zeros(ClPb.CLASS_UNITS_VECTOR_SIZE)
FIXED_CONSTS        = [1.]
FIXED_CONSTS_UNITS  = [dimensionless_units]
CLASS_FREE_CONSTS_NAMES = ["c1",]
CLASS_FREE_CONSTS_UNITS = [dimensionless_units,]
SPE_FREE_CONSTS_NAMES   = ["k1", "k2"]
SPE_FREE_CONSTS_UNITS   = [dimensionless_units, dimensionless_units]

# ----- HYPERPARAMS : OPERATORS -----
OP_NAMES = ["mul", "add", "sub", "div", "inv", "n2", "sqrt", "neg", "exp", "log", "sin", "cos"]

# ----- HYPERPARAMS : DATA -----
N_SAMPLES = int(1e2)

# Nb of samples for testing results
N_SAMPLES_TEST = int(1e4)

# ----- HYPERPARAMS : CONFIG -----
CONFIG = physo.config.config2b.config2b

# ----- HYPERPARAMS : MAX NUMBER OF EVALUATIONS -----
MAX_N_EVALUATIONS = 200_000 + 1
# Allowed to search in an infinitely large search space, research will be stopped by MAX_N_EVALUATIONS
N_EPOCHS = int(1e99)
#int(MAX_N_EVALUATIONS/CONFIG["learning_config"]["batch_size"])