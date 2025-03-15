import argparse

# Internal imports
import physo.benchmark.FeynmanDataset.FeynmanProblem as Feyn

# Local imports
from benchmarking import utils as bu
import feynman_configs as fconfigs

# Parallel config :
# Parallel mode may cause issues due to the number of samples, non-parallel mode is recommended
# Single core with so many samples will actually use up to 10 cores via pytorch parallelization along sample dim
PARALLEL_MODE_DEFAULT = False
N_CPUS_DEFAULT        = 1

# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------
parser = argparse.ArgumentParser (description     = "Creates a jobfile to run all Feynman problems at specified noise level.",
                                  formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-n", "--noise", default = 0.,
                    help = "Noise level.")
parser.add_argument("-c", "--feynman_config", default = "feynman_config_r10",
                    help = "Feynman config file to use, using feynman_config_r10 by default.")
parser.add_argument("-p", "--parallel_mode", default = PARALLEL_MODE_DEFAULT,
                    help = "Should parallel mode be used.")
parser.add_argument("-ncpus", "--ncpus", default = N_CPUS_DEFAULT,
                    help = "Nb. of CPUs to use")
config = vars(parser.parse_args())

# Noise level
NOISE_LEVEL = float(config["noise"])
# Feynman config
FCONFIG = str(config["feynman_config"])
# Parallel config
PARALLEL_MODE = int(bool(config["parallel_mode"]))
N_CPUS        = int(config["ncpus"])

# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------

# Expected performances on unistra HPC
# With N_SAMPLES = 1e5 on 1 CPU core -> 40min/10k evaluations
# With 1M expressions -> each run .log -> 400 Mo
fconfig = fconfigs.configs[FCONFIG]
N_TRIALS = fconfig.N_TRIALS
ORIGINAL_VAR_NAMES = fconfig.ORIGINAL_VAR_NAMES
EXCLUDED_IN_SRBENCH_EQS_FILENAMES = fconfig.EXCLUDED_IN_SRBENCH_EQS_FILENAMES


# Output jobfile name
PATH_OUT_JOBFILE = "jobfile"

commands = []
# Iterating through Feynman problems
for i_eq in range (Feyn.N_EQS):
    print("\nProblem #%i"%(i_eq))
    # Loading a problem
    pb = Feyn.FeynmanProblem(i_eq, original_var_names=ORIGINAL_VAR_NAMES)
    # Making run file only if it is not in excluded problems
    if pb.eq_filename not in EXCLUDED_IN_SRBENCH_EQS_FILENAMES:
        print(pb)
        # Iterating through trials
        for i_trial in range (N_TRIALS):
            # File name
            command = "python feynman_run.py -i %i -t %i -n %f -c %s -p %i -ncpus %i"%(i_eq, i_trial, NOISE_LEVEL, FCONFIG, PARALLEL_MODE, N_CPUS)
            commands.append(command)
    else:
        print("Problem excluded.")

bu.make_jobfile_from_command_list(PATH_OUT_JOBFILE, commands)

n_jobs = len(commands)
print("\nSuccessfully created a jobile with %i commands : %s"%(n_jobs, PATH_OUT_JOBFILE))





