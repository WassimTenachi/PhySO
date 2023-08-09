import argparse

# Internal imports
import physo.benchmark.FeynmanDataset.FeynmanProblem as Feyn

# Local imports
from benchmarking import utils as bu
import feynman_config as fconfig

# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------
parser = argparse.ArgumentParser (description     = "Creates a jobfile to run all Feynman problems at specified noise level.",
                                  formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-n", "--noise", default = 0.,
                    help = "Noise level.")
config = vars(parser.parse_args())

NOISE_LEVEL = float(config["noise"])
# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------

# Expected performances on unistra HPC
# With N_SAMPLES = 1e5 on 1 CPU core -> 40min/10k evaluations
# With 1M expressions -> each run .log -> 400 Mo

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
            command = "python feynman_run.py -i %i -t %i -n %f"%(i_eq, i_trial, NOISE_LEVEL)
            commands.append(command)
    else:
        print("Problem excluded.")

bu.make_jobfile_from_command_list(PATH_OUT_JOBFILE, commands)

n_jobs = len(commands)
print("\nSuccessfully created a jobile with %i commands : %s"%(n_jobs, PATH_OUT_JOBFILE))





