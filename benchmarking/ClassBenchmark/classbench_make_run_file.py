import argparse

# Internal imports
import physo.benchmark.ClassDataset.ClassProblem as ClPb

# Local imports
from benchmarking import utils as bu
import classbench_config as fconfig

# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------
parser = argparse.ArgumentParser (description     = "Creates a jobfile to run all Class benchmark problems.",
                                  formatter_class = argparse.ArgumentDefaultsHelpFormatter)
# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------


ORIGINAL_VAR_NAMES = fconfig.ORIGINAL_VAR_NAMES
EXCLUDED_EQS = fconfig.EXCLUDED_EQS

N_TRIALS       = fconfig.N_TRIALS
NOISE_LEVELS   = fconfig.NOISE_LEVELS
N_REALIZATIONS = fconfig.N_REALIZATIONS

# Output jobfile name
PATH_OUT_JOBFILE = "jobfile"

commands = []
# Iterating through Class problems
for i_eq in range (ClPb.N_EQS):
    print("\nProblem #%i"%(i_eq))
    # Loading a problem
    pb = ClPb.ClassProblem(i_eq, original_var_names=ORIGINAL_VAR_NAMES)
    # Making run file only if it is not in excluded problems
    if pb.eq_name not in EXCLUDED_EQS:
        print(pb)
        # Iterating through trials
        for i_trial in range (N_TRIALS):
            for noise_lvl in NOISE_LEVELS:
                for n_reals in N_REALIZATIONS:
                    # File name
                    command = "python classbench_run.py -i %i -t %i -n %f -r %i"%(i_eq, i_trial, noise_lvl, n_reals)
                    commands.append(command)
    else:
        print("Problem excluded.")

bu.make_jobfile_from_command_list(PATH_OUT_JOBFILE, commands)

n_jobs = len(commands)
print("\nSuccessfully created a jobile with %i commands : %s"%(n_jobs, PATH_OUT_JOBFILE))





