import numpy as np
import os
import shutil
import argparse

import physo.benchmark.FeynmanDataset.FeynmanProblem as Feyn

# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------
parser = argparse.ArgumentParser (description     = "Creates a jobfile to run all Feynman problems.",
                                  formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-n", "--noize", default = 0.,
                    help = "Noize level.")
config = vars(parser.parse_args())

NOIZE_LEVEL = float(config["noize"])
# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------

# Expected performances on unistra HPC
# With N_SAMPLES = 1e5 on 1 CPU core -> 40min/10k evaluations
# With 1M expressions -> each run .log -> 400 Mo

# Nb of trials per problem
N_TRIALS = 5

# Output jobfile name
PATH_OUT_JOBFILE = "jobfile"

# Equations that are excluded in SRBench (see section Feynman datasets of https://arxiv.org/abs/2107.14351)
EXCLUDED_IN_SRBENCH_EQS_FILENAMES = ['I.26.2', 'I.30.5', 'II.11.17', 'test_10']

commands = []
# Iterating through Feynman problems
for i_eq in range (Feyn.N_EQS):
    print("\nProblem #%i"%(i_eq))
    # Loading a problem
    pb = Feyn.FeynmanProblem(i_eq)
    # Making run file only if it is not in excluded problems
    if pb.eq_filename not in EXCLUDED_IN_SRBENCH_EQS_FILENAMES:
        print(pb)
        # Iterating through trials
        for i_trial in range (N_TRIALS):
            # File name
            command = "python feynman_run.py -i %i -t %i -n %f"%(i_eq, i_trial, NOIZE_LEVEL)
            commands.append(command)

# Creating a jobfile containing all commands to run
jobfile_name    = PATH_OUT_JOBFILE
jobfile_content = ''.join('%s\n'%com for com in commands)
f = open(jobfile_name, "w")
f.write(jobfile_content)
f.close()

n_jobs = len(commands)
print("\nSuccessfully created a jobile with %i commands : %s"%(n_jobs, PATH_OUT_JOBFILE))





