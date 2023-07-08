import numpy as np
import os
import shutil
import physo.benchmark.FeynmanDataset.FeynmanProblem as Feyn

# Expected performances on unistra HPC
# With N_SAMPLES = 1e5 on 1 CPU core -> 40min/10k evaluations
# With 1M expressions -> each run .log -> 400 Mo
# todo: Checkout produce file with noize

# Path to .py template
PATH_TO_TEMPLATE_PY = os.path.join(os.path.dirname(__file__), 'FR_x_x_template.template')
# Name of output folder
PATH_OUT_FOLDER = "run_files"

# Reading template
f = open(PATH_TO_TEMPLATE_PY,'r')
template_str = f.read()
f.close()

# Making a directory for this run and run in it
if not os.path.exists(PATH_OUT_FOLDER):
    os.makedirs(PATH_OUT_FOLDER)
os.chdir(PATH_OUT_FOLDER)

# Copying .py this script to the directory
shutil.copy2(src=__file__, dst=os.path.join(os.path.dirname(__file__), PATH_OUT_FOLDER))

# Nb of trials per problems
N_TRIALS = 5

# Equations that are excluded in SRBench (see section Feynman datasets of https://arxiv.org/abs/2107.14351)
EXCLUDED_IN_SRBENCH_EQS_FILENAMES = ['I.26.2', 'I.30.5', 'II.11.17', 'test_10']

filenames = []
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
            # Formatting iterators
            i_eq_str    = str(i_eq)    .zfill(3)
            i_trial_str = str(i_trial) .zfill(2)
            # File name
            filename = "FR_%s_%s.py"%(i_eq_str, i_trial_str)
            filenames.append(filename)
            # File content
            file_content = template_str
            file_content = file_content.replace("[I_TRIAL]", str(i_trial) )
            file_content = file_content.replace("[I_FEYN]" , str(i_eq)    )
            # Making file
            print("->", filename)
            f = open(filename, "w")
            f.write(file_content)
            f.close()

# Creating a jobfile containing all commands to run
jobfile_name    = "jobfile"
jobfile_content = ''.join('python %s\n'%fname for fname in filenames)
print("\n->", jobfile_name)
f = open(jobfile_name, "w")
f.write(jobfile_content)
f.close()

n_run_files = len(filenames)
print("\n%i run files successfully created in %s"%(n_run_files, os.getcwd()))





