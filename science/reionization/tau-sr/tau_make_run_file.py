# Local imports
from benchmarking import utils as bu

# Output jobfile name
PATH_OUT_JOBFILE = "jobfile"

# GRID SEARCH RANGES
# Seed
SEED = [0, 1, 2, 3]
# Number of free parameters
N_FREE_PARAMS = [5, 10, 20]
# Soft length prior params
LENGTH_LOC_ARG   = [35, 75, 125]
LENGTH_SCALE_ARG = [5, 12, 10]
# Whether to only use essential x variables
X_ESSENTIAL_ONLY = [False, True]

# GENERATING COMMANDS
commands = []
for xe in X_ESSENTIAL_ONLY:
    for fp in N_FREE_PARAMS:
        for ll, ls in zip (LENGTH_LOC_ARG, LENGTH_SCALE_ARG):
            for s in SEED:
                command = "python tau_sr.py -xe %i -fp %d -ll %d -ls %d -s %d"%(int(xe), fp, ll, ls, s)
                commands.append(command)

bu.make_jobfile_from_command_list(PATH_OUT_JOBFILE, commands)

n_jobs = len(commands)
print("\nSuccessfully created a jobile with %i commands : %s"%(n_jobs, PATH_OUT_JOBFILE))



