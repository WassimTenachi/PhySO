from benchmarking import utils as bu

# Expected performances on unistra HPC
# With N_SAMPLES = 1e3
# With 1M expressions -> each run .log -> 400 Mo

# Output jobfile name
PATH_OUT_JOBFILE = "jobfile"

# BENCHMARKING PARAMETERS
N_TRIALS          = 12
NOISE_LEVELS      = [0., 0.001, 0.01, 0.1]
eps = 1e-6 # -> Will result in only one realization being used
FRAC_REALIZATIONS = [1., 0.5, 0.25, eps]

commands = []

for noise in NOISE_LEVELS:
    for frac_real in FRAC_REALIZATIONS:
        for i_trial in range(N_TRIALS):
            command = "python MW_streams_run.py --trial %i --noise %f --frac_real %f"%(i_trial, noise, frac_real)
            commands.append(command)

bu.make_jobfile_from_command_list(PATH_OUT_JOBFILE, commands)

n_jobs = len(commands)
print("\nSuccessfully created a jobile with %i commands : %s"%(n_jobs, PATH_OUT_JOBFILE))





