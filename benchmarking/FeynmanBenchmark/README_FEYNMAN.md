# Feynman benchmarking readme

## Running a single Feynman problem run:

E.g. Running problem 5 with a trial seed 1, at a noize level of 0.0, parallel mode activated and 4 CPUs:
```
python feynman_run.py --equation 5 --trial 1 --noize 0.0 --parallel_mode 1 --ncpus 4
```

## Making an HPC jobfile

Making a jobile to run all Feynman problems at a noize level 0.0:
```
python feynman_make_run_file.py --noize 0.0
```

## Analyzing results

Analyzing a results folder:
```

```
