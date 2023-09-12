## Feynman benchmarking

### Running a single Feynman problem

E.g. Running problem 5 with a trial seed 1, at a noise level of 0.0, parallel mode activated and 4 CPUs:
```
python feynman_run.py --equation 5 --trial 1 --noise 0.0 --parallel_mode 1 --ncpus 4
```

### Making an HPC jobfile

Making a jobile to run all Feynman problems at a noise level 0.0:
```
python feynman_make_run_file.py --noise 0.0
```

### Analyzing results

Analyzing a results folder:
```
python feynman_results_analysis.py --path [results folder]
```

### Results

![logo](https://raw.githubusercontent.com/WassimTenachi/PhySO/main/benchmarking/FeynmanBenchmark/results/feynman_results.png)