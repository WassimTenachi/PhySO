Here, we provide instructions to reproduce the results of the benchmarking experiments presented in our papers.
See the [Benchmarks](https://physo.readthedocs.io/en/latest/r_benchmarks.html) section of the documentation accessing the challenges in the benchmarks (eg. if you want to benchmark your own method).


## Feynman benchmarking

The purpose of the Feynman benchmark is to evaluate symbolic regression systems, in particular methods built for scientific discovery.
That is methods able to produce compact, predictive and interpretable expressions from potentially noisy data.

See [[Udrescu 2019]](https://arxiv.org/abs/1905.11481) which introduced the benchmark, [[La Cava 2021]](https://arxiv.org/abs/2107.14351) which formalized it and [[Tenachi 2023]](https://arxiv.org/abs/2303.03192) for the evaluation of `physo` on this benchmark.

### Running a single Feynman problem

Running `physo` on challenge number `i` $\in$ {0, 1, ..., 119} of the Feynman benchmark, with a trial seed `t` $\in \mathbb{N}$, employing a noise level of `n` $\in [0,1]$.
```
python feynman_run.py --equation i --trial t --noise n

```

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

## Class benchmarking

The purpose of the Class benchmark is to evaluate Class symbolic regression systems, that is: methods for automatically finding a single analytical functional form that accurately fits multiple datasets - each governed by its own (possibly) unique set of fitting parameters.

See [[Tenachi 2024]](https://arxiv.org/abs/2312.01816) in which we introduce this first benchmark for Class SR methods and evaluate `physo` on it.

### Running a single class problem

Running `physo` on challenge number `i` $\in$ {0, 1, ..., 7} of the Class benchmark, using a trial seed `t` $\in \mathbb{N}$, employing a noise level of `n` $\in [0,1]$ and exploiting `Nr` $\in \mathbb{N}$ realizations. 
```
python classbench_run.py --equation i --trial t --noise n --n_reals Nr
```

Eg. Running problem 7 with a trial seed 3, at a noise level of 0.001, and 10 realizations.

```
python classbench_run.py --equation 7 --trial 3 --noise 0.001 --n_reals 10
```

### Making an HPC jobfile

Making a jobile to run all Class problems.

```
python classbench_make_run_file.py
```

### Analyzing results

Analyzing a results folder:
```
python classbench_results_analysis.py --path [results folder]
```

### Results

![logo](https://raw.githubusercontent.com/WassimTenachi/PhySO/main/benchmarking/ClassBenchmark/results/class_results.png)

## Class benchmarking (MW bonus)

The purpose of this bonus is to evaluate Class symbolic regression systems on a real-world problem, that is the discovery of the Milky Way dark matter distribution from observed positions and velocities of stellar streams.

![logo](https://raw.githubusercontent.com/WassimTenachi/PhySO/main/docs/assets/demo_streams_plot.png)


See [[Tenachi 2024]](https://arxiv.org/abs/2312.01816) in which we introduce this bonus and evaluate `physo` on it.

### Running a single configuration

Running `physo` on the Milky Way streams problem, using a trial seed `t` $\in \mathbb{N}$, employing a noise level of `n` $\in [0,1]$ and exploiting a fraction of `fr` $\in [0,1]$ realizations. 
```
python MW_streams_run.py --noise n --trial t --frac_real fr
```

### Making an HPC jobfile

Making a jobile to run all Class problems.

```
python MW_streams_make_run_file.py
```

### Analyzing results

Analyzing a results folder:
```
python MW_streams_results_analysis.py --path [results folder]
```

### Results

![logo](https://raw.githubusercontent.com/WassimTenachi/PhySO/main/demos/class_sr/demo_milky_way_streams/results/MW_benchmark.png)



