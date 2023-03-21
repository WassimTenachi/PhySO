
# $\Phi$-SO : Physical Symbolic Optimization

The physical symbolic regression ( $\Phi$-SO ) package `physo` is a symbolic regression package that fully leverages physical units constraints. For more details see: [[Tenachi et al 2023]](https://arxiv.org/abs/2303.03192).

https://user-images.githubusercontent.com/63928316/225642347-a07127da-a84e-4af3-96f4-4c7fef5a673b.mp4


# Installation

### Virtual environment

The package has been tested on:
- Linux
- OSX (ARM & Intel)

Running `physo` on Windows is not recommended.

To install the package it is recommended to first create a conda virtual environment:
```
conda create -n PhySO python=3.8
```
And activate it:
```
conda activate PhySO
```
### Dependencies
From the repository root:

Installing essential dependencies :
```
conda install --file requirements.txt
```
Installing optional dependencies (for advanced debugging in tree representation) :
```
conda install --file requirements_display1.txt
```
```
pip install -r requirements_display2.txt
```

#####  Side note regarding CUDA acceleration:

$\Phi$-SO supports CUDA but it should be noted that since the bottleneck of the code is free constant optimization, using CUDA (even on a very high-end GPU) does not improve performances over a CPU and can actually hinder performances.

### Installing $\Phi$-SO

Installing `physo` (from the repository root):
```
pip install -e .
```

### Testing install

#####  Import test:
```
python3
>>> import physo
```
This should result in `physo` being successfully imported.

#####  Unit tests:

From the repository root:
```
python -m unittest discover -p "*UnitTest.py"
```
This should result in all tests being successfully passed (except for program_display_UnitTest tests if optional dependencies were not installed). 

# Getting started

### Symbolic regression with default hyperparameters
Symbolic regression (SR) consists in the inference of a free-form symbolic analytical function $f: \mathbb{R}^n \longrightarrow \mathbb{R}$ that fits $y = f(x_0,..., x_n)$ given $(x_0,..., x_n, y)$ data.

Given a dataset $(x_0,..., x_n, y)$:
```
z = np.random.uniform(-10, 10, 50)
v = np.random.uniform(-10, 10, 50)
X = np.stack((z, v), axis=0)
y = 1.234*9.807*z + 1.234*v**2
```
Given the units input variables $(x_0,..., x_n)$ (here $(z, v)$ ), the root variable $y$ (here $E$) as well as free and fixed constants, you can run an SR task to recover $f$ via:
```
expression, logs = physo.SR(X, y,
                            X_units = [ [1, 0, 0] , [1, -1, 0] ],
                            y_units = [2, -2, 1],
                            fixed_consts       = [ 1.      ],
                            fixed_consts_units = [ [0,0,0] ],
                            free_consts_units  = [ [0, 0, 1] , [1, -2, 0] ],                          
)
```

It should be noted that SR capabilities of `physo` are heavily dependent on hyperparameters, it is therefore recommended to tune hyperparameters to your own specific problem for doing science.
Summary of available currently configurations:

<div align="center">

| Configuration |                           Notes                           |
|:-------------:|:---------------------------------------------------------:|
|    config0    | Light config for demo purposes.                           |
|    config1    | Tuned on a few physical cases.                            |
|    config2    | [work in progress] Good starting point for doing science. |

</div>

By default, `config0` is used, however it is recommended to use the latest configuration currently available (`config1`) as a starting point for doing science by specifying it:

```
expression, logs = physo.SR(X, y,
                            X_units = [ [1, 0, 0] , [1, -1, 0] ],
                            y_units = [2, -2, 1],
                            fixed_consts       = [ 1.      ],
                            fixed_consts_units = [ [0,0,0] ],
                            free_consts_units  = [ [0, 0, 1] , [1, -2, 0] ],      
                            run_config = physo.config.config1.config1                    
)
```

You can also specify the choosable symbolic operations for the construction of $f$ and give the names of variables for display purposes by filling in optional arguments:
```
expression, logs = physo.SR(X, y,
                            X_names = [ "z"       , "v"        ],
                            X_units = [ [1, 0, 0] , [1, -1, 0] ],
                            y_name  = "E",
                            y_units = [2, -2, 1],
                            fixed_consts       = [ 1.      ],
                            fixed_consts_units = [ [0,0,0] ],
                            free_consts_names = [ "m"       , "g"        ],
                            free_consts_units = [ [0, 0, 1] , [1, -2, 0] ],
                            op_names = ["mul", "add", "sub", "div", "inv", "n2", "sqrt", "neg", "exp", "log", "sin", "cos"]
)
```

`physo.SR` saves monitoring curves, the pareto front (complexity vs accuracy optimums) and the logs.
It also returns the best fitting expression found during the search which can be inspected in regular infix notation (eg. in ascii or latex) via:

```
>>> print(expression.get_infix_pretty(do_simplify=True))
  ⎛       2⎞
m⋅⎝g⋅z + v ⎠
>>> print(expression.get_infix_latex(do_simplify=True))
'm \\left(g z + v^{2}\\right)'
```
Free constants can be inspected via:
```
>>> print(expression.free_const_values.cpu().detach().numpy())
array([9.80699996, 1.234     ])
```
`physo.SR` also returns the log of the run from which one can inspect Pareto front expressions:
```

for i, prog in enumerate(pareto_front_expressions):
    # Showing expression
    print(prog.get_infix_pretty(do_simplify=True))
    # Showing free constant
    free_consts = prog.free_const_values.detach().cpu().numpy()
    for j in range (len(free_consts)):
        print("%s = %f"%(prog.library.free_const_names[j], free_consts[j]))
    # Showing RMSE
    print("RMSE = {:e}".format(pareto_front_rmse[i]))
    print("-------------")
```
Returning:
```
   2
m⋅v 
g = 1.000000
m = 1.486251
RMSE = 6.510109e+01
-------------
g⋅m⋅z
g = 3.741130
m = 3.741130
RMSE = 5.696636e+01
-------------
  ⎛       2⎞
m⋅⎝g⋅z + v ⎠
g = 9.807000
m = 1.234000
RMSE = 1.675142e-07
-------------
```

This demo can be found in `demo/demo_quick_sr.ipynb`.

### Symbolic regression
[Coming soon] 
In the meantime you can have a look at our demo folder ! :)

### Custom symbolic optimization task
[Coming soon]
### Using custom functions
[Coming soon]

# About performances

The main performance bottleneck of `physo` is free constant optimization, therefore, performances are almost linearly dependent on the number of free constant optimization steps and on the number of trial expressions per epoch (ie. the batch size).

In addition, it should be noted that generating monitoring plots takes ~3s, therefore we suggest making monitoring plots every >10 epochs for low time / epoch cases. 

Summary of expected performances with `physo`:

| Time / epoch | Batch size | # free const | free const <br>opti steps | Example                             | Device                                      |
|--------------|------------|--------------|---------------------------|-------------------------------------|---------------------------------------------|
| ~20s         | 10k        | 2            | 15                        | eg: demo_damped_harmonic_oscillator | CPU: Mac M1 <br>RAM: 16 Go                  |
| ~30s         | 10k        | 2            | 15                        | eg: demo_damped_harmonic_oscillator | CPU: Intel W-2155 10c/20t <br>RAM: 128 Go   |
| ~250s        | 10k        | 2            | 15                        | eg: demo_damped_harmonic_oscillator | GPU: Nvidia GV100 <br>VRAM : 32 Go          |
| ~3s          | 1k         | 2            | 15                        | eg: demo_mechanical_energy          | CPU: Mac M1 <br>RAM: 16 Go                  |
| ~3s          | 1k         | 2            | 15                        | eg: demo_mechanical_energy          | CPU: Intel W-2155 10c/20t <br>RAM: 128 Go   |
| ~4s          | 1k         | 2            | 15                        | eg: demo_mechanical_energy          | GPU: Nvidia GV100 <br>VRAM : 32 Go          |

Please note that using a CPU typically results in higher performances than when using a GPU.

# Uninstalling
Uninstalling the package.
```
conda deactivate
conda env remove -n PhySO
```

# Citing this work

```
@ARTICLE{2023arXiv230303192T,
       author = {{Tenachi}, Wassim and {Ibata}, Rodrigo and {Diakogiannis}, Foivos I.},
        title = "{Deep symbolic regression for physics guided by units constraints: toward the automated discovery of physical laws}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Computer Science - Machine Learning, Physics - Computational Physics},
         year = 2023,
        month = mar,
          eid = {arXiv:2303.03192},
        pages = {arXiv:2303.03192},
          doi = {10.48550/arXiv.2303.03192},
archivePrefix = {arXiv},
       eprint = {2303.03192},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230303192T},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
