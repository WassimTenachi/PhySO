# $\Phi$-SO : Physical Symbolic Optimization
![logo](https://raw.githubusercontent.com/WassimTenachi/PhySO/main/docs/assets/logo_dark.png)
Physical symbolic optimization ( $\Phi$-SO ) - A symbolic optimization package built for physics.

[![GitHub Repo stars](https://img.shields.io/github/stars/WassimTenachi/PhySO?style=social)](https://github.com/WassimTenachi/PhySO)
[![Documentation Status](https://readthedocs.org/projects/physo/badge/?version=latest)](https://physo.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/WassimTenachi/PhySO/badge.svg?branch=main)](https://coveralls.io/github/WassimTenachi/PhySO?branch=main)
[![Twitter Follow](https://img.shields.io/twitter/follow/WassimTenachi?style=social)](https://twitter.com/WassimTenachi)
[![Paper](https://img.shields.io/badge/arXiv-2305.01582-b31b1b)](https://arxiv.org/abs/2303.03192)

Source code: [WassimTenachi/PhySO](https://github.com/WassimTenachi/PhySO)\
Documentation: [physo.readthedocs.io](https://physo.readthedocs.io/en/latest/)

## Highlights

$\Phi$-SO's symbolic regression module fully leverages physical units constraints in order to infer analytical physical laws from data points, searching in the space of functional forms. For more details see: [[Tenachi et al 2023]](https://arxiv.org/abs/2303.03192).

$\Phi$-SO recovering the equation for a damped harmonic oscillator:

https://github.com/WassimTenachi/PhySO/assets/63928316/655b0eea-70ba-4975-8a80-00553a6e2786

Performances on the standard Feynman benchmark from [SRBench](https://github.com/cavalab/srbench/tree/master)) comprising 120 expressions from the Feynman Lectures on Physics against popular SR packages.

$\Phi$-SO achieves state-of-the-art performance in the presence of noise (exceeding 0.1%) and shows robust performances even in the presence of substantial (10%) noise:

![feynman_results](https://github.com/WassimTenachi/PhySO/assets/63928316/bbb051a2-2737-40ca-bfbf-ed185c48aa71)

# Installation

The package has been tested on:
- Linux
- OSX (ARM & Intel)
- Windows

## Install procedure

### Virtual environment

To install the package it is recommended to first create a conda virtual environment:
```
conda create -n PhySO python=3.8
```
And activate it:
```
conda activate PhySO
```
### Downloading

`physo` can be downloaded using git:
```
git clone https://github.com/WassimTenachi/PhySO
```

Or by direct downloading a zip of the repository: [here](https://github.com/WassimTenachi/PhySO/zipball/master)

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

In addition, latex should be installed on the system.

---

**NOTE** : `physo` supports CUDA but it should be noted that since the bottleneck of the code is free constant optimization, using CUDA (even on a very high-end GPU) does not improve performances over a CPU and can actually hinder performances.

---

### Installing PhySO

Installing `physo` (from the repository root):
```
pip install -e .
```

### Testing install

Import test:
```
python3
>>> import physo
```
This should result in `physo` being successfully imported.

Unit tests:

Running all unit tests except parallel mode ones (from the repository root):
```
python -m unittest discover -p "*UnitTest.py"
```
This should result in all tests being successfully passed (except for program_display_UnitTest tests if optional dependencies were not installed).

Running all unit tests (from the repository root):
```
python -m unittest discover -p "*Test.py"
```
This  should take 5-15 min depending on your system (as if you have a lot of CPU cores, it will take longer to make the efficiency curves).

## Uninstalling
Uninstalling the package.
```
conda deactivate
conda env remove -n PhySO
```
# Getting Started
## A simple symbolic regression task
Symbolic regression (SR) consists in the inference of a free-form symbolic analytical function $f: \mathbb{R}^n \longrightarrow \mathbb{R}$ that fits $y = f(x_0,..., x_n)$ given $(x_0,..., x_n, y)$ data.

Given a dataset $(x_0,..., x_n, y)$:
```
import numpy as np

z = np.random.uniform(-10, 10, 50)
v = np.random.uniform(-10, 10, 50)
X = np.stack((z, v), axis=0)
y = 1.234*9.807*z + 1.234*v**2
```
Where $X=(z,v)$, $z$ being a length of dimension $L^{1}, T^{0}, M^{0}$, v a velocity of dimension $L^{1}, T^{-1}, M^{0}$, $y=E$ if an energy of dimension $L^{2}, T^{-2}, M^{1}$.

Given the units input variables $(x_0,..., x_n)$ (here $(z, v)$ ), the root variable $y$ (here $E$) as well as free and fixed constants, you can run an SR task to recover $f$ via:
```
import physo

expression, logs = physo.SR(X, y,
                            X_units = [ [1, 0, 0] , [1, -1, 0] ],
                            y_units = [2, -2, 1],
                            fixed_consts       = [ 1.      ],
                            fixed_consts_units = [ [0,0,0] ],
                            free_consts_units  = [ [0, 0, 1] , [1, -2, 0] ],                          
)
```
(Allowing the use of a fixed constant $1$ of dimension $L^{0}, T^{0}, M^{0}$ (ie dimensionless) and free constants $m$ of dimension $L^{0}, T^{0}, M^{1}$ and $g$ of dimension $L^{1}, T^{-2}, M^{0}$.)

It should be noted that here the units vector are of size 3 (eg: `[1, 0, 0]`) as in this example the variables have units dependent on length, time and mass only.
However, units vectors can be of any size $\leq 7$ as long as it is consistent across X, y and constants, allowing the user to express any units (dependent on length, time, mass, temperature, electric current, amount of light, or amount of matter).
In addition, dimensional analysis can be performed regardless of the order in which units are given, allowing the user to use any convention ([length, mass, time] or [mass, time, length] etc.) as long as it is consistent across X,y and constants.

---

**NOTE** : If you are not working on a physics problem and all your variables/constants are dimensionless, do not specify any of the `xx_units` arguments (or specify them as `[0,0]` for all variables/constants) and `physo` will perform a dimensionless symbolic regression task.

---

It should also be noted that free constants search starts around 1. by default.
Therefore when using default hyperparameters, normalizing the data around an order of magnitude of 1 is strongly recommended.

Finally, please note that SR capabilities of `physo` are heavily dependent on hyperparameters, it is therefore recommended to tune hyperparameters to your own specific problem for doing science.
Summary of currently available hyperparameters configurations:

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

In addition, `physo.SR` can create log files which can be inspected later on.
Pareto front expressions and their constants can be loaded into a list of sympy expressions via:
```
sympy_expressions = physo.read_pareto_csv("SR_curves_pareto.csv")
```

This demo can be found in [here](https://github.com/WassimTenachi/PhySO/blob/main/demos/demos_sr/demo_quick_sr.ipynb).

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
