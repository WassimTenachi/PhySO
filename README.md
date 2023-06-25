
# $\Phi$-SO : Physical Symbolic Optimization

The physical symbolic optimization ( $\Phi$-SO ) package `physo` is a symbolic regression package that fully leverages physical units constraints in order to infer analytical physical laws from data points, searching in the space of functional forms. For more details see: [[Tenachi et al 2023]](https://arxiv.org/abs/2303.03192).

https://user-images.githubusercontent.com/63928316/225642347-a07127da-a84e-4af3-96f4-4c7fef5a673b.mp4


# Installation

The package has been tested on:
- Linux
- OSX (ARM & Intel)
- Windows

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
Or by direct downloading a zip of the repo:
```
https://github.com/WassimTenachi/PhySO/zipball/master
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

In addition, latex should be installed on the system.

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
This  should take 5-15 min depending on your system (as if you have a lot of CPU cores, it will take longer to make the efficiency curves).

# Getting started

### Symbolic regression with default hyperparameters
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

This demo can be found in `demo/demo_quick_sr.ipynb`.

### Symbolic regression
[Coming soon] 
In the meantime you can have a look at our demo folder ! :)

### Custom symbolic optimization task
[Coming soon]

### Adding custom functions

1. Defining function token

If you want to add a custom choosable function to `physo`, you can do so by adding you own [Token](physo/physym/token.py) to the list `OPS_UNPROTECTED` in [functions.py](physo/physym/functions.py).

For example a token such as $f(x) = x^5$ can be added via:
```
OPS_UNPROTECTED = [
...
Token (name = "n5"     , sympy_repr = "n5"     , arity = 1 , complexity = 1 , var_type = 0, function = lambda x :torch.pow(x, 5)         ),
...
]
```
Where:
- `name` (str) is the name of the token (used for selecting it in the config of a run).
- `sympy_repr` (str) is the name of the token to use when producing the sympy / latex representation. 
- `arity` (int) is the number of arguments that the function takes.
- `complexity` (float) is the value to consider for expression complexity considerations (1 by default).
- `var_type` (int) is the type of token, it should always be 0 when defining functions like here.
- `function` (callable) is the function, it should be written in pytorch to support auto-differentiation.

More details about Token attributes can be found in the documentation of the Token object : [here](physo/physym/token.py)

2. Behavior in dimensional analysis

The newly added custom function should also be listed in its corresponding behavior (in the context of dimensional analysis) in the list of behaviors in [`OP_UNIT_BEHAVIORS_DICT`](physo/physym/functions.py).

3. Additional information

In addition, the custom function should be :
- Listed in [`TRIGONOMETRIC_OP`]((physo/physym/functions.py)) if it is a trigonometric operation (eg. cos, sinh, arcos etc.) so it can be treated as such by priors if necessary.
- Listed in [`INVERSE_OP_DICT`]((physo/physym/functions.py)) along with its corresponding inverse operation $f^{-1}$ if it has one (eg. arcos for cos) so they can be treated as such by priors if necessary.
- Listed in [`OP_POWER_VALUE_DICT`]((physo/physym/functions.py)) along with its power value (float) if it is a power token (eg. 0.5 for sqrt) so it can be used in dimensional analysis.

4. Protected version (optional)

If your custom function has a protected version ie. a version defined everywhere on $\mathbb{R}$ (eg. using $f(x) = log(abs(x))$ instead of $f(x) = log(x)$ ) to smooth expression search and avoid undefined rewards, you should also add the protected version in to the list `OPS_PROTECTED` in [functions.py](physo/physym/functions.py) with the similar attributes but with the protected version of the function for the `function` attribute.

5. Running the functions unit test (optional)

After adding a new function, running the functions unit test via `python ./physo/physym/tests/functions_UnitTest.py` is highly recommended.

If you found the function you have added useful, don't hesitate to make a pull request so other people can use it too !

# About computational performances

The main performance bottleneck of `physo` is free constant optimization, therefore, in non-parallel execution mode, performances are almost linearly dependent on the number of free constant optimization steps and on the number of trial expressions per epoch (ie. the batch size).

In addition, it should be noted that generating monitoring plots takes ~3s, therefore we suggest making monitoring plots every >10 epochs for low time / epoch cases. 

## Expected computational performances

Summary of expected performances with `physo` (in parallel mode):

| Time / epoch | Batch size | # free const | free const <br>opti steps | Example                             | Device                                    |
|--------------|------------|--------------|---------------------------|-------------------------------------|-------------------------------------------|
| ~5s          | 10k        | 2            | 15                        | eg: demo_damped_harmonic_oscillator | CPU: Intel W-2155 10c/20t <br>RAM: 128 Go |
| ~10s         | 10k        | 2            | 15                        | eg: demo_damped_harmonic_oscillator | CPU: Mac M1 <br>RAM: 16 Go                |
| ~30s         | 10k        | 2            | 15                        | eg: demo_damped_harmonic_oscillator | CPU: Intel i7 4770 <br>RAM: 16 Go         |
| ~250s        | 10k        | 2            | 15                        | eg: demo_damped_harmonic_oscillator | GPU: Nvidia GV100 <br>VRAM : 32 Go        |
| ~1s          | 1k         | 2            | 15                        | eg: demo_mechanical_energy          | CPU: Intel W-2155 10c/20t <br>RAM: 128 Go |
| ~6s          | 1k         | 2            | 15                        | eg: demo_mechanical_energy          | CPU: Mac M1 <br>RAM: 16 Go                |
| ~30s         | 1k         | 2            | 15                        | eg: demo_mechanical_energy          | CPU: Intel i7 4770 <br>RAM: 16 Go         |
| ~4s          | 1k         | 2            | 15                        | eg: demo_mechanical_energy          | GPU: Nvidia GV100 <br>VRAM : 32 Go        |

Please note that using a CPU typically results in higher performances than when using a GPU.

## Parallel mode

1. Parallel free constant optimization

Parallel constant optimization is enabled if and only if :
- The system is compatible (checked by `physo.physym.execute.ParallelExeAvailability`).
- `parallel_mode = True` in the reward computation configuration.
- `physo.physym.reward.USE_PARALLEL_OPTI_CONST = True`.

By default, both of these are true as parallel mode is typically faster for this task.
However, if you are using a batch size <10k, due to communication overhead it might be worth it to disable it for this task via:
```
physo.physym.reward.USE_PARALLEL_OPTI_CONST = False
```

2. Parallel reward computation

Parallel reward computation is enabled if and only if :
- The system is compatible (checked by `physo.physym.execute.ParallelExeAvailability`).
- `parallel_mode = True` in the reward computation configuration.
- `physo.physym.reward.USE_PARALLEL_EXE = True`.

By default, `physo.physym.reward.USE_PARALLEL_EXE = False`, i.e. parallelization is not used for this task due to communication overhead making it typically slower for such individually inexpensive tasks.
However, if you are using $>10^6$ data points it tends to be faster, so we recommend enabling it by setting:
```
physo.physym.reward.USE_PARALLEL_EXE = True
```

3. Miscellaneous

- Efficiency curves (nb. of CPUs vs individual task time) are produced by `execute_UnitTest.py` in a realistic toy case with batch size = 10k and $10^3$ data points.
- Parallel mode is not available from jupyter notebooks on spawn multiprocessing systems (typically MACs/Windows), run .py scripts on those.
- The use of `parallel_mode` can be managed in the configuration of the reward which can itself be managed through a hyperparameter config file (see `config` folder) which is handy for running a benchmark on an HPC with a predetermined number of CPUs.
- Disabling parallel mode entirely via `USE_PARALLEL_EXE=False` `USE_PARALLEL_OPTI_CONST=False` is recommended before running `physo` in a debugger.

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
