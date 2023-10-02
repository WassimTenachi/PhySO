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

