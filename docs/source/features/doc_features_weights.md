## Weighting Data Points

### About weights

`physo` provides a feature that allows the user to weight data points when performing symbolic regression.
This feature can be used to give more importance to certain data points over others. 
The weights are used during free constant optimization as well as for computing the reward driving the symbolic regression process.

Weights can simply be passed through the `y_weights` argument of `physo.SR`, they should have the same shape as the target values `y`.

---
__Class SR notes:__
If you are using Class SR, the weights should be passed through the `multi_y_weights` argument of `physo.ClassSR`.
They can have the same shape as the target values `multi_y` or simply the length of the number of realizations (for giving more weights to certain realizations over others), as indicated in the [docstring](https://physo.readthedocs.io/en/latest/r_class_sr.html#function-docstring) of `physo.ClassSR` for more details).
---

### Example

Example of weight usage in SR.
The reference notebook for this tutorial can be found here: [demo_y_weights.ipynb](https://github.com/WassimTenachi/PhySO/blob/main/demos/sr/demo_y_weights/demo_y_weights.ipynb).


#### Setup

Importing the necessary libraries:
```
# External packages
import numpy as np
import matplotlib.pyplot as plt
import torch
```

Importing `physo`:
```
# Internal code import
import physo
import physo.learn.monitoring as monitoring
```

It is recommended to fix the seed for reproducibility:
```
# Seed
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
```

#### Making synthetic datasets

Making a toy synthetic dataset:
```
# Making toy synthetic data
t = np.random.uniform(0.1, 10, 1000)
X = np.stack((t,), axis=0)
y = np.exp(-1.45*t) + 0.5*np.cos(3.7*t)
```

Let's give more weight to the latter part of the dataset which should favor the cosine term:
```
y_weights = 0.01 + (t > 5.).astype(float)
```

Datasets plot:
```
n_dim = X.shape[0]
fig, ax = plt.subplots(n_dim, 1, figsize=(10,5))
for i in range (n_dim):
    curr_ax = ax if n_dim==1 else ax[i]
    curr_ax.plot(X[i], y, 'k.',)
    curr_ax.set_xlabel("X[%i]"%(i))
    curr_ax.set_ylabel("y")
    # Weights
    curr_ax.plot(X[i], y_weights, 'r.', label="weights")
    curr_ax.legend()
plt.show()
```
![logo](https://raw.githubusercontent.com/WassimTenachi/PhySO/main/docs/assets/demo_weights_data_plot.png)

#### SR configuration

Logging and visualisation setup:
```
save_path_training_curves = 'demo_curves.png'
save_path_log             = 'demo.log'

run_logger     = lambda : monitoring.RunLogger(save_path = save_path_log,
                                                do_save = True)

run_visualiser = lambda : monitoring.RunVisualiser (epoch_refresh_rate = 1,
                                           save_path = save_path_training_curves,
                                           do_show   = False,
                                           do_prints = True,
                                           do_save   = True, )
```

#### Running SR


```
# Running SR task
expression, logs = physo.SR(X, y,
                            y_weights = y_weights,
                            # Giving names of variables (for display purposes)
                            X_names = [ "t"   ],
                            # Giving name of root variable (for display purposes)
                            y_name  = "y",
                            # Fixed constants
                            fixed_consts       = [ 1.      ],
                            # Free constants names (for display purposes)
                            free_consts_names = [ "c1", "c2", "c3",],
                            # Symbolic operations that can be used to make f
                            op_names = ["mul", "add", "sub", "div", "inv", "n2", "sqrt", "neg", "exp", "log", "sin", "cos"],
                            get_run_logger     = run_logger,
                            get_run_visualiser = run_visualiser,
                            # Run config
                            run_config = physo.config.config0.config0,
                            # Parallel mode (only available when running from python scripts, not notebooks)
                            parallel_mode = False,
                            # Number of iterations
                            epochs = 30
)
```

#### Inspecting the best expression found

__Getting best expression:__

The best expression found (in accuracy) is returned in the `expression` variable:
```
best_expr = expression
print(best_expr.get_infix_pretty())
```
```
>>> 
       ⎛     c₃⋅c₃⋅(c₂ + c₂ + c₃ + c₃ + t)⎞
    sin⎜c₂ + ─────────────────────────────⎟
       ⎝                  1.0             ⎠
    ───────────────────────────────────────
                       c₃    
```

It can also be loaded later on from log files:
```
import physo
from physo.benchmark.utils import symbolic_utils as su
import sympy

# Loading pareto front expressions
pareto_expressions = physo.read_pareto_pkl("demo_curves_pareto.pkl")
# Most accurate expression is the last in the Pareto front:
best_expr = pareto_expressions[-1]
print(best_expr.get_infix_pretty())
```

#### Checking exact symbolic recovery

```
# To sympy
best_expr = best_expr.get_infix_sympy(evaluate_consts=True)

best_expr = best_expr[0]

# Printing best expression simplified and with rounded constants
print("best_expr : ", su.clean_sympy_expr(best_expr, round_decimal = 4))

# Target expression was:
target_expr = sympy.parse_expr("0.5*cos(3.7*t)")
print("target_expr : ", su.clean_sympy_expr(target_expr, round_decimal = 4))

# Check equivalence
print("\nChecking equivalence:")
is_equivalent, log = su.compare_expression(
                        trial_expr  = best_expr,
                        target_expr = target_expr,
                        round_decimal = 1,
                        handle_trigo            = True,
                        prevent_zero_frac       = True,
                        prevent_inf_equivalence = True,
                        verbose                 = True,
)
print("Is equivalent:", is_equivalent)
```

```
>>> best_expr :  0.5194*sin(3.7062*t + 26.6575)
    target_expr :  0.5*cos(3.7*t)
    
    Checking equivalence:
      -> Assessing if 0.5*cos(3.7*t) (target) is equivalent to 0.519443317596468*sin(3.70615580351008*t + 26.6574530128894) (trial)
       -> Simplified expression : 0.5*sin(3.7*t + 26.7)
       -> Symbolic error        : -0.5*sin(3.7*t + 26.7) + 0.5*cos(3.7*t)
       -> Symbolic fraction     : cos(3.7*t)/sin(3.7*t + 26.7)
       -> Trigo symbolic error        : 0
       -> Trigo symbolic fraction     : 1
       -> Equivalent : True
    Is equivalent: True
```

#### Fit plot

```
# Reloading expression
best_expr = pareto_expressions[-1]
# Making predictions
X_extended = np.stack( [np.linspace(X.min(), X.max(), 1000),] )
y_pred = best_expr(torch.tensor(X_extended))
```

Plot:
```
n_dim = X.shape[0]
fig, ax = plt.subplots(n_dim, 1, figsize=(10,5))
for i in range (n_dim):
    curr_ax = ax if n_dim==1 else ax[i]
    # Data
    curr_ax.plot(X[i], y, 'k.', label="Data")
    # Prediction
    curr_ax.plot(X_extended[i], y_pred, 'g-', label="Prediction")
    # Weights
    curr_ax.plot(X[i], y_weights, 'r.', label="Weights")
    curr_ax.legend(loc="upper right")
    # Labels
    curr_ax.set_xlabel("X[%i]"%(i))
    curr_ax.set_ylabel("y")
plt.show()
```

![logo](https://raw.githubusercontent.com/WassimTenachi/PhySO/main/docs/assets/demo_weights_results_plot.png)



