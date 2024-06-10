
This section presents the benchmarking utilities available within `physo`.
I.e. user-friendly ways to access benchmarking challenges in a reproducible and standardized manner (using standard data ranges etc.).
For each challenge, we present ways to generate data and compare candidate expressions to the ground truth.

Our interfaces adhere to the standard benchmarking practices presented in [[La Cava 2021]](https://arxiv.org/abs/2107.14351).

## Feynman benchmark

The purpose of the Feynman benchmark is to evaluate symbolic regression systems, in particular methods built for scientific discovery.
That is methods able to produce compact, predictive and interpretable expressions from potentially noisy data.

See [[Udrescu 2019]](https://arxiv.org/abs/1905.11481) which introduced the benchmark, [[La Cava 2021]](https://arxiv.org/abs/2107.14351) which formalized it and [[Tenachi 2023]](https://arxiv.org/abs/2303.03192) for the evaluation of `physo` on this benchmark.

Note that `physo` reads Feynman challenges from the original files (FeynmanEquations.csv, BonusEquations.csv, units.csv) produced by [[Udrescu 2019]](https://arxiv.org/abs/1905.11481) and correcting some mistakes (incorrect units, columns etc.) in the original files.

### Loading a Feynman challenge

This benchmark comprises 120 challenges, each with a unique ground truth expression.

Challenge number `i_eq` $\in$ {0, 1, ..., 119} of the Feynman benchmark, {0, ..., 99} corresponding to bulk challenges and {100, ..., 119} to bonus challenges can be accessed as follows:

```
import physo
import physo.benchmark.FeynmanDataset.FeynmanProblem as Feyn

i_eq = 42
pb = Feyn.FeynmanProblem(i_eq=i_eq)
print(pb)
```

```
>>> FeynmanProblem : I.42 : I.42
>>> x0*exp((-x1)*x2*x4/((x3*x5)))
```

It is also possible to access the problem with the original variable names:

```
print(Feyn.FeynmanProblem(i_eq, original_var_names=True))
```
```
>>> FeynmanProblem : I.40.1 : I.40.1
>>> n_0*exp(g*(-m)*x/((T*kb)))
```

### Getting formula and variable names

Getting sympy formula, also encodes assumptions on variables (positivity, etc.):
```
print(pb.formula_sympy)
>>> x0*exp((-x1)*x2*x4/((x3*x5)))
```
Getting latex formula:
```
print(pb.formula_latex)
>>> 'x_{0} e^{\\frac{- x_{1} x_{2} x_{4}}{x_{3} x_{5}}}'
```

Getting names of variables and their units:
```
print(pb.X_names)
>>> array(['x0', 'x1', 'x2', 'x3', 'x4', 'x5'], dtype='<U2')
print(pb.X_units)
>>> array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.],
           [ 1., -2.,  0.,  0.,  0.],
           [ 2., -2.,  1., -1.,  0.]])
print(pb.y_name)
>>> 'y'
print(pb.y_units)
>>> array([0., 0., 0., 0., 0.])
```

### Generating data

Generating $1000$ data points:
```
X, y = pb.generate_data_points(n_samples=1000)
```
```
print(X.shape)
>>> (6, 1000)
print(y.shape)
>>> (1000,)
```

Showing 1000 data points:
```
pb.show_sample(1000)
```

### Target function

It is possible to evaluate the target function using:
```
y = pb.target_function(X)
```

### Assessing symbolic equivalence of a candidate expression


Given a candidate expression, we can assess its symbolic equivalence to the ground truth expression.
```
# Expression returned by SR system:
candidate_expr_str = "1.0002*x0*exp(0.0004-x1*x2*0.9993*x4)**(1/(x3*x5))+0.00034"
```

Getting a sympy expression from the string:
```
import sympy
candidate_expr = sympy.parse_expr(candidate_expr_str,
                                 # Encoding assumptions about variables (positivity etc.)
                                 local_dict = pb.sympy_X_symbols_dict)
```
Comparing it to the ground truth:
```
is_equivalent, report = pb.compare_expression(trial_expr    = candidate_expr, 
                                              round_decimal = 3,
                                              verbose       = True
                                             )
```
```
>>>   -> Assessing if x0*exp(-x1*x2*x4/(x3*x5)) (target) is equivalent to 1.0002*x0*(1.00040008001067*exp(-0.9993*x1*x2*x4))**(1/(x3*x5)) + 0.00034 (trial)
       -> Simplified expression : x0*exp(-x1*x2*x4/(x3*x5))
       -> Symbolic error        : 0
       -> Symbolic fraction     : 1
       -> Trigo symbolic error        : 0
       -> Trigo symbolic fraction     : 1
       -> Equivalent : True
```