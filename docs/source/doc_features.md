## Adding custom functions

### Defining function token

If you want to add a custom choosable function to `physo`, you can do so by adding you own [Token](https://github.com/WassimTenachi/PhySO/blob/main/physo/physym/token.py) to the list `OPS_UNPROTECTED` in [functions.py](https://github.com/WassimTenachi/PhySO/blob/main/physo/physym/functions.py).

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

More details about Token attributes can be found in the documentation of the Token object : [here](https://github.com/WassimTenachi/PhySO/blob/main/physo/physym/token.py)

### Behavior in dimensional analysis

The newly added custom function should also be listed in its corresponding behavior (in the context of dimensional analysis) in the list of behaviors in `OP_UNIT_BEHAVIORS_DICT` [(here)](https://github.com/WassimTenachi/PhySO/blob/main/physo/physym/functions.py).

### Additional information

In addition, the custom function should be :
- Listed in `TRIGONOMETRIC_OP` [(here)](https://github.com/WassimTenachi/PhySO/blob/main/physo/physym/functions.py) if it is a trigonometric operation (eg. cos, sinh, arcos etc.) so it can be treated as such by priors if necessary.
- Listed in `INVERSE_OP_DICT` [(here)](https://github.com/WassimTenachi/PhySO/blob/main/physo/physym/functions.py) along with its corresponding inverse operation $f^{-1}$ if it has one (eg. arcos for cos) so they can be treated as such by priors if necessary.
- Listed in `OP_POWER_VALUE_DICT` [(here)](https://github.com/WassimTenachi/PhySO/blob/main/physo/physym/functions.py) along with its power value (float) if it is a power token (eg. 0.5 for sqrt) so it can be used in dimensional analysis.

### Protected version (optional)

If your custom function has a protected version ie. a version defined everywhere on $\mathbb{R}$ (eg. using $f(x) = log(abs(x))$ instead of $f(x) = log(x)$ ) to smooth expression search and avoid undefined rewards, you should also add the protected version in to the list `OPS_PROTECTED` in [functions.py](https://github.com/WassimTenachi/PhySO/blob/main/physo/physym/functions.py) with the similar attributes but with the protected version of the function for the `function` attribute.

### Running the functions unit test (optional)

After adding a new function, running the functions unit test is highly recommended:
```
python ./physo/physym/tests/functions_UnitTest.py
```

If you found the function you have added useful, don't hesitate to make a pull request so other people can use it too !


## Custom symbolic optimization task

[Coming soon...]

