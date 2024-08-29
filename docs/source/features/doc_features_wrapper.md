## Candidate wrapper

The `candidate_wrapper` argument can be passed to the `physo.SR` or `physo.ClassSR` to apply a wrapper function $g$ to the candidate symbolic function's output $f(X)$.

The wrapper function $g$ should be a callable taking the candidate symbolic function callable $f$ and the input data $X$ as arguments, and returning the wrapped output $g(f(X))$.
By default `candidate_wrapper = None`, no wrapper is applied (identity).

Note that the wrapper function should be differentiable and written in `pytorch` if free constants are to be optimized since the free constants are optimized using gradient-based optimization.

In addition, it is recommended to use protected functions when writing the wrapper function to avoid evaluating the symbolic function on invalid points (eg. using log abs instead of log).
See the [protected functions](https://physo.readthedocs.io/en/latest/r_features.html#protected-version-optional) documentation for more details. 