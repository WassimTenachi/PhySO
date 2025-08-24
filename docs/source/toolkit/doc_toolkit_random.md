##  Generating random expressions

`physo.toolkit` can be used to randomly generate symbolic mathematical equations.
These equations are constructed from a customizable library of tokens-including mathematical operators, variables, and numerical constants-and are encoded in a format suitable for training machine learning models.

Key features include:
- 📏 __Length-controlled sampling__: Equations are sampled with a Gaussian prior over expression length.
- 🏗️ __Custom structural priors__: You can enforce specific structural properties, such as prohibiting nested trigonometric functions or setting token occurrence constraints.
- ⚙️ __Dimensional analysis__: Equations can be generated with physically consistent units, and unit information is preserved throughout the expression tree.
- 📦 __Flexible export__: Expressions can be exported to various formats including (differentiable) python functions, SymPy objects, LaTeX, strings and saved on disk.
- ⚡️ __Evaluation and parameter fitting__: Expressions can be numerically evaluated, and their free parameters optimized as needed (in parallel across a batch of equations if desired). Uncertainty can also be taken into account via weights.
- 🔢 __Encoding__: Expressions' numerical encoding can be accessed easily, facilitating the use of expressions for machine learning purposes.
- ⛓️ __Auto-differentiable structures__: Expression trees are compatible with automatic differentiation.
- 🌳 __Tree structure navigation__: The expression tree can be displayed and navigated. This can be used to access e.g. parent, children, and sibling nodes of any token, and even list their ancestors in a vectorized way across all expressions in the batch.
- ⚖️ __Physical Units information__: Physical units of each token is dynamically computed and stored in the expression tree.
- 📚 __One equation, multiple datasets__: Expressions can contain dataset-specific free constant values through, allowing for a single equation to be evaluated and fitted across multiple datasets.

### Video tutorial (Random sampling)

(Coming soon)

### Getting started

Reference notebook for this tutorial: [📙demo_random_sampler.ipynb](https://github.com/WassimTenachi/PhySO/blob/main/demos/toolkit/demo_random_sampler.ipynb).

A complementary tutorial that shows how to manipulate the symbolic expressions, including inspecting, showing, exporting, evaluating, and more is available in [this section of the documentation](https://physo.readthedocs.io/en/latest/r_toolkit.html#expression-manipulation).