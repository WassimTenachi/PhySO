
## Encoding and decoding expressions

This notebook demonstrates how to numerically encode and decode mathematical expressions using `physo.toolkit`. 
This is useful for any machine learning (ML) tasks that involves symbolic mathematical expressions, such as symbolic regression, equation discovery, or any task that requires the manipulation of mathematical formulas.

Key features include:
- 🧠 __Priors__ : `physo` includes many deterministic priors that are computed after each token generation that can be used to e.g. bias the search towards certain expressions : this includes length priors, structural priors (e.g. excluding nesting of trigonometric functions such as $\text{cos}(a+\text{sin}(1/\text{tan}(x)))$), dimensional analysis priors, prior about the number of occurrences of a token, prior about sub-functions, and more.
- 📦 __Flexible export__: Expressions can be exported to various formats including (differentiable) python functions, SymPy objects, LaTeX, strings and saved on disk.
- ⚡️ __Evaluation and parameter fitting__: Expressions can be numerically evaluated, and their free parameters optimized as needed (in parallel across a batch of equations if desired). Uncertainty can also be taken into account via weights.
- 🔢 __Encoding__: Expressions' numerical encoding can be accessed easily, facilitating the use of expressions for machine learning purposes.
- ⛓️ __Auto-differentiable structures__: Expression trees are compatible with automatic differentiation.
- 🌳 __Tree structure navigation__: The expression tree can be displayed and navigated. This can be used to access e.g. parent, children, and sibling nodes of any token, and even list their ancestors in a vectorized way across all expressions in the batch.
- ⚖️ __Physical Units information__: Physical units of each token is dynamically computed and stored in the expression tree.
- 📚 __One equation, multiple datasets__: Expressions can contain dataset-specific free constant values through, allowing for a single equation to be evaluated and fitted across multiple datasets.


### Video tutorial

(Coming soon)

### Getting started

Reference notebook for this tutorial: [📙demo_encode_decode.ipynb](https://github.com/WassimTenachi/PhySO/blob/main/demos/toolkit/demo_encode_decode.ipynb).

A complementary tutorial that shows how to manipulate the symbolic expressions, including inspecting, showing, exporting, evaluating, and more is available in [this section of the documentation](https://physo.readthedocs.io/en/latest/r_toolkit.html#expression-manipulation).