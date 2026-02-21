
## Expression manipulation

Refer to this section you wish to learn more about how to manipulate `physo` expressions (even outside of SR tasks).

Features include:
- 📦 __Flexible export__: Expressions can be exported to various formats including (differentiable) python functions, SymPy objects, LaTeX, strings and saved on disk.
- ⚡️ __Evaluation and parameter fitting__: Expressions can be numerically evaluated, and their free parameters optimized as needed (in parallel across a batch of equations if desired). Uncertainty can also be taken into account via weights.
- 🔢 __Encoding__: Expressions' numerical encoding can be accessed easily, facilitating the use of expressions for machine learning purposes.
- ⛓️ __Auto-differentiable structures__: Expression trees are compatible with automatic differentiation.
- 🌳 __Tree structure navigation__: The expression tree can be displayed and navigated. This can be used to access e.g. parent, children, and sibling nodes of any token, and even list their ancestors in a vectorized way across all expressions in the batch.
- ⚖️ __Physical Units information__: Physical units of each token is dynamically computed and stored in the expression tree.
- 📚 __One equation, multiple datasets__: Expressions can contain dataset-specific free constant values through, allowing for a single equation to be evaluated and fitted across multiple datasets.

### Video tutorial (Expressions)

(Coming soon)

### Getting started

Reference notebook for this tutorial: [📙demo_expressions.ipynb](https://github.com/WassimTenachi/PhySO/blob/main/demos/toolkit/demo_expressions.ipynb).

