
`physo.toolkit` is a comprehensive platform that bridges machine learning methods with the automated formulation of analytical symbolic expressions in scientific contexts. 
The toolkit lowers the barrier to entry for researchers in AI for Science by providing high-level interfaces, extensive tutorials, and ready-to-use components requiring minimal background in computational mathematics.

## Overview

`physo.toolkit`'s embedding enables the decoding of symbolic equations from any machine learning technique - such as neural networks or genetic programming - based on numerical objectives (e.g., fitness, limit values, simulation behavior) or symbolic properties (e.g., derivatives, primitives, symmetries). Priors and constraints - such as length, composition, or dimensional analysis - can be incorporated to guide or restrict expression decoding. 
The toolkit also provides efficient utilities for generating large sets of random equations, allowing users to construct datasets or explore functional search spaces under custom constraints.

We notably provide a generic for loop `insert your ML model here 👇` that you can use to insert your own generative ML model to generate expressions that satisfy any criteria you may have while ensuring that the generated expressions are valid and can be decoded back to their symbolic form.

Workflow of a generative analytic expression model using `physo.toolkit`:
![workflow](https://raw.githubusercontent.com/WassimTenachi/PhySO/main/docs/assets/toolkit_workflow.png)
The user's model produces probability distributions over tokens, which are modulated by deterministic priors to enforce structural or domain-specific constraints. 
Tokens are then sampled from the `library` according to these distributions and appended to the current `expression`. 
The resulting expression can be evaluated for various properties - either formal or based on numerical predictions - to define arbitrary objectives that serve as feedback for the generative process.
