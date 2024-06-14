## Priors

### About Priors

We provide a number of priors that can be used to guide the symbolic regression process and reduce the search space by tuning the probability of selecting certain symbols during the generation of the candidate expressions.  

For using a prior, a tuple should be added to the `priors_config` list in the run configuration (see [physo/config/](https://github.com/WassimTenachi/PhySO/tree/main/physo/config) for examples of configurations).
The first element of the tuple should be the name of the prior class, and the second element should be the parameters to be passed to the prior.
These should parametrize the prior class as required in their `__init__` method (except for library, program args which are handled by the algorithm itself).

### List of priors