## Dimensional analysis

`physo` provides a physical units feature that can be used to reduce the symbolic expression search space by enforcing dimensional analysis constraints.   
Reference paper for this feature: [[Tenachi 2023]](https://arxiv.org/abs/2303.03192).

![logo](https://raw.githubusercontent.com/WassimTenachi/PhySO/main/docs/assets/da_space_reduction_illustration.png)

This feature can simply be used by passing the units of the variables and root through `X_units` and `y_units` arguments, as well as units of constants through `fixed_consts_units` and `free_constants_units` (and if necessary through `spe_free_consts_units` and `class_free_consts_units` for Class SR).
In addition, `PhysicalUnitsPrior` should be enabled in the `priors_config` list of the run configuration, this is the case for all available configurations in the [physo/config/](https://github.com/WassimTenachi/PhySO/tree/main/physo/config) directory.

See the [getting-started-sr](https://physo.readthedocs.io/en/latest/r_sr.html#getting-started-sr) tutorial where this feature is used.