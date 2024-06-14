Uniform Arity Prior
~~~~~~~~~~~~~~~~~~~

.. autoclass:: physo.physym.prior.UniformArityPrior
    :members: __init__

Example of usage from the config file:

.. code-block:: python

    priors_config  = [
                    ...
                    ("UniformArityPrior", None),
                    ...
                     ]


Hard Length Prior
~~~~~~~~~~~~~~~~~

.. autoclass:: physo.physym.prior.HardLengthPrior
    :members: __init__

Example of usage from the config file:

.. code-block:: python

    priors_config  = [
                    ...
                    ("HardLengthPrior"  , {"min_length": 4, "max_length": 35, }),
                    ...
                     ]




Soft Length Prior
~~~~~~~~~~~~~~~~~

.. autoclass:: physo.physym.prior.SoftLengthPrior
    :members: __init__

Example of usage from the config file:

.. code-block:: python

    priors_config  = [
                    ...
                    ("SoftLengthPrior"  , {"length_loc": 8, "scale": 5, }),
                    ...
                     ]





Relationship Constraint Prior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physo.physym.prior.RelationshipConstraintPrior
    :members: __init__

Example of usage from the config file:

.. code-block:: python

    priors_config  = [
                    ...
                    ("RelationshipConstraintPrior"  , {"effectors"    : ["exp" , "log", "sin", "exp", "sub"],
                                                       "targets"      : ["log" , "exp", "sin", "cos", "neg"],
                                                       "relationship" : "child",
                                                       }),
                    ...
                     ]



No Useless Inverse Prior
""""""""""""""""""""""""

.. autoclass:: physo.physym.prior.NoUselessInversePrior
    :members: __init__

Example of usage from the config file:

.. code-block:: python

    priors_config  = [
                    ...
                    ("NoUselessInversePrior"  , None),
                    ...
                     ]



Nested Functions
~~~~~~~~~~~~~~~~

.. autoclass:: physo.physym.prior.NestedFunctions
    :members: __init__

Example of usage from the config file:

.. code-block:: python

    priors_config  = [
                    ...
                    ("NestedFunctions", {"functions":["exp",], "max_nesting" : 1}),
                    ("NestedFunctions", {"functions":["log",], "max_nesting" : 1}),
                    ...
                     ]




Nested Trigonometry Prior
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physo.physym.prior.NestedTrigonometryPrior
    :members: __init__

Example of usage from the config file:

.. code-block:: python

    priors_config  = [
                    ...
                    ("NestedTrigonometryPrior", {"max_nesting" : 1}),
                    ...
                     ]



Occurrences Prior
~~~~~~~~~~~~~~~~~

.. autoclass:: physo.physym.prior.OccurrencesPrior
    :members: __init__

Example of usage from the config file:

.. code-block:: python

    priors_config  = [
                    ...
                    ("OccurrencesPrior", {"targets" : ["1",], "max" : [3,] }),
                    ...
                     ]



Symbolic Prior
~~~~~~~~~~~~~~

.. autoclass:: physo.physym.prior.SymbolicPrior
    :members: __init__

Example of usage from the config file:

.. code-block:: python

    priors_config  = [
                    ...
                    # Forcing expression to look like a + ...
                    ('SymbolicPrior', {'expression': ["+", "a", ]}),
                    ...
                     ]




Physical Units Prior
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physo.physym.prior.PhysicalUnitsPrior
    :members: __init__

Example of usage from the config file:

.. code-block:: python

    priors_config  = [
                    ...
                    # Zeroing out to epsilon level unphysical symbolic choices
                    ("PhysicalUnitsPrior", {"prob_eps": np.finfo(np.float32).eps}),
                    ...
                     ]


