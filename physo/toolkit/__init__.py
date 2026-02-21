from . import random_sampler
from . import codec

# Making important functions accessible:
get_library      = codec.get_library
get_prior        = codec.get_prior
get_expressions  = codec.get_expressions
sample_random_expressions = random_sampler.sample_random_expressions