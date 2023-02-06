import torch
import numpy as np

class FreeConstantsTable:
    """
    Contains free constants values.
    """
    def __init__(self, batch_size, library):
        # Library
        self.library = library

        # Shape
        self.batch_size = batch_size
        self.n_free_const = self.library.n_free_const # Number of free constants
        self.shape = (self.batch_size, self.n_free_const,)

        # Free constants values for each program # as torch tensor for fast computation (sent to device in Batch)
        self.values = torch.tensor(np.tile(self.library.free_constants_init_val,
                                      reps=(self.batch_size, 1)))                      # (batch_size, n_free_const,) of float
        # mask : is free constant optimized
        self.is_opti   = np.full(shape=self.batch_size, fill_value=False, dtype=bool)  # (batch_size,) of bool
        # Number of epochs necessary to optimize free constant
        self.opti_time = np.full(shape=self.batch_size, fill_value=False, dtype=int )  # (batch_size,) of int

    def __repr__(self):
        s = "FreeConstantsTable for %s : %s"%(self.library.free_constants_tokens, self.shape,)
        return s


