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
        self.n_free_const = self.library.n_free_constants # Number of free constants
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

class Dataset:
    """
    Contains a dataset and runs assertions.
    """
    def __init__(self, library, X, y_target):
        """
        Parameters
        ----------
        library : Library.Library
            Library of choosable tokens.
        X : torch.tensor of shape (n_dim, ?,) of float
            Values of the input variables of the problem with n_dim = nb of input variables.
        y_target : torch.tensor of shape (?,) of float
            Values of the target symbolic function on input variables contained in X_target.
        """
        self.library = library

        # ---------------------- ASSERTIONS ----------------------

        # --- ASSERT TORCH TYPE ---
        assert torch.is_tensor(X),        "X        must be a torch.tensor"
        assert torch.is_tensor(y_target), "y_target must be a torch.tensor"
        # --- ASSERT FLOAT TYPE ---
        assert X       .dtype == torch.float64 or X       .dtype == torch.float32, "X        must contain floats."
        assert y_target.dtype == torch.float64 or y_target.dtype == torch.float32, "y_target must contain floats."
        # --- ASSERT SHAPE ---
        assert len(X.shape)        == 2, "X        must have shape = (n_dim, data_size,)"
        assert len(y_target.shape) == 1, "y_target must have shape = (data_size,)"
        assert X.shape[1] == y_target.shape[0], "X must have shape = (n_dim, data_size,) and y_target must have " \
                                                "shape = (data_size,) with the same data_size."
        n_dim, data_size = X.shape
        # --- ASSERT VARIABLE ID ---
        # Check that all tokens in the library have id < n_dim
        # Is id input_var_id wrong : mask.
        # Ie. is_input_var is True AND id >= n_dim
        mask_wrong_id = np.logical_and(library.is_input_var, library.input_var_id >= n_dim)
        assert mask_wrong_id.sum() == 0, "Can not access input variable data X by X[input_var_id] of tokens :" \
                                         "\n %s\n as they have out of range input_var_id >= X.shape[0] = n_dim = %i," \
                                         " input_var_id :\n %s" % (library.lib_name[mask_wrong_id], n_dim, library.input_var_id [mask_wrong_id])


        # ---------------------- ATTRIBUTES ----------------------
        self.X        = X
        self.y_target = y_target

    def __repr__(self):
        s = "X        : %s \n" \
            "y_target : %s"%(self.X.shape, self.y_target.shape)
        return s