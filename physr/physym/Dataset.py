import torch
import numpy as np


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