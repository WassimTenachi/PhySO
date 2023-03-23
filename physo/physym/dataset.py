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
        library : library.Library
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
        assert torch.isnan(X).any()        == False, "X should not contain any Nans"
        assert torch.isnan(y_target).any() == False, "y should not contain any Nans"
        # --- ASSERT SHAPE ---
        assert len(X.shape)        == 2, "X        must have shape = (n_dim, data_size,)"
        assert len(y_target.shape) == 1, "y_target must have shape = (data_size,)"
        assert X.shape[1] == y_target.shape[0], "X must have shape = (n_dim, data_size,) and y_target must have " \
                                                "shape = (data_size,) with the same data_size."
        n_dim, data_size = X.shape
        # --- ASSERT VARIABLE ID ---
        # Check that all tokens in the library have id < n_dim
        # Is id var_id wrong : mask.
        # Ie. var_type is that of input var AND id >= n_dim
        mask_wrong_id = np.logical_and(library.var_type == 1, library.var_id >= n_dim)
        assert mask_wrong_id.sum() == 0, "Can not access input variable data X by X[var_id] of tokens :" \
                                         "\n %s\n as they have out of range var_id >= X.shape[0] = n_dim = %i," \
                                         " var_id :\n %s" % (library.lib_name[mask_wrong_id], n_dim, library.var_id [mask_wrong_id])


        # ---------------------- ATTRIBUTES ----------------------
        self.X               = X
        self.y_target        = y_target
        self.detected_device = X.device

    def __repr__(self):
        s = "X        : %s \n" \
            "y_target : %s"%(self.X.shape, self.y_target.shape)
        return s