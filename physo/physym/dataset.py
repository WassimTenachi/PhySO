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
        self.n_dim            = n_dim
        self.data_size       = data_size
        self.y_target        = y_target
        self.detected_device = X.device

    def __repr__(self):
        s = "Dataset: X (dim=%i), y_target (dim=%i), data_size = %i" % (self.n_dim, 1, self.data_size)
        return s


class MoDataset:
    """
    Contains multiple datasets referring to multiple objects that should obey the same symbolic function.
    """
    def __init__(self, library, multi_X, multi_y_target):
        """

        Parameters
        ----------
        library : library.Library
            Library of choosable tokens.
        multi_X
        multi_y_target
        n_dim : int
            Number of input variables.
        """

        # Checking that multi_X is a list or array_like
        assert isinstance(multi_X, list) or isinstance(multi_X, np.ndarray), "X must be a list or array_like."
        # Checking that multi_y_target is a list or array_like
        assert isinstance(multi_y_target, list) or isinstance(multi_y_target, np.ndarray), "y_target must be a list or array_like."
        # Checking that multi_X and multi_y_target have the same length ie same number of objects
        assert len(multi_X) == len(multi_y_target), "X and y_target must have the same length."

        # Attributes
        self.n_objects = len(multi_X)
        self.datasets = []
        for i in range(self.n_objects):
            self.datasets.append(Dataset(library, multi_X[i], multi_y_target[i]))

        # Checking that all datasets have the same n_dim
        n_dims = [dataset.n_dim for dataset in self.datasets]
        assert len(set(n_dims)) == 1, "All datasets must have the same number of input variables."
        self.n_dim = n_dims[0]
        return None

    def __repr__(self):
        s = "MODataset: n_objects = %i, X(n_dim=%i), y(n_dim=1)\n" % (self.n_objects, self.n_dim)
        for i in range(self.n_objects):
            s += "   Object %i: %s\n" % (i, self.datasets[i].__repr__())
        return s


