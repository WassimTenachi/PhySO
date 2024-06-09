import torch
import numpy as np
# Internal imports
from physo.physym import token as Tok

def flatten_multi_data (multi_data,):
    """
    Flattens multiple datasets into a single one for vectorized evaluation.
    Parameters
    ----------
    multi_data : list of length (n_realizations,) of torch.tensor of shape (..., [n_samples depends on dataset],)
        List of datasets to be flattened.
    Returns
    -------
    torch.tensor of shape (..., n_all_samples)
        Flattened data (n_all_samples = sum([n_samples depends on dataset])).
    """
    flattened_data = torch.cat(multi_data, axis=-1) # (..., n_all_samples)
    return flattened_data

def unflatten_multi_data (flattened_data, n_samples_per_dataset):
    """
    Unflattens a single data into multiple ones.
    Parameters
    ----------
    flattened_data : torch.tensor of shape (..., n_all_samples)
        Flattened data (n_all_samples = sum([n_samples depends on dataset])).
    n_samples_per_dataset : array_like of shape (n_realizations,) of int
        flattened_data contains multiple datasets with samples of each dataset following each other and each portion of
        flattened_data corresponding to a dataset. n_samples_per_dataset is the number of sample in each portion.
         Eg. [90, 100, 110] for 3 datasets, this will assume that the first 90 samples of flattened_data are for the
         first dataset, the next 100 for the second and the last 110 for the third.
    Returns
    -------
    list of len (n_realizations,) of torch.tensor of shape (..., [n_samples depends on dataset],)
        Unflattened data.
    """
    unflattened_data = list(torch.split(flattened_data, n_samples_per_dataset.tolist(), dim=-1)) # (n_realizations,) of (..., [n_samples depends on dataset],)
    return unflattened_data

def inspect_Xy (X, y):
    """
    Runs assertions and analyzes shape of a single dataset corresponding to a single realization.
    Converts to torch if necessary.
    Parameters
    ----------
    X : torch.tensor of shape (n_dim, data_size,) of float
        Values of the input variables of the problem with n_dim = nb of input variables, and data_size = nb of data points.
    y : torch.tensor of shape (data_size,) of float
        Values of the target symbolic function on input variables contained in X, and data_size = nb of data points.
    Returns
    -------
    n_dim : int
        Number of input variables.
    data_size : int
        Number of data points.
    X : torch.tensor of shape (n_dim, data_size,) of float
    y : torch.tensor of shape (data_size,) of float
    """
    # --- CONVERSION ---
    # Conversion to torch if necessary
    try:
        if not torch.is_tensor(X):
            X = torch.tensor(X)
        if not torch.is_tensor(y):
            y = torch.tensor(y)
    except Exception as e:
        print("X and y must be convertible to torch.tensor.")
        raise e
    # --- ASSERTIONS ---
    assert torch.is_tensor(X), "X must be a torch.tensor"
    assert torch.is_tensor(y), "y must be a torch.tensor"
    # --- ASSERT FLOAT TYPE ---
    assert X.dtype == torch.float64 or X.dtype == torch.float32, "X must contain floats."
    assert y.dtype == torch.float64 or y.dtype == torch.float32, "y must contain floats."
    assert torch.isnan(X).any() == False, "X should not contain any Nans"
    assert torch.isnan(y).any() == False, "y should not contain any Nans"
    # --- ASSERT SHAPE ---
    assert len(X.shape) == 2, "X must have shape = (n_dim, data_size,)"
    assert len(y.shape) == 1, "y must have shape = (data_size,)"
    assert X.shape[1] == y.shape[0], "X must have shape = (n_dim, data_size,) and y must have " \
                                            "shape = (data_size,) with the same data_size."
    n_dim, data_size = X.shape
    return n_dim, data_size, X, y

def inspect_multi_y_weights (multi_y_weights, multi_y):
    """
    Runs assertions and analyzes shape of multi_y_weights.
    Converts to torch if necessary.
    Parameters
    ----------
    multi_y_weights :  list of len (n_realizations,) of torch.tensor of shape (?,) of float
                   or array_like of (n_realizations,) of float
                   or float, optional
        List of y_weights (one per realization). With y_weights being weights to apply to y data.
        Or list of weights one per entire realization.
        Or single float to apply to all.
    multi_y :  list of len (n_realizations,) of torch.tensor of shape (?,) of float
        List of y (one per realization). With y being values of the target symbolic function on input variables
        contained in X.
    Returns
    -------
    processed_multi_y_weights : list of len (n_realizations,) of torch.tensor of shape (?,) of float
        List of y_weights (one per realization). With y_weights being weights to apply to y data.
    """

    # SR task       : y_weights is used       -> [y_weights,]    (A) is transmitted OR not used -> [1,] (B) is transmitted
    # Class SR task : multi_y_weights is used -> [y_weights,...] (A) is transmitted OR not used -> 1.   (C) is transmitted

    # If multi_y_weights is a single float expand to (n_realizations,) which will be further expanded in one by
    # one (C) -> (B)
    n_realizations = len(multi_y)
    if isinstance(multi_y_weights, float) or isinstance(multi_y_weights, (int, np.integer)):
        multi_y_weights = np.full(shape=(n_realizations,), fill_value=float(multi_y_weights))

    # Asserting that multi_y_weights is a list or array_like
    assert isinstance(multi_y_weights, list) or isinstance(multi_y_weights, np.ndarray), "multi_y_weights must be a list or array_like."
    # Asserting that multi_y_weights has the same length ie same number of realizations as multi_y
    assert len(multi_y_weights) == len(multi_y), "multi_y_weights and multi_y must have the same length ie. same number of realizations."

    processed_multi_y_weights = []
    for i in range (len(multi_y_weights)):
        # Expanding weights in cases where multi_y_weights is a (n_realizations,) list of floats (B) -> (A)
        if isinstance(multi_y_weights[i], float) or isinstance(multi_y_weights[i], (int, np.integer)):
            y_weights = torch.full_like(multi_y[i], fill_value=float(multi_y_weights[i]))
        else:
            # If multi_y_weights is a list of shape (n_realizations,) of tensors of shape
            # ([n_samples depends on dataset,]) (A) -> (A)
            assert len(multi_y_weights[i]) == len(multi_y[i]), "multi_y_weights[%i] must have the same length as multi_y[%i] ie. same number of data points." % (i, i)
            y_weights = multi_y_weights[i]
        # Conversion to torch if necessary
        if not torch.is_tensor(y_weights):
            try:
                y_weights = torch.tensor(y_weights)
            except Exception as e:
                print("y_weights must be convertible to torch.tensor.")
                raise e
        # Asserting that y_weights is a torch.tensor
        assert torch.is_tensor(y_weights), "y_weights must be a torch.tensor"
        # Asserting float type
        assert y_weights.dtype == torch.float64 or y_weights.dtype == torch.float32, "y_weights must contain floats."
        assert torch.isnan(y_weights).any() == False, "y_weights should not contain any Nans"
        # Appending the processed y_weights
        processed_multi_y_weights.append(y_weights)
    return processed_multi_y_weights

class Dataset:
    """
    Contains a dataset and runs assertions.
    Converts to torch if necessary.
    """
    def __init__(self, multi_X, multi_y, multi_y_weights=1., library=None):
        """
        Parameters
        ----------
        multi_X : list of len (n_realizations,) of torch.tensor of shape (n_dim, ?,) of float
            List of X (one per realization). With X being values of the input variables of the problem with n_dim = nb
            of input variables.
        multi_y :  list of len (n_realizations,) of torch.tensor of shape (?,) of float
            List of y (one per realization). With y being values of the target symbolic function on input variables
            contained in X.
        multi_y_weights :  list of len (n_realizations,) of torch.tensor of shape (?,) of float
                           or array_like of (n_realizations,) of float
                           or float, optional
            List of y_weights (one per realization). With y_weights being weights to apply to y data.
            Or list of weights one per entire realization.
            Or single float to apply to all (for default value = 1.).
        library : library.Library or None, optional
            Library of choosable tokens. This is used for assertions, some assertions are not performed if library is
            None.
        """
        self.library = library

        # ---------------------- ASSERTIONS ----------------------

        # Asserting that multi_X is a list or array_like
        assert isinstance(multi_X, list) or isinstance(multi_X, np.ndarray), "multi_X must be a list or array_like."
        # Asserting that multi_y is a list or array_like
        assert isinstance(multi_y, list) or isinstance(multi_y, np.ndarray), "multi_y must be a list or array_like."
        # Asserting that multi_X and multi_y have the same length ie same number of realizations
        assert len(multi_X) == len(multi_y), "multi_X and multi_y must have the same length ie. same number of realizations."

        # Converting multi_X and multi_y to lists in case they are np.array (as slicing torch.tensor into np.array
        # will convert them back to np.array)
        if isinstance(multi_X, np.ndarray):
            multi_X = multi_X.tolist()
        if isinstance(multi_y, np.ndarray):
            multi_y = multi_y.tolist()

        # Saving the number of realizations
        self.n_realizations = len(multi_X)

        # Asserting each dataset
        list_of_n_samples = np.full(self.n_realizations, -1, dtype=int)
        list_of_n_dim     = np.full(self.n_realizations, -1, dtype=int)
        for i in range(self.n_realizations):
            n_dim, data_size, X, y = inspect_Xy(X=multi_X[i], y=multi_y[i])
            multi_X           [i] = X
            multi_y           [i] = y
            list_of_n_samples [i] = data_size
            list_of_n_dim     [i] = n_dim

        # Checking that all realizations have the same n_dim
        assert (list_of_n_dim == list_of_n_dim[0]).all(), "Each X in multi_X must have the same first dimension (n_dim) ie. all realizations must have the same number of input variables."
        n_dim = list_of_n_dim[0]
        # Checking that all tokens in the library have id < n_dim
        if self.library is not None:
            # Is id var_id wrong : mask.
            # Ie. var_type is that of input var AND id >= n_dim
            mask_wrong_id = np.logical_and(library.var_type == Tok.VAR_TYPE_INPUT_VAR, library.var_id >= n_dim)
            assert mask_wrong_id.sum() == 0, "Can not access input variable data X by X[var_id] of tokens :" \
                                             "\n %s\n as they have out of range var_id >= X.shape[0] = n_dim = %i," \
                                             " var_id :\n %s" % (library.lib_name[mask_wrong_id], n_dim, library.var_id [mask_wrong_id])
        # Saving the number of input variables
        self.n_dim = n_dim

        # Saving the number of data points
        self.n_samples_per_dataset = list_of_n_samples
        self.n_all_samples         = self.n_samples_per_dataset.sum()

        # Handling multi_y_weights (must be done before device detection)
        multi_y_weights = inspect_multi_y_weights (multi_y_weights=multi_y_weights, multi_y=multi_y)

        # Device detection
        list_of_devices = np.array([X.device for X in multi_X] + [y.device for y in multi_y] + [y_weights.device for y_weights in multi_y_weights])
        assert (list_of_devices[0] == list_of_devices).all(), "All X, y and y_weights datasets must be on the same device."
        # Saving the detected device
        self.device = list_of_devices[0]

        # Saving the datasets
        self.multi_X         = multi_X                                  # (n_realizations) of (n_dim, [n_samples depends on dataset])
        self.multi_y         = multi_y                                  # (n_realizations) of ([n_samples depends on dataset])
        self.multi_y_weights = multi_y_weights                          # (n_realizations) of ([n_samples depends on dataset])

        # Saving the flattened datasets
        self.multi_X_flatten         = flatten_multi_data(self.multi_X)          # (n_dim, n_all_samples)
        self.multi_y_flatten         = flatten_multi_data(self.multi_y)          # (n_all_samples,)
        self.multi_y_weights_flatten = flatten_multi_data(self.multi_y_weights)  # (n_all_samples,)

        return None

    @property
    def detected_n_realizations(self):
        return self.n_realizations

    @property
    def detected_device(self):
        return self.device

    def to(self, device):
        """
        Send all values to device.
        """
        for i in range(self.n_realizations):
            self.multi_X[i] = self.multi_X[i].to(device)
        for i in range(self.n_realizations):
            self.multi_y[i] = self.multi_y[i].to(device)
        for i in range(self.n_realizations):
            self.multi_y_weights[i] = self.multi_y_weights[i].to(device)
        self.device = device
        return None

    def __repr__(self):
        s = "Dataset: n_realizations=%i, X(n_dim=%i), y(n_dim=1)\n" % (self.n_realizations, self.n_dim)
        for i in range(self.n_realizations):
            s += " -> Realization %i (n_samples = %i)\n" % (i, self.n_samples_per_dataset[i])
        return s
