import numpy as np

# Metrics
def r2(y_target, y_pred, weights=None):
    if weights is None:
        r2 = 1 - ((y_target - y_pred) ** 2).sum() / ((y_target - y_target.mean()) ** 2).sum()
    else:
        mean = np.sum(weights * y_target) / np.sum(weights)
        r2 = 1 - np.sum(weights * (y_target - y_pred) ** 2) / np.sum(weights * (y_target - mean) ** 2)
    return r2

def r2_zero(y_target, y_pred):
    res = r2(y_target, y_pred)
    if res < 0.:
        res = 0.
    return res

def MAE(y_target, y_pred):
    return np.mean(np.abs(y_target - y_pred))

def MSE(y_target, y_pred):
    return ((y_target - y_pred) ** 2).mean()

# Pareto front

def get_pareto_front(x, y):
    """
    Get indices of Pareto front points.
    Parameters
    ----------
    x : array-like
        First objective
    y : array-like
        Second objective
    Returns
    -------
    pareto_indices : list
        Indices of Pareto front points
    """
    pareto_indices = []
    for i in range(len(x)):
        xi, yi = x[i], y[i]
        # Check if any other point is better in both objectives
        is_dominated = False
        for j in range(len(x)):
            if j != i:
                xj, yj = x[j], y[j]
                if (xj <= xi and yj <= yi) and (xj < xi or yj < yi):
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_indices.append(i)
    return pareto_indices