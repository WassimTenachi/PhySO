import numpy as np

# Metrics
def r2(y_target, y_pred):
    return 1 - ((y_target - y_pred) ** 2).sum() / ((y_target - y_target.mean()) ** 2).sum()

def r2_zero(y_target, y_pred):
    res = r2(y_target, y_pred)
    if res < 0.:
        res = 0.
    return res

def MAE(y_target, y_pred):
    return np.mean(np.abs(y_target - y_pred))

def MSE(y_target, y_pred):
    return ((y_target - y_pred) ** 2).mean()