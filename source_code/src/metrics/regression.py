import numpy as np


def mean_squared_error(y_true : np.array, y_pred : np.array) -> float:
    """Calculate MSE between prediction result and true value

    Args:
        y_true (np.array) (n,):
            true value with n size
        y_pred (np.array) (n,):
            prediction value with n size 

    Returns:
        float: result of MSE calculation
    """
    loss = np.mean((y_true-y_pred)**2, axis=0)

    return loss