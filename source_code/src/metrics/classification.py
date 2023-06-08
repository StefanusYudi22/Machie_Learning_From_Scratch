import numpy as np


def accuracy_score(y_true : np.array, y_pred : np.array) -> float:
    """Calculate Accuracy between prediction value and true value

    Args:
        y_true (np.array) (n,):
            true value with n size
        y_pred (np.array) (n,):
            prediction value with n size 

    Returns:
        float: result of accuracy calculation. ranged from 0 - 1
    """
    # Compute accuracy score
    n_true = np.sum(y_true == y_pred)
    n_total = len(y_true)

    return n_true/n_total
