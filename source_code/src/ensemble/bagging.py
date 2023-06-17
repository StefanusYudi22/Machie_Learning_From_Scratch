import copy
import numpy as np

from .base import BaseEnsemble
from ..tree import DecisionTreeClassifier
from ..tree import DecisionTreeRegressor

def _calculate_majority_vote(y : np.array) -> float:
    """Calculate mode from an array

    Args:
        y (np.array):
            data input

    Returns:
        float:
            mode from an array
    """

    # Extract output
    vals, counts = np.unique(y, return_counts = True)

    # Find the majority vote
    ind_max = np.argmax(counts)
    y_pred = vals[ind_max]

    return y_pred

def _calculate_average_vote(y : np.array) -> float:
    """Calculate average of an array

    Args:
        y (np.array):
            data input

    Returns:
        float:
            mean of an array
    """
    y_pred = np.mean(y)
    return y_pred

class BaggingClassifier(BaseEnsemble):
    """
    A Bagging Classifier

    Parameters
    ----------
    estimator : object, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a
        :class:`~ml_from_scratch.tree.DecisionTreeClassifier`.

    n_estimators : int, default=10
        The number of base estimators in the ensemble.

    random_state : int, default=None
        The random seed
    """
    def __init__(
        self,
        estimator=None,
        n_estimators=10,
        random_state=None,
    ):
        # Generate estimator
        if estimator is None:
            estimator = DecisionTreeClassifier()
        
        # Generate the aggregate function for prediction
        self.aggregate_function = _calculate_majority_vote

        # Initialize the others
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            random_state=random_state
        )

class BaggingRegressor(BaseEnsemble):
    """
    A Bagging Regressor

    Parameters
    ----------
    estimator : object, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a
        :class:`~ml_from_scratch.tree.DecisionTreeRegressor`.

    n_estimators : int, default=10
        The number of base estimators in the ensemble.

    random_state : int, default=None
        The random seed
    """
    def __init__(
        self,
        estimator=None,
        n_estimators=10,
        random_state=None,
    ):
        # Generate estimator
        if estimator is None:
            estimator = DecisionTreeRegressor()
        
        # Generate the aggregate function for prediction
        self.aggregate_function = _calculate_average_vote

        # Initialize the others
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            random_state=random_state
        )
    