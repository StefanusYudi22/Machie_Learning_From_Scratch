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


class RandomForestClassifier(BaseEnsemble):
    """
    A random forest classifier

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest

    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are exapnded untill
        all leaves are pure.

    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node

    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.

    max_features : {"sqrt", "log2", None}, int, default="sqrt"
        The number of features to consider when looking for the best split
        - If int, then consider `max_features` features at each split
        - If "sqrt", then `max_features = sqrt(n_features)`
        - If "log2", then `max_features = log2(n_features)`
        - If None, then `max_features = n_features`

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        or greater than or equal to this value.

    random_state : int, default=None
        Controls both the randomness of the bootstrapping of the samples
        and the sampling of the features to consider

    """
    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        min_impurity_decrease=0.0,
        random_state=None
    ):
        # Generate estimator
        self.base_estimator = DecisionTreeClassifier(criterion=criterion,
                                                     max_depth=max_depth,
                                                     min_samples_split=min_samples_split,
                                                     min_samples_leaf=min_samples_leaf,
                                                     min_impurity_decrease=min_impurity_decrease)

        # Generate the aggregate function for prediction
        self.aggregate_function = _calculate_majority_vote

        super().__init__(
            estimator=self.base_estimator,
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=random_state
        )

class RandomForestRegressor(BaseEnsemble):
    """
    A random forest classifier

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest

    criterion : {"squared_error", "absolute_error"}, default="squared_error"
        The function to measure the quality of a split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are exapnded untill
        all leaves are pure.

    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node

    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.

    max_features : {"sqrt", "log2", None}, int, default="sqrt"
        The number of features to consider when looking for the best split
        - If int, then consider `max_features` features at each split
        - If "sqrt", then `max_features = sqrt(n_features)`
        - If "log2", then `max_features = log2(n_features)`
        - If None, then `max_features = n_features`

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        or greater than or equal to this value.

    random_state : int, default=None
        Controls both the randomness of the bootstrapping of the samples
        and the sampling of the features to consider

    """
    def __init__(
        self,
        n_estimators=100,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        min_impurity_decrease=0.0,
        random_state=None
    ):
        # Generate estimator
        self.base_estimator = DecisionTreeRegressor(criterion=criterion,
                                                    max_depth=max_depth,
                                                    min_samples_split=min_samples_split,
                                                    min_samples_leaf=min_samples_leaf,
                                                    min_impurity_decrease=min_impurity_decrease)

        # Generate the aggregate function for prediction
        self.aggregate_function = _calculate_average_vote

        super().__init__(
            estimator=self.base_estimator,
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=random_state
        )