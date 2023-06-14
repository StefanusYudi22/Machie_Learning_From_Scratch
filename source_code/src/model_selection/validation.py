import numpy as np
import copy

from .split import KFold
from ..metrics import __all__


def cross_val_score(
    estimator,
    X : np.array,
    y : np.array,
    cv : int = 5,
    scoring : str = "mean_squared_error"
) -> (tuple(list, list)):
    """Evaluate model using KFold cross validation

    Args:
        estimator (object): 
            Model object that will be cross validated

        X (np.array) (n,k):
            Data input for model object, with n data point
            and k number of feature

        y (np.array) (n,):
            Data target for model object, with n data point

        cv (int, optional): 
            Amount of fold for cross validation. Defaults to 5.

        scoring (str, optional): 
            Metrics used to calculate the model performance.
            Defaults to "mean_squared_error".

    Returns:
        list, list :
            list of data train evaluation, and data validation
            evaluation
    """
    # Extract data
    X = np.array(X).copy()
    y = np.array(y).copy()
    
    # Split data
    kf = KFold(n_splits=cv)

    scoring = __all__[scoring]
    score_train_list = []
    score_val_list = []
    for _, (ind_train, ind_val) in enumerate(kf.split(X)):
        # Extract data
        X_train = X[ind_train]
        y_train = y[ind_train]
        X_val = X[ind_val]
        y_val = y[ind_val]

        # Create & fit model
        mdl = copy.deepcopy(estimator)
        mdl.fit(X_train, y_train)

        # Predict
        y_pred_train = mdl.predict(X_train)
        y_pred_val = mdl.predict(X_val)

        # Calculate error
        score_train = scoring(y_train, y_pred_train)
        score_test = scoring(y_val, y_pred_val)
        
        # Append
        score_train_list.append(score_train)
        score_val_list.append(score_test)

    return score_train_list, score_val_list
    
