import copy
import numpy as np
from typing import Any

# max number of int32 type
MAX_INT = np.iinfo(np.int32).max

def _generate_random_seed(random_state : int) -> int:
    """Generate random seed

    Args:
        random_state (int):
            The condition of random_state {'None' or 'int}

    Returns:
        int:
            Seed number
    """
    if random_state is None:
        seed = np.random.randint(0, MAX_INT)
    else:
        seed = random_state

    return seed

def _generate_ensemble_estimators(base_estimator : object,
                                  n_estimators : int) -> np.array :
    """Generate array of object estimator

    Args:
        base_estimator (object):
            model or estimator object

        n_estimators (int):
            how much the model will be 
            generated

    Returns:
        np.array:
            array of model or estimator object
    """
    # copy the estimator
    # into an array of estimator

    estimators = [copy.deepcopy(base_estimator) for i in range(n_estimators)]

    return estimators

def _generate_sample_indices(seed : int,
                             n_estimators : int,
                             n_population_data : int,
                             n_samples : int,
                             bootstrap : bool = True) -> np.array:
    """Generate the bootstraped sample indices

    Args:
        seed (int):
            random seed for reproducibility

        n_estimators (int):
            the number of object model or estimator

        n_population (int):
            the number of maximum data available

        n_samples (int):
            the number of samples to generate in 
            each bootstraped samples

        bootstrap (bool, optional):
            sampling with replacement condition
            True with replacement, and false without
            replacement. Default to True

    Returns:
        np.array (n_estimator, n_samples):
            index of bootstrapped sample for each estimator
            for n_samples amount of index.
    """

    # Get the seed
    np.random.seed(seed)

    # Get the bagging indices
    sample_indices = np.random.choice(n_population_data,
                                      size = (n_estimators, n_samples),
                                      replace = bootstrap)
    
    return sample_indices

def _generate_feature_indices(seed : int,
                              n_estimators : int,
                              n_population_feature : int,
                              n_features : int,
                              bootstrap=False) -> np.array :
    """Generate the Bootstrapped sample indices

    Args:
        seed (int):
            random seed for reproducibility

        n_estimators (int):
            the number of object model or estimator

        n_population_feature (int):
            the number of maximum feature available

        n_features (int):
            the number of feature from the dataset

        bootstrap (bool, optional):
            sampling with replacement condition
            True with replacement, and false without
            replacement. Default to False

    Returns:
        np.array: 
            index of bootstrapped sample for each estimator
            for n_sfeatures amount of index.

    """
    np.random.seed(seed)

    # Get the bagging indices
    feature_indices = np.empty((n_estimators, n_features), dtype="int")
    for i in range(n_estimators):
        feature_indices[i] = np.random.choice(n_population_feature, 
                                              n_features, 
                                              replace=bootstrap)
        feature_indices[i].sort()

    return feature_indices

def _predict_ensemble(estimators : np.array,
                      feature_indices : np.array,
                      X : np.array) -> np.array :
    """Predict X input using respective object model or estimator

    Args:
        estimators (np.array) (,m):
            array of model or estimator object with
            m object

        feature_indices (np.array) (m,n):
            array of feature used for every model
            or estimator object, with m number of model 
            or estimator object and n number of feature

        X (np.array) (k, j):
            data input for prediction with k number of
            data point and j number of data feature

    Returns:
        np.array (m,k):
            prediction result from every model object
            with k number of data point
    """

    # Prepare the data
    X = np.array(X).copy()
    n_samples = X.shape[0]

    # Prepare the ensemble model
    n_estimators = len(estimators)

    # Create the output
    y_preds = np.empty((n_estimators, n_samples))

    # Fill the output with the given ensemble model
    for i, estimator in enumerate(estimators):
        # Extract the estimators
        X_ = X[:, feature_indices[i]]

        # Get the predictions
        y_preds[i] = estimator.predict(X_)

    return y_preds

def _predict_aggregate(y_ensemble : np.array,
                       aggregate_func) -> Any:
    """Aggregate the result of predict ensemble
 
    Args:
        y_ensemble (np.array) (m,k) :
            calculation result from function
            predict_ensemble

        aggregate_func (function):
            aggregage_func used for ensembling

    Returns:
        Any (,k):
            Aggregate value prediction from every
            model object with k number of prediction
    """

    # Extract the predicted data
    n_estimators, n_samples = y_ensemble.shape

    # Find the majority vote for each samples
    y_pred = np.empty(n_samples)

    # loop over each result from 
    # every model object
    for i in range(n_samples):
        # Extract the ensemble results on
        y_samples = y_ensemble[:, i]

        # Predict the aggregate
        y_pred[i] = aggregate_func(y_samples)

    return y_pred

class BaseEnsemble:
    """
    Base class for Ensemble Model
    """
    def __init__(
        self,
        estimator,
        n_estimators,
        max_features=None,
        random_state=None
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X : np.array, y : np.array):
        """
        Build a Ensemble of estimators from the training set (X, y)

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The training input samples.

        y : {array-like} of shape (n_samples,)
            The target values.
            - Class labels in classification
            - Real number in regression

        Returns
        -------
        self : object
            Fitted estimator
        """
        # Convert data
        X = np.array(X).copy()
        y = np.array(y).copy()

        # Extract number of samples & features
        self.n_samples, self.n_features = X.shape

        # Generate the Ensemble estimators
        self.estimators_ = _generate_ensemble_estimators(base_estimator = self.estimator,
                                                         n_estimators = self.n_estimators)
        
        # Generate the random seed
        seed = _generate_random_seed(random_state = self.random_state)

        # Generate the ensemble sample indices
        sample_indices = _generate_sample_indices(seed = seed,
                                                  n_estimators = self.n_estimators,
                                                  n_population_data = self.n_samples,
                                                  n_samples = self.n_samples,
                                                  bootstrap = True)
        
        # set max feature for bootstraping
        if isinstance(self.max_features, int):
            max_features = self.max_features
        elif self.max_features == "sqrt":
            max_features = int(np.sqrt(self.n_features))
        elif self.max_features == "log2":
            max_features = int(np.log2(self.n_features))
        else:
            max_features = self.n_features

        # Generate the ensemble feature indices
        self.feature_indices = _generate_feature_indices(seed = seed,
                                                         n_estimators = self.n_estimators,
                                                         n_population_feature = self.n_features,
                                                         n_features = max_features,
                                                         bootstrap = False)
        
        # Fit the model
        # loop for every estimator in array
        for b in range(self.n_estimators):
            # Get the bootstrapped features
            X_bootstrap = X[:, self.feature_indices[b]]

            # Get the bootstrapped samples
            X_bootstrap = X_bootstrap[sample_indices[b], :]
            y_bootstrap = y[sample_indices[b]]

            
            # Fit the model from the bootstrapped sample
            estimator = self.estimators_[b]
            estimator.fit(X_bootstrap, y_bootstrap)

    def predict(self, X : np.array) -> Any :
        """Predict the outcome from input data

        Args:
            X (np.array) (n,m):
                Input data for prediction with n number
                of data point and m number of feature

        Returns:
            Any (n,):
                Return from fitted model, any 
                datatype based on the label y
                with n number of prediction
        """

        # Predict the ensemble
        y_pred_ensemble = _predict_ensemble(estimators = self.estimators_,
                                            feature_indices = self.feature_indices,
                                            X = X)
        
        # Aggregate the ensemble
        y_pred = _predict_aggregate(y_pred_ensemble,
                                    self.aggregate_function)

        return y_pred