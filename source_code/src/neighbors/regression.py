""" Regressor for NearestNeighbor """

import numpy as np
from .base import NearestNeighbor

class KNeighborsRegressor(NearestNeighbor):
    """KNeighborsRegressor Class and Method Predict

    Args:
        NearestNeighbor (class): Parent Class for Regressor
    """

    def __init__(self,
                 n_neighbors : int = 5,
                 weights : str = 'uniform',
                 distance_type : str = "eucledian"):
        """Initialize NeighborRegressor object

        Args:
            n_neighbors (int, optional): N neighbors fetch from registered data inside the model. Default to 5
            weights (str, optional): uniform weights or non uniform weights. Default to uniform
            distance_type (str, optional): determine distance formula for calculation. eucledian, manhattan. Defaults to eucledian.
        """
        # initialize NearestNeighbor parent class
        super().__init__(
            n_neighbors=n_neighbors,
            distance_type = distance_type,
            weights = weights 
        )

    def predict(self, X_input : np.array) -> np.array:
        """Predict the output from the input data

        Args:
            X (np.array) (k,n): data input for prediction

        Returns:
            np.array (1,k): data output from prediction
        """
        # Convert input to ndarray
        X = np.array(X_input)

        # Calculate weights
        if self.weights == 'uniform':
            # In that case, we do not need the distance to perform
            # the weighting so we do not compute them
            neighbors_index = self.get_neighbors(X, return_distance=False)
            neighbors_distance = None
        else:
            neighbors_distance, neighbors_index = self.get_neighbors(X, return_distance=True)

        weights = self.get_weights(neighbors_distance)

        # Get the prediction
        # create the container for prediction result
        y_pred = np.zeros(X.shape[0])

        if self.weights == 'uniform':
            y_pred = np.mean(self.y_model[neighbors_index], axis=1)
        else:
            y_pred = np.sum(self.y_model[neighbors_index]*weights) / np.sum(weights, axis=1)
        
        return y_pred