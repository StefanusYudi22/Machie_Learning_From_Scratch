""" Classifier with Nearest Neighbors """

import numpy as np

from .base import NearestNeighbor

class KNeighborsClassifier(NearestNeighbor):
    """KNeighborsClassifier Class and Method Predict

    Args:
        NearestNeighbor (class): Parent Class for Regressor
    """

    def __init__(self,
                 n_neighbors : int = 5,
                 weights : str = 'uniform',
                 distance_type : str = "eucledian"):
        """Initialize NeighborRegressor object

        Args:
            n_neighbors (int, optional): How many neighbors will be fetch from the registered data inside the model object. Default to 5
            weights (str, optional): uniform weights or non uniform weights. Default to uniform
            distance_type (str, optional): determine distance formula for calculation. eucledian, manhattan. Defaults to eucledian.
        """
        # initialize NearestNeighbor parent class
        super().__init__(
            n_neighbors=n_neighbors,
            distance_type = distance_type,
            weights = weights
        )

    def predict_proba(self, X_input : np.array) -> np.array:
        """Calculate every neighbors probability

        Args:
            X (np.array) (k,n): data input for prediction

        Returns:
            np.array (k,n_neighbors): data output from prediction
        """
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

        # Get the label data using index
        neighbors_y = self.y_model[neighbors_index]

        # take unique class from the matrix label
        self.classes = np.unique(neighbors_y)

        # create container for unique class probability
        count_classes = len(self.classes)
        count_data = X.shape[0]
        matrix_proba = np.ones((count_data, count_classes))

        # iterate over input data neighbors
        for i in range(count_data):
            # Exctract neighbor output
            neigh_y_i = neighbors_y[i]

            # Iterate over class
            for j, class_ in enumerate(self.classes):

                # Calculate the I(y = class) for every neighbors
                i_class = (neigh_y_i == class_).astype(int)

                # Calculate the class counts or weighted count
                if self.weights == 'uniform':
                    class_counts_ij = np.sum(i_class)
                else:
                    weights_i = weights[i]
                    class_counts_ij = np.dot(weights_i, i_class)

                # Append
                matrix_proba[i, j] = class_counts_ij

        # Normalize counts --> get probability
        for i in range(count_data):
            sum_i = np.sum(matrix_proba[i])
            matrix_proba[i] /= sum_i

        return matrix_proba
        
    def predict(self, X : np.array) -> np.array:
        """Predict class output from data input

        Args:
            X (np.array) (k,n): data input 

        Returns:
            np.array (1,k): class output
        """
        # Predict neighbor probability
        matrix_proba = self.predict_proba(X)

        # Predict y
        index_max = np.argmax(matrix_proba, axis=1)
        y_pred = self.classes[index_max]

        return y_pred
            