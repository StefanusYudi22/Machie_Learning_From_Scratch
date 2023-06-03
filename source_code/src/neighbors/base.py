""" Base for Nearest Neighbors"""

import numpy as np

class NearestNeighbor:
    """Base NearestNeighbor Class
    """

    def __init__(self,
                 n_neighbors : int =5,
                 distance_type : str = "eucledian",
                 weights : str = "uniform") -> None:
        """Initialize NearestNeighbor object

        Args:
            n_neighbors (int, optional): How many neighbors will be fetch from the registered data inside the model object. Default to 5
            distance_type (str, optional): Determine distance formula for calculation. eucledian, manhattan. Defaults to eucledian.
        """
        self.n_neighbors = n_neighbors
        self.distance_type = distance_type
        self.weights = weights

    def compute_distance(self,
                         X_input : np.array,
                         X_model : np.array) -> np.array :
        """Compute distance between data input and registered data

        Args:
            X_input (np.array) (1,n): Input data array
            X_model (np.array) (m,n): Registered data array

        Returns:
            np.array (1,m): Distance of X_input to every registered data
        """
        if self.distance_type == "manhattan":
            distance = np.sum(np.abs(X_model-X_input), axis=1)
        else :
            distance = np.linalg.norm(X_model - X_input, axis=1)

        return distance
    
    def get_neighbors(self,
                   X_input : np.array,
                   return_distance : str = True) -> np.array:
        """Method to get neighbors index from registered data in model object

        Args:
            X (np.array) (k,n) : Data Input for searching neighbors
            return_distance (str, optional): State return the result of distance calculation or not. Defaults to True.

        Returns:
            np.array (k, n_neighbors): matrix of neighbors data
        """
        # Check for input dimension

        # alter the input dimension if X_input dimension is like (m,)
        try :
            X_input.shape[1]
        except IndexError:
            X_input = X_input.reshape(-1,1)

        assert self.X_dim == X_input.shape[1], "input dimension for X differ with registered data"

        # Create the container for distance matrix
        n_data_input = X_input.shape[0]
        n_data_register = self.X_model.shape[0]
        matrix_distance = np.zeros((n_data_input, n_data_register))

        # Loop for every data requested
        for index, input in enumerate(X_input):
            # calculate the distance between input and every registered data
            matrix_distance[index] = self.compute_distance(X_input = input, X_model = self.X_model)

        # sort the distance, ascending order and extract the n neighbor data
        neighbors_index = np.argsort(matrix_distance, axis=1)[:, :self.n_neighbors]

        # return calculated distance and index
        if return_distance:
            neighbors_distance = np.sort(matrix_distance, axis=1)[:, :self.n_neighbors]
            return neighbors_distance, neighbors_index
        # return calculated index
        else:
            return neighbors_index

    def fit(self, X : np.array, y : np.array):
        """Register data into model object

        Args:
            X (np.array) (m,n): Array for input
            y (np.array) (1,m): Array for output 
        """
        try:
        # check for data train with incomplete dimension (m,)
            np.array(X).shape[1]
            # create the design matrix
            self.X_model = np.array(X)
            self.y_model = np.array(y)
        except IndexError:
        # create addtional dimension for registered data with incomplete dimension
            self.X_model = np.array(X).reshape(-1,1)
            self.y_model = np.array(y)

        # create input criteria
        try:
            # take the count feature n from registered matrix size (m,n)
            self.X_dim = self.X_model.shape[1]
        except IndexError:
            # set input criteria to 1 if matrix size is (m,)
            self.X_dim = 1

    def get_weights(self, distance_matrix : np.array) -> np.array:
        """Calculate weights from distance data

        Args:
            distance_matrix (np.array) (k,n_neighbors): distance from data input to neighbors 

        Returns:
            np.array (k,n_neighbors): weights based on distance  
        """
        if self.weights == 'uniform':
            return None
        else:
            return 1.0/(distance_matrix**2)
