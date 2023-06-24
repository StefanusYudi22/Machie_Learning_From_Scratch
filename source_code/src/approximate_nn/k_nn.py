import numpy as np

class KNearestNeighbor:
    """KNearestNeighbor Class
    """

    def __init__(self,
                 distance_type : str = "eucledian") -> None:
        """KNN Initialization
        Args:
            distance_type (str, optional): 
                Distance algorithm to rank the neighbor.. Defaults to "eucledian".
        """
        self.distance_type = distance_type

    def fit(self,
            X : np.array,
            y : np.array):
        """ Register data into KNN class

        Args:
            X (np.array) (m,n):
                Feature of data train with m data point
                and n feature

            y (np.array) (m,):
                Label of data train, with m data label
        """
        # Pass the data by value
        self.X = np.array(X).copy()
        self.y = np.array(y).copy()
        
    def find_similar_items(self,
                           X_in : np.array,
                           n_items : int = 10) -> np.array:
        """Find similar item within forest of tree

        Args:
            X_in (np.array) (k, ): 
                Data input into the environment with 1
                data point and k number of feature

            n_items (int, optional): 
                How many similar item will be querried
                from the registered data. Defaults to 10.

        Returns:
            np.array (n_items,) :
                array of similar item with counts n_items
        """
        # search in registered data
        ranked_index = self.rank_neighbors(X_in, self.X)
        
        if len(ranked_index) < n_items:
            return self.y[ranked_index]
        else:
            return self.y[ranked_index[:n_items]]

    def rank_neighbors(self,
                       X_in : np.array,
                       X_env : np.array) -> np.array :
        """Method to compute distance between data input
           and registered data in a leaf, then rank the 
           environment

        Args:
            X_in (np.array) (k,):
                Data input into the environment with 1
                data point and k number of feature
            X_env (np.array) (n,k):
                Data registered in the environment with
                n data point registered in the environment
                and k number of feature

        Returns:
            np.array (n,): 
                array of index of nearest item calculated 
                between the input and environment data 
                with size n data index
        """
        
        if self.distance_type == "eucledian":
            distance = np.linalg.norm(X_env - X_in, axis=1)
            rank_index = np.argsort(distance)

        elif self.distance_type == "manhattan":
            distance = np.sum(np.abs(X_env - X_in), axis=1)
            rank_index = np.argsort(distance)

        elif self.distance_type == "cosine-similarity":
            distance = np.squeeze(X_env @ X_in.T)/(np.linalg.norm(X_env, axis=1)*np.linalg.norm(X_in))
            rank_index = np.argsort(-distance)

        else:
            distance = np.squeeze(X_env @ X_in.T)
            rank_index = np.argsort(-distance)

        return rank_index
