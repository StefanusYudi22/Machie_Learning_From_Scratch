import numpy as np


def _calculate_distance(point_1, point_2, p=2):
    """
    Function to calculate a Minkowski distance
    between two point

    Parameters
    ----------
    point_1 : {array-like} of size (n_features,)
        Point 1

    point_2 : {array-like} of size (n_features,)
        Point 2

    p : int, default=2
        The Minkowski power

    Returns
    -------
    distance : float
        The Minkowski distance of power p
    """
    distance = (np.sum((np.abs(point_1 - point_2))**p))**(1/p)

    return distance

def _get_true_indices(sample):
    """
    Return True (1) indices from a given sample

    Parameters
    ----------
    sample : {array-like} of shape (n_samples,)
        The given sample

    Returns
    -------
    indices : set
        The indices of True or (1) value
    """
    indices = set(np.where(sample==1)[0])

    return indices


class DBSCAN:
    """
    Perform DBSCAN clustering.

    Parameters
    ----------
    eps : float, default=0.25
        The maximum distance between two samples for one to be
        considered as in the neighborhood of the other.

    min_samples : int, default=15
        The number of samples (or total weight) in a neighborhood
        to be considered as a core point.
        This includes the point itself.

    p : float, default=2
        The power of Minkowski distance metric.
        If 
        - `p=1` --> Manhattan distance
        - `p=2` --> Euclidean distance

    Attributes
    ----------
    core_sample_indices_ : {array-like} of shape (n_core_samples,)
        Indices of core sample.

    labels_ : {array-like} of shape (n_samples,)
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1
    """
    def __init__(
        self,
        eps=0.25,
        min_samples=15,
        p=2
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.p = p

    def _region_query(self, p, X):
        """
        Given a point p, we want to find whether other point in X
        is lies withing eps-sized ball.

        Parameters
        ----------
        p : {array-like} of shape (n_features,)
            The centered point

        X : {array-like} of shape (n_samples, n_features)
            The whole sample point

        Returns
        -------
        adj : {array-like} of shape (n_samples,)
            A 0 or 1 list.
            If X[i] is lies within eps-radius centered at p -> 1
            Otherwise -> 0
        """
        # Initialize
        n_samples, n_features = X.shape

        # Start iterating
        adj = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            # Find distance between p & X[i]
            dist_i = _calculate_distance(point_1 = p,
                                         point_2 = X[i],
                                         p = self.p)
            
            # Add distance to adj if it lies within eps
            if dist_i <= self.eps:
                adj[i] = 1

        return adj

    def _find_neighbors(self, X):
        """
        Function to find set of eps-neighbors for every data
        in sample data

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The sample data

        Returns
        -------
        neighbors : list, {array-like}
            The neighbors of each data in sample data
        """
        # Initialize
        n_samples, n_features = X.shape
        neighbors = []

        # Iterate over all data
        for i in range(n_samples):
            # Find point that lies within eps-radius cetered at X[i]
            adj_i = self._region_query(p = X[i], X = X)
            ind_i = _get_true_indices(sample = adj_i)

            # Append the indices of neighbors point to neighbors
            neighbors.append(ind_i)

        return neighbors

    def _find_core_points(self, neighbors):
        """
        Find core point from given neighbors information

        Paramaters
        ----------
        neighbors : list
            The neighbors information

        Returns
        -------
        core_ind : set
            The indices set of core point
        """
        # Initialize
        core_ind = set()

        # Start iterating over all neighbors
        for i, neigh_i in enumerate(neighbors):
            # point_i is a core point if
            # it has minimum points within eps-radius
            if len(neigh_i) >= self.min_samples:
                core_ind.add(i)

        return core_ind

    def _expand_cluster(self, p, neighbors, core_ind, visited, assignment):
        """
        Function to grow/expand cluster --> density reachable

        Parameters
        ----------
        p : {array-like} of shape (n_features,)
            A sample point

        neighbors : list
            The neighbors information of the whole data

        core_ind : set
            The indices set of core points

        visited : set
            The indices set of visited points

        assignment : dict
            The cluster assignment of each visited points
        """
        # Set the reachable points from p
        # We want to update this thus we know the density reachable
        reachable = set(neighbors[p])

        # Iterate
        while reachable:
            # Remove any point q from reachable set, to expand the cluster obviously
            q = reachable.pop()

            # Check if q has not yet been visited
            if q not in visited:
                # Mark q as being visited
                visited.add(q)

                # Check if q also a core point
                if q in core_ind:
                    # Then add all of q neighbors to p reachable set
                    # With this, we can create a "reachability"
                    reachable |= neighbors[q]

                # Finally, check if q is not yet assigned to any cluster
                if q not in assignment:
                    # Then assign q with the same cluster as p
                    assignment[q] = assignment[p]

    def _assignment_to_labels(self, assignment, X):
        """
        Convert the assignment to labels

        Parameters
        ----------
        assignment : dict
            The cluster assignment of each visited points

        X : {array-like} of shape (n_samples, n_features)
            The sample input data

        Returns
        -------
        labels : list of shape (n_samples,)
            The cluster labels
            if -1, then it is an outlier
        """
        # Initialize
        n_samples, _ = X.shape
        labels = -1 * np.ones(n_samples, dtype=int)

        # Start iterating
        for i, cluster_i in assignment.items():
            # Update the value of the labels
            labels[i] = cluster_i

        return labels

    def fit(self, X):
        """
        Fit a DBSCAN clustering

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The sample data

        Returns
        -------
        self : object
            Returns a fitted instance of self
        """
        # Initialize
        X = np.array(X).copy()

        # Find neighbors & core sample indices
        neighbors = self._find_neighbors(X)
        core_ind = self._find_core_points(neighbors)

        # Start the iterations
        assignment = {}
        next_cluster_id = 0
        visited = set()

        # Visit all core point
        for i in core_ind:
            if i not in visited:
                visited.add(i)
                assignment[i] = next_cluster_id
                self._expand_cluster(i, neighbors, core_ind, visited, assignment)
                next_cluster_id += 1

        # Finishing
        self.core_sample_indices_ = core_ind
        self.labels_ = self._assignment_to_labels(assignment, X)
