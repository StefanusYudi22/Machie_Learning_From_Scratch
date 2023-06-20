import numpy as np


MAX_INT = np.iinfo(np.int32).max

def _euclidean_distance(point_1, point_2):
    """
    Calculate the Euclidean distance between two points

    Parameters
    ----------
    point_1 : {array-like} of size (n_features,)
        point-1

    point-2 : {array-like} of size (n_features,)
        point-2

    Returns
    -------
    dist : float
        The Euclidean distance between point-1 and point-2
    """
    dist = np.linalg.norm(point_1 - point_2)

    return dist

def _generate_random_seed(random_state):
    """
    Generate the random seed

    Parameters
    -----------
    random_state : int
        The condition of random_state {`None` or `int`}

    Returns
    -------
    seed : int
        The random seed
    """
    if random_state is None:
        seed = np.random.randint(0, MAX_INT)
    else:
        seed = random_state

    return seed


class KMeans:
    """
    K-Means clustering

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of 
        centroids to generate

    init : {'k-means++', 'random'}, default='k-means++'
        Initialization of the centroid
        - if `init='k-means++'`, we use k-means++ initialization
        - else we use random initialization

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run

    tol : float, default=1e-4
        Relative tolerance with regards to 

    random_state : int, default=None
        Random state for random number generation for centroid initialization
    """
    def __init__(
        self,
        n_clusters=8,
        init="k-means++",
        max_iter=300,
        tol=1e-4,
        random_state=None
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _init_cluster_centers(self, X):
        """
        Initialize centroids / cluster centers

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training instances to cluster

        Returns
        -------
        cluster_centers : {array-like} of shape (n_clusters, n_features)
            The cluster centers
        """
        # Generate random seed for centroid initialization
        seed = _generate_random_seed(random_state = self.random_state)
        np.random.seed(seed)

        # Check the initialization scheme
        if self.init == "random":
            # We initialize randomly
            cluster_centers_ind = np.random.choice(self.n_samples,
                                                   size = self.n_clusters,
                                                   replace = False)
            
            cluster_centers = X[cluster_centers_ind]        
        else:
            # We initialize using K-Means++
            # Initialization
            cluster_centers = []
            dist_sample = np.zeros(self.n_samples)
            proba_sample = np.zeros(self.n_samples)

            # Choose c1 randomly from data
            c_0_ind = np.random.choice(self.n_samples)
            cluster_centers.append(X[c_0_ind])

            # Choose c2, c3, etch
            for j in range(1, self.n_clusters):
                # Compute distance between data to the closest available centroid
                for i, sample_i in enumerate(X):
                    # Find closest cluster centers
                    ind_i = self._closest_cluster_centers(sample_i,
                                                          cluster_centers)
                    cluster_centers_i = cluster_centers[int(ind_i)]

                    # Find the distance
                    dist_i = np.linalg.norm(sample_i-cluster_centers_i)

                    # Append
                    dist_sample[i] = dist_i

                # Compute probability of a point
                proba_sample = (dist_sample**2) / np.sum(dist_sample**2)

                # Generate random cluster centers based on the probability
                c_j_ind = np.random.choice(self.n_samples, p=proba_sample)

                # Append the cluster centers
                cluster_centers.append(X[c_j_ind])

            cluster_centers = np.array(cluster_centers)
                

        return cluster_centers

    def _closest_cluster_centers(self, sample, cluster_centers):
        """
        Return the index of the closest centroid (cluster centers) to the sample

        Parameters
        ----------
        sample : {array-like} of shape (1, n_features)
            Data input

        cluster_centers : {array-like} of shape (n_clusters, n_features)
            The cluster centers

        Returns
        -------
        closest_ind : int
            Index of closest centroid
        """
        closest_i = 0
        closest_dist = float('inf')

        for i, cluster_center in enumerate(cluster_centers):
            # Calculate distance
            distance_i = _euclidean_distance(point_1 = sample,
                                             point_2 = cluster_centers[i])
            
            # Check for the minimal distance
            if distance_i < closest_dist:
                closest_dist = distance_i
                closest_i = i

        return closest_i
    
    def _assign_clusters(self, X, cluster_centers):
        """
        Assign the samples to the closest centroids

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training instances to cluster

        cluster_centers : {array-like} of shape (n_clusters, n_features)
            The cluster centers

        Returns
        -------
        labels : {array-like} of shape (n_samples,)
            The cluster assignment
        """
        labels = np.zeros(self.n_samples)
        for i, sample_i in enumerate(X):
            # Find the closest cluster centers of the i-th sample
            cluster_centers_i = self._closest_cluster_centers(sample = sample_i,
                                                              cluster_centers = cluster_centers)
            
            # Assign the cluster
            labels[i] = cluster_centers_i

        return labels

    def _calculate_cluster_centers(self, X, cluster_centers, labels):
        """
        Calculate the cluster centers by taking its cluster mean

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training instances to cluster

        cluster_centers : {array-like} of shape (n_clusters, n_features)
            The cluster centers

        labels : {array-like} of shape (n_samples,)
            The cluster assignment

        Returns
        -------
        cluster_centers : {array-like} of shape (n_clusters, n_features)
            The cluster centers
        """
        cluster_centers = cluster_centers.copy()

        for i in range(self.n_clusters):
            # Filter the member of cluster
            X_i = X[labels==i]

            # Find the average of cluster member
            cluster_centers[i] = np.mean(X_i, axis=0)
        
        return cluster_centers

    def _calculate_inertia(self, X, cluster_centers, labels):
        """
        Calculate the inertia
            sum of squared distances of samples
            to their closest cluster center


        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training instances to cluster

        cluster_centers : {array-like} of shape (n_clusters, n_features)
            The cluster centers

        labels : {array-like} of shape (n_samples,)
            The cluster assignment

        Returns
        -------
        inertia : float
            The inertia
        """
        dist_sample = np.zeros(self.n_samples)

        for i, sample_i in enumerate(X):
            # Obtain closest distance
            dist_i = _euclidean_distance(point_1 = sample_i,
                                         point_2 = cluster_centers[int(labels[i])])
            
            # Append
            dist_sample[i] = dist_i

        # Calculate the inertia
        inertia = np.sum(dist_sample**2)

        return inertia

    def fit(self, X):
        """
        Compute k-means clustering

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training instances to cluster

        Returns
        -------
        self : object
            Fitted estimator
        """
        # Validate X
        X = np.array(X).copy()
        self.n_samples, self.n_features = X.shape

        # Subtract of mean of x for more accurate distance computation
        #X_mean = X.mean(axis=0)
        #X -= X_mean

        # Initialization centroids
        cluster_centers_ = self._init_cluster_centers(X)

        # Start learning
        for i in range(self.max_iter):
            # Assign samples to the closest centroids (cluster centers)
            labels_ = self._assign_clusters(X,
                                            cluster_centers_)
            
            # Save current cluster centers for convergence check
            prev_cluster_centers = cluster_centers_.copy()

            # Calculate new cluster centers
            cluster_centers_ = self._calculate_cluster_centers(X,
                                                               cluster_centers_,
                                                               labels_)
            
            # Calculate inertia
            inertia_ = self._calculate_inertia(X,
                                               cluster_centers_,
                                               labels_)
            
            # If no cluster centers have changed => convergence
            diff = np.linalg.norm(cluster_centers_ - prev_cluster_centers, axis=0)
            if all(diff < self.tol):
                 break
        
        # Summarize
        self.labels_ = labels_
        self.cluster_centers_ = cluster_centers_
        self.inertia_ = inertia_

    def predict(self, X):
        """
        Predict the colsest cluster each sample in X belong to.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            New data to predict

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to
        """
        X = np.array(X).copy()
        n_samples = X.shape[0]

        # Predict
        labels = np.zeros(n_samples)
        for i, X_i in enumerate(X):
            # Predict cluster
            cluster_i = self._closest_cluster_centers(sample = X_i, 
                                                      cluster_centers = self.cluster_centers_)
            
            # Append
            labels[i] = cluster_i

        return labels
