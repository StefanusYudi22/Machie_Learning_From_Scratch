import numpy as np


class PCA:
    """
    Principal component analysis (PCA)

    A linear dimensionality reduction of the data
    to project it to a lower dimensional space.

    Parameters
    ----------
    n_components : int, default=None
        Number of components to keep
        If `None`:
            n_components == n_features

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data.

    explained_variance_ : ndarray of shape (n_components,)
        The amount of variance explained by each of the selected components.

    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.
    """
    def __init__(
        self,
        n_components=None
    ):
        self.n_components = n_components

    def fit(self, X):
        """
        Fit the model with X

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        """
        # Initialize
        X = np.array(X).copy()
        n_samples, n_features = X.shape
        
        # Handle the number of components
        if self.n_components is None:
            n_components = n_features
        else:
            n_components = self.n_components

        self._fit(X, n_components)

    def _fit(self, X, n_components):
        """
        Fit the model

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        n_components : int
            The number of principal components to choose
        """
        # START FITTING
        # 1. Build the covariance matrix
        covariance_matrix = self._calculate_covariance_matrix(X)

        # 2. Decompose the covariance matrix
        eig_val, eig_vec = np.linalg.eig(covariance_matrix)

        # 3. Sort the eigen vector based on eigen values in descending order
        sorted_ind = eig_val.argsort()[::-1]
        eig_val_sorted = eig_val[sorted_ind]
        eig_vec_sorted = eig_vec[:, sorted_ind]

        # 4. Get the decomposition information
        explained_variance_ = eig_val_sorted
        total_var = np.sum(explained_variance_)
        explained_variance_ratio_ = explained_variance_ / total_var

        # 5. Return information
        self.n_components = n_components
        self.components_ = eig_vec_sorted[:, :n_components].T
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]


    def _calculate_covariance_matrix(self, X):
        """
        Fit the model

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        Returns
        -------
        cov_mat : ndarray of shape (n_features, n_features)
            The covariance matrix
        """
        # Initialize
        n_samples, n_features = X.shape

        # 1. Centered the features data (X_i - mean(X_i))
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. Build the covariance matrix
        cov_mat = np.cov(X_centered, rowvar=False)

        return cov_mat

    def transform(self, X):
        """
        Apply dimensionality reduction to X

        X is projected to the PC

        Parameters
        -----------
        X : array-like of shape (n_samples, n_features)
            The data to be projected

        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_components)
            The projection of X in the PC
        """
        # Centered the data first
        if self.mean_ is not None:
            X = X - self.mean_

        # Transformed
        X_transformed = np.dot(X, self.components_.T)

        return X_transformed
    
    def fit_transform(self, X):
        """
        Fit and transform X to the principal components

        Parameters
        -----------
        X : array-like of shape (n_samples, n_features)
            The data to be projected

        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_components)
            The projection of X in the PC
        """
        # Fit the model
        self.fit(X)

        # Transform the model
        X_transformed = self.transform(X)

        return X_transformed
    
    def inverse_transform(self, X):
        """
        Transform data back to its original space.
        
        Remember
        X_transformed = (X_original - mean) . PC.T

        Then
        X_transformed . PC = (X_original - mean) . (PC.T . PC)
        X_transformed . PC = X_original - mean
        X_original = X_transformed . PC + mean

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            The data in PC axis

        Returns
        -------
        X_original : array-like of shape (n_samples, n_features)
            The data in original axis
        """
        X_original = np.dot(X, self.components_) + self.mean_

        return X_original
