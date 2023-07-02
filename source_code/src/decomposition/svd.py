import numpy as np


def _generate_unit_vector(n):
    """
    Function to generate a normally distributed n-vector
    with mean = 0.0 and sigma = 1.0

    Parameters
    ----------
    n : int
        Size of vector

    Returns
    -------
    vec : array-like of shape (n,)
        The generated vector
    """
    # Initialize
    mu = 0.0
    sigma = 1.0

    # Generate the vector
    vec = np.random.normal(mu, sigma, n)

    # Normalize the vector
    vec = vec / np.linalg.norm(vec)

    return vec


class SVD:
    """
    Dimensionality reduction using SVD.
    The eigen is obtained by power method

    Parameters
    ----------
    n_components : int, default=None
        Number of components to keep
        If `None`:
            n_components == n_features

    tol : float, default=1e-10
        The tolerance for power method iterations

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
        n_components=None,
        tol = 1e-10
    ):
        self.n_components = n_components
        self.tol = tol

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
        # Initialize
        n_samples, n_features = X.shape
        change_of_basis = []

        # Iterate
        for i in range(n_features):
            A = X.copy()

            # Generate the data matrix
            for u, sigma, v in change_of_basis[:i]:
                A = A - sigma * np.outer(u, v)

            # Do the power analysis
            u, sigma, v = self._power_iterate(A)

            # Append the results
            change_of_basis.append((u, sigma, v))            

        # Summarize
        U, Sigma, VT = [np.array(x) for x in zip(*change_of_basis)]

        # Extract data
        explained_variance_ = Sigma**2
        total_var = np.sum(explained_variance_)
        explained_variance_ratio_ = explained_variance_ / total_var
        
        # Return information
        self.n_components = n_components
        self.components_ = VT[:n_components]
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]

    def _power_iterate(self, A):
        """
        Recursively compute A^T.A dot x to compute u, v, and sigma

        See more: http://www.cs.yale.edu/homes/el327/datamining2013aFiles/07_singular_value_decomposition.pdf

        Parameters
        ----------
        A : array-like of shape (n_samples, n_features)
            The sample data

        Returns
        -------
        u : array-like of shape (n_samples,)
            The u vector

        sigma : float
            The sigma

        v : array-like of shape (n_features,)
            The v vector
        """
        # Initialize
        n_samples, n_features = A.shape

        # 1. Generate x0 s.t. x0(i) ~ N(0, 1) (step #1)
        x_0 = _generate_unit_vector(n = n_features)

        # 2. Do the iteration (step #2 & #3)
        # Actually we have to find s, but we skip it for now
        # and iterate until converge
        x_prev = None
        x_curr = x_0
        error = np.inf
        while error > self.tol:
            # set
            x_prev = x_curr

            # Update the v0 (step #4)
            x_curr = np.dot(A.T @ A, x_prev)
            
            # Normalize the x_curr
            x_curr = x_curr / np.linalg.norm(x_curr)

            # Calculate error
            error = np.linalg.norm(x_curr - x_prev)
            if error < self.tol:
                break

        # Summarize the results
        # Find SVD decomposition (step #5 - #7)
        v = x_curr / np.linalg.norm(x_curr)
        sigma = np.linalg.norm(A @ v)   # --> ||A.v|| = s ||u|| = s
        u = (A @ v) / sigma             # --> A.v = u.s --> u = A.v/s

        return u, sigma, v
    
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
        # Transformed
        X_transformed = np.dot(X, self.components_.T)

        return X_transformed

    def inverse_transform(self, X):
        """
        Transform data back to its original space.
        
        Remember
        X_transformed = X_original . PC.T

        Then
        X_transformed . PC = X_original . (PC.T . PC)
        X_transformed . PC = X_original
        X_original = X_transformed . PC 

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            The data in PC axis

        Returns
        -------
        X_original : array-like of shape (n_samples, n_features)
            The data in original axis
        """
        X_original = np.dot(X, self.components_)

        return X_original
    
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
