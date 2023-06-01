import numpy as np

class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        # Prepare data
        X = np.array(X).copy()
        y = np.array(y).copy()

        # Extract Size
        n_samples, n_features = X.shape

        # Create the design matrix, A
        if self.fit_intercept:
            # Create A
            A = np.column_stack((X, np.ones(n_samples)))
        else:
            # Create A
            A = X
        
        # Solve for model parameters, theta
        theta = np.linalg.inv(A.T @ A) @ A.T @ y 