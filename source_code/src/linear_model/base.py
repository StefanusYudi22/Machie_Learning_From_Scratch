"""base class for linear model"""

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
        
        # Find model parameter using closed form solution
        theta = np.linalg.inv(A.T @ A) @ A.T @ y 

        # Extract model parameters
        if self.fit_intercept:
            self.coef_ = theta[:n_features]
            self.intercept_ = theta[-1]
        else:
            self.coef_ = theta
            self.intercept_ = 0.0

    def predict(self, X):
        X = np.array(X)
        y_pred = np.dot(X,self.coef_) + self.intercept_

        return y_pred