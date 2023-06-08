import numpy as np

from .base import LinearRegression

class Lasso(LinearRegression):
    
    def __init__(self,
                 fit_intercept : str = True,
                 optimizer : str = "ols",
                 learning_rate : float = 1e-5,
                 num_iters : int = 10000,
                 max_iter_lasso : int = 1000,
                 cost_tolerance : float = 1e-4, 
                 lamda : float = 1.0) :
        super().__init__(
            fit_intercept = fit_intercept,
            optimizer = optimizer,
            learning_rate = learning_rate,
            num_iters = num_iters
        )
        self.lamda = lamda
        self.max_iter_lasso = max_iter_lasso
        self.cost_tolerance = cost_tolerance

    def fit(self,
            X : np.array,
            y : np.array) -> None:
        """Method to calculate model parameter from given
           data input nad target

        Args:
            X (np.array) (k,n): 
                input data to calculate model parameter with k is the number
                of data point and n is the number of feature data

            y (np.array) (k,):
                vector target data to calculate model parameter with
                k is the number of data point
        """
        # prepare data
        X = np.array(X).copy()
        y = np.array(y).copy()

        # extract size
        n_samples, n_features = X.shape

        # initialize the design matrix, A
        if self.fit_intercept:
            A = np.column_stack((X, np.ones(n_samples)))
            n_features += 1
        else:
            A = X

        # initialize theta
        theta = np.zeros(n_features)

        # start the coordinate descent
        for iter in range(self.max_iter_lasso):
            for j in range(n_features):
                # extract needed data
                X_j = A[:, j]
                X_k = np.delete(A, j, axis=1)
                theta_k = np.delete(theta, j)
                
                # Calculate rho_j
                res_j = y - X_k @ theta_k
                rho_j = np.dot(X_j, res_j)

                # compute z_j
                z_j = np.sum(X_j**2)

                # compute new theta_j from soft thresholding
                if self.fit_intercept:
                    if j == (n_features-1):
                        theta[j] = rho_j
                    else:
                        theta[j] = self.soft_threshold(rho_j,
                                                  n_samples*self.lamda)
                else:
                    theta[j] = self.soft_threshold(rho_j,
                                              n_samples*self.lamda)
                theta[j] /= z_j
                
            # stopping criteria
            cost_current = self.lasso_cost_function(A, y, theta)
            if cost_current < self.cost_tolerance:
                break
                
        # extract parameters
        if self.fit_intercept:
            self.coef_ = theta[:n_features-1]
            self.intercept_ = theta[-1]
        else:
            self.coef_ = theta
            self.intercept_ = 0.0

    def lasso_cost_function(self,
                            X : np.array, 
                            y : np.array,
                            theta : np.array) -> float :
        """Calculate cost function for lasso regression

        Args:
            X (np.array) (k,n): 
                input data to calculate model parameter with k is the number
                of data point and n is the number of feature data

            y (np.array) (k,):
                vector target data to calculate model parameter with
                k is the number of data point

            theta (np.array) (n,):
                vector model parameter to calculate prediction from input data X

        Returns:
            float: cost between prediction and true value of y
        """

        # enumerate the input and feature
        n_samples, n_features = X.shape

        # Calculate OLS cost
        err_rss =  (1/(2*n_samples)) * (np.sum((y - X@theta)**2))

        # calculate lasso cost with intercept
        if self.fit_intercept:
            err_l1 = self.lamda * np.sum(np.abs(theta[:n_features-1]))
        # calculate lasso cost without intercept
        else:
            err_l1 = self.lamda * np.sum(np.abs(theta))

        # sum OLS and Lasso cost
        cost = err_rss + err_l1

        return cost
    
    def soft_threshold(self, rho_j, lamda):
        if (rho_j < - lamda):
            theta_j = (rho_j + lamda)
        elif (-lamda <= rho_j) and (rho_j <= lamda):
            theta_j = 0
        else:
            theta_j = (rho_j - lamda)

        return theta_j