"""Base class for linear model"""

import numpy as np

class LinearRegression:
    """Linear Regression Class
    """
    def __init__(self,
                 fit_intercept : str = True,
                 optimizer : str ="ols",
                 learning_rate : float = 1e-5,
                 num_iters : int = 10000):
        """Class initialization for Linear Regression

        Args:
            fit_intercept (str, optional):
                True for Model with intercept, False with no intercept. Defaults to True.

            optimizer (str, optional):
                Optimization algorithm ols or gradient-descent. Defaults to "ols".

            learning_rate (float, optional):
                Learning rate for gradient-descent algorithm, used if gradient-descent
                used as optimization algorithm. Defaults to 1e-5.

            num_iters (int, optional):
                Number of iterations until gradient-descent finish, used if gradient-descent
                used as optimization algoritm. Defaults to 10000.
        """
        self.fit_intercept = fit_intercept
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.num_iters = num_iters

    def fit(self,
            X : np.array,
            y : np.array) -> np.array :
        """Method to calculate linear model parameter

        Args:
            X (np.array) (k,n): 
                input data to calculate model parameter with k is the number
                of data point and n is the number of feature data

            y (np.array) (k,):
                vector target data to calculate model parameter with
                k is the number of data point

        Returns:
            np.array (n, ) or (n+1,):
                the result of model parameter calculation
                (n, ) size without intercept
                (n+1, ) size with intercept
        """
        # prepare data
        X = np.array(X).copy()
        y = np.array(y).copy()

        # extract Size
        n_features = X.shape[1]

        if self.optimizer == "ols":
            # find model parameter using ols equation
            theta = self.ols_solution(X, y)
        else :
            # find model parameter using grandient-descent
            theta = self.gradient_descent(X,
                                          y,
                                          self.learning_rate,
                                          self.num_iters)
            
        self.coef_ = theta[:n_features]
        self.intercept_ = theta[-1]

    def predict(self,
                X : np.array) -> np.array :
        """Predict data input using calculated model parameter

        Args:
            X (np.array) (k,n):
                input data for prediction with k number of data point
                and n feature
        Returns:
            np.array (k, ):
                result of the predict function whit k number of 
                prediction
        """
        # prepare data
        X = np.array(X)

        # calculate prediction
        y_pred = np.dot(X,self.coef_) + self.intercept_

        return y_pred
    
    def ols_solution(self,
                     X : np.array,
                     y : np.array) -> np.array :
        """Find model parameter using ols equation

        Args:
            X (np.array) (k,n):
                input data for model parameter calculation, with k
                is the number of data point and n number of feature

            y (np.array) (k,): 
                input target for model parameter calculation, with k
                is the number of data point

        Returns:
            np.array (n,) or (n+1, ):
                result of model parameter calculation, with n shape
                if no intercept and n+1 shape with intercept

        """
        n_features = X.shape[1]

        # if the model require intercept
        if self.fit_intercept:
            # Create Design Matrix
            A = np.column_stack((X, np.ones(n_features)))
        # els if the model don;t require intercept
        else :
            A = X

        return np.linalg.pinv(A.T @ A) @ A.T @ y
    
    def gradient_descent(self,
                         X : np.array,
                         y : np.array,
                         alpha : float,
                         num_iters : int) -> np.array : 
        """Find model parameter using gradient descent

        Args:
            X (np.array) (k,n):
                input data for model parameter calculation, with k
                is the number of data point and n number of feature

            y (np.array) (k,):
                input target for model parameter calculation, with k
                is the number of data point

            alpha (float):
                learning rate for gradient-descent algorithm, used if gradient-descent
                used as optimization algorithm.

            num_iters (int):
                number of iterations until gradient-descent finish, used if gradient-descent
                used as optimization algoritm. Defaults to 10000.

        Returns:
            np.array (n,) or (n+1, ):
                result of model parameter calculation, with n shape
                if no intercept and n+1 shape with intercept
        """

        # initialize gain and intercept
        w = np.zeros(X.shape[1])
        b = np.zeros(1)
        
        for i in range(num_iters):

            # calculate error
            err = X@w + b - y

            # calculate and update gain
            dj_dw = (1/len(X))*np.sum(X*err.reshape(-1,1), axis=0)
            w = w - alpha * dj_dw

            # If model require intercept
            # calculate and update intercept
            if self.fit_intercept:
                dj_db = (1/len(X))*np.sum(err)
                b = b - alpha * dj_db

            # print cost function for monitoring
            if i%(num_iters/10) == 0:
                cost = (1/(2*len(X)))*np.sum((X@w + b - y)**2)
                print(f"Cost at iteration {i} is : {cost}")
        
        # append the gain and bias parameter
        theta = np.append(w,b)
        
        return theta