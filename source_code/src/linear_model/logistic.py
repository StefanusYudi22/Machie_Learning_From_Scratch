import numpy as np


def _compute_cost_function(y, y_pred):
        """
        Compute the average log-loss
            -(1/m) * sum [y*log(y_hat) + (1-y)*log(1-y_hat)]
        """
        # Extract number of samples
        m = len(y)

        # Calcualte the log-loss
        log_loss = 0
        for i in range(m):
            log_loss += y[i]*np.log(y_pred[i]) + (1-y[i])*np.log(1-y_pred[i])

        log_loss *= -(1/m)

        return log_loss


class LogisticRegression:
    """
    Logistic Regression with Gradient Descent optimization

    Parameters
    ----------
    learning_rate: float, default = 0.01
        Learning rate of gradient descent
    
    max_iter : int, default = 100_000
        Maximum number of iterations taken for the solvers to converge

    tol : float, default = 1e-4
        Tolerance for stopping criteria
        The iterations will stop when 
        all(abs(NLL gradient)) < tol
    
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Coefficient of the features in the decision function
    
    intercept_ : ndarray of shape (1,)
        Intercept or bias added to the decision function

    Examples
    --------
    >>> import pandas as pd
    >>> from ml_from_scratch.linear_model import LogisticRegression
    >>> data = pd.DataFrame({"x1": [0, 1, 1, 0], "x2": [0, 0, 1, 1], "y": [0, 1, 1, 1]})
    >>> X = data[["x1", "x2"]]
    >>> y = data["y"]
    >>> clf = LogisticRegression()
    >>> clf.fit(X, y)
    >>> clf.predict(X)
    array([0 1 1 1])
    >>> clf.predict_proba(X)
    array([0.02053742 0.99179754 0.99999857 0.99179754])
    """
    def __init__(
        self,
        learning_rate=0.01,
        max_iter=100_000,
        tol=1e-4
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        """
        Fit the model according to the given training data
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features
        
        y : array-like of shape (n_samples,)
            Target vector relative to X
        
        Returns
        --------
        self
            Fitted estimator
        """
        # Update the data
        X = np.array(X).copy()
        y = np.array(y).copy()

        # Extract the data & feature size
        self.n_samples, self.n_features = X.shape

        # Initialize the parameters
        self.coef_ = np.zeros(self.n_features)
        self.intercept_ = 0

        # Tune the parameters
        for i in range(self.max_iter):
            # Make a new prediction
            y_pred = self.predict_proba(X)

            # Calculate the gradient
            grad_coef_ = -(y - y_pred).dot(X) / self.n_samples
            grad_intercept_ = -(y - y_pred).dot(np.ones(self.n_samples)) / self.n_samples

            # Update parameters
            self.coef_ -= self.learning_rate * grad_coef_
            self.intercept_ -= self.learning_rate * grad_intercept_

            # Break the iteration
            grad_stack_ = np.hstack((grad_coef_, grad_intercept_))      # stack the gradient
            if all(np.abs(grad_stack_) < self.tol):
                 break

    def predict_proba(self, X):
        """
        Probability estimates.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features

        Returns
        -------
        proba : array-like of shape (n_samples,)
            Returns the probability of the sample for each class in the model,
        """
        # Calculate the log odds
        logits = np.dot(X, self.coef_) + self.intercept_

        # Calculate the probability using sigmoid function
        # sigmoid(x) = 1 / (1 + e^(-x))
        proba = 1. / (1 + np.exp(-logits))

        return proba
    
    def predict(self, X):
        """
        Predict class
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features
        Returns
        -------
        T : array-like of shape (n_samples,)
            Returns the probability of the sample for each class in the model,
        """
        return (self.predict_proba(X)>0.5).astype("int")
