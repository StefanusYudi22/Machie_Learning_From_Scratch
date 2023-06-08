""" Base class for support vector machine algorithm"""

import numpy as np

class SVC:
    """Class for Support Vector Machine Algorithm"""
    def __init__(self,
                 C : float = 1.0,
                 tolerance : float = 1e-5,
                 max_passes : int = 10) -> None:
        """Initialize SVM Class

        Args:
            C (float, optional):
                regularization parameter for SVM. Defaults to 1.0.
            tolerance (float, optional):
                error tolerance for iteration process. Defaults to 1e-5.
            max_passes (int, optional):
                max of times to iterate over alpha without changing. Defaults to 10.
        """
        self.C = C
        self.tol = tolerance
        self.max_passes = max_passes
    
    def initialize_parameters(self, X : np.array) -> None:
        """Initialize iteration parameter with 0 value

        Args:
            X (np.array) (k,n):
                Input data for shape calculation
        """
        n_samples, n_features = X.shape

        # create alpha as long as data point
        self.alpha = np.zeros(n_samples)

        # create coeficient as long as data feature
        self.coef = np.zeros(n_features)

        # create intecept
        self.intercept = 0

    def get_random_index(self, current_index : int, n : int) -> int :
        """Get random index

        Args:
            current_index (int):
                current data input index at iteration
            n (int): 
                number of sample

        Returns:
            int: 
                another index not current_index
        """
        rand_index = np.random.choice(n)
        # keep j != i, if same iterate again
        while rand_index == current_index:
            rand_index = np.random.choice(n)
        
        return rand_index
    
    def compute_boundary(self, y_i : float , y_j : float , a_i : float, a_j : float) -> tuple((float, float)) :
        """Compute Lower boundary (L) and Higher boundary (H)

        Args:
            y_i (float): 
                target value i
            y_j (float):
                target value j
            a_i (float):
                alpha for data point i
            a_j (float):
                alpha for data point j

        Returns:
            L, H (float, float):
                value of lower boundary and higher boundary 
                
        """
        if y_i != y_j :
            L = max(0, a_j-a_i)
            H = min(self.C, self.C + (a_j-a_i))
        else:
            L = max(0, a_i+a_j - self.C)
            H = min(self.C, a_i+a_j)

        return L, H
    
    def compute_coef(self, X : np.array, y : np.array) -> None:
        """Compute model parameter coefficient

        Args:
            X (np.array) (n,k) :
                Data input with n number of data point
                and k number of feature

            y (np.array) (n,) :
                Data target with n number of data point
        """
        # calculate coeficient
        self.coef = np.dot(self.alpha*y, X)
    
    
    def calculate_error(self, X_i : np.array, y_i : float, X : np.array, y : np.array) -> float:
        """ Calculate error from predicted value and real value

        Args:
            X_i (np.array) (k,):
                One data point with k feature
            y_i (float): 
                One target value
            X (np.array) (n,k): 
                n data point with k feature
            y (np.array) (n,): 
                n target value

        Returns:
            float : 
                error from predicted value (f_x) and
                real value y_i
        """
        # calculate 
        f_x = np.dot(self.alpha*y, X @ X_i.T) + self.intercept
        err = f_x - y_i
        return err
    
    def fit(self, X : np.array, y : np.array) -> None:
        """Calculate model parameter using SMO
            (Sequential Minimal Optimization) Algorith

        Args:
            X (np.array) (n,k):
                Data input for calculating model parameter
                n is number of data points, k is number of feature
            y (np.array) (n,):
                Target value from every data point
                n is number of data
        """
        # Prepare the data 
        X = np.array(X).copy()
        y = np.array(y).copy()
        n_samples, _ = X.shape

        # Initialize variables
        self.initialize_parameters(X)
        passes = 0

        # Start tuning the parameters
        while (passes < self.max_passes):
            num_changed_alphas = 0

            # Iterate for every data input
            for i in range(n_samples):
                # Extract data X and y and alpha for every data input X = (k,) y = (1), self.alpha = (1)
                X_i, y_i, a_i = X[i,:], y[i], self.alpha[i]

                # Calculate E_i
                E_i = self.calculate_error(X_i, y_i, X, y)

                # Check condition
                cond_1 = (y_i*E_i < -self.tol) and (a_i < self.C)
                cond_2 = (y_i*E_i > self.tol) and (a_i > 0)
                if cond_1 or cond_2:
                    # Select j randomly
                    j = self.get_random_index(i, n_samples)

                    # Extract data
                    X_j, y_j, a_j = X[j,:], y[j], self.alpha[j]

                    # Calculate E_j
                    E_j = self.calculate_error(X_j, y_j, X, y)

                    # Save old a's
                    a_i_old, a_j_old = a_i, a_j

                    # Compute L and H
                    L, H = self.compute_boundary(y_i, y_j, a_i_old, a_j_old)
                    if L == H:
                        continue

                    # Compute eta
                    eta = 2*np.dot(X_i, X_j) - np.dot(X_i,X_i) - np.dot(X_j,X_j)
                    if eta >= 0:
                        continue

                    # Clip value for a_j
                    # Get a_j_unclipped
                    a_j_unclipped = a_j_old - (y_j*(E_i-E_j))/eta

                    # Get a_j_clipped
                    if a_j_unclipped > H:
                        a_j_new = H
                    elif (a_j_unclipped >= L) and (a_j_unclipped <= H):
                        a_j_new = a_j_unclipped
                    else:
                        a_j_new = L

                    if np.abs(a_j_new - a_j_old) < self.tol:
                        continue

                    # Get the a_i
                    a_i_new = a_i_old + (y_i*y_j)*(a_j_old-a_j_new)

                    # Compute b_1 and b_2
                    b_old = self.intercept

                    # Compute b_1
                    b_1 = b_old - E_i \
                            - (y_i*(a_i_new - a_i_old)) * np.dot(X_i,X_i) \
                            - (y_j*(a_j_new - a_j_old)) * np.dot(X_i,X_j)
                    
                    # Compute b_2
                    b_2 = b_old - E_j \
                            - (y_i*(a_i_new - a_i_old)) * np.dot(X_i,X_i) \
                            - (y_j*(a_j_new - a_j_old)) * np.dot(X_i,X_j)
                    
                    # compute b
                    if (a_i > 0) & (a_i < self.C):
                        b_new = b_1
                    elif (a_j > 0) & (a_j < self.C):
                        b_new = b_2
                    else:
                        b_new = 0.5*(b_1 + b_2)

                    # Update variables
                    self.alpha[i], self.alpha[j] = a_i_new, a_j_new
                    self.intercept = b_new
                    self.compute_coef(X, y)

                    # Update counter
                    num_changed_alphas += 1
            
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

    def predict(self, X : np.array) -> np.array :
        """Predict target data from input data

        Args:
            X (np.array) (n,k):
                Data input with n data point, and k
                number of feature

        Returns:
            np.array :
                Prediction result from data Input
                and model parameter

        """
        return np.sign(np.dot(X, self.coef) + self.intercept).astype("int")
