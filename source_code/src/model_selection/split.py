import numpy as np

class KFold:
    """Class to provide train data and
        validation data for training 
        machine learning model
    """
    def __init__(
            self,
            n_splits : int = 5,
            shuffle : str = False,
            random_state : int =66
    ):
        """Initialization Class

        Args:
            n_splits (int, optional):
                Amount of data fold within a set of data. Defaults to 5 fold.
            shuffle (str, optional): 
                Randomize shuffle data within a set of data. Defaults to False.
            random_state (int, optional): 
                Random state for shuffle index. Defaults to 66.
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def iter_val_indices(self, X : np.array) -> np.array:
        """Generate data fold index

        Args:
            X (np.array) (n,k):
                Data input with n number of data point
                and k number of feature

        Returns:
            np.array (m, ):
                array of index for every fold data.

        Yields:
            Iterator[np.array]: _description_
        """

        # calculate data count and 
        # create list of index using arange
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.shuffle:

            # set random seed for shuffeling
            np.random.seed(self.random_state)

            # shuffle the indices of train data
            np.random.shuffle(indices)

        # calculate amount of data for every fold
        fold_sizes = np.full(self.n_splits, n_samples//self.n_splits, dtype=int)
        fold_sizes[: n_samples%self.n_splits] += 1

        # give the start and stop indices every iteration
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current+fold_size
            yield indices[start:stop]
            current = stop

    def split(self, X : np.array) -> (tuple(np.array, np.array)) :
        """Arange train data index and test data index

        Args:
            X (np.array) (n,k):
                Data input with n number of data point
                and k number of feature

        Yields:
            np.array, np.array:
                array of train data, and validation data
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        for val_index in self.iter_val_indices(X):
            train_index = np.array([ind for ind in indices if ind not in val_index])

            yield (train_index, val_index)