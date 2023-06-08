import numpy as np


# CLASSIFICATION IMPURITY
def Gini(y : np.array) -> float :
    """Calculate impurity of a node using Gini Index

    Args:
        y (np.array) (n,):
            target data in a node

    Returns:
        float: The impurity of the node
    """
    # Extract class and count of each class
    num_data = len(y)
    class_, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(class_, counts))
    
    # Calculate the proportion every class in a node
    p_class = {k : class_counts[k]/num_data for k in class_}

    # Calculate the node impurity
    node_impurity = np.sum([p*(1-p) for p in p_class.values()])

    return node_impurity

def Log_Loss(y : np.array) -> float:
    """Calculate impurity of a node using Log Loss

    Args:
        y (np.array) (n,):
            target data in a node

    Returns:
        float: the impurity of the node
    """
    # Extract class and count of each class
    num_data = len(y)
    class_, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(class_, counts))
    
    # Calculate the proportion every class in a node
    p_class = {k : class_counts[k]/num_data for k in class_}

    # Find the majority class in the node
    ind_max = np.argmax(counts)
    class_max = class_[ind_max]

    # Calculate the node impurity
    node_impurity = 1 - p_class[class_max]

    return node_impurity

def Entropy(y : np.array) -> float:
    """Calculate impurity of a node using Entropy

    Args:
        y (np.array) (n,):
            target data in a node

    Returns:
        float: the impurity of the node
    """
    # Extract class and count of each class
    num_data = len(y)
    class_, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(class_, counts))
    
    # Calculate the proportion every class in a node
    p_class = {k : class_counts[k]/num_data for k in class_}

    # Calculate the node impurity
    node_impurity = np.sum([p*np.log(p) for p in p_class.values()])

    return -node_impurity


# REGRESSION IMPURITY
def MSE(y : np.array) -> float:
    """Calculate impurity of a node using MSE

    Args:
        y (np.array) (n,):
            target data in a node

    Returns:
        float: the impurity of the node
    """
    # Calculate the mean of the node
    node_mean = np.mean(y)

    # Calculate the node-impurity (variance)
    node_impurity = np.mean([(y_i - node_mean)**2 for y_i in y])

    return node_impurity

def MAE(y : np.array) -> float:
    """Calculate impurity of a node using MAE

    Args:
        y (np.array) (n,):
            target data in a node

    Returns:
        float: the impurity of the node
    """
    # Calculate the node median
    node_median = np.median(y)

    # Calculate the node-impurity (variance)
    node_impurity = np.mean([np.abs(y_i - node_median) for y_i in y])

    return node_impurity
