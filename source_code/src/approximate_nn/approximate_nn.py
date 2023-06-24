import numpy as np

class Node:
    def __init__(self,
                 children_left : object = None,
                 children_right : object = None,
                 size : int = None,
                 is_leaf : bool = None,
                 hyperplane : np.array = None,
                 feature_registered : np.array = None,
                 label_registered : np.array = None):
        """Node to store information about clustered data

        Args:
            children_left (object, optional):
                Children node object which data is below parent node hyperplane. Defaults to None.

            children_right (object, optional):
                children node object which data is above parent node hyperplance. Defaults to None.

            size (int, optional): 
                Data registered count within the node, sum of data registered 
                in children_left and children_right. Defaults to None.

            is_leaf (bool, optional):
                Identifier of the leaf node. True for Leaf node, False if not leaf node. Defaults to None.

            hyperplane (np.array, optional):
                Hyperplane coefficient and offsett of the node. Defaults to None.

            feature_registered (np.array, optional):
                Feature registered from data train in the leaf node,
                no feature if the node is not leaf node. Defaults to None.

            label_registered (np.array, optional): 
                Label registered from data train in the leaf node,
                no feature if the node is not leaf node. Defaults to None.
        """
        self.children_left = children_left
        self.children_right = children_right
        self.size = size
        self.is_leaf = is_leaf
        self.hyperplane = hyperplane
        self.feature_registered = feature_registered
        self.label_registered = label_registered

class ApproximateNearestNeighbor:
    """Approximate Nearest Neighbor Class
    """
    def __init__(self,
                 n_tree : str,
                 min_size_split : int = 10,
                 distance_type : str = "eucledian"):
        """Initialization of ANN Object

        Args:
            min_size_split (int, optional): 
                Minimum size of the node data registered to split
                into children_right and children_left. Defaults to 10.

            distance_type (str, optional):
                Distance algorithm to rank the neighbor. Defaults to "eucledian".
        """
        self.min_size_split = min_size_split
        self.distance_type = distance_type
        self.n_tree = n_tree

    def fit(self,
            X : np.array,
            y : np.array) -> None :
        """ Index data train into ANN index

        Args:
            X (np.array) (m,n):
                Feature of data train with m data point
                and n feature

            y (np.array) (m,):
                Label of data train, with m data label
        """
        # Pass the data by value
        X = np.array(X).copy()
        y = np.array(y).copy()

        # Grow indexing Tree
        self.forest = self.create_forest(X, y, n_tree=self.n_tree)

    def create_forest(self,
                      X : np.array,
                      y : np.array,
                      n_tree : int) -> np.array :
        """Create trees object for randomly clustering
        data training

        Args:
             X (np.array) (m,n):
                Feature of data train with m data point
                and n feature

            y (np.array) (m,):
                Label of data train, with m data label

            n_tree (int, optional):. Defaults to 
                Count of root node tree generated from the data train.
                Default to10.

        Returns:
            np.array:
                array of root node object
        """
        # prepare the container Node objects
        trees = []

        # create tree as much as n_tree
        for _ in range(n_tree):
            tree = self.grow_index(X,y)
            trees.append(tree)
        
        return trees

    def grow_index(self,
                   X : np.array,
                   y : np.array,
                   depth : int = 0) -> object:
        """Grow the tree until min_size_split reached

        Args:
             X (np.array) (m,n):
                Feature of data train with m data point
                and n feature

            y (np.array) (m,):
                Label of data train, with m data label

            depth (int, optional):
                Variable to state the depth of the node.
                Defaults to 0.

        Returns:
            object:
                Node object with growed children_left and 
                children right
        """
        # calculate node size
        node_size = len(X)

        # if the node size larger than minimum data
        if node_size > self.min_size_split:

            # create Tree object without
            # registering data
            node = Node(size = node_size, is_leaf = False)

            # split the node
            # and calculate hyperplane parameter
            X_left, X_right, y_left, y_right, hyperplane = self.node_split(X, y)
            node.hyperplane = hyperplane

            # grow recursively the branch
            node.children_left = self.grow_index(X_left, y_left, depth+1)
            node.children_right = self.grow_index(X_right, y_right, depth+1)
        
        else :
            # create Tree object with
            # registered data
            node = Node(size=node_size,
                        is_leaf = True,
                        feature_registered = X,
                        label_registered = y,
                        hyperplane=None)
            
        return node
    
    def node_split(self,
                         X : np.array,
                         y :np.array) -> np.array:
        """Split node into children_left and children_right

        Args:
             X (np.array) (m,n):
                Feature of data train with m data point
                and n feature

            y (np.array) (m,):
                Label of data train, with m data label

        Returns:
            X_left (s, n) :
                Feature of data train for children_left with s data point
                and n feature
            y_left (s,):
                Label of data train, with s data label
            X_right (m-s, n):
                Feature of data train for children_right, with m-s data point
                and n feature
            y_right :
                Label of data train, with m-s data label
            hyperplane: 
                Coefficient and biar of the hiperplane splitting 
                children_left and children_right
        """

        # pick 2 random data point from data using
        random_index = np.random.randint(low=0, high=len(X)-1, size=2)
        random_point = X[random_index]

        # calculate the middle point
        random_middle = np.sum(random_point, axis=0)/2

        # vector random point
        vector = random_point[0] - random_point[1]

        # Create the hyperplane equation
        c = -np.dot(vector,random_middle)
        hyperplane = np.append(vector, c)

        # calculate the sign for every data 
        result_sign = np.sign(np.dot(X, hyperplane[:-1].T) + hyperplane[-1])

        # separate the data into left and right data
        X_left = X[np.where(result_sign==-1)]
        X_right = X[np.where(result_sign==1)]
        y_left = y[np.where(result_sign==-1)]
        y_right = y[np.where(result_sign==1)]

        return X_left, X_right, y_left, y_right, hyperplane
    
    def find_similar_items(self,
                          X_in : np.array,
                          n_items : int = 10) -> np.array:
        """Find similar item within forest of tree

        Args:
            X_in (np.array) (k, ): 
                Data input into the environment with 1
                data point and k number of feature

            n_items (int, optional): 
                How many similar item will be querried
                from the registered data. Defaults to 10.

        Returns:
            np.array (n_items,) :
                array of similar item with counts n_items
        """

        # crate empty container for store 
        # item from each tree with feature 
        # of X plus 1 for y
        item_forest = np.empty((0,X_in.shape[1]+1), dtype=float)

        # populate data from leaf
        for tree in self.forest:
            item_tree = self.find_node_leaf(X_in, node=tree)
            item_forest = np.append(item_forest, item_tree, axis=0)

        # eliminate duplicate data
        item_forest = np.unique(item_forest, axis=0)

        # separate the item feature and item_label
        X_forest = item_forest[:,:X_in.shape[1]].astype("float")
        y_forest = item_forest[:,-1]

        # rank the X_env and retrieve the y
        ranked_index = self.rank_neighbors(X_in,X_forest)
        
        if len(ranked_index) < n_items:
            return y_forest[ranked_index]
        else:
            return y_forest[ranked_index[:n_items]]
        
    def find_node_leaf(self,
                       X_in : np.array,
                       node : object) -> object:
        """Finding the leaf node and extract the data and 
           label registered within the node

        Args:
            X_in (np.array) (k,):
                Data input into the environment with 1
                data point and k number of feature

            tree (onject):
                Root node object being searched 
                for leaf node

        Returns:
            object :
                Leaf node object associated with data 
                input X_in
        """
        
        if node.is_leaf:
            return np.column_stack((node.feature_registered, node.label_registered))
        
        else:
            sign_check = np.sign(np.dot(X_in, node.hyperplane[:-1]) + node.hyperplane[-1])
            if sign_check == 1:
                branch = node.children_right
            else:
                branch = node.children_left
            return self.find_node_leaf(X_in, branch)
        
    
    def rank_neighbors(self,
                       X_in : np.array,
                       X_env : np.array) -> np.array :
        """Method to compute distance between data input
           and registered data in a leaf, then rank the 
           environment

        Args:
            X_in (np.array) (k,):
                Data input into the environment with 1
                data point and k number of feature
            X_env (np.array) (n,k):
                Data registered in the environment with
                n data point registered in the environment
                and k number of feature

        Returns:
            np.array (n,): 
                array of index of nearest item calculated 
                between the input and environment data 
                with size n data index
        """
        
        if self.distance_type == "eucledian":
            distance = np.linalg.norm(X_env - X_in, axis=1)
            rank_index = np.argsort(distance)

        elif self.distance_type == "manhattan":
            distance = np.sum(np.abs(X_env - X_in), axis=1)
            rank_index = np.argsort(distance)

        elif self.distance_type == "cosine-similarity":
            distance = np.squeeze(X_env @ X_in.T)/(np.linalg.norm(X_env, axis=1)*np.linalg.norm(X_in))
            rank_index = np.argsort(-distance)

        else:
            distance = np.squeeze(X_env @ X_in.T)
            rank_index = np.argsort(-distance)

        return rank_index