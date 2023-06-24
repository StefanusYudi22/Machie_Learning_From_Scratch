# APPROXIMATE NEAREST NEIGHBOR
Approximate nearet neighbor is an searching algorithm intended to solve the slowness of K-Nearest neighbor algorithm when searching neighbors of an item based on its feature. This form of algorithm broadly used in vector database ([redis](https://redis.io/docs/stack/search/), [qdrant](https://qdrant.tech/)) to efficiently search nearest neighbor or similar item. In this ANN algorithm I will demonstrate my understanding about library [ANNOY](https://github.com/spotify/annoy) which is used by [SPOTIFY](https://open.spotify.com/) to recommend songs to users.
<br>
In shorts ANN use tree based rule to randomly cluster the data training and then focusing the searching algorithm (distance calculation) into the match the data test with available cluster. Here I will explain about how this ANN algorithms work 

## 1. How The ANN Works
### Random Cluster Data Training
Trained data in ANN model will be clustered into nodes using randomly created hyperplane until every nodes have registered data less than minimum_size_split. Lets us depict how this algorithm work for fitting the data into ANN model. Let's say we have this bunch of data in 2 dimension this is the root node of the ANN, the whole data point (although this ANN model could be use by more that 2 dimension of feature)

<p align="center">
<img src="../../../docs/picture/cluster1.png">
</P>

We first select 2 random points from the root node data, calculate the middle point from this 2 random points and draw perpendicular hiperplane which go through the middle point

<p align="center">
<img src="../../../docs/picture/cluster2.png">
</P>

After the first hyperplane formed, then the root node data can be separated into children_left data (let's just say yellow one) and the children_right data (the blue one).

<p align="center">
<img src="../../../docs/picture/cluster3.png">
<img src="../../../docs/picture/cluster4.png">
</P>

This process is being continued (like creating tree) until at certain node the amount of data clustered is less than or equal to the min_size_split, then no more hyperplane will be formed. At the end we will have a tree representing the data clustering method

<p align="center">
<img src="../../../docs/picture/cluster5.png">
</P>

### Search Cluster Using Hiperplane
Besides calculating the data input to every registered data inside the machine learning model. ANN will choose the data input and direct the searching into certain leaf node inside its data structure. 

<p align="center">
<img src="../../../docs/picture/search1.png">
</P>

How can the ANN select the rights leaf_node? It is done by knowing the data input relative location to each hiperplane from the root hyperplane until the last hyperplane before the leaf node. 

<p align="center">
<img src="../../../docs/picture/search2.png">
</P>

The searching process will direct the data input into the depth of the tree data structure until it's find the leaf node. This searching algorithm will make the ANN model search faster than brute-force algorithm and save time in big dataset and large number of feature

### Calculate Distance Between Data Test and Leaf Node Data
When the data input has arrive at the leaf node, distance between every registered data in the leaf_node and input data will be calculated by certain kind of algorithm. You can use like eucledian distance, cosine similarity, manhattan distance, etc. 

### Rank The Result Based on Distance
The last part is to rank the Leaf node data by distance from the nearest data point until the farthes data point. 

## 2. ANN Limitations
- Although ANN seems more powerfull than the ordinary brute force KNN algorithm, this mode trade its performance with time consumption. As you understand before the data is randomly clustered into nodes. Unfortunatelly there is posibility that the "Nearest Neighbor" is located in difference leaf node, besides the "Less Nearest Neighbor" this is depends on the model itselfs. To overcome this limitations I create a bunc of tree (forest) so one ANN model has more that one random tree data structure, with the intentions to broaden the scope of the cluster. 

<p align="center">
<img src="../../../docs/picture/forest1.png">
</P>

As you can see in the picture above, this region is the union of several leaf node which come from different trees, then the distance between the data input and every data inside this region is calculated and ranked

<p align="center">
<img src="../../../docs/picture/forest2.png">
</P>

 with this technique the performance of the ann could be lifted up, but the increase in the performance is paid by decreased time to search the neighbors inside the forest.

## 3. My Library approximate_nn
### class Node
class Node is used to store information about the node in clustered data ann, attributes like children_left, children_right, size, is_leaf, hyperplane, feature_registered, and label_registered shape this class to store the data about certain node.
### class ApproximateNearestNeighbors
class ApproximateNearestNeighbors is used to create ANN model. To define this model you need to specify how many minimum size split (min_size_split) in the leaf node, how many tree (n_tree) that you want to build and distance calculation for the neighbors item (distance_type). There also some method like "fit" to register the data train for this ANN model and create the trees object, and search_similar_item method to search the most similar item with the input data based on registered data inside the ANN model

### class KNearestNeighbors
class KNearestNeighbors here is an bruteforce searching method which time execution will be compared to the ANN class. This class will calculate similarity between data input and every registered data inside the KNN model. Expected result time from KNN model will be longger than the ANN model, because there are no clustered data.

## 3. Source
Annoy Developer Page : https://erikbern.com/2015/10/01/nearest-neighbors-and-vector-models-part-2-how-to-search-in-high-dimensional-spaces.html