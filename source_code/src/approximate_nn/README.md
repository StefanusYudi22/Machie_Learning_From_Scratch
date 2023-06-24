# APPROXIMATE NEAREST NEIGHBOR
Approximate nearet neighbor is an searching algorithm intended to solve the slowness of K-Nearest neighbor algorithm when searching neighbors of an item based on its feature. This form of algorithm broadly used in vector database ([redis](https://redis.io/docs/stack/search/), [qdrant](https://qdrant.tech/)) to efficiently search nearest neighbor or similar item. In this ANN algorithm I will demonstrate my understanding about library [ANNOY](https://github.com/spotify/annoy) which is used by [SPOTIFY](https://open.spotify.com/) to recommend songs to users.
<br>
In shorts ANN use tree based rule to randomly cluster the data training and then focusing the searching algorithm (distance calculation) into the match the data test with available cluster. Here I will explain about how this ANN algorithms work 

## 1. How The ANN Works
- Random Cluster Data Training
- Search Cluster Using Hiperplane
- Calculate Distance Between Data Test and Clustered Data Training
- Rank The Result Based on Distance

## 2. My Library approximate_nn
- class Node
- class ApproximateNearestNeighbors