# Import Library
from src.neighbors import KNeighborsRegressor
from src.neighbors import KNeighborsClassifier

# REGRESSION CASE
X = [[0,1],[1,1],[2,1],[3,1]]
y = [0,0,1,1]

neigh = KNeighborsRegressor(n_neighbors=3, weights="uniform", distance_type="eucledian")
neigh.fit(X,y)

print("Regression Case: ")
print(neigh.predict([[2.2,1],[5,1],[10,1]]))
print("")

# CLASSIFICATION CASE
X = [[0],[1],[2],[3]]
y = [0,0,1,1]

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X,y)

print("Classification case")
print(neigh.predict([[1.1]]))
print(neigh.predict_proba([[1.1]]))
print(neigh.classes)