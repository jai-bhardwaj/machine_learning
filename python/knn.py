from functions import readFeatureFile
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(
   n_neighbors=67,
)

X,y=readFeatureFile("../data/dataset/training.csv")

knn.fit(X,y)

X,y=readFeatureFile("../data/dataset/testing.csv")

accuracy=knn.score(X,y)
print ("Accuracy:",accuracy)