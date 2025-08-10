"""

KNN (K-nearest neighbours):
        In this the prediction for the new sample is decided by the distance from the K neighbours.
        K is an integer value popular k value is  k = 5
        There are two methods to measure distances
        1) Euclidean Distance:
                   This works in the principle of Pythagoras. Measures distance in straight line from one point to other
                   AC^2 = AB^2 + BC^2
                   AC^2 = (x2 - x1)^2 + (y2 - y1)^2
                   Formula:
                   ---------
                        d = sqrt((x2 - x1)^2 + (y2 - y1)^2 )
        2) Manhattan Distance:
                   In this instead of measuring the distance in straight line. It calculates the distance in
                   as a path from one point to other point by assuming the 2D graph as a map.
                   Formula:
                   --------
                         d = |x1 - x2| + |y1 - y2|

"""

# Importing the Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the data in to train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Training train set with KNN model
knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_classifier.fit(x_train, y_train)

# Predicting the new sample value
print(knn_classifier.predict(sc.transform([[35, 87000]])))
y_predict = knn_classifier.predict(x_test)
print(np.concatenate((y_predict.reshape(len(y_predict), 1), y_test.reshape(len(y_test), 1)), 1))

# Confusion Matrix
cm = confusion_matrix(y_test, y_predict)
print(cm)

# Accuracy Score
print(accuracy_score(y_test, y_predict))

# Visualizing the training set data
X_set, y_set = sc.inverse_transform(x_test), y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                     np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
plt.contourf(X1, X2, knn_classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(['r', 'g']))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(['r', 'g'])(i), label=j)
plt.title('Logistic_Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
