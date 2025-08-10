"""

Principal Component Analysis (PCA) :
-------------------------------------
                Principal component analysis (PCA) is a popular technique for analyzing large datasets containing a
                high number of dimensions/features per observation, increasing the interpretability of data while
                preserving the maximum amount of information, and enabling the visualization of multidimensional data.

                PCA is Unsupervised Learning Algorithm.

"""

# Importing the Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Importing the dataset
dataset = pd.read_csv('Wine.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset to train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling on dataset
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Applying PCA
pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# Applying Logistic Regression
lr_classifier = LogisticRegression(random_state=0)
lr_classifier.fit(x_train, y_train)

# Prediction on test set
y_predict = lr_classifier.predict(x_test)

# Accuracy Score
print(accuracy_score(y_test, y_predict))

# Confusion Matrix
print(confusion_matrix(y_test, y_predict))
