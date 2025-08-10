"""

Principal Component Analysis with Kernel :
-----------------------------------------
                Kernel PCA is an extension of PCA that allows for the separability of nonlinear data by making use of
                kernels. The basic idea behind it is to project the linearly inseparable data onto a higher dimensional
                space where it becomes linearly separable.

"""

# Importing the Libraries
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Importing the dataset
dataset = pd.read_csv('Wine.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting dataset to train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Applying feature scaling Standardization on xtrain, xtest set
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Deploying train set in LDA model
k_pca = KernelPCA(kernel='rbf', n_components=2)
x_train = k_pca.fit_transform(x_train)
x_test = k_pca.transform(x_test)

# Applying Logistic Regression
lr_classifier = LogisticRegression(random_state=0)
lr_classifier.fit(x_train, y_train)

# Predicting the model on test set
y_predict = lr_classifier.predict(x_test)

# Accuracy Score
print(accuracy_score(y_test, y_predict))

# Confusion Matrix
print(confusion_matrix(y_test, y_predict))
