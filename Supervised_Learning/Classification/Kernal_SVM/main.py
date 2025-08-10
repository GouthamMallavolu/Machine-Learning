"""

Kernel SVM :
-----------
        SVM model are usually worked on Linearly separable data, but sometimes if the dataset contains non-linearly
        separable data we use kernel functions in SVM models.
        Kernels usually a used for the projection of data on the 3D plane with some function and create a hyperplane
        that separates the data using SVM. Thus, it becomes linearly separable.

        SVM Kernel functions :
        -----------------------------------
        * Linear Kernel ( It is used when the data is linearly separable.)
           K(x1, x2) = x1 . x2

        * Polynomial Kernel (It is used when the data is not linearly separable.)
           K(x1, x2) = (x1 . x2 + 1)d

        * Gaussian Kernel or RBF (The Gaussian kernel is an example of a radial basis function kernel.)
           k(xi, xj) = exp(-𝛾||xi - xj||2)

        * Exponential Kernel (Similar to RBF kernel, but it decays much more quickly.)
           k(x, y) =exp(-||x -y||22)

        * Laplacian Kernel (Similar to RBF Kernel, it has a sharper peak and faster decay.)
           k(x, y) = exp(- ||x - y||)

        * Hyperbolic or the Sigmoid Kernel (It is used for non-linear classification problems.
                                            It transforms the input data into a higher-dimensional
                                            space using the Sigmoid kernel.)
           k(x, y) = tanh(xTy + c)

        * Anova radial basis kernel (It is a multiple-input kernel function that can be used for feature selection.)
           k(x, y) = k=1nexp(-(xk -yk)2)d

        * Radial-basis function kernel (It maps the input data into end to end infinite-dimensional space.)
           K(x, y) = exp(-γ ||x - y||^2)

        * Wavelet kernel (It is a non-stationary kernel function that can be used for time-series analysis.)
           K(x, y) = ∑φ(i,j) Ψ(x(i),y(j))

        * Spectral kernel (This function is based on eigenvalues & eigen vectors of a similarity matrix.)
           K(x, y) = ∑λi φi(x) φi(y)

        * Mahalonibus kernel (This function takes into account the covariance structure of the data.)
           K(x, y) = exp(-1/2 (x - y)T S^-1 (x - y))


"""

# Importing the Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.colors import ListedColormap

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting dataset in to train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Implementing SVM with Kernels

kernels = ['linear', 'rbf', 'poly', 'sigmoid']
y_prediction, cm, accuracy, dict_final = [], [], [], {}

for i in range(len(kernels)):
    svm_classifier = SVC(kernel=kernels[i], random_state=0)
    svm_classifier.fit(x_train, y_train)
    y_prediction.append(svm_classifier.predict(x_test))
    cm.append(confusion_matrix(y_prediction[i], y_test))
    accuracy.append(accuracy_score(y_prediction[i], y_test))
    dict_final[kernels[i]] = accuracy[i]

accuracy.sort(reverse=True)

# print kernels and their Confusion Matrix

print(f"\n--------\nKernels and their Confusion Matrix\n--------\n")
for i in range(len(kernels)):
    print(f"Kernel : {kernels[i]}\nConfusion Matrix : \n{cm[i]}\n")

# Print Kernel and their Accuracy Score
print(f"\n--------\nKernels and their Accuracy\n--------\n")
for key, value in dict_final.items():
    print(f"Kernel : {key}\nAccuracy Score : {value}\n")

# Print best kernel with the highest accuracy score
for key, value in dict_final.items():
    if accuracy[0] == value:
        print(f"\n--------\nBest kernel in SVM for the given non-linear Dataset\n--------\n"
              f"Dataset : Social_Network_Ads.csv \nKernel : {key}\nAccuracy Score : {value}")
        break
