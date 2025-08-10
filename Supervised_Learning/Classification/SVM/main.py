"""

SVM:
 Support vector machine. In this it will create a bigger boundary between the different samples dividing the +ve
 hyperplane and -ve Hyperplane

 xi . wi + b >= 1 ,
 -xi . wi + b <= -1

"""

# Importing the Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset in to train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Fitting the train set to SVM model
svm_classifier = SVC(kernel='linear', random_state=0)
svm_classifier.fit(x_train, y_train)

# Predicting for new sample
print(svm_classifier.predict(sc.transform([[30, 87000]])))
y_predict = svm_classifier.predict(x_test)
print(np.concatenate((y_predict.reshape(len(y_predict), 1), y_test.reshape(len(y_test), 1)), 1))

# Confusion Matrix
cm = confusion_matrix(y_test, y_predict)
print(cm)

# Accuracy Score
print(accuracy_score(y_test, y_predict))
