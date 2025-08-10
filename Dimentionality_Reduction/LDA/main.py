"""

Linear Discriminant Analysis (LDA) :
------------------------------------
            Linear Discriminant Analysis (LDA) is a supervised learning algorithm used for classification tasks in
            machine learning. It is a technique used to find a linear combination of features that best separates the
            classes in a dataset.


"""

# Importing the Libraries
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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
lda = LDA(n_components=2)
x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)

# Applying the Logistic Regression
lr_classifier = LogisticRegression(random_state=0)
lr_classifier.fit(x_train, y_train)

# Predicting the test set
y_predict = lr_classifier.predict(x_test)

# Accuracy Score
print(accuracy_score(y_test, y_predict))

# Confusion Matrix
print(confusion_matrix(y_test, y_predict))
