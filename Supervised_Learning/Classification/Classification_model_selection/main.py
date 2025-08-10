# Template to select the best classification model just by entering dataset name

# Importing the Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

list_classifiers_models = ['Decision Tree', 'Random Forest', 'Logistic Regression', 'KNN', 'Naive Bayes',
                           'SVM with linear Kernel', 'SVM with rbf/Gaussian Kernel', 'SVM with polynomial Kernel',
                           'SVM with sigmoid Kernel']

list_classifiers_ac, list_classifiers_cm, dict_classifier_models = [], [], {}

print("""
 █████╗ ██╗      █████╗  ██████╗ ██████╗██╗███████╗██╗ █████╗  █████╗ ████████╗██╗ █████╗ ███╗  ██╗
██╔══██╗██║     ██╔══██╗██╔════╝██╔════╝██║██╔════╝██║██╔══██╗██╔══██╗╚══██╔══╝██║██╔══██╗████╗ ██║
██║  ╚═╝██║     ███████║╚█████╗ ╚█████╗ ██║█████╗  ██║██║  ╚═╝███████║   ██║   ██║██║  ██║██╔██╗██║
██║  ██╗██║     ██╔══██║ ╚═══██╗ ╚═══██╗██║██╔══╝  ██║██║  ██╗██╔══██║   ██║   ██║██║  ██║██║╚████║
╚█████╔╝███████╗██║  ██║██████╔╝██████╔╝██║██║     ██║╚█████╔╝██║  ██║   ██║   ██║╚█████╔╝██║ ╚███║
 ╚════╝ ╚══════╝╚═╝  ╚═╝╚═════╝ ╚═════╝ ╚═╝╚═╝     ╚═╝ ╚════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚════╝ ╚═╝  ╚══╝\n""")

# Importing the dataset
dataset = input("Enter the dataset name with extension : ")

try:
    data = pd.read_csv(dataset)
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
except FileNotFoundError:
    print(f"There is no such file with name {dataset} in the path")
    exit(0)

# Splitting the data in to train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Feature Scaling
print("\n--------\nPlease select Scaling Method\n--------\n'N' for Normalization\n'S' for Standardization\n--------\n")
scalar = input("Please enter the scaling method : ").lower()

if scalar == 's':
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
elif scalar == 'n':
    mm = MinMaxScaler()
    x_train = mm.fit_transform(x_train)
    x_test = mm.transform(x_test)
else:
    print("Invalid Choice !")
    exit(0)

# Decision Tree
dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt_classifier.fit(x_train, y_train)
y_predict = dt_classifier.predict(x_test)
list_classifiers_cm.append(confusion_matrix(y_test, y_predict))
list_classifiers_ac.append(accuracy_score(y_test, y_predict))

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rf_classifier.fit(x_train, y_train)
y_predict = rf_classifier.predict(x_test)
list_classifiers_cm.append(confusion_matrix(y_test, y_predict))
list_classifiers_ac.append(accuracy_score(y_test, y_predict))

# Logistic Regression
lr_classifier = LogisticRegression(random_state=0)
lr_classifier.fit(x_train, y_train)
y_predict = lr_classifier.predict(x_test)
list_classifiers_cm.append(confusion_matrix(y_test, y_predict))
list_classifiers_ac.append(accuracy_score(y_test, y_predict))

# KNN
knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_classifier.fit(x_train, y_train)
y_predict = knn_classifier.predict(x_test)
list_classifiers_cm.append(confusion_matrix(y_test, y_predict))
list_classifiers_ac.append(accuracy_score(y_test, y_predict))

# Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)
y_predict = nb_classifier.predict(x_test)
list_classifiers_cm.append(confusion_matrix(y_test, y_predict))
list_classifiers_ac.append(accuracy_score(y_test, y_predict))

# Kernel SVM
kernels = ['linear', 'rbf', 'poly', 'sigmoid']

for i in range(len(kernels)):
    svm_classifier = SVC(kernel=kernels[i], random_state=0)
    svm_classifier.fit(x_train, y_train)
    y_predict = svm_classifier.predict(x_test)
    list_classifiers_cm.append(confusion_matrix(y_test, y_predict))
    list_classifiers_ac.append(accuracy_score(y_test, y_predict))


# Creating dictionary for all models
for i in range(len(list_classifiers_models)):
    dict_classifier_models[list_classifiers_models[i]] = list_classifiers_ac[i]

# Printing the confusion matrix and the accuracy score of the models
print("\n--------\nConfusion Matrix and the Accuracy Score of the models\n--------\n")
for i in range(len(list_classifiers_models)):
    print(f"Model : {list_classifiers_models[i]}\nConfusion Matrix : \n{list_classifiers_cm[i]}\n"
          f"Accuracy Score : {list_classifiers_ac[i]}\n")

# Sorting the list to find the highest accuracy score among models
list_classifiers_ac.sort(reverse=True)

# Printing the model which has the highest accuracy score
list_high_ac = []
for key, value in dict_classifier_models.items():
    if list_classifiers_ac[0] == value:
        list_high_ac.append(key)

print("\n--------\nModels with high accuracy score\n--------\n")
print(f"Dataset : {dataset}\nModels : ")
for i in list_high_ac:
    print(f"\t{i}")
print(f"Accuracy Score : {list_classifiers_ac[0]}\n")
