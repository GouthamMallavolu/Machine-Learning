"""

XGBoost:
-----------------------


"""

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
from sklearn.model_selection import cross_val_score
from tabulate import tabulate
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

list_classifiers_models = ['Decision Tree', 'Random Forest', 'Logistic Regression', 'KNN', 'Naive Bayes',
                           'SVM with linear Kernel', 'SVM with rbf/Gaussian Kernel', 'SVM with polynomial Kernel',
                           'SVM with sigmoid Kernel', 'XGBoost', 'CatBoost']

kernels = ['linear', 'rbf', 'poly', 'sigmoid']

list_classifiers_ac, list_classifiers_cm, dict_classifier_models, kfc_validation = [], [], {}, []
kfc_accuracy, kfc_SD = [], []


# Importing the dataset
dataset = input("Enter the dataset name with extension : ")

try:
    data = pd.read_csv(dataset)
    # Replacing 2 and 4 with 0 and 1 as e are using classifiers and XGBoost only accept dependent variable with values
    # 0 and 1
    data.iloc[:, -1] = data.iloc[:, -1].replace({2: 0, 4: 1})
    # Diving independent values (x) and dependent values (y)
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

print("\n--------\nDo you want to perform Dimensionality Reduction (Y/N) ?\n")
dr = input(": ").lower()
if dr == 'y':
    print("\n--------\nPlease select method for Dimensionality Reduction\n--------\n'K' for Kernel PCA\n"
          "'L' for Linear Discriminant Analysis (LDA)\n'P' for Principal Component Analysis (PCA)\n")
    dr_method = input(": ").lower()
    if dr_method == 'k':
        print("\n-------\nKernel PCA selected\n--------\nWhich Kernel you need (poly/linear/rbf/sigmoid) ?\n")
        k = input(": ")
        if k in kernels:
            k_pca = KernelPCA(n_components=2, kernel=k)
            x_train = k_pca.fit_transform(x_train)
            x_test = k_pca.transform(x_test)
        else:
            print("\nInvalid kernel selected !! Proceeding without dimensionality reduction....")
    elif dr_method == 'p':
        pca = PCA(n_components=2)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)
    elif dr_method == 'l':
        lda = LDA(n_components=2)
        x_train = lda.fit_transform(x_train)
        x_test = lda.transform(x_test)
    else:
        print("\n Invalid Selection !!")
        exit(0)
elif dr == 'n':
    print("\nNo Dimensionality Reduction performed proceeding to next steps...")
else:
    print("Invalid Choice !!")
    exit(0)
print("\n--------")

folds = int(input("\nPlease enter the number of folds you need for K-Fold Cross Validation : "))

# Decision Tree
dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt_classifier.fit(x_train, y_train)
y_predict = dt_classifier.predict(x_test)
list_classifiers_cm.append(confusion_matrix(y_test, y_predict))
list_classifiers_ac.append(accuracy_score(y_test, y_predict) * 100)
kfc_validation.append(cross_val_score(estimator=dt_classifier, X=x_train, y=y_train, cv=folds))

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rf_classifier.fit(x_train, y_train)
y_predict = rf_classifier.predict(x_test)
list_classifiers_cm.append(confusion_matrix(y_test, y_predict))
list_classifiers_ac.append(accuracy_score(y_test, y_predict) * 100)
kfc_validation.append(cross_val_score(estimator=rf_classifier, X=x_train, y=y_train, cv=folds))

# Logistic Regression
lr_classifier = LogisticRegression(random_state=0)
lr_classifier.fit(x_train, y_train)
y_predict = lr_classifier.predict(x_test)
list_classifiers_cm.append(confusion_matrix(y_test, y_predict))
list_classifiers_ac.append(accuracy_score(y_test, y_predict) * 100)
kfc_validation.append(cross_val_score(estimator=lr_classifier, X=x_train, y=y_train, cv=folds))

# KNN
knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_classifier.fit(x_train, y_train)
y_predict = knn_classifier.predict(x_test)
list_classifiers_cm.append(confusion_matrix(y_test, y_predict))
list_classifiers_ac.append(accuracy_score(y_test, y_predict) * 100)
kfc_validation.append(cross_val_score(estimator=knn_classifier, X=x_train, y=y_train, cv=folds))

# Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)
y_predict = nb_classifier.predict(x_test)
list_classifiers_cm.append(confusion_matrix(y_test, y_predict))
list_classifiers_ac.append(accuracy_score(y_test, y_predict) * 100)
kfc_validation.append(cross_val_score(estimator=nb_classifier, X=x_train, y=y_train, cv=folds))

# Kernel SVM
for i in range(len(kernels)):
    svm_classifier = SVC(kernel=kernels[i], random_state=0)
    svm_classifier.fit(x_train, y_train)
    y_predict = svm_classifier.predict(x_test)
    list_classifiers_cm.append(confusion_matrix(y_test, y_predict))
    list_classifiers_ac.append(accuracy_score(y_test, y_predict) * 100)
    kfc_validation.append(cross_val_score(estimator=svm_classifier, X=x_train, y=y_train, cv=folds))

# XGBoost
xgb_classifier = XGBClassifier()
xgb_classifier.fit(x_train, y_train)
y_predict = xgb_classifier.predict(x_test)
list_classifiers_cm.append(confusion_matrix(y_test, y_predict))
list_classifiers_ac.append(accuracy_score(y_test, y_predict) * 100)
kfc_validation.append(cross_val_score(estimator=xgb_classifier, X=x_train, y=y_train, cv=folds))

# CatBoost
cb_classifier = CatBoostClassifier()
cb_classifier.fit(x_train, y_train)
y_predict = cb_classifier.predict(x_test)
list_classifiers_cm.append(confusion_matrix(y_test, y_predict))
list_classifiers_ac.append(accuracy_score(y_test, y_predict) * 100)
kfc_validation.append(cross_val_score(estimator=cb_classifier, X=x_train, y=y_train, cv=folds))

# Creating dictionary for all models
for i in range(len(list_classifiers_models)):
    kfc_accuracy.append(kfc_validation[i].mean() * 100)
    kfc_SD.append(kfc_validation[i].std() * 100)
    dict_classifier_models[list_classifiers_models[i]] = kfc_accuracy[i]

# Printing the confusion matrix and the accuracy score of the models
print("\n--------\nFinal Results\n--------\n")
dataframe = pd.DataFrame(list(zip(list_classifiers_models, list_classifiers_ac,
                                  list_classifiers_cm, kfc_accuracy, kfc_SD)),
                         columns=['Classification Model', 'Model Accuracy Score (%)', 'confusion matrix',
                                  'K-Fold_Cross Accuracy (%)', 'K-Fold_Cross Standard Deviation (%)'])

print(tabulate(dataframe, headers='keys', tablefmt='psql'))

# Sorting the list to find the highest accuracy score among models
kfc_accuracy.sort(reverse=True)

# Printing the model which has the highest accuracy score
list_high_ac = []
for key, value in dict_classifier_models.items():
    if kfc_accuracy[0] == value:
        list_high_ac.append(key)

print("\n--------\nModels with high k-fold cross accuracy score\n--------\n")
print(f"Dataset : {dataset}\nModels : ")
for i in list_high_ac:
    print(f"\t{i}")
print(f"Accuracy Score : {kfc_accuracy[0]}\n")
