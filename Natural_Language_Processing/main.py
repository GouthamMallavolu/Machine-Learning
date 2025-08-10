# Importing the Libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate

nltk.download('stopwords')

list_classifiers_models = ['Decision Tree', 'Random Forest', 'Logistic Regression', 'KNN', 'Naive Bayes',
                           'SVM with linear Kernel', 'SVM with rbf/Gaussian Kernel', 'SVM with polynomial Kernel',
                           'SVM with sigmoid Kernel']

list_classifiers_ac, list_classifiers_cm, list_classifiers_pc, list_classifiers_f1s, list_classifiers_rs = \
    ([] for _ in range(5))
dict_classifier_models = {}

# Importing the dataset
data = 'Restaurant_Reviews.tsv'
dataset = pd.read_csv(data, delimiter='\t', quoting=3)

# Cleaning the text inside dataset
ps = PorterStemmer()
corpus = []
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in all_stopwords]
    review = ' '.join(review)
    corpus.append(review)

# Creating a bag of words models
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# splitting dataset in to train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training the train set on all classifier models

# Decision Tree
dt_classifier = DecisionTreeClassifier(min_samples_split=30, criterion='entropy', random_state=0)
dt_classifier.fit(x_train, y_train)
y_predict = dt_classifier.predict(x_test)
list_classifiers_cm.append(confusion_matrix(y_test, y_predict))
list_classifiers_ac.append(accuracy_score(y_test, y_predict))
list_classifiers_pc.append(precision_score(y_test, y_predict))
list_classifiers_f1s.append(f1_score(y_test, y_predict))
list_classifiers_rs.append(recall_score(y_test, y_predict))

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
rf_classifier.fit(x_train, y_train)
y_predict = rf_classifier.predict(x_test)
list_classifiers_cm.append(confusion_matrix(y_test, y_predict))
list_classifiers_ac.append(accuracy_score(y_test, y_predict))
list_classifiers_pc.append(precision_score(y_test, y_predict))
list_classifiers_f1s.append(f1_score(y_test, y_predict))
list_classifiers_rs.append(recall_score(y_test, y_predict))

# Logistic Regression
lr_classifier = LogisticRegression(random_state=0)
lr_classifier.fit(x_train, y_train)
y_predict = lr_classifier.predict(x_test)
list_classifiers_cm.append(confusion_matrix(y_test, y_predict))
list_classifiers_ac.append(accuracy_score(y_test, y_predict))
list_classifiers_pc.append(precision_score(y_test, y_predict))
list_classifiers_f1s.append(f1_score(y_test, y_predict))
list_classifiers_rs.append(recall_score(y_test, y_predict))

# KNN
knn_classifier = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2)
knn_classifier.fit(x_train, y_train)
y_predict = knn_classifier.predict(x_test)
list_classifiers_cm.append(confusion_matrix(y_test, y_predict))
list_classifiers_ac.append(accuracy_score(y_test, y_predict))
list_classifiers_pc.append(precision_score(y_test, y_predict))
list_classifiers_f1s.append(f1_score(y_test, y_predict))
list_classifiers_rs.append(recall_score(y_test, y_predict))

# Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)
y_predict = nb_classifier.predict(x_test)
list_classifiers_cm.append(confusion_matrix(y_test, y_predict))
list_classifiers_ac.append(accuracy_score(y_test, y_predict))
list_classifiers_pc.append(precision_score(y_test, y_predict))
list_classifiers_f1s.append(f1_score(y_test, y_predict))
list_classifiers_rs.append(recall_score(y_test, y_predict))

# Kernel SVM
kernels = ['linear', 'rbf', 'poly', 'sigmoid']

for i in range(len(kernels)):
    svm_classifier = SVC(kernel=kernels[i], random_state=0)
    svm_classifier.fit(x_train, y_train)
    y_predict = svm_classifier.predict(x_test)
    list_classifiers_cm.append(confusion_matrix(y_test, y_predict))
    list_classifiers_ac.append(accuracy_score(y_test, y_predict))
    list_classifiers_pc.append(precision_score(y_test, y_predict))
    list_classifiers_f1s.append(f1_score(y_test, y_predict))
    list_classifiers_rs.append(recall_score(y_test, y_predict))

# Creating dictionary for all models
for i in range(len(list_classifiers_models)):
    dict_classifier_models[list_classifiers_models[i]] = list_classifiers_ac[i]

# Final result from all the models
# pd.set_option('display.max_columns', None)
print("\n--------\nFinal Results\n--------\n")
dataframe = pd.DataFrame(list(zip(list_classifiers_models, list_classifiers_ac,
                                  list_classifiers_cm, list_classifiers_pc, list_classifiers_f1s, list_classifiers_rs)),
                         columns=['Model', 'Accuracy Score', 'confusion matrix',
                                  'Precision Score', 'F1 Score', 'Recall Score'])

print(tabulate(dataframe, headers='keys', tablefmt='psql'))

# Sorting the list to find the highest accuracy score among models
list_classifiers_ac.sort(reverse=True)

# Printing the model which has the highest accuracy score
list_high_ac = []
for key, value in dict_classifier_models.items():
    if list_classifiers_ac[0] == value:
        list_high_ac.append(key)

print("\n--------\nModels with high accuracy score\n--------\n")
print(f"Dataset : {data}\nModels : ")
for i in list_high_ac:
    print(f"\t{i}")
print(f"Accuracy Score : {list_classifiers_ac[0]}\n")
