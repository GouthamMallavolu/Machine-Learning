"""

Deep Learning (DL):
---------------
        Deep learning is a method in artificial intelligence (AI) that teaches computers to process data in a way that
        is inspired by the human brain. Deep learning models can recognize complex patterns in pictures, text, sounds,
        and other data to produce accurate insights and predictions.

        Activation Functions:
        ---------------------
        1. Threshold Function
        2. Sigmoid Function
        3. Rectified Function
        4. Hyperbolic Tangent Function

Artificial Neural Network (ANN):
--------------------------------
        An artificial neural network (ANN) is an information processing paradigm that is inspired by the
        way biological nervous systems, such as the brain, process information.

        Artificial neural network (ANN) model involves computations and mathematics, which simulate the human–brain
        processes. Many of the recently achieved advancements are related to the artificial intelligence research area
        such as image and voice recognition, robotics, and using ANNs.

Link:
https://www.superdatascience.com/blogs/the-ultimate-guide-to-artificial-neural-networks-ann

"""

# Importing the Libraries
import os
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Importing the dataset
data = 'Churn_Modelling.csv'
dataset = pd.read_csv(data)
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data for columns 'Geography' with OneHotEncoder and 'Gender' with LabelEncoder
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# splitting dataset in to train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

selection = int(input("Enter the following to proceed\n1. Classification Models\n2. ANN\n: "))

if selection == 1:
    # Training the train set on all classifier models
    list_classifiers_models = ['Decision Tree', 'Random Forest', 'Logistic Regression', 'KNN', 'Naive Bayes',
                               'SVM with linear Kernel', 'SVM with rbf/Gaussian Kernel', 'SVM with polynomial Kernel',
                               'SVM with sigmoid Kernel']

    list_classifiers_ac, list_classifiers_cm, list_classifiers_pc, list_classifiers_f1s, list_classifiers_rs = \
        ([] for _ in range(5))
    dict_classifier_models = {}

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
                                      list_classifiers_cm, list_classifiers_pc, list_classifiers_f1s,\
                                      list_classifiers_rs)),
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

elif selection == 2:
    """
    Training the ANN with Stochastic Gradient Descent.
    
    STEP 1: Randomly initialise the weights to small numbers close to 0 (but not 0).
    STEP 2: Input the first observation of your dataset in the input layer, each feature in one imput node.
    STEP 3: Forward-Propagation: from left to right, the neurons are activated in a way that the impact of each 
            neuron's activation is limited by the weights. Propagate the activations until getting the predicted 
            result y.
    STEP 4: Compare the predicted result to the actual result. Measure the generated error.
    STEP 5: Back-Propagation: from right to left, the error is back-propagated. Update the weights according to how 
            much they are responsible for the error. The learning rate decides by how much we update the weights.
    STEP 6: Repeat Steps 1 to 5 and update the weights after each observation (Reinforcement Learning). Or:
            Repeat Steps 1 to 5 but update the weights only after a batch of observations (Batch Learning).
    STEP 7: When the whole training set passed through the ANN, that makes an epoch. Redo more epochs.
    
    """
    print(tf.__version__)

    # Initializing ANN Model
    ann = keras.models.Sequential()

    # Adding the Input Layer, Hidden Layers and Output Layer
    ann.add(keras.layers.Dense(units=6, activation='relu'))
    ann.add(keras.layers.Dense(units=6, activation='relu'))
    ann.add(keras.layers.Dense(units=1, activation='sigmoid'))

    # Compiling the ANN
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Training the ANN on training set
    ann.fit(x_train, y_train, batch_size=32, epochs=50)

    # Predicting for one input
    print(ann.predict(sc.transform(ct.transform([[600, 'France', le.transform([['Male']]),
                                                  40, 3, 6000, 2, 1, 1, 50000]]))) > 0.5)

    # Predicting for the test sets
    y_predict = ann.predict(x_test)
    y_predict = y_predict > 0.5
    print(np.concatenate((y_predict.reshape(len(y_predict), 1), y_test.reshape(len(y_test), 1)), 1))

    print("\n--------\nFinal Results\n--------\n")
    ac, cm, ps, f1, rs = (accuracy_score(y_test, y_predict), confusion_matrix(y_test, y_predict),
                          precision_score(y_test, y_predict), f1_score(y_test, y_predict),
                          recall_score(y_test, y_predict))
    fr = ['ANN', ac, cm, ps, f1, rs]

    dataframe = pd.DataFrame([fr],
                             columns=['Model', 'Accuracy Score', 'confusion matrix', 'Precision Score', 'F1 Score',
                                      'Recall Score'])
    print(tabulate(dataframe, headers='keys', tablefmt='psql'))

else:
    print("Invalid Selection !")
