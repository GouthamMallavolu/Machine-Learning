# Place your dataset and quickly get the best regression model that will fit

# Importing the libraries

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Import the Dataset
data = input('Enter the dataset name with extension : ')
dataset = pd.read_csv(data)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Checking for categorical data
count = 0
list_category = []
for i in range(len(x[0])):
    if isinstance(x[0][i], str):
        list_category.append(i)
        count += 1
print(f"The total Count of categorical data Columns : {count}")

# Regression Models stats
if count > 0:
    for i in range(count):
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [list_category[i]])], remainder='passthrough')
        x = np.array(ct.fit_transform(x))

# Splitting the data for training and testing

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Multi Linear Regression Model

print("--------\nMultiple Linear Regression\n--------")
np.set_printoptions(precision=2)
lr_regressor = LinearRegression()
lr_regressor.fit(x_train, y_train)
lr_predict = lr_regressor.predict(x_test)
lr_score = r2_score(y_test, lr_predict)
print(f"****\nThe prediction for Multiple Linear Regression is as follows\nPredicted  Actual\n"
      f"{np.concatenate((lr_predict.reshape(len(lr_predict), 1), y_test.reshape(len(y_test), 1)), 1)}\n")

# Polynomial Regression Model

print("--------\nPolynomial Regression\n--------")
deg = int(input("Enter a degree for Polynomial Regression : "))
poly_regressor = PolynomialFeatures(degree=deg)
x_poly = poly_regressor.fit_transform(x_train)
# ply_regressor = LinearRegression()
lr_regressor.fit(x_poly, y_train)
ply_predict = lr_regressor.predict(poly_regressor.fit_transform(x_test))
ply_score = r2_score(y_test, ply_predict)
print(f"****\nThe prediction for Polynomial Regression is as follows\nPredicted  Actual\n"
      f"{np.concatenate((ply_predict.reshape(len(ply_predict), 1), y_test.reshape(len(y_test), 1)), 1)}\n")

# Decision Tree Regression Model

print("--------\nDecision Tree Regression\n--------")
dt_regressor = DecisionTreeRegressor(random_state=0)
dt_regressor.fit(x_train, y_train)
dt_predict = dt_regressor.predict(x_test)
dt_score = r2_score(y_test, dt_predict)
print(f"****\nThe prediction for Decision Tree Regression Model is as follows\nPredicted  Actual\n"
      f"{np.concatenate((dt_predict.reshape(len(dt_predict), 1), y_test.reshape(len(y_test), 1)), 1)}\n")

# Random Forest Regression Model

print("--------\nRandom Forest Regression\n--------")
rf_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
rf_regressor.fit(x_train, y_train)
rf_predict = rf_regressor.predict(x_test)
rf_score = r2_score(y_test, rf_predict)
print(f"****\nThe prediction for Random Forest Regression Model is as follows\nPredicted  Actual\n"
      f"{np.concatenate((rf_predict.reshape(len(rf_predict), 1), y_test.reshape(len(y_test), 1)), 1)}\n")

# Support Vector Regression Model

print("--------\nSVR Regression\n--------")
# converting 1D array to 2D array
# y = y.reshape(len(y), 1)

# Splitting the data again as we converted y from 1D array to 2D array
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# -----

sc_x = StandardScaler()
sc_y = StandardScaler()
sc_x_train = sc_x.fit_transform(x_train)
sc_y_train = sc_y.fit_transform(y_train.reshape(len(y_train), 1))  # converting 1D array to 2D array

svr_regressor = SVR(kernel='rbf')
svr_regressor.fit(sc_x_train, sc_y_train.ravel())
# .ravel will convert that array shape to (n, ) (i.e. flatten it)

svr_predict = sc_y.inverse_transform(svr_regressor.predict(sc_x.fit_transform(x_test)).reshape(-1, 1))
svr_score = r2_score(y_test, svr_predict)

print(f"****\nThe prediction for SVR Regression Model is as follows\nPredicted  Actual\n"
      f"{np.concatenate((svr_predict.reshape(len(svr_predict), 1), y_test.reshape(len(y_test), 1)), 1)}\n")

# Score for Regression Models

print("--------\nR2 Scores for Models\n--------\n")
print(f"Notes : \n"
      f"------\n"
      f"Normal Regression projection and taking difference between prediction value and actual value\n"
      f"SSres = SUM(yi - yi') ^ 2  (i.e., SSres = Residue sum of squares )\n"
      f"Taking average of y axis value and then taking the difference between"
      f" the actual value and avg of y projection\n"
      f"SStot = SUM(yi - yavg) ^ 2 (i.e., SStot = Total sum of squares )\n"
      f"Final Formula : R2 = 1 - ( SSres / SStot )  ( *Note :  R2 values will be between 0 and 1 )\n"
      f"**Important\n"
      f"As we increase the independent variables the SSres decrease or stay the same because of the Ordinary Least "
      f"SquaresSolution to this is Adjusting R2\n"
      f"Adjusted Formula: Adj R2 = 1 - ( 1 - R2 ) * ( (n - 1) / ( n - k - 1) )"
      f" (i.e., k = Number of independent variables, n = sample size)\n")
print(f"Rule of thumb for R2 value : \n"
      f"= 1.0 = Perfect fit (suspicious)\n"
      f"~ 0.9 = Very Good\n"
      f"< 0.7 = Not great\n"
      f"< 0.4 = Terrible\n"
      f"< 0.0 = Model Makes no sense on the data")
print("--------\n")

print(f"Multiple Linear Regression : {lr_score}")
print(f"Polynomial Regression : {ply_score}")
print(f"SVR (Support Vector Regression) : {svr_score}")
print(f"Decision Tree Regression : {dt_score}")
print(f"Random Forest Regression : {rf_score}")

dict_scores = {
    "Multiple Linear Regression": lr_score,
    "Polynomial Regression": ply_score,
    "SVR": svr_score,
    "Decision Tree Regression": dt_score,
    "Random Forest Regression": rf_score
}

list_scores = [lr_score, ply_score, svr_score, dt_score, rf_score]
list_scores.sort(reverse=True)

for key, value in dict_scores.items():
    if list_scores[0] == value:
        print(f"\n--------\nBest Model for the given Dataset\n--------\n"
              f"Dataset : {data} \nModel : {key}\nR2 Score : {value}")
        break
