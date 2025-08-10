# Non-Linear Regression Model

# Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset to training and test set
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training the Linear regression model on whole dataset
linear_regressor = LinearRegression()
linear_regressor.fit(x, y)

# Training the Polynomial Regression on whole dataset
polynomial_regressor = PolynomialFeatures(degree=6)
x_polynomial = polynomial_regressor.fit_transform(x)
linear_regressor_poly = LinearRegression()
linear_regressor_poly.fit(x_polynomial, y)

# Graph for Linear Regression model
plt.scatter(x, y, color='red')
plt.plot(x, linear_regressor.predict(x), color='blue')
plt.title('Truth or Bluff (Linear Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()

# Graph for Polynomial Linear Regression model
plt.scatter(x, y, color='red')
plt.plot(x, linear_regressor_poly.predict(x_polynomial), color='blue')
plt.title('Truth or Bluff (Polynomial Linear Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()

# Predicting the outcome with Linear Regression Model
print(f"Prediction for Linear Regression : {linear_regressor.predict([[6.5]])}")

# Predicting the outcome with Polynomial Regression Model
print(f"Prediction for Polynomial Regression : {linear_regressor_poly.predict(polynomial_regressor.fit_transform([[6.5]]))}")

# -----------------------------------
# ********* TESTING ***************
# -----------------------------------

print("--------------------------------")

predict = 6.5

# Backend process happening in Linear Regression model
# y = b0 + b1 x1 (i.e., b0 = interception, b1 = coefficent, x1 = Feature)

linear_tmp = predict * linear_regressor.coef_

print(f"Prediction for Linear Regression manually : {linear_regressor.intercept_ + linear_tmp}")

print("---------------------------------")

# Backend process happening in Polynomial Linear Regression model
# y = b0 + (b1 * x1 ^ 1) + (b2 * x1 ^ 2) ........... (bn * x1 ^ n)
# (i.e., b0 = interception, b1....n = coefficent, x1 = Feature, n = degree we want to give)

polynomial_tmp = 0
n = int(input("enter a degree : "))
predict_list = [predict ** i for i in range(1, n+1)]

for i in range(len(predict_list)):
    polynomial_tmp += predict_list[i] * linear_regressor_poly.coef_[i+1]

print(f"Prediction for Polynomial Regresssion manually : {[linear_regressor_poly.intercept_ + polynomial_tmp]}")
