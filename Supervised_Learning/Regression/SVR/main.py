# Support Vector Regression ( Invented by Vladimir Vapnik at Bell Labs in 90's )

# Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# ***************** Optional Part ***********************
# Added this part to Visualize the difference between Linear, Polynomial Linear, SVR Models prediction

# Training the Linear regression model on whole dataset
linear_regressor = LinearRegression()
linear_regressor.fit(x, y)

# Training the Polynomial Regression on whole dataset
polynomial_regressor = PolynomialFeatures(degree=6)
x_polynomial = polynomial_regressor.fit_transform(x)
linear_regressor_poly = LinearRegression()
linear_regressor_poly.fit(x_polynomial, y)

# ********************************************************

# Changing the 1D array to 2D array and reshaping the data of y for SVR Model
y = y.reshape(len(y), 1)

# Feature Scaling required for SVR Model
sc_x = StandardScaler()
x = sc_x.fit_transform(x)

sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# Training the SVR Model on whole dataset
regressor = SVR(kernel='rbf')
regressor.fit(x, y)

# # Visualizing SVR result
# plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
# plt.plot(sc_x.inverse_transform(x), predict, color='blue', label='SVR Model')
# plt.title('Truth or Bluff (SVR Model)')
# plt.xlabel('Position Levels')
# plt.ylabel('Salaries')
# plt.show()

# # Visualizing in High resolution and smooth curve for SVR Model
# x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
# x_grid = x_grid.reshape((len(x_grid), 1))
# plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
# plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid)).reshape(-1, 1)), color='blue')
# plt.title('Truth or Bluff (SVR Model)')
# plt.xlabel('Position Levels')
# plt.ylabel('Salaries')
# plt.show()

# Visualizing for Linear, Polynomial, SVR Model Regressions
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')

# Linear Regression Plot
plt.plot(sc_x.inverse_transform(x), linear_regressor.predict(sc_x.inverse_transform(x)), color='cyan',
         label='Linear Regression')

# Polynomial Linear Regression Plot
plt.plot(sc_x.inverse_transform(x), linear_regressor_poly.predict(x_polynomial), color='magenta',
         label='Polynomial Regression')

# SVR Model Plot
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x).reshape(-1, 1)), color='blue',
         label='SVR Model')

plt.title('Truth or Bluff ( Linear, Polynomial, SVR )')
plt.xlabel('Position Levels')
plt.ylabel('Salaries')
plt.legend()
plt.show()
