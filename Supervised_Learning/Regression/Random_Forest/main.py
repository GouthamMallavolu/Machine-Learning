# Random Forest Regression

# Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the dataset with Random Forest Regression
rf_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
rf_regressor.fit(x, y)

# Predict the data
print(rf_regressor.predict([[8]]))

# Visualizing the prediction of Decision Tree Regression Model with High Resolution
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, rf_regressor.predict(x_grid), color='blue', label='Random Forest Regression Model')
plt.title('Random Forest Regression Model')
plt.xlabel('Position Levels')
plt.ylabel('Salaries')
plt.show()