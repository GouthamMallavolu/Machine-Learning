"""
Decision Tree Intuition
-----------------------
         |-------> Classification Trees
Cart ---|
       |--------> Regression Trees
"""

# ---- Decision Tree Regression Model -------
# There is no need of Feature Scaling for this Model
# Note: Decision Tree Regression Model is not recommended if the data has only one feature

# Import the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression Model on the whole dataset
dt_regressor = DecisionTreeRegressor(random_state=0)
dt_regressor.fit(x, y)

# Visualizing the prediction of Decision Tree Regression Model with High Resolution
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, dt_regressor.predict(x_grid), color='blue', label='Decision Tree Regression Model')
plt.title('Decision Tree Regression Model')
plt.xlabel('Position Levels')
plt.ylabel('Salaries')
plt.show()
