# Import the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import the dataset
dataset = pd.read_csv('Salary_Data.csv')

# Splitting the dataset to Independent and Dependent sets
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the sets to training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Linear regression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

print(lr.coef_)
print(lr.intercept_)

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, lr.predict(x_train), color='blue')
plt.title('Salary vs Experience ( Training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
