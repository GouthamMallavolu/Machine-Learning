# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# encoding the categorical data
ct = ColumnTransformer(transformers=[('encoder',  OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Split training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# train the model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

model_predict = regressor.predict(x_test)

np.set_printoptions(precision=2)
print(np.concatenate((model_predict.reshape(len(model_predict), 1), y_test.reshape(len(y_test), 1)), axis=1))

print("--------------------------------")

# The profit of a startup with R&D Spend = 160000, Administration Spend = 130000,
# Marketing Spend = 300000 and State = California

print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

print("-------------------------------")

# The final regression equation y = b0 + b1 x1 + b2 x2 + ... with the final values of the coefficients

inp = [1, 0, 0, 160000, 130000, 300000]
coef = regressor.coef_
tmp = 0

for i in range(len(inp)):
    tmp += coef[i] * inp[i]

print(regressor.intercept_ + tmp)