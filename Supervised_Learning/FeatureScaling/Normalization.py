"""
Normalization:
--------------
           The is also called Min - Max Normalization. In this approach we will scale down the values between 0 and 1.

Formula:
--------
        Xnom = (X - Xmin) / (Xmax - Xmin)
        (i.e., Xmin = minimum value in the column, Xmax = Maximum value in the column)

Python package for Normalization :
---------------------------------
from sklearn.preprocessing import MinMaxScaler

"""

# Importing the Libraries

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Importing the dataset

dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Manually calculating the Normalization values
list_age, list_age_1 = [x[i][0] for i in range(len(x))], [x[i][0] for i in range(len(x))]
list_sal, list_sal_1 = [x[i][1] for i in range(len(x))], [x[i][1] for i in range(len(x))]
list_age.sort()
list_sal.sort()
min_age, max_age, min_sal, max_sal = list_age[0], list_age[len(list_age) - 1],  list_sal[0], list_sal[len(list_sal) - 1]

for i in range(len(list_age_1)):
    list_age_1[i] = (list_age_1[i] - min_age) / (max_age - min_age)
    list_sal_1[i] = (list_sal_1[i] - min_sal) / (max_sal - min_sal)

df = pd.DataFrame(list(zip(list_age_1, list_sal_1)), columns=['Age', 'Salary'])
print(df)

# Now Using Python Package for Normalization
mm = MinMaxScaler()
x = mm.fit_transform(x)
print(x)
