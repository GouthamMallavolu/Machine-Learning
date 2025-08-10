"""

Standardization:
-----------------
               Also called Z-score Normalization.
               In this values will be transformed such a way that it will have the properties of standard normal
               distribution with mean(μ) = 0 and standard deviation(σ) = 1

Formula:
--------
              z = (x - μ) / σ

              ( i.e., μ = mean (formula: sum of values / total no of values),
                      σ = Standard deviation ( formula : sqrt ( ∑1 to n ( X − μ ) ^ 2 ) / n ) )

Python Library and function :
------------------------------
from sklearn.preprocessing import StandardScaler

"""

# Importing the Libraries
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Manually calculating standardization

# Calculating Mean (i.e., sum of values / Total No of values )

# Creating the lists for Age and salary and creating duplicates for those lists
list_age, list_age_1 = [x[i][0] for i in range(len(x))], [x[i][0] for i in range(len(x))]
list_sal, list_sal_1 = [x[i][1] for i in range(len(x))], [x[i][1] for i in range(len(x))]

# Calculating sum and average for both salary and age
age_sum, sal_sum = sum(list_age), sum(list_sal)
mean_age, mean_sal = age_sum / len(list_age), sal_sum / len(list_sal)

# Calculating Standard deviation Calculation (i.e., σ = sqrt ( ∑1 to n ( X − μ ) ^ 2 ) / n )

# Calculating the difference for the values with mean and squaring them (i.e., ( actual value - mean ) ^ 2 )
for i in range(len(list_age)):
    list_age[i] = (list_age[i] - mean_age) ** 2
    list_sal[i] = (list_sal[i] - mean_sal) ** 2

# Calculating the variance (i.e., sum and average after the difference)
age_v, sal_v = sum(list_age), sum(list_sal)
mean_age_v, mean_sal_v = age_v / len(list_age), sal_v / len(list_sal)

# Calculating standard deviation (i.e., sqrt (variance) )
age_sd, sal_sd = math.sqrt(mean_age_v), math.sqrt(mean_sal_v)

# Final Formula

# Applying to final formula  z = (x - μ) / σ
for i in range(len(list_age_1)):
    list_age_1[i] = (list_age_1[i] - mean_age) / age_sd
    list_sal_1[i] = (list_sal_1[i] - mean_sal) / sal_sd

# Printing the fina result
df = pd.DataFrame(list(zip(list_age_1, list_sal_1)), columns=['Age', 'Salary'])
print(df)

# Using Python package
sc = StandardScaler()
x = sc.fit_transform(x)
print(x)
