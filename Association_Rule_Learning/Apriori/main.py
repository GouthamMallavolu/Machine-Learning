# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

# Converting dataframe content to list as apyori function only accepts data in list format and values must be strings
# list_dataset = str(dataset.values.tolist())
list_dataset = []
for i in range(len(dataset)):
    list_dataset.append([str(dataset.values[i, j]) for j in range(len(dataset.values[i,]))])

# Training the dataset on apriori function
ap_rules = apriori(transactions=list_dataset, min_support=0.003, min_confidence=0.2,
                   min_lift=3, min_length=2, max_length=2)

ap_rules = list(ap_rules)


# Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))


results_DataFrame = pd.DataFrame(inspect(ap_rules),
                                 columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

# Visualizing data unsorted
print(results_DataFrame)

# Sorting data as per the preference
preference = input("\nEnter a column name you prefer to sort dataframe with : ")

print(results_DataFrame.nlargest(n=len(ap_rules), columns=preference))
