"""

Eclat:
    This is similar to Apriori, but we only consider support in Eclat
    support = p(item)/total no of transactions

"""

# Importing the Libraries
import pandas as pd
from apyori import apriori

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

# converting the dataset from dataframe to list as apriori function only accepts data in list of strings
# list_dataset = dataset.values.tolist()
# or
list_dataset = []
for i in range(len(dataset)):
    list_dataset.append([str(dataset.values[i, j]) for j in range(len(dataset.values[i, ]))])

# Training the dataset on apriori function
ap_rules = apriori(transactions=list_dataset, min_support=0.003, min_confidence=0.2,
                   min_lift=3, min_length=2, max_length=2)

ap_rules = list(ap_rules)


# Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))


results_DataFrame = pd.DataFrame(inspect(ap_rules), columns=['Product1', 'Product2', 'Support'])

# Visualizing data unsorted
print(results_DataFrame)

# Sorting data as per the support
print("\n")
print(results_DataFrame.nlargest(n=10, columns='Support'))
