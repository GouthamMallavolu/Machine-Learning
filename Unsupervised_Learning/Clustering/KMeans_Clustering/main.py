"""

K-Means Clustering:
                In this unlabelled data is sorted using no of clusters we want to use based on the distance between
                data and cluster points.
                The best method to know how many clusters we need is by using Elbow method

                Formula:
                WCSS (Within cluster sum of squares)
                WCSS = Σ 1ton distance(i, b0) ^ 2 + Σ 1ton distance(i, b1) ^ 2
                (i.e., i = data point, b0, b1 = clusters )

"""

# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
# Selecting only two columns just to best visualize in 2D plot
x = dataset.iloc[:, [3, 4]].values

# Using the Elbow method to get the best optimal cluster need to used on the dataset by identifying
# which one has less WSCC value
n = int(input("Enter the Max number of clusters you need to test on : "))
WSCC = []
for i in range(1, n + 1):
    km_cluster = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=n)
    km_cluster.fit(x)
    # Now we need to append WSCC of each iteration with different number of clusters in to a list
    WSCC.append(km_cluster.inertia_)

# Visualizing the WSCC range
plt.scatter(range(1, n + 1), WSCC, color='red')
plt.plot(range(1, n + 1), WSCC)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS Values')
plt.show()

# Training the dataset with optimal clusters we observed in previous graph
km_cluster = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=5)
x_predict = km_cluster.fit_predict(x)

# Visualizing
colors = ['red', 'green', 'cyan', 'orange', 'yellow']
label = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']
for i in range(5):
    plt.scatter(x[x_predict == i, 0], x[x_predict == i, 1], color=colors[i], s=50, label=label[i])

plt.scatter(km_cluster.cluster_centers_[:, 0], km_cluster.cluster_centers_[:, 1], s=70, label='Cluster Center')
plt.title('K-Means Clustering on Dataset')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
