# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy as sch

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
# Selecting only two columns just to best visualize in 2D plot
x = dataset.iloc[:, [3, 4]].values

# Creating dendrogram to get optimal number of clusters
dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

# Training the dataset on Hierarchical Clustering
hc_cluster = AgglomerativeClustering(n_clusters=5, linkage='ward', metric='euclidean')
x_predict = hc_cluster.fit_predict(x)

# Visualizing
colors = ['red', 'green', 'cyan', 'orange', 'yellow']
label = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']
for i in range(5):
    plt.scatter(x[x_predict == i, 0], x[x_predict == i, 1], color=colors[i], s=50, label=label[i])

# plt.scatter(hc_cluster.n_clusters_[:, 0], hc_cluster.n_clusters_[:, 1], s=70, label='Cluster Center')
plt.title('Hierarchical Clustering on Dataset')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
