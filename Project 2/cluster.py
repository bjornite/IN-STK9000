import numpy as np
import pandas

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Read the data
features = pandas.read_csv('./medical/historical_X.dat', header=None, sep=" ").values
actions = pandas.read_csv('./medical/historical_A.dat', header=None, sep=" ").values
outcome = pandas.read_csv('./medical/historical_Y.dat', header=None, sep=" ").values
observations = features[:, :128]
labels = features[:,128] + features[:,129]*2

print(features.shape)

from sklearn import cluster
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances_argmin

np.random.seed(200)
Xtrain, Xholdout, ytrain, yholdout = train_test_split(features, outcome, test_size = 0.2)

"""
# K-means clustering
k = 5
kmeans = cluster.KMeans(n_clusters=k)
kmeans.fit(features)
y_kmeans = kmeans.predict(features)

def find_clusters(X, n_clusters, rseed = 2):
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        labels = pairwise_distances_argmin(X, centers)
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])

        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels

centers, labels = find_clusters(features, k)
"""
# Simple hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(Xtest, method='ward'))
plt.show()

k = 5
cluster = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
#euclidean = distance between datapoints, 'ward' = minimizes the variant between the clusters
cluster.fit_predict(features)

print(cluster.labels_)
