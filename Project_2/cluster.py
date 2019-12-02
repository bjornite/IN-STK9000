import numpy as np
import pandas

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Read the data
features = pandas.read_csv('./medical/historical_X.dat', header=None, sep=" ").values
actions = pandas.read_csv('./medical/historical_A.dat', header=None, sep=" ").values
outcome = pandas.read_csv('./medical/historical_Y.dat', header=None, sep=" ").values
observations = features[:, :128]
symptoms = features[:,128] + features[:,129]*2

from sklearn import cluster
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering


np.random.seed(200)
Xtrain, Xholdout, ytrain, yholdout = train_test_split(features, outcome, test_size = 0.2)

df = np.append(observations, symptoms[:,None], 1)
symptoms_true = df[np.where(df[:,128] != 0)]
symptoms_false = df[np.where(df[:,128] == 0)]

data = symptoms_true[:,:128]


from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import cdist
from yellowbrick.cluster import KElbowVisualizer


#km = KModes(n_clusters=4, init='Cao', n_init=5, verbose=1)
#clusters = km.fit_predict(observations)
"""
model = KModes(init='Cao', n_init = 5, n_jobs=-1, verbose = 1)
visualizer = KElbowVisualizer(model, k=(2,15), timings = False, locate_elbow = False)
visualizer.fit(observations)        # Fit the data to the visualizer
visualizer.show(outpath = "Images/kelbow_kmodes.png")

cost = []
for num_clusters in list(range(1,15)):
    kmode = KModes(n_clusters=num_clusters, init = "Cao", verbose=1)
    kmode.fit_predict(observations)
    cost.append(kmode.cost_)

y = np.array([i for i in range(1,15,1)])
plt.plot(y,cost)
plt.xlabel('Clusters')
plt.ylabel('Cost')
plt.savefig('Images/cost_KModes')
plt.show()
"""

# Use PCA to reduce the dimensionality
from sklearn.decomposition import PCA

pca = PCA(.70)
data_red = pca.fit_transform(observations)
print(pca.n_components_)

model = KModes(init='Cao', n_jobs=-1, verbose = 1)
visualizer = KElbowVisualizer(model, k=(2,15), timings = False, locate_elbow = False)
visualizer.fit(data_red)        # Fit the data to the visualizer
visualizer.show(outpath = "Images/kelbow_kmodes_pca.png")
plt.show()

cost = []
for num_clusters in list(range(1,15)):
    kmode = KModes(n_clusters=num_clusters, init = "Cao", verbose=1)
    kmode.fit_predict(data_red)
    cost.append(kmode.cost_)

y = np.array([i for i in range(1,15,1)])
plt.plot(y,cost)
plt.xlabel('Clusters')
plt.ylabel('Cost')
plt.savefig('Images/cost_KModes_PCA')


"""
k_max = 10
models = [cluster.KMeans(k).fit(observations) for k in range(1, k_max)]
dplot = m.inertia_ for m in models

plt.plot(range(1, k_max), ([(m.inertia_/k for k in range(1, k_max)) for m in models]))
plt.title('Elbow plot KMeans PCA(.50)')
plt.xlabel('k-clusters')
#plt.ylabel('')
plt.show()

model = cluster.KMeans()
visualizer = KElbowVisualizer(model, k=(2,15), timings = False)
visualizer.fit(observations)        # Fit the data to the visualizer
visualizer.show()

#divide by k. > inertia per cluster



# K-means clustering
k_max = 10
models = [cluster.KMeans(k).fit(observations) for k in range(1, k_max)]
print(models)
print(type(models))

plt.plot(range(1, k_max), [m.inertia_ for m in models])
plt.show()

clusters = models[3].labels_

# Hierarchical clustering -> uses dendrogram to find clusters
model1 = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward').fit_predict(observations)
print(models1)



obs_clusters = np.append(observations, clusters[:,None], 1)
obs_clusters = np.append(obs_clusters, symptoms[:,None], 1)

data_symptoms_true = obs_clusters[np.where(obs_clusters[:,129] != 0)]
data_symptoms_false = obs_clusters[np.where(obs_clusters[:,129] == 0)]

clusters_no_symp = data_symptoms_false[:,128]
clusters_symp = data_symptoms_true[:,128]

fig1, ax = plt.subplots(figsize = (15, 5), ncols = 3)
sns.countplot(clusters, ax = ax[0])
sns.countplot(clusters_no_symp, ax = ax[1])
sns.countplot(clusters_symp, ax = ax[2])

fig1.suptitle("KMeans clusters", fontsize=14)
ax[0].set_title('Countplot of data per cluster')
ax[1].set_title('Countplot no symptoms data per cluster')
ax[2].set_title('Countplot symptoms data per cluster')
plt.show()
"""
