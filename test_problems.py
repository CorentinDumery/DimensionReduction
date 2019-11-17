import numpy as np
# from sklearn.svm import SVC
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.spatial.distance import cdist

# K-Means
def kmeans_(data, n_clusters, y):
    """K-means"""
    kmeans = KMeans(
        n_clusters=n_clusters,
        # , random_state=4,
        n_jobs=-1,
        ).fit(data)
    return kmeans.labels_

# K-Nearest Neighbors
def kneigh_(data, n_clusters, y):
    """KNN"""
    k=max(10, int(data.shape[0]/n_clusters/3))
    neigh = KNeighborsClassifier(
        n_neighbors=k,
        n_jobs=-1,
        )
    neigh.fit(data, y)
    return neigh.predict(data)


###########################################################################
# Other Attempts...
###########################################################################

def spectr_(data, n_clusters, y):
    """Spectral Clustering"""
    sc = SpectralClustering(
        n_clusters=n_clusters,
        assign_labels="discretize",
        n_jobs=-1,
        ).fit(data)
    return sc.labels_



def mshift_(data, n_clusters, y):
    """MeanShift"""
    # The following bandwidth can be automatically detected using
    # Check parameters ...
    n,d = data.shape
    # bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=10)

    ms = MeanShift(
        bandwidth=1.,
        seeds=data[np.random.randint(n, size=int(n/10))],
        bin_seeding=False
    ).fit(data)

    cluster_centers = ms.cluster_centers_
    labels = ms.labels_

    return labels

    # Merge cluster_centers until you have the desired number
    while len(cluster_centers) > n_clusters:
        # print(cluster_centers.shape)
        dists = cdist(cluster_centers, cluster_centers, 'euclidean')
        min_dists = dists.argsort(axis=1)[:,1]

        min_i = 0
        for i,min_j in enumerate(min_dists):
            if dists[i,min_j] < dists[min_i,min_dists[min_i]]:
                min_i = i
        # Min: dists[min_i,min_dists[min_i]]
        # join clusters min_i and min_dists[min_i]
        min_j = min_dists[min_i]

        # print()
        # print(len(cluster_centers))
        # print(len(np.unique(labels)))
        # print(labels.shape, data.shape, cluster_centers.shape)
        # print("merging")
        # print(min_i, min_j)
        # print(labels[np.any([labels == min_i, labels == min_j], axis = 0)])
        new_center = data[np.any([labels == min_i, labels == min_j], axis = 0)].mean(axis=0)

        cluster_centers[min_i] = new_center
        labels[labels == min_j] = min_i

        cluster_centers = np.delete(cluster_centers, min_j,0)
        for j in range(min_j, len(cluster_centers)):
            labels[labels == j+1] = j


        if not len(cluster_centers) == len(np.unique(labels)):
            raise ValueError("ciao")


    # print(len(np.unique(labels)))
    # print(len(cluster_centers))
    return labels
