import numpy as np
# from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth

def kmeans_(data, k):
    """K-means"""
    kmeans = KMeans(
        n_clusters=k
        # , random_state=4
        ).fit(data)
    return kmeans.labels_

def meanshift_(data, k):
    """MeanShift"""
    # The following bandwidth can be automatically detected using
    # Check parameters ...
    # bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=500)

    ms = MeanShift(
        # bandwidth=bandwidth,
        bin_seeding=True
    ).fit(data)
    return ms.labels_
