#!/usr/bin/env python3

## Note : the current file reading is sequential, need to find how to directly
##  access row i
## GIO: what do you mean? is it about NIST? is it about the fact that digits is a "bunch" and not a numpy array?

## TODO : find better dataset ( https://leon.bottou.org/projects/infimnist )

from math import log,sqrt
import numpy as np
from random import random
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_digits
from sklearn import metrics
import sys

from JLT_methods import *
from digits_data import *
#from taxis_data import *

##########################################################
####################### ENTRY POINT ######################
##########################################################

# This script compares the performance of dimensionality reduction
#  by means of JLT transforms methods over some available datasets

def main():

    print()
    print("### Johnson-Lindenstrauss Transform ###")
    print()

    #########################################################################
    #########################################################################

    # Number of repeated runs
    R = 5

    data, labels = loadDigits()
    #data, labels = loadMNIST()

    load_phi = Achlioptas_phi
    # load_phi = FJLT_phi
    cs = [1, 10, 100]
    epss = [0.1, 0.5, 1., 2., 3.]

    #########################################################################
    #########################################################################

    n, d = data.shape
    n_clusters = len(np.unique(labels))

    print("n, d = %d * %d" % (n, d))
    print("# of clusters = %d" % n_clusters)

    print()
    print("R = %d" % (R))
    if load_phi == Achlioptas_phi:
        print("Using Achlioptas' coin flip method")
    elif load_phi == FJLT_phi:
        print("Using Fast JL Transform method")
    print()
    print("Pure KMeans...")

    #data = [descriptor(entry) for entry in data]

    evaluateKMeans(data, n_clusters, labels, R)

    for c in cs:
        for eps in epss:
            k = int(c*log(n)/(eps**2))+1
            if k > d:
                continue

            print()
            print("LJT (c, eps =  %d, %.2f) + KMeans..." % (c, eps))
            print("\tDim. reduction: %d -> %d" % (d, k))

            phi = load_phi(n, d, k)

            #reduced_data = [phi.dot(descriptor(entry)) for entry in data]

            reduced_data = data.dot(phi.T)
            evaluateKMeans(reduced_data, n_clusters, labels, R)

    print()

# Use KMeans as clustering method for comparison
def evaluateKMeans(data, k, labels, R=2):
    n, d = data.shape

    time_elapsed  = 0.
    accuracy = 0
    score = 0
    score2 = 0
    error = 0

    # Source https://stackoverflow.com/questions/28344660/how-to-identify-cluster-labels-in-kmeans-scikit-learn
    # Rosenberg and Hirschberg (2007)
    # https://scikit-learn.org/stable/modules/clustering.html#homogeneity-completeness
    homogeneity = 0
    completeness = 0
    v_measure = 0

    for r in range(R):

        time_elapsed -= time.time()
        kmeans = KMeans(n_clusters=k
            # , random_state=4
            ).fit(data)
        time_elapsed += time.time()

        predictions = kmeans.predict(data)

        # instead of computing the accuracy...
        #  labels_reassignment = [1,5,2,7,0,...]
        #  accuracy += metrics.accuracy_score(predictions[labels_reassignment], labels)

        # adhoc_metric = 0
        # for prediction in predictions:
        #     siblings_pred = [i for i,x in enumerate(predictions) if x == prediction]
        #     siblings = [i for i,x in enumerate(labels) if x == prediction]
        #     adhoc_metric += len(set(siblings_pred) & set(siblings))

        # let's compute the rate of good pairs clustered together
        total_good = 0
        good = 0
        for i in range(n):
            for j in range(i+1, n):
                if labels[i] == labels[j]:
                    total_good += 1
                    if predictions[i] != predictions[j]:
                        good += 1

        #score        += adhoc_metric/(2 * n**2)
        score2       += good/total_good
        error        += kmeans.score(data)
        homogeneity  += metrics.homogeneity_score(labels, kmeans.labels_)
        completeness += metrics.completeness_score(labels, kmeans.labels_)
        v_measure    += metrics.v_measure_score(labels, kmeans.labels_)

    print("\tTime: %f secs." % (time_elapsed/R))
    # print("\tAccuracy: %f" % (score/R))
    #print("\tScore: %f" % (score/R))
    # print("\tScore2: %f" % (score2/R))
    # print("\tEnergy: %.2f"     % (error/R))

    # print("Homogeneity: %0.3f" % (homogeneity/R))
    # print("Completeness: %0.3f" % (completeness/R))
    print("V-measure: %0.3f" % (v_measure/R))
    #print(kmeans.labels_)
    #print(kmeans.cluster_centers_)

    """
    for center in kmeans.cluster_centers_:
        #try to see what they correspond to
        best = (-1,np.inf)
        for i in range(len(data)):
            #l = linearizeMatrix(im)
            l = data[i]
            dist = 0
            for j in range(len(l)):
                dist += abs(center[j]-l[j])
            if dist < best[1]:
                best = (i,dist)
    """

if __name__ == "__main__":
    main()
