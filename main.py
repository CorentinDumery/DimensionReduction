#!/usr/bin/env python3

## Note : the current file reading is sequential, need to find how to directly
##  access row i
## GIO: what do you mean? is it about NIST? is it about the fact that digits is a "bunch" and not a numpy array?

## TODO : find better dataset ( https://leon.bottou.org/projects/infimnist )

from math import log,sqrt
import numpy as np
from random import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_digits
import sys

from JLT_methods import *
from digits_data import *
#from taxis_data import *

##########################################################
####################### ENTRY POINT ######################
##########################################################

def main():

    print()
    print("### Johnson-Lindenstrauss Transform ###")
    print()

    #########################################################################
    #########################################################################

    # Number of repeated runs
    R = 5

    data, labels, n_clusters = loadDigits()
    #data, labels, n_clusters = loadMNIST()

    load_phi = Achlioptas_phi
    # load_phi = FJLT_phi
    cs = [1, 10, 100]
    epss = [0.1, 0.5, 1., 2., 3.]

    #########################################################################
    #########################################################################

    n, d = data.shape

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
            print("FLJT + KMeans... (c, eps =  %d, %f)" % (c, eps))
            print("Dim. reduction: %d -> %d" % (d, k))

            phi = load_phi(n, d, k)

            #reduced_data = [phi.dot(descriptor(entry)) for entry in data]

            reduced_data = data.dot(phi.T)
            evaluateKMeans(reduced_data, n_clusters, labels, R)

    print()

def evaluateKMeans(data, k, labels, R=2):
    n = data.shape[0]

    time_elapsed  = 0.
    score = 0
    error = 0

    for i in range(R):
        time_elapsed -= time.time()
        kmeans = KMeans(n_clusters=k
            # , random_state=4
            ).fit(data)
        time_elapsed += time.time()

        predictions = kmeans.predict(data)
        score += sum([1 for i,j in zip(predictions, labels) if i == j])
        error += kmeans.score(data)

    print("KMeans took %f secs." % (time_elapsed/R))
    print("KMeans precision: %f" % (score/R/n))
    print("KMeans score: %f" % (error/R))
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
