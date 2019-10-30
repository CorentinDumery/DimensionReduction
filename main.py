#!/usr/bin/env python3

## Note : the current file reading is sequential, need to find how to directly
##  access row i
## GIO: what do you mean? is it about NIST? is it about the fact that digits is a "bunch" and not a numpy array?

## TODO : find better dataset ( https://leon.bottou.org/projects/infimnist )

from math import log,sqrt
import numpy as np
from random import random
import matplotlib.pyplot as plt
import time
from sklearn import metrics
import sys

from JLT_methods import *
from digits_data import *
from test_problems import *
#from taxis_data import *

##########################################################
####################### ENTRY POINT ######################
##########################################################

# This script compares the performance of dimensionality reduction
#  by means of JLT transforms methods over some available datasets

def main():

    print()
    print("#######################################")
    print("### Johnson-Lindenstrauss Transform ###")
    print("#######################################")
    print()

    #########################################################################
    #########################################################################

    # Number of repeated runs
    R = 5

    data, labels = loadDigits()
    # data, labels = loadMNIST()

    methods = [Achlioptas_phi, FJLT_phi]
    test_problems = [kmeans_]

    cs = [1, 10, 100]
    epss = [0.1, 0.5, 1., 2., 3.]

    #########################################################################
    #########################################################################

    n, d = data.shape
    n_clusters = len(np.unique(labels))

    print("n * d = %d * %d" % (n, d))
    print("k = %d" % n_clusters)
    print("R = %d" % (R))
    print()

    # For each method
    for method_fun in methods:

        if method_fun == Achlioptas_phi:
            print("Achlioptas' coin flip method")
        elif method_fun == FJLT_phi:
            print("Fast JL Transform method")

        print()

        # For test problem
        for test_problem in test_problems:
            print("\tPure " + test_problem.__doc__ + "...")
            naive_labels = evaluate(data, test_problem, n_clusters, labels, R)

            # For some dimensionality ... TODO: remove c and eps
            for c in cs:
                for eps in epss:
                    k = int(c*log(n)/(eps**2))+1
                    if k > d:
                        continue

                    print()
                    print("\t\tLJT (c, eps =  %d, %.2f) + %s..." % (c, eps, test_problem.__doc__))
                    print("\t\tDim. reduction: %d -> %d" % (d, k))

                    reduced_data = method_fun(data, k)

                    # evaluate(reduced_data, test_problem, n_clusters, labels, R)
                    evaluate(reduced_data, test_problem, n_clusters, naive_labels, R)

    print()

# Use KMeans as clustering method for comparison
def evaluate(data, test_problem, k, gt_labels, R=2):

    time_elapsed  = 0.
    v_measure = 0.

    for r in range(R):
        time_elapsed -= time.time()
        clust = test_problem(data, k)
        time_elapsed += time.time()
        v_measure    += metrics.v_measure_score(gt_labels, clust)

    time_elapsed /= R
    v_measure    /= R

    print("\t\tTime: %f secs." % (time_elapsed))
    print("\t\tV-measure: %0.3f" % (v_measure))

    return time_elapsed, v_measure

if __name__ == "__main__":
    main()
