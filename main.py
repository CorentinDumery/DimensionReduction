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

from JLT_methods   import *
from test_problems import *
from data_digits   import *
from data_highd    import *
#from data_taxis import *

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

    dim_red_factor = [0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    
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
            naive_labels = evaluate(data, test_problem, n_clusters, gt_labels=labels, R=R)

            # For some dim. red
            for kn in dim_red_factor:
                k = int(kn * n)
                if k > d or k < 1:
                    continue

                print()
                print("\t\tLJT %f + %s..." % (kn, test_problem.__doc__))
                print("\t\tDim. reduction: %d -> %d" % (d, k))

                reduced_data = method_fun(data, k)

                # if(labels != None)
                # evaluate(reduced_data, test_problem, n_clusters, gt_labels=labels, R=R)
                evaluate(reduced_data, test_problem, n_clusters, gt_labels=naive_labels, R=R)

    print()

# Use KMeans as clustering method for comparison
def evaluate(data, test_problem, k, gt_labels, R=2):

    time_elapsed  = 0.
    v_measure_avg = 0.
    v_measure_max = 0.
    v_measure_min = 1.

    for r in range(R):
        time_elapsed -= time.time()
        clust = test_problem(data, k)
        time_elapsed += time.time()
        if(gt_labels != None)
            v_measure = metrics.v_measure_score(gt_labels, clust)
            v_measure_avg += v_measure
            v_measure_max = max(v_measure, v_measure_max)
            v_measure_min = min(v_measure, v_measure_min)

    time_elapsed  /= R
    v_measure_avg /= R

    print("\t\tTime: %f secs." % (time_elapsed))
    print("\t\tv-measure: %0.3f" % (v_measure_min, v_measure_avg, v_measure_max))

    return time_elapsed, v_measure_avg

if __name__ == "__main__":
    main()
