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
import random

from JLT_methods   import *
from test_problems import *

from data_digits   import *
from data_highd    import *
from data_taxis    import *

import warnings
def warn(*args, **kwargs):
    pass


#########################################################################
#########################################################################

# Number of repeated runs
R = 3

data, labels, n_clusters = loadDigits()
# data, labels, n_clusters = loadMNIST()
# data, labels, n_clusters = loadTaxis()
# data, labels, n_clusters = loadHighD_1024()

# TODO: real-world datasets
#  http://archive.ics.uci.edu/ml/datasets/Gisette
#  http://archive.ics.uci.edu/ml/datasets/Dorothea
#  http://archive.ics.uci.edu/ml/datasets/Dexter

# methods = [Achlioptas, FJLT, sampleDimensions]
methods = [Achlioptas, FJLT, sampleDimensions, selectMostVarDims, useLDA, usePCA, useFactorAnalysis]

DontTestAgainstGT = True
CleanOutput = True

# test_problems = [kmeans_]
test_problems = [kmeans_, meanshift_]
# test_problems = [meanshift_]

# dim_red_factor = [0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
dim_red_factor = [0.5, 0.2, 0.1, 0.01, 0.001, 0.0001]

#########################################################################
#########################################################################

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

    np.random.seed(0)

    methods_names = {
        "Achlioptas"        : "Achlioptas' coin flip method",
        "FJLT"              : "Fast JL Transform method",
        "sampleDimensions"  : "Simple dimensions sampling",
        "selectMostVarDims" : "Select most variant dimensions",
        "useLDA"            : "LDA",
        "usePCA"            : "PCA",
        "useFactorAnalysis" : "FactorAnalysis",
    }

    n, d = data.shape

    print("n * d = %d * %d" % (n, d))
    print("n_clusters = %d" % n_clusters)
    print("R = %d" % (R))

    # For each method
    for method_fun in methods:

        print()

        try:
             print(methods_names[method_fun.__name__])
        except ValueError:
             print("Please select a transform method")

        # For test problem
        for test_problem in test_problems:
            print("\tPure " + test_problem.__doc__ + "...")

            test_against = [{'name' : "gr-truth", 'labs' : labels}]
            if labels is None:
                test_against = []
            naive_labels = evaluate(data, test_problem, n_clusters, test_against, R=R)

            # For some dim. red
            for kd in dim_red_factor:
                k = int(kd * d)
                if k > d or k < 1:
                    continue
                if method_fun == useLDA and k > min(d, len(np.unique(n_clusters))-1):
                    continue

                print("\tLJT %g + %s..." % (kd, test_problem.__doc__))
                print("\t\tDim. reduction: %d -> %d" % (d, k))


                test_against = [{'name' : "gr-truth", 'labs' : labels}, {'name' : "baseline", 'labs' : naive_labels}]
                if labels is None or DontTestAgainstGT:
                    test_against = [{'name' : "baseline", 'labs' : naive_labels}]

                evaluate(data, test_problem, n_clusters, test_against, R=R, reduce_by=(method_fun, k))

    print()

# Use KMeans as clustering method for comparison
def evaluate(data, test_problem, n_clusters, base_labels_arrs, R=2, reduce_by=None):
    b = len(base_labels_arrs)

    time_elapsed_avg  = 0.

    # For each labels to test against
    for i,base in enumerate(base_labels_arrs):
        labels = base['labs']

        v_measure_avg = 0
        v_measure_min = 1
        v_measure_max = 0

        # REPEAT R times: test & evaluate the reduced data
        for r in range(R):

            if reduce_by != None:
                method_fun = reduce_by[0]
                k = reduce_by[1]
                reduced_data = method_fun(data, k, labels, silent=CleanOutput)
                if reduced_data.shape[1] != k:
                    ValueError("Error! reduction failed:")
                    print(data.shape, reduced_data.shape)
            else:
                reduced_data = data

            time_elapsed_avg -= time.time()
            clust = test_problem(reduced_data, n_clusters)
            time_elapsed_avg += time.time()

            v_measure = metrics.v_measure_score(labels, clust)
            v_measure_avg += v_measure
            v_measure_min = min(v_measure, v_measure_min)
            v_measure_max = max(v_measure, v_measure_max)

        v_measure_avg  /= R

        # print("\t\tv-measure/%s:\t[%0.3f {%0.3f} %0.3f]" % (base['name'], v_measure_min, v_measure_avg, v_measure_max))
        print("\t\tv-measure/%s:\t%0.3f +/- %0.2f" % (base['name'], v_measure_avg, (v_measure_max-v_measure_min)/2/v_measure_avg))

    time_elapsed_avg /= (R*len(base_labels_arrs))

    print("\t\tAvg. time: %f secs." % (time_elapsed_avg))

    return clust

if __name__ == "__main__":
    main()
