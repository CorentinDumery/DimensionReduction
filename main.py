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

CleanOutput = True
# Number of repeated runs
R = 3

load_dataset_fun = loadDigits
# load_dataset_fun = loadMNIST
# load_dataset_fun = loadHighD_1024

# TODO: real-world datasets
# load_dataset_fun = loadTaxis
#  http://archive.ics.uci.edu/ml/datasets/Gisette
#  http://archive.ics.uci.edu/ml/datasets/Dorothea
#  http://archive.ics.uci.edu/ml/datasets/Dexter

methods = [Achliop, FastJLT, sampDim, HVarDim, useLDA, usePCA, FactAna]
# methods = [Achliop, FastJLT, sampDim]

# One might not want to eval against the ground truth
#  because the results against the baseline are sufficient (and very similar)
DontEvalAgainstGT = True

test_problems = [kneigh_, kmeans_] # spectr_
# test_problems = [kmeans_]
# test_problems = [mshift_]

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
    print("###################################################")
    print("### Dimensionality Reduction Methods Comparison ###")
    print("###################################################")
    print()

    np.random.seed(0)

    methods_names = {
        "Achliop"     : "Achlioptas' coin flip method",
        "FastJLT"     : "Fast Johnson-Lindenstrauss Transform method",
        "sampDim"     : "Simple dimensions sampling",
        "HVarDim"     : "Select high-variance dimensions",
        "useLDA"      : "LDA",
        "usePCA"      : "PCA",
        "FactAna"     : "FactorAnalysis",
    }


    data, labels, n_clusters = load_dataset_fun()
    n, d = data.shape

    print("n * d = %d * %d" % (n, d))
    print("n_clusters = %d" % n_clusters)
    print("R = %d" % (R))

    if CleanOutput:
        print("n\td\tn_clust\ttest_pr\teval_against\tmethod\tk\tn/k\tvmeas\tvmeas_e\ttime")
        # E.g 10000	784	mshift_	baseline	Achliop	392	0.5	0.000 +/- 0.00	14.589523

    # For each method
    for method_fun in methods:

        if not CleanOutput:
            print()
            try:
                 print(methods_names[method_fun.__name__])
            except ValueError:
                 print("Please select a transform method")

        # For test problem
        for test_problem in test_problems:

            if not CleanOutput:
                print("\tPure " + test_problem.__doc__ + "...")

            eval_against = [{'name' : "gr-truth", 'labs' : labels}]
            if labels is None:
                eval_against = []
            naive_labels = evaluate(data, test_problem, n_clusters, eval_against, R=R)

            # For some dim. red
            ks = np.unique((np.array(dim_red_factor)*100).astype(int).clip(1, d))
            ks[::-1].sort()
            for k in ks:
                kd = k/n

                if method_fun == useLDA and k > min(d, len(np.unique(n_clusters))-1):
                    continue

                if not CleanOutput:
                    print("\tLJT %g + %s..." % (kd, test_problem.__doc__))
                    print("\t\tDim. reduction: %d -> %d" % (d, k))


                eval_against = [{'name' : "gr-truth", 'labs' : labels}, {'name' : "baseline", 'labs' : naive_labels}]
                if labels is None or DontEvalAgainstGT:
                    eval_against = [{'name' : "baseline", 'labs' : naive_labels}]

                evaluate(data, test_problem, n_clusters, eval_against, reduce_by=(method_fun, k), R=R)

    print()

# Use KMeans as clustering method for comparison
def evaluate(data, test_problem, n_clusters, base_labels_arrs, reduce_by=None, R=2):
    n, d = data.shape
    b = len(base_labels_arrs)


    method_name = '-'
    k_name = '-'
    kd_name = '-'
    if reduce_by != None:
        method_fun = reduce_by[0]
        k = reduce_by[1]

        method_name = method_fun.__name__
        k_name = str(k)
        kd_name = "%.5f" % (k/d)

    # For each labels to evaluate against
    for i,base in enumerate(base_labels_arrs):
        labels = base['labs']

        time_elapsed_avg  = 0.
        v_measure_avg = 0
        v_measure_min = 1
        v_measure_max = 0

        # REPEAT R times: test & evaluate the reduced data
        for r in range(R):

            if reduce_by != None:
                reduced_data = method_fun(data, k, labels, silent=CleanOutput)
                if reduced_data.shape[1] != k:
                    print(data.shape, reduced_data.shape)
                    raise ValueError("Error! reduction failed:")
            else:
                reduced_data = data

            time_elapsed_avg -= time.time()
            clust = test_problem(reduced_data, n_clusters, labels)
            time_elapsed_avg += time.time()

            if len(np.unique(clust)) != n_clusters:
                print("Error! Check the code. The clustering routine only used %d of the %d available cluster(s)." % (len(np.unique(clust)), n_clusters))

            v_measure = metrics.v_measure_score(labels, clust)
            v_measure_avg += v_measure
            v_measure_min = min(v_measure, v_measure_min)
            v_measure_max = max(v_measure, v_measure_max)

        v_measure_avg    /= R
        time_elapsed_avg /= R

        # print("\t\tv-measure/%s:\t[%0.3f {%0.3f} %0.3f]" % (base['name'], v_measure_min, v_measure_avg, v_measure_max))
        if not CleanOutput:
            print("\t\tv-measure/%s:\t%0.3f +/- %0.2f" % (base['name'], v_measure_avg, (v_measure_max-v_measure_min)/2/v_measure_avg))
            print("\t\tAvg. time: %f secs." % (time_elapsed_avg))
        else:
            print("%d\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%0.3f\t%0.2f\t%f" % (
                n,
                d,
                n_clusters,
                test_problem.__name__,
                base['name'],
                method_name,
                k_name,
                kd_name,
                v_measure_avg,
                ((v_measure_max-v_measure_min)/2/v_measure_avg),
                time_elapsed_avg,
                ))


    return clust

if __name__ == "__main__":
    main()
