#!/usr/bin/env python3

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
from data_real     import *

import warnings
def warn(*args, **kwargs):
    pass


#########################################################################
#########################################################################

# One might not want to eval against the ground truth (baseline is often enough)
DontEvalAgainstGT = False

CleanOutput = True

# Number of repeated runs
R = 3


load_dataset_funs = [
    # loadDigits,
    # loadMNIST_tiny,
    # loadGISETTE_tiny,
    # TODO FIX loadDEXTER_tiny,
    # TODO FIX loadDOROT_tiny,
    loadHighD_128,
    loadHighD_256,
    loadHighD_512,
    loadHighD_1024]

# Methods used for dimensionality reduction
methods = [Achliop, FastJLT, sampDim, HVarDim, useLDA, usePCA, FactAna]
# methods = [Achliop, FastJLT, sampDim]

test_problems = [kneigh_, kmeans_]
# test_problems = [kmeans_]

dim_red_factor = [0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
# dim_red_factor = [0.5, 0.2, 0.1, 0.01, 0.001, 0.0001]

#########################################################################
#########################################################################

# This script compares the performance of dimensionality reduction
#  by means of JLT transforms methods over some available datasets

def main():

    print()
    print("###################################################")
    print("### Dimensionality Reduction Methods Comparison ###")
    print("###################################################")
    print()

    for load_dataset_fun in load_dataset_funs:
        test_on_dataset(load_dataset_fun)

def test_on_dataset(load_dataset_fun):

    np.random.seed(0)

    data, labels, n_clusters = load_dataset_fun()
    n, d = data.shape

    print("n * d = %d * %d" % (n, d))
    print("n_clusters = %d" % n_clusters)
    print("R = %d" % (R))

    if CleanOutput:
        print("n\td\tn_clust\ttest_pr\teval_against\tmethod\tk\tk/d\tvmeas\tvmeas_e\ttime")
        # E.g 10000	784	mshift_	baseline	Achliop	392	0.5	0.000 +/- 0.00	14.589523

    # For test problem
    for test_problem in test_problems:

        if not CleanOutput:
            print("\tPure " + test_problem.__doc__ + "...")

        # Obtain the labels according to the algorithm (without dim. reduction)
        eval_against = [{'name' : "gr-truth", 'labs' : labels}]
        if labels is None:
            eval_against = []
        naive_labels = evaluate(data, test_problem, n_clusters, eval_against, R=R)

        # For each method
        for method_fun in methods:

            if not CleanOutput:
                print()
                print(method_fun.__doc__)

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


                # Obtain the labels according to the algorithm with dim. reduction
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

    if len(base_labels_arrs) == 0:
        raise ValueError("Error! Cannot evaluate against nothing.")

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
                reduced_data, str_note = method_fun(data, k, labels, silent=CleanOutput)
                if reduced_data.shape[1] != k:
                    print(data.shape, reduced_data.shape)
                    raise ValueError("Error! reduction failed:")
            else:
                reduced_data = data
                str_note = ''

            time_elapsed_avg -= time.time()
            clust = test_problem(reduced_data, n_clusters, labels)
            time_elapsed_avg += time.time()

            if len(np.unique(clust)) != n_clusters:
                print("Error! Check the code. The clustering routine used %d of the %d available cluster(s)." % (len(np.unique(clust)), n_clusters))

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
            print("%d\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%0.3f\t%0.2f\t%f\t%s" % (
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
                str_note,
                ))

    return clust

if __name__ == "__main__":
    main()
