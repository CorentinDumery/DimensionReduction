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

##########################################################
######################## DATASETS ########################
##########################################################

def loadDigits():
    print("Loading sklearn's 'digit' dataset...")

    n_clusters = 10
    # image_size = 8
    # image_pixels = image_size * image_size

    digits = load_digits()
    return digits.data, digits.target, n_clusters

# This dataset was derived by concatenating the following two files
# - https://www.python-course.eu/data/mnist/mnist_train.csv
# - https://www.python-course.eu/data/mnist/mnist_test.csv
# into data/mnist.csv
# Source: https://www.python-course.eu/neural_network_mnist.php
def loadMNIST():
    print("Loading MNIST dataset...")

    n_clusters = 10
    # image_size = 28
    # image_pixels = image_size * image_size

    # Any of these will work, choose depending on the size you want
    mnist = np.loadtxt("data/mnist/mnist_test.csv", delimiter=",")     # 18MB
    # mnist = np.loadtxt("data/mnist/mnist_train.csv", delimiter=",")  # 109MB
    # mnist = np.loadtxt("data/mnist.csv", delimiter=",")              # 127MB

    fac = 0.99 / 255
    data   = np.asfarray(mnist[:, 1:]) * fac + 0.01
    labels = np.asfarray(mnist[:, :1])

    return data, labels, n_clusters

##########################################################
####################### ENTRY POINT ######################
##########################################################

def main():

    print()
    print("### Fast Johnson-Lindenstrauss Transform ###")
    print()

    #########################################################################
    #########################################################################

    data, labels, n_clusters = loadDigits()
    # data, labels, n_clusters = loadMNIST()

    c = 1
    epss = [1, 10, 100, 1000, 10000]

    #########################################################################
    #########################################################################

    n, d = data.shape

    print("n, d = %d * %d" % (n, d))
    print("# of clusters = %d" % n_clusters)

    print()
    print()
    print("Pure KMeans...")
    evaluateKMeans(data, n_clusters, labels)

    for eps in epss:
        print()
        print("FLJT + KMeans... (c, eps =  %d, %d)" % (c, eps))
        k = int(c*log(n)/(eps**2))+1
        print("Dim. reduction: %d -> %d" % (d, k))

        phi = FJLT_phi(n,k,d)
        reduced_data = [phi.dot(entry) for entry in data]
        evaluateKMeans(reduced_data, n_clusters, labels)

    print()

### --- FJLT --- ###

def binaryDot(x,y):
    xb = bin(x)[2:]
    yb = bin(y)[2:]
    res = 0
    for i in range(min(len(xb),len(yb))):
        if xb[i]==yb[i] and xb[i]==1:
            res += 1
    return res % 2

def FJLT_phi(n, k, d):
    # Note : assume the p in the article is 2
    q = min(1,(log(n)**2)/d)
    P = np.zeros((k,d))
    for i in range(k):
        for j in range(d):
            if random() < q:
                P[i][j] = np.random.normal(0,1/q)
    H = np.zeros((d,d))
    D = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            H[i][j] = pow(d,-1/2)*pow(-1,binaryDot(i-1,j-1))

        if random() > 1/2:
            D[i][i] =  1
        else:
            D[i][i] = -1
    return P.dot(H.dot(D))

### --- End of FJLT --- ###

def evaluateKMeans(data, k, labels):
    n = len(data)

    start = time.time()
    kmeans = KMeans(n_clusters=k
        # , random_state=4
        ).fit(data)
    end = time.time()

    predictions = kmeans.predict(data)

    score = sum([1 for i,j in zip(predictions, labels) if i == j])

    print("KMeans took %f secs." % (end-start))
    print("KMeans precision: %f" % (score/n))
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
        plotImg(orig_data[best[0]])
    """

def plotImgs(digits):
    for i in range(1, len(digits)):
        plotImg(digits[i])

def plotImg(digit):
    plt.imshow(digit, cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()
