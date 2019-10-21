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

    #data, labels, n_clusters = loadDigits()
    data, labels, n_clusters = loadMNIST()

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
    #orig_data = [descriptor(entry) for entry in data]
    orig_data = [entry for entry in data]
    evaluateKMeans(orig_data, n_clusters, labels, 5)

    for eps in epss:
        print()
        print("FLJT + KMeans... (c, eps =  %d, %d)" % (c, eps))
        k = int(c*log(n)/(eps**2))+1
        print("Dim. reduction: %d -> %d" % (d, k))

        phi = FJLT_phi(n,k,d)
        #reduced_data = [phi.dot(descriptor(entry)) for entry in data]
        reduced_data = [phi.dot(entry) for entry in data]
        evaluateKMeans(reduced_data, n_clusters, labels, 5)

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

def evaluateKMeans(data, k, labels, R=2):
    n = len(data)

    score = 0
    time_elapsed  = 0.

    for i in range(R):
        time_elapsed -= time.time()
        kmeans = KMeans(n_clusters=k
            # , random_state=4
            ).fit(data)
        time_elapsed += time.time()

        predictions = kmeans.predict(data)
        score += sum([1 for i,j in zip(predictions, labels) if i == j])

    print("KMeans took %f secs." % (time_elapsed/R))
    print("KMeans precision: %f" % (score/R/n))
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

#### OTHER ####
# This serves for linearizing the matrix without losing too much
#  of its 2d intter structure
def descriptor(img, depth=3):
    return img

    feats = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################
    L = depth-1

        # This recursive function performs the spacial pyramid
    #  It accepts the image, the depth level and the reference of the list where to
    #  append the histograms.
    def deep_describe(img, l, current_feats):
        def level_weight(l):
            if l == 0: return 1./2**L
            else:      return 1./2**(L-l+1)

        # Base case
        if l > L: return

        H, W = img.shape

        # For each descriptor, find the nearest neighbor, construct the histogram
        #  and append it to our global feature vector
        descriptor = np.mean(img)
        current_feats.append(descriptor * level_weight(l))

        # Iterate on the 4 subimages
        #  Note: there is a potential out-of-one error caused by odd img dimensions
        #   but I thought this iterative code would generalize more easily
        #   and would be more elegant than hardcoding 4 static calls to deep_describe.
        #   The price to pay for the elegance here is just for some thin
        #   central areas of the picture to be potentially ignored for dense sift
        h_2 = H//2
        w_2 = W//2
        for h in [0, H-h_2]:
            for w in [0, W-w_2]:
                deep_describe(img[h:h+h_2,w:w+w_2], l+1, current_feats)

    img_feats = []
    img_l = int(np.sqrt(len(img)))
    if int(np.prod(img.shape)) != img_l*img_l:
        print("ERROR delinearizing: %d != %d" % int(np.prod(img.shape)), img_l*img_l)

    deep_describe(img.reshape(img_l, img_l), 0, img_feats)
    img_feats /= np.linalg.norm(np.array(img_feats))
    return np.array(img_feats)

if __name__ == "__main__":
    main()
