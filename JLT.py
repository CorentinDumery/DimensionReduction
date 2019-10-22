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
    print("### Johnson-Lindenstrauss Transform ###")
    print()

    #########################################################################
    #########################################################################

    data, labels, n_clusters = loadDigits()
    #data, labels, n_clusters = loadMNIST()

    cs = [1, 10, 100]
    epss = [0.1, 0.5, 1., 2., 3.]

    #########################################################################
    #########################################################################

    n, d = data.shape

    print("n, d = %d * %d" % (n, d))
    print("# of clusters = %d" % n_clusters)

    print()
    print()
    print("Pure KMeans...")

    #data = [descriptor(entry) for entry in data]

    evaluateKMeans(data, n_clusters, labels, 5)

    for c in cs:
        for eps in epss:
            k = int(c*log(n)/(eps**2))+1
            if k > n:
                continue

            print()
            print("FLJT + KMeans... (c, eps =  %d, %f)" % (c, eps))
            print("Dim. reduction: %d -> %d" % (d, k))

            # phi = FJLT_phi(n, d, k)
            phi = Achlioptas_phi(n, d, k)

            #reduced_data = [phi.dot(descriptor(entry)) for entry in data]

            reduced_data = data.dot(phi.T)
            evaluateKMeans(reduced_data, n_clusters, labels, 5)

    print()

### --- FJLT --- ###

# Note : in the article they only prove their JLT works for k > k0 > 25 so it's
# kinda useless, but I guess we can still use it as a comparison for low k, in
# question 2 if I remember correctly we don't need to prove stuff

def Achlioptas_phi(n, d, k):
    print("Using Achlioptas' coin flip method")
    non_zero = (d/3)**(-1/2)

    phi = np.random.rand(k,d)

    for i in range(k):
        for j in range(d):
            if phi[i,j] < 1/6:
                phi[i,j] = non_zero
            elif phi[i,j] < 2/6:
                phi[i,j] = -non_zero
            else:
                phi[i,j] = 0
    return phi

# Fast Johnson-Lindenstrauss Transform

def FJLT_phi(n, d, k):
    print("Using Fast JL Transform method")

    # Note : assume the p in the article is 2

    def binaryDot(x,y):
        xb = bin(x)[2:]
        yb = bin(y)[2:]
        res = 0
        for i in range(min(len(xb),len(yb))):
            if xb[i]==yb[i] and xb[i]==1:
                res += 1
        return res % 2

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
