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


def loadDigits():
    n_clusters = 10
    image_size = 8
    image_pixels = image_size * image_size

    digits = load_digits()
    return digits.data, digits.target, n_clusters

def loadMNIST():
    n_clusters = 10
    image_size = 28
    image_pixels = image_size * image_size

    mnist = np.loadtxt("data/mnist/mnist_test.csv", delimiter=",")/255
    # ... mnist = np.loadtxt("data/mnist.csv", delimiter=",")/255

    fac = 0.99 / 255
    data   = np.asfarray(mnist[:, 1:]) * fac + 0.01
    labels = np.asfarray(mnist[:, :1])

    return data, labels, n_clusters

def main():

    # data, labels, n_clusters = loadDigits()
    data, labels, n_clusters = loadMNIST()

    n = len(data)

    c = 1
    eps = 1
    d = data.shape[1]
    k = int(c*log(n)/(eps**2))+1

    print()
    print("Fast Johnson-Lindenstrauss Transform applied to MNIST digit dataset")
    print("n = %d" % n)
    print("c = %d" % n)
    print("eps = %d" % eps)
    print("d = %d" % d)
    print()
    print("k = %d" % k)


    # Apply FJLT
    phi = FJLT_phi(n,k,d)
    reduced_data = [phi.dot(entry) for entry in data]

    print()
    print("KMeans on the original data...")
    evaluateKMeans(data, n_clusters, labels)

    print()
    print("FLJT + KMeans...")
    evaluateKMeans(reduced_data, n_clusters, labels)

    print()

### --- FJLT --- ###

def binaryDot(x,y):
    xb = bin(x)[2:]
    yb = bin(y)[2:]
    res =0
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
            if random()<q:
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

    err = float(n-score)/n

    print("KMeans took %f secs." % (end-start))
    print("KMeans error: %f" % err)
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
        printDigit(orig_data[best[0]])
    """

def printDigits(digits):
    for i in range(1, len(digits)):
        printDigit(digits[i])

def printDigit(digit):
    plt.imshow(digit, cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()
