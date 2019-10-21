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




def main():
    digits = load_digits()

    n = len(digits.images)

    c = 1
    eps = 1
    d = 64
    k = int(c*log(n)/(eps**2))+1

    print()
    print("Fast Johnson-Lindenstrauss Transform applied to MNIST digit datase")
    print("n = %d" % n)
    print("c = %d" % n)
    print("eps = %d" % eps)
    print("d = %d" % d)
    print()
    print("k = %d" % k)

    # printDigits(digits)

    phi = buildPhi(n,k,d)


    print()
    print("KMeans on the original data...")
    array = []
    for i in range(len(digits.images)):
        array.append(matToList(digits.images[i]))
    computeKMeans(array, 10, orig_data=digits)

    print()
    print("FLJT + KMeans...")
    array = []
    for i in range(len(digits.images)):
        array.append(phi.dot(matToList(digits.images[i])))
    computeKMeans(array, 10, orig_data=digits)

### --- FJLT --- ###

def binaryDot(x,y):
    xb = bin(x)[2:]
    yb = bin(y)[2:]
    res =0
    for i in range(min(len(xb),len(yb))):
        if xb[i]==yb[i] and xb[i]==1:
            res += 1
    return res % 2

def matToList(mat):
    res = []
    for i in mat:
        for elem in i:
            res.append(elem)
    return np.array(res)

def buildPhi(n, k, d):
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


def evalKMeans(array,centers): #average of points to centers
    total = 0
    for i in range(len(array)):
        best = -1
        for j in range(len(centers)):
            dist = 0
            for k in range(len(centers[j])):
                dist += abs(array[i][k]-centers[j][k])
            if dist<best or best ==-1:
                best = dist
        total += best
    return total/len(array)

def computeKMeans(array, k, orig_data=digits):
    start = time.time()
    kmeans = KMeans(n_clusters=k
        # , random_state=4
        ).fit(array)
    end = time.time()
    print("KMeans took "+str(end-start) +" secs.")
    print("KMeans avg. error: "+str(evalKMeans(array, kmeans.cluster_centers_)))
    #print(kmeans.labels_)
    #print(kmeans.cluster_centers_)

    """
    for center in kmeans.cluster_centers_:
        #try to see what they correspond to
        best = (-1,np.inf)
        for i in range(len(array)):
            #l = matToList(im)
            l = array[i]
            dist = 0
            for j in range(len(l)):
                dist += abs(center[j]-l[j])
            if dist < best[1]:
                best = (i,dist)
        printDigit(digits.images[best[0]])
    """

def printDigits(digits):
    for i in range(1, len(digits)):
        printDigit(digits.images[i])

def printDigit(digit):
    plt.imshow(digit, cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()
