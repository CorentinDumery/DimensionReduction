#!/usr/bin/env python3

from math import log,sqrt
import numpy as np
from random import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

loadMNIST_tiny = lambda : loadMNIST(set="test")
loadMNIST_big  = lambda : loadMNIST(set="train")
loadMNIST_huge = lambda : loadMNIST(set="all")

def loadDigits():
    print("Loading sklearn's 'digit' dataset...")

    digits = load_digits()
    return digits.data, digits.target, len(np.unique(digits.target))


def loadMNIST(set):
    """
    Source: https://www.python-course.eu/neural_network_mnist.php

    train : N=6000, k=10, d=784
    valid : N=1000, k=10, d=784
    -
    all   : N=7000, k=10, d=784
    """

    print("Loading MNIST dataset...")

    mnist = np.loadtxt("data/mnist/mnist_%s.csv" % (set), delimiter=",")

    # Normalize values
    fac = 0.99 / 255
    data   = np.asfarray(mnist[:, 1:]) * fac + 0.01
    labels = np.asfarray(mnist[:, :1]).flatten()

    return data, labels, len(np.unique(labels))

def plotImgs(digits):
    for i in range(1, len(digits)):
        plotImg(digits[i])

def plotImg(digit):
    plt.imshow(digit, cmap='gray')
    plt.show()

############################ OTHER ############################
# This serves for linearizing the matrix without losing too much
#  of its 2d intter structure
def descriptor(img, depth=3):

    feats = []

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
