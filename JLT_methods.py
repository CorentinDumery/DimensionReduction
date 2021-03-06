#!/usr/bin/env python3

from math import log,sqrt
import numpy as np
from random import random
import time
from michaelmathenFJLT import *

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA, FactorAnalysis

# Note : in the article they only prove their JLT works for k > k0 > 25 so it's
# kinda useless, but I guess we can still use it as a comparison for low k, in
# question 2 if I remember correctly we don't need to prove stuff

def Achliop(data, k, y=None, silent=False):
    """Achlioptas' coin flip method"""
    n, d = data.shape

    # Estimation of the transform's "accuracy":
    #  beta is set such that we get "good" results with a >= 2/3 probability
    #  then epsilon is deduced from the rest and defines how "good" it actually is
    def estEpsilon(k,n):
        x0 = 0
        x1 = 1
        beta = log(3)/log(n)
        objective = log(n)*(4+2*beta)/k
        if (objective>0.1666):
            if not silent:
                print("\t\tWarning : output's dimension is too low, the proof doesn't hold.")
            return str('/')
        else:
            precision = 0.001
            while True:
                x = (x0+x1)/2
                fx = (x**2)/2 - (x**3)/3
                if fx<objective-precision:
                    x0 = (x0+x1)/2
                elif fx>objective+precision:
                    x1 = (x0+x1)/2
                else:
                    break
            eps = x
            if not silent:
                print("\t\tEpsilon factor of the approximation: %f" % (eps))
            return str(eps)

    str_eps = estEpsilon(k,n)

    # Core routine
    phi = np.random.rand(k,d)

    non_zero = (d/3)**(-1/2)
    for i in range(k):
        for j in range(d):
            if phi[i,j] < 1/6:
                phi[i,j] = non_zero
            elif phi[i,j] < 2/6:
                phi[i,j] = -non_zero
            else:
                phi[i,j] = 0

    return 1/sqrt(k) * data.dot(phi.T), str_eps

# Fast Johnson-Lindenstrauss Transform
def FastJLT(data, k, y=None, silent=False):
    """Fast Johnson-Lindenstrauss Transform method"""
    n, d = data.shape
    eps = 1/(sqrt(k))
    # fjlt(data, k)
    return fjlt_usp(data, k), ("%0.3f" % (eps))

# Fast Johnson-Lindenstrauss Transform
def FastJLT2(data, k, y=None, silent=False):
    """Fast Johnson-Lindenstrauss Transform method"""
    n, d = data.shape
    eps = 1/(sqrt(k))
    q = min(1,(log(n)**2)/d)
    # fjlt(data, k)
    return fjlt(data, k, q), ("%0.3f" % (eps))

# Our failed attempt at Fast JLT
def FastJLT3(data, k, y=None, silent=False):
    """Fast Johnson-Lindenstrauss Transform method"""
    d, n = data.shape
    # Note : assume the p in the article is 2

    eps = 1/(sqrt(k))

    if not silent:
        print("\t\tEpsilon factor of the approximation: %f" % (eps))

    def binaryDot(x,y):
        xb = bin(x)[2:]
        yb = bin(y)[2:]
        res = 0
        for i in range(min(len(xb),len(yb))):
            if xb[i]==yb[i] and xb[i]==1:
                res += 1
        return res % 2

    q = min(1,(log(n)**2)/d)
    H = np.zeros((d,d))
    D = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            H[i][j] = pow(d,-1/2)*pow(-1,binaryDot(i-1,j-1))

    for i in range(d):
        if random() > .5:
            D[i][i] =  1
        else:
            D[i][i] = -1

    sample_size = npr.binomial(k * d, q)
    indc = fast_sample(k * d, sample_size)
    p_rows, p_cols = np.unravel_index(indc, (k, d))
    p_data = npr.normal(loc=0, scale=math.sqrt(1/q), size=len(p_rows))
    P = sparse.csr_matrix((p_data, (p_rows, p_cols)), shape=(k, d))

    phi = P.dot(H.dot(D))
    return phi.dot(data), ("%0.3f" % (eps))

# Randomly sample a subset of dimensions
def sampDim(data, k, y=None, silent=False):
    """Simple dimensions sampling"""
    n, d = data.shape
    nonzero = np.random.choice(d,k,replace=False)
    return data[:,nonzero], ""

# Select the features with highest (normalized) variance
def HVarDim(data, k, y=None, silent=False):
    """Select high-variance dimensions"""
    n, d = data.shape
    disp = data.var(axis=0)/data.mean(axis=0)
    mostdisp = disp.argsort()[-k:]
    return data[:,mostdisp], ""

### Out of the box methods ###

def useLDA(data, k, y=None, silent=False):
    """LDA"""
    clf = LinearDiscriminantAnalysis(n_components=k)
    return clf.fit_transform(data, y), ""

def usePCA(data, k, y=None, silent=False):
    """PCA"""
    pca = PCA(n_components=k)
    return pca.fit_transform(data, y), ""

def FactAna(data, k, y=None, silent=False):
    """FactorAnalysis"""
    fa = FactorAnalysis(n_components=k)
    return fa.fit_transform(data, y), ""


### --- End --- ###
