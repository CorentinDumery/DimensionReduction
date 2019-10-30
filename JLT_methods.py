#!/usr/bin/env python3

from math import log,sqrt
import numpy as np
from random import random
import time

### --- FJLT --- ###

# Note : in the article they only prove their JLT works for k > k0 > 25 so it's
# kinda useless, but I guess we can still use it as a comparison for low k, in
# question 2 if I remember correctly we don't need to prove stuff

def Achlioptas_phi(data, k):
    n, d = data.shape

    # Estimation of the transform's accuracy:
    #  beta is set such that we get "good" results with a >= 2/3 probability
    #  then epsilon is deduced from the rest and defines how "good" it actually is
    beta = log(3)/log(n)

    x0 = 0
    x1 = 1
    objective = log(n)*(4+2*beta)/k
    if (objective>0.1666):
        print("\t\tWarning : output's dimension is too low, the proof doesn't hold.")
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
        print("\t\tEpsilon factor of the approximation: %f" % (eps))

    # Core routing
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

    phi *= 1/sqrt(k)

    return data.dot(phi.T)

# Fast Johnson-Lindenstrauss Transform

def FJLT_phi(data, k):
    n, d = data.shape
    # Note : assume the p in the article is 2

    print("\t\tEpsilon factor of the approximation: %f" % (1/(sqrt(k))))

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

    phi = P.dot(H.dot(D))
    ... what to do with phi?
    return data.dot(phi.T) ?
    or return ...


### --- End of FJLT --- ###
