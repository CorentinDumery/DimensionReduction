# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 15:53:09 2019

"""

## Note : the current file reading is sequential, need to find how to directly 
## access row i


## TODO : find better dataset ( https://leon.bottou.org/projects/infimnist )




from math import log,sqrt
import numpy as np
from random import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import ezodf
import time
doc = ezodf.opendoc('taxiSample.ods')
sheet = doc.sheets[0]


### --- This section is only useful for cabs --- ###

valRange = []

def timestampToFloat(timeString):
    return float(timeString[-2:])+float(timeString[-5:-3])*60+float(timeString[-8:-6])*60*60

def computeScale(): # for each column, find a factor to scale values in [0,1]
    global valRange
    valRange = []
    for i, row in enumerate(sheet.rows()):
        if i == 0:
            continue
        if i == 1:
            for j, cell in enumerate(row):
                if not(j==1 or j==2 or j==6):
                    valRange.append([float(cell.value),float(cell.value)])
                else :
                    valRange.append("That column is for floats")
            continue
        for j, cell in enumerate(row):
            if not(j==1 or j==2 or j==6):
                c = float(cell.value)
                if c < valRange[j][0]:
                    valRange[j][0] = c
                if c > valRange[j][1]:
                    valRange[j][1] = c
    

def distance(columnId,val1,val2): #where val1 and val2 were taken from column columnId
    global valRange
    if len(valRange)==0:
        valRange = computeScale()
    if columnId==1 or columnId==2:
        v1 = timestampToFloat(val1)
        v2 = timestampToFloat(val2)
        return min(max(v1,v2)-min(v1,v2),min(v1,v2)+60*60*24-max(v1,v2))/(60*60*24/2)
    elif columnId==6:
        if val1==val2:
            return 0
        return 1
    elif valRange[columnId][1]==valRange[columnId][0]:
        return 0 #that column always has the same value
    else:
        val1 = float(val1)
        val2 = float(val2)
        return abs(val1-val2)/(valRange[columnId][1]-valRange[columnId][0])

def compareTest(row1,row2): #prints all the distances between two entries
    l1 = []
    l2 = []
    for i, row in enumerate(sheet.rows()):
        if i>max(row1,row2):
            break
        for j, cell in enumerate(row):
            if i == row1:
                l1.append(cell.value)
            if i == row2:
                l2.append(cell.value)
    
    for i in range(len(l1)):
        print("Distance("+str(l1[i])+" , "+str(l2[i])+") = " +str(distance(i,l1[i],l2[i])))
        

def entryToArray(rowNumber):
    l = []
    for i,row in enumerate(sheet.rows()):
        if i == rowNumber:
            for j, cell in enumerate(row):
                if j==1 or j==2:
                    l.append(timestampToFloat(cell.value)/60*60*24)
                elif j==6:
                    if cell.value=="N":
                        l.append(0)
                    else:
                        l.append(1)
                elif valRange[j][1]==valRange[j][0]:
                    l.append(0)
                else:
                    l.append((float(cell.value)-valRange[j][0])/valRange[j][1])
    res = np.array([l])
    return res

### --- End of cabs section --- ###


### --- Coin flips JLT --- ###
    
# Note : in the article they only prove their JLT works for k > k0 > 25 so it's
# kinda useless, but I guess we can still use it as a comparison for low k, in 
# question 2 if I remember correctly we don't need to prove stuff
    
R = np.zeros((0,0))
n = 25
d = 17
k = -1
eps = 0.99
beta = 0.01
    
def buildR():
    global k,R
    k0 = log(n)*(4+2*beta)/(eps*eps/2 - (eps**3)/3)
    k = int(k0)+1
    R = np.zeros((d,k))
    for i in range(d):
        for j in range(k):
            die = random()
            if die < 1/6:
                R[i][j] = 1
            elif die < 2/6:
                R[i][j] = -1
            else :
                R[i][j] = 0

def JLTransform(rowList): # f described in the article
    if k==-1:
        buildR()
    print(k)
    return (1/sqrt(k))*rowList.dot(R)


### --- End of Coin flips JLT --- ###
    
### --- FJLT --- ###


notImportedYet = True
if notImportedYet:
    from sklearn.datasets import load_digits
    digits = load_digits()
    notImportedYet = False

n = len(digits.images)
c = 1
eps = 1
d = 64
k = int(c*log(n)/(eps**2))+1

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

def buildPhi():
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
        if random()>1/2:
            D[i][i] =  1
        else :
            D[i][i] = -1
    return P.dot(H.dot(D))



phi=buildPhi()

#vector = np.array(matToList(digits.images[4]))

### --- End of FJLT --- ###


def evalKMeans(array,centers):
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



array = []
for i in range(len(digits.images)):
    array.append(phi.dot(matToList(digits.images[i])))


start = time.time()
kmeans = KMeans(n_clusters=5, random_state=4).fit(array)
end = time.time()
print("KMeans took "+str(end-start) +"s.")
print("KMeans score : "+str(evalKMeans(array,kmeans.cluster_centers_)))
kmeans.labels_
#kmeans.predict([[0, 0], [12, 3]])
kmeans.cluster_centers_

for center in kmeans.cluster_centers_:
    #try to see what they correspond to
    best = (-1,pow(10,10))
    for i in range(len(array)):
        #l = matToList(im)
        l = array[i]
        dist = 0
        for j in range(len(l)):
            dist += abs(center[j]-l[j])
        if dist < best[1]:
            best = (i,dist)
    plt.gray() 
    plt.matshow(digits.images[best[0]]) 
    plt.show()


    
array = []
for i in range(len(digits.images)):
    array.append(matToList(digits.images[i]))

start = time.time()
kmeans = KMeans(n_clusters=5, random_state=4).fit(array)
end = time.time()
print("KMeans took "+str(end-start) +"s.")
print("KMeans score : "+str(evalKMeans(array,kmeans.cluster_centers_)))

for center in kmeans.cluster_centers_:
    #try to see what they correspond to
    best = (-1,pow(10,10))
    for i in range(len(array)):
        #l = matToList(im)
        l = array[i]
        dist = 0
        for j in range(len(l)):
            dist += abs(center[j]-l[j])
        if dist < best[1]:
            best = (i,dist)
    plt.gray() 
    plt.matshow(digits.images[best[0]]) 
    plt.show()








