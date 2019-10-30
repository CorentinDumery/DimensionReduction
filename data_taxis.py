#!/usr/bin/env python3

from math import log,sqrt
import numpy as np
from random import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

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
