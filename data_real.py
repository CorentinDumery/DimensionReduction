#!/usr/bin/env python3

from math import log,sqrt
import numpy as np
import pandas as pd
from random import random

loadGISETTE_tiny = lambda : loadGISETTE(set="valid")   # N=1000, k=2, d=5000
loadGISETTE_big  = lambda : loadGISETTE(set="train")   # N=6000, k=2, d=5000
loadGISETTE_huge = lambda : loadGISETTE(set="all")     # N=7000, k=2, d=5000

loadDEXTER_tiny  = lambda : loadDEXTER(set="train")    # N=300,  k=2, d=20000
loadDEXTER_big   = lambda : loadDEXTER(set="all")      # N=600,  k=2, d=20000

loadDOROT_tiny   = lambda : loadDOROT(set="valid")     # N=350,  k=2, d=100000
loadDOROT_big    = lambda : loadDOROT(set="train")     # N=800,  k=2, d=100000
loadDOROT_huge   = lambda : loadDOROT(set="all")       # N=1150, k=2, d=100000

def loadGISETTE(set):
    """
    Source: http://archive.ics.uci.edu/ml/datasets/Gisette

    train : N=6000, k=2, d=5000
    valid : N=1000, k=2, d=5000
    -
    all   : N=7000, k=2, d=5000
    """

    print("Loading GISETTE dataset...")
    set = "all"

    data   = pd.read_csv("data/gisette/gisette_%s.data"   % (set), header=None, sep='\s+').to_numpy()
    labels = pd.read_csv("data/gisette/gisette_%s.labels" % (set), header=None, sep='\s+').to_numpy().flatten()
    n_clust = len(np.unique(labels))
    return data, labels, n_clust

def loadDEXTER(set):
    """
    Source: http://archive.ics.uci.edu/ml/datasets/Dexter

    train : N=300, k=2, d=20000
    valid : N=300, k=2, d=20000
    -
    all   : N=600, k=2, d=20000
    """

    print("Loading DEXTER dataset...")
    set = "all"

    data   = pd.read_csv("data/dexter/dexter_%s.data"   % (set), header=None, sep='\s+').to_numpy()
    labels = pd.read_csv("data/dexter/dexter_%s.labels" % (set), header=None, sep='\s+').to_numpy().flatten()
    n_clust = len(np.unique(labels))
    return data, labels, n_clust

def loadDOROT(set):
    """
    Source: http://archive.ics.uci.edu/ml/datasets/Dorothea

    train : N=800, k=2, d=100000
    valid : N=350, k=2, d=100000
    -
    all   : N=1150, k=2, d=100000
    """

    print("Loading DOROTHEA dataset...")

    data   = pd.read_csv("data/dorothea/dorothea_%s.data"   % (set), header=None, sep='\s+').to_numpy()
    labels = pd.read_csv("data/dorothea/dorothea_%s.labels" % (set), header=None, sep='\s+').to_numpy().flatten()
    n_clust = len(np.unique(labels))
    return data, labels, n_clust
