import pandas as pd

loadHighD_32   = lambda : loadHighD(32)
loadHighD_64   = lambda : loadHighD(64)
loadHighD_128  = lambda : loadHighD(128)
loadHighD_256  = lambda : loadHighD(256)
loadHighD_512  = lambda : loadHighD(512)
loadHighD_1024 = lambda : loadHighD(1024)

def loadHighD(d):
    """
    N=1024 and k=16

    https://cs.uef.fi/sipu/datasets/
    DIM-sets (high)


P. Fränti, O. Virmajoki and V. Hautamäki, "Fast agglomerative clustering using a k-nearest neighbor graph", IEEE Trans. on Pattern Analysis and Machine Intelligence, 28 (11), 1875-1881, November 2006. (Bibtex)

    @article{DIMsets,
        author = {P. Fr\"anti and O. Virmajoki and V. Hautam\"aki},
        title = {Fast agglomerative clustering using a k-nearest neighbor graph},
        journal = {IEEE Trans. on Pattern Analysis and Machine Intelligence},
        year = {2006},
        volume = {28},
        number = {11},
        pages = {1875--1881}
    }
    """

    print("Loading highd (d=%d) dataset..." % d)

    data   = pd.read_csv("dim%03d.txt" % d, header=None, sep='\s+').to_numpy()
    labels = pd.read_csv("dim%03d.txt" % d, header=None, sep='\s+').to_numpy()
    return data, labels
