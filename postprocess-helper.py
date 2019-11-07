#!/usr/bin/env python3

from math import log,sqrt
import numpy as np
import pandas as pd
from random import random
import sys

def main():
    output_file = sys.argv[1]
    print(output_file)

    #       n   d  n_clust  test_pr eval_against   method   k      k/d  vmeas  vmeas_e       time                  eps

    data   = pd.read_csv(output_file, index_col=False, header=0, sep='\t').to_numpy()

    newdata = []
    for row in data:
        def methodId(met):
            if met == "Base":
                return 0
            elif met == "HVarDim":
                return 1
            elif met == "sampDim":
                return 2
            elif met == "Achliop":
                return 3
            elif met == "FastJLT":
                return 4
        def methodName(met):
            if met == "Base":
                return "Baseline"
            elif met == "sampDim":
                return "Sample dimensions"
            elif met == "Achliop":
                return "Coin flip"
            elif met == "HVarDim":
                return "Highest-variance"
            elif met == "FastJLT":
                return "Fast JL Transform"

        newrow = [row[0], row[1], row[2], row[3], row[4], methodId(row[5]), methodName(row[5]), row[6], np.around(row[7], 5), row[8]]
        err_abs = row[8]*row[9]
        if err_abs > row[8]*0.5:
            err_abs = row[8]*0.5

        newrow.append( np.clip(err_abs, 0, 1) )
        newrow.append(row[10])
        newrow.append(row[11])
        # print(row)
        # print(newrow)
        newdata.append(newrow)
    newdata = np.array(newdata)

    print(pd.DataFrame(newdata).to_csv(path_or_buf=None, index=False, header=False))

    # print(newdata)

if __name__ == "__main__":
    main()
