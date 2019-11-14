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
    #       n  d  n_clust  test_pr  eval_against   method   k      k/d  vmeas  vm_min  vm_max  t_red  t_test  t_total  eps

    data   = pd.read_csv(output_file, index_col=False, header=0, sep='\t')
    #data   = pd.read_csv(output_file, index_col=False, header=0, sep='\t').to_numpy()
    newdata = data

    """
    for row in data:
        newrow = [row[0], row[1], row[2], row[3], row[4], methodId(row[5]), methodName(row[5]), row[6], row[7], row[8]]
        newrow.append(row[9])
        newrow.append(row[10])
        newrow.append(row[11])
        newrow.append(row[12])
        newrow.append(row[13])
        newrow.append(row[14])
        # print(row)
        # print(newrow)
        newdata.append(newrow)
    newdata = np.array(newdata)
    """
    # newdata = pd.DataFrame(newdata)data.sort_values(by=["methodid", "k/d"])
    #print(newdata.to_csv(path_or_buf=None, index=False, header=False, float_format='%.5f'))

    round_dec = {"vmeas" : 5,"vm_min" : 5,"vm_max" : 5,"t_red" : 5,"tred_min" : 5,"tred_max" : 5,"t_test" : 5,"ttest_min" : 5,"ttest_max" : 5,"t_tot" : 5,"ttot_min" : 5,"ttot_max" : 5, "eps" : 5}

    data = data.sort_values(by=["methodid", "k/d"], ascending=[True, False]).round({"k/d": 5}).round(round_dec)

    print(data.to_csv(path_or_buf=None, index=False, header=False))

    # print(newdata)

if __name__ == "__main__":
    main()
