from numba import njit
import sys
import math
import time
import numpy as np
import os.path
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", ".."))
import statistics_python.src.statistics_observables as stat
from astropy.stats import jackknife_resampling, jackknife_stats, bootstrap


def read_test_data():
    file_path = f'/home/ilya/soft/lattice/fortran/POTEN/wl_mon.dat'
    if(os.path.isfile(file_path)):
        with open(file_path, 'r') as f:
            lines = f.readlines()

            conf_num = 1
            line_num = 0
            data_full = []
            data_conf = []

            for i in range(len(lines)):
                line_num += 1
                data_conf.append(list(map(float, lines[i].split()[1:])))
                if line_num == 24:
                    line_num = 0
                    conf_num += 1
                    data_full.append(data_conf)
                    data_conf = []

            data_full = np.array(data_full)
            data_full = np.transpose(data_full, (1, 2, 0))
            # for i in range(len(data_full)):
            #     for j in range(i):
            #         for k in range(len(data_full[i][j])):
            #             data_full[i][j] = (
            #                 data_full[i][j] + data_full[j][i]) / 2

        return data_full


def get_statistics():
    data = read_test_data()

    x = np.vstack((data[23][0], data[23][1]))

    field, err = stat.jackknife_var_numba(x, potential_numba)

    print(field, err)


@njit
def potential_numba(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        fraction = x[0][i] / x[1][i]
        if(fraction >= 0):
            # if(fraction < 0):
            #     print(fraction)
            y[i] = math.log(fraction)
        else:
            y[i] = 0
    return y


def potential(x):
    a = np.mean(x, axis=1)
    fraction = a[0] / a[1]
    if(fraction >= 0):
        return math.log(fraction)
    else:
        return 0


get_statistics()
