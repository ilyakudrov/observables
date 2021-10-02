import pandas as pd
import os.path
import numpy as np
import math
import sys
import os.path
from numba import njit
sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", ".."))
import statistics_python.src.statistics_observables as stat

conf_type = "qc2dstag"


def get_field(data, df1):
    x = data[['wilson-plaket-correlator',
              'wilson-loop', 'plaket']].to_numpy()

    field, err = jackknife_var(x, field_electric)

    new_row = {'field': field, 'err': math.sqrt(err)}

    df1 = df1.append(new_row, ignore_index=True)

    return df1


@njit
def electric_field_numba(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = x[0][i] / x[1][i] - x[2][i]
    return y


def field_electric(x):
    a = x.mean(axis=0)
    return a[0] / a[1] - a[2]


def jackknife(x, func):
    n = len(x)
    idx = np.arange(n)
    return sum(func(x[idx != i]) for i in range(n)) / float(n)


def jackknife_var(x, func):
    n = len(x)
    idx = np.arange(n)
    j_est = jackknife(x, func)
    return j_est, (n - 1) / (n + 0.0) * sum((func(x[idx != i]) - j_est)**2.0
                                            for i in range(n))


# T1 = [8, 10, 12]
# R1 = [8, 10, 12, 14, 16, 18]
T1 = [8]
R1 = [8]

conf_type = "qc2dstag"
# conf_type = "SU2_dinam"
# conf_sizes = ["40^4", "32^4"]
conf_sizes = ["32^4"]
# conf_sizes = ["40^4"]

for monopole in ['/', 'monopoless', 'monopole']:
    # for monopole in ['/']:
    # for monopole in ['monopole']:
    for conf_size in conf_sizes:
        if conf_size == '40^4':
            conf_max = 700
            # mu1 = ['0.05', '0.35', '0.45']
            mu1 = ['0.05']
            chains = {"s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"}
        elif conf_size == '32^4':
            conf_max = 2800
            mu1 = ['0.00']
            chains = {"/"}
        for mu in mu1:
            data = []
            for chain in chains:
                for T in T1:
                    for R in R1:
                        for i in range(0, conf_max):
                            # file_path = f"../../data/flux_tube/qc2dstag/{conf_size}/HYP_APE/mu{mu}/s{chain}/T={T}/R={R}/electric_{i:04}"
                            # file_path = f"../../data/flux_tube/qc2dstag/{conf_size}/HYP_APE/mu{mu}/T={T}/R={R}/electric_{i:04}"
                            file_path = f"../../data/flux_tube_wilson/{monopole}/qc2dstag/{conf_size}/mu{mu}/{chain}/T={T}/R={R}/electric_{i:04}"
                            if(os.path.isfile(file_path)):
                                data.append(pd.read_csv(file_path))
                                data[-1]["R"] = R
                                data[-1]["T"] = T
                                # data[-1]['d'] = data[-1]['d'].transform(
                                #     lambda x: x - R/2)

            df = pd.concat(data)

            df1 = pd.DataFrame(columns=['field', 'err'])

            df1 = df.groupby(['d', 'R', 'T']).apply(
                get_field, df1).reset_index()

            df1 = df1[['d', 'R', 'T', 'field', 'err']]

            # print(df1)

            path_output = f"../../result/flux_tube_wilson/{monopole}/qc2dstag/{conf_size}"

            try:
                os.makedirs(path_output)
            except:
                pass

            df1.to_csv(
                f"{path_output}/flux_tube_electric_mu={mu}.csv", index=False)
