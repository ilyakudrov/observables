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


def get_wilson(data):
    x = np.array(data['wilson_loop'].to_numpy())
    x = np.array([x])

    field, err = stat.jackknife_var_numba(x, trivial)

    return pd.Series([field, err], index=['field', 'err'])


@njit
def potential_numba(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        # if (x[1][i] == 0):
        #     print(i)
        fraction = x[0][i] / x[1][i]
        if(fraction >= 0):
            y[i] = math.log(fraction)
        else:
            y[i] = 0
    return y


@njit
def trivial(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = x[0][i]
    return y


def potential(x):
    a = np.mean(x, axis=1)
    fraction = a[0] / a[1]
    if(fraction >= 0):
        return math.log(fraction)
    else:
        return 0


# conf_type = "qc2dstag"
conf_type = 'su2_suzuki'
# conf_type = "SU2_dinam"
# conf_sizes = ["40^4", "32^4"]
# conf_sizes = ["40^4"]
conf_sizes = ["24^4"]
# conf_sizes = ["32^4"]

axis = 'on-axis'

# for monopole in ['/', 'monopoless']:
# for monopole in ['monopoless']:
for beta in [2.4, 2.5, 2.6]:
    for monopole in ['/', 'monopole', 'monopoless']:
        for conf_size in conf_sizes:
            if conf_size == '40^4':
                conf_max = 1200
                # conf_max = 1000
                # mu1 = ['0.05', '0.35', '0.45']
                mu1 = ['0.45']
                # mu1 = ['0.00']
                chains = {"s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"}
                # chains = {"/"}
            elif conf_size == '32^4':
                conf_max = 2800
                mu1 = ['0.00']
                chains = {"/"}
            elif conf_size == '24^4':
                conf_max = 100
                mu1 = ['0.00']
                chains = {"/"}
            for mu in mu1:
                # print(monopole, mu)
                data = []
                for chain in chains:
                    for i in range(0, conf_max):
                        # file_path = f"../../data/wilson_loop/{axis}/{monopole}/qc2dstag/{conf_size}/mu{mu}/{chain}/wilson_loop_{i:04}"
                        file_path = f"../../data/wilson_loop/{axis}/{monopole}/{conf_type}/{conf_size}/beta{beta}/mu{mu}/{chain}/wilson_loop_{i:04}"
                        # print(file_path)

                        if(os.path.isfile(file_path)):
                            data.append(pd.read_csv(file_path, header=0,
                                                    names=["T", "R", "wilson_loop"]))
                            data[-1]["conf_num"] = i
                if len(data) == 0:
                    print("no data", conf_size, mu, beta)
                elif len(data) != 0:
                    data = pd.concat(data)

                    start = time.time()

                    data = data.groupby(['T', 'R']).apply(
                        get_wilson).reset_index()

                    end = time.time()
                    print("execution time = %s" % (end - start))

                    path_output = f"../../result/wilson_loop/{monopole}/{conf_type}/{conf_size}"
                    # path_output = f"../../result/potential_spatial/{test}/{axis}/{monopole}/qc2dstag/{conf_size}"
                    # path_output = f"../../result/potential/{test}/{axis}/{monopole}/su2_dynam/{conf_size}/{smearing}"

                    try:
                        os.makedirs(path_output)
                    except:
                        pass

                    data.to_csv(
                        f"{path_output}/wilson_loop_beta={beta}.csv", index=False)
