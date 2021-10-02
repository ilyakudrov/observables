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
    x = data['polyakov_loop'].to_numpy()

    x = np.array([x])

    potential, err = stat.jackknife_var_numba(x, potential_polyakov_numba)

    # print("r:", data["r/a"].iloc[0], "potential:",
    #       potential, "err:", math.sqrt(err))

    new_row = {'aV(r)': potential, 'err': math.sqrt(err)}

    df1 = df1.append(new_row, ignore_index=True)

    return df1


@njit
def potential_polyakov_numba(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        if x[0][i] > 0:
            y[i] = -math.log(x[0][i])
        else:
            y[i] = 0
    return y


conf_type = "qc2dstag"
# conf_type = "SU2_dinam"
conf_sizes = ["40^4", "32^4"]
# conf_sizes = ["32^4"]
# conf_sizes = ["40^4"]

# for monopole in ['/', 'monopoless', 'monopole']:
for monopole in ['/', 'monopoless']:
    for conf_size in conf_sizes:
        if conf_size == '40^4':
            conf_max = 700
            mu1 = ['0.05', '0.35', '0.45']
            # mu1 = ['0.05']
            chains = {"s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"}
        elif conf_size == '32^4':
            conf_max = 2800
            mu1 = ['0.00']
            chains = {"/"}
        for mu in mu1:
            data = []
            for chain in chains:
                for i in range(0, conf_max):
                    # file_path = f"../../data/flux_tube/qc2dstag/{conf_size}/HYP_APE/mu{mu}/s{chain}/T={T}/R={R}/electric_{i:04}"
                    # file_path = f"../../data/flux_tube/qc2dstag/{conf_size}/HYP_APE/mu{mu}/T={T}/R={R}/electric_{i:04}"
                    file_path = f"../../data/polyakov_loop/{monopole}/qc2dstag/{conf_size}/mu{mu}/HYP6_APE0/{chain}/polyakov_loop_{i:04}"

                    # print(file_path)
                    if(os.path.isfile(file_path)):
                        data.append(pd.read_csv(file_path, header=0,
                                    names=["r/a", "polyakov_loop"]))

            df = pd.concat(data)

            df1 = pd.DataFrame(columns=['aV(r)', 'err'])

            df1 = df.groupby(['r/a']).apply(get_field, df1).reset_index()

            df1 = df1[['r/a', 'aV(r)', 'err']]

            # print(df1)

            path_output = f"../../result/potential_polyakov/{monopole}/qc2dstag/{conf_size}"

            try:
                os.makedirs(path_output)
            except:
                pass

            df1.to_csv(
                f"{path_output}/potential_polyakov_HYP6_APE0_mu={mu}.csv", index=False)
