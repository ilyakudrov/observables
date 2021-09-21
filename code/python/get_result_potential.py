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


def get_field(data, df1, df, time_size_max):

    time_size = data["T"].iloc[0]
    space_size = data["r/a"].iloc[0]

    if time_size < time_size_max:

        x1 = data['wilson_loop'].to_numpy()

        x2 = df[(df["T"] == time_size + 1) & (df["r/a"]
                                              == space_size)]['wilson_loop'].to_numpy()

        x3 = np.vstack((x1, x2))

        # field, err = stat.jackknife_var(x3, potential)
        field, err = stat.jackknife_var_numba(x3, potential_numba)
        # print(field)

        # field1 = average(x3)

        new_row = {'aV(r)': field, 'err': err}

        df1 = df1.append(new_row, ignore_index=True)

        return df1


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


axis = 'on-axis'
conf_type = "qc2dstag"
# conf_type = "SU2_dinam"
# conf_sizes = ["40^4", "32^4"]
conf_sizes = ["32^4"]
# conf_sizes = ["40^4"]

for monopole in ['/', 'monopoless', 'monopole']:
    # for monopole in ['/']:
    # for monopole in ['monopole']:
    if monopole == '/':
        smearing = 'smeared'
    elif monopole == 'monopoless':
        smearing = 'smeared'
    elif monopole == 'monopole':
        smearing = 'unsmeared'
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
                for i in range(0, conf_max):
                    file_path = f"../../data/wilson_loop/{axis}/{monopole}/{conf_type}/{smearing}/{conf_size}/mu{mu}/{chain}/wilson_loop_{i:04}"
                    # file_path = f"../../data/wilson_loop/{axis}/{monopole}/{conf_type}/{conf_size}/mu{mu}/wilson_loop_{i:04}"
                    # print(file_path)

                    if(os.path.isfile(file_path)):
                        # print(file_path)
                        data.append(pd.read_csv(file_path, header=0,
                                    names=["T", "r/a", "wilson_loop"]))
                        data[-1]["conf_num"] = i
            if len(data) == 0:
                print("no data", conf_size, mu, monopole)
            elif len(data) != 0:
                df = pd.concat(data)

                # df = df[df['T'] <= 16]
                # df = df[df['r/a'] <= 16]

                # df_test = df[np.isnan(df['wilson_loop']) ]

                # print(df_test)

                # print(df)

                # wilson = df[['wilson_loop']].to_numpy()
                # conf_num = df[['conf_num']].to_numpy()

                # for i in range(len(wilson)):
                #     if(math.isnan(wilson[i])):
                #         print(conf_num[i])

                df1 = pd.DataFrame(columns=["aV(r)", "err"])

                time_size_max = df["T"].max()

                start = time.time()

                df1 = df.groupby(['T', 'r/a']).apply(get_field, df1,
                                                     df, time_size_max).reset_index()

                end = time.time()
                print("execution time = %s" % (end - start))

                df1 = df1[['T', 'r/a', 'aV(r)', 'err']]

                path_output = f"../../result/potential/{axis}/{monopole}/{conf_type}/{smearing}/{conf_size}"
                # path_output = f"../../result/potential_spatial/{test}/{axis}/{monopole}/qc2dstag/{conf_size}"
                # path_output = f"../../result/potential/{test}/{axis}/{monopole}/su2_dynam/{conf_size}/{smearing}"

                try:
                    os.makedirs(path_output)
                except:
                    pass

                # df1.to_csv(f"{path_output}/potential_spatial_mu={mu}.csv", index=False)
                df1.to_csv(f"{path_output}/potential_mu={mu}.csv", index=False)
