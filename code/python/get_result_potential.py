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


def get_field(data, df1, df, time_size_max):

    time_size = data["T"].iloc[0]
    space_size = data["r/a"].iloc[0]

    if time_size < time_size_max:

        x1 = data['wilson_loop'].to_numpy()

        x2 = df[(df["T"] == time_size + 1) & (df["r/a"]
                                              == space_size)]['wilson_loop'].to_numpy()

        x3 = np.vstack((x1, x2))

        # print(x3)

        # field, err = stat.jackknife_var(x3, potential)
        field, err = stat.jackknife_var_numba(x3, potential_numba)
        # print(field)

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


def estimate_composite(x, y, err_x, err_y):
    return math.log(x / y), math.sqrt((err_x / x) ** 2 + (err_y / y) ** 2)


axis = 'on-axis'
conf_type = "qc2dstag"
# conf_type = "su2_suzuki"
# conf_type = "SU2_dinam"
# conf_sizes = ["40^4", "32^4"]
# conf_sizes = ["24^4"]
# conf_sizes = ["32^4"]
conf_sizes = ["40^4"]
# conf_sizes = ["48^4"]
theory_type = 'su2'
# betas = ['beta2.7']
# betas = ['beta2.4', 'beta2.5', 'beta2.6']
betas = ['']
# smeared = 'smeared'
# smeared = 'HYP1_alpha=1_1_0.5_APE400_alpha=0.5'
# smeared = 'HYP1_alpha=1_0.5_0.5_APE100_alpha=0.5'
# smeared = 'HYP0_alpha=1_1_0.5_APE100_alpha=0.5'
smeared = 'HYP0_alpha=1_1_0.5_APE200_alpha=0.5'
wilson_loop_type = 'wilson_loop'
potential_type = 'potential'
# wilson_loop_type = 'wilson_loop_adjoint'
# potential_type = 'potential_adjoint'

# adjoint_fix = True
adjoint_fix = False


for beta in betas:
    for monopole in ['monopole', 'abelian', 'photon']:
        # for monopole in ['/', 'monopoless']:
        # for monopole in ['/']:
        # for monopole in ['monopole']:
        if monopole == '/':
            monopole1 = theory_type
            # smearing = 'smeared'
        # smearing = 'HYP1_alpha=1_1_0.5_APE60_APE_alpha=0.75'
        else:
            monopole1 = monopole
        for conf_size in conf_sizes:
            if conf_size == '40^4':
                conf_max = 1000
                mu1 = ['mu0.05', 'mu0.20', 'mu0.25',
                       'mu0.30', 'mu0.35', 'mu0.45']
                # mu1 = ['mu0.25']
                # mu1 = ['mu0.00']
                chains = {"s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"}
                # chains = {"/"}
            elif conf_size == '32^4':
                conf_max = 2800
                mu1 = ['0.00']
                chains = {"/"}
            elif conf_size == '48^4':
                conf_max = 50
                mu1 = ['']
                chains = {'/'}
            elif conf_size == '24^4':
                conf_max = 100
                mu1 = ['']
                chains = {'/'}
            for mu in mu1:
                print(monopole, conf_size, mu)
                data = []
                for chain in chains:
                    for i in range(0, conf_max):
                        file_path = f"../../data/{wilson_loop_type}/{axis}/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{monopole}/{smeared}/{chain}/wilson_loop_{i:04}"

                        if(os.path.isfile(file_path)):
                            data.append(pd.read_csv(file_path, header=0,
                                        names=["T", "r/a", "wilson_loop"]))
                            data[-1]["conf_num"] = i
                            if adjoint_fix:
                                data[-1]["wilson_loop"] = data[-1]["wilson_loop"] + 1
                if len(data) == 0:
                    print("no data", monopole, conf_size, mu)
                elif len(data) != 0:
                    df = pd.concat(data)

                    # print(df)

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

                    path_output = f"../../result/{potential_type}/{axis}/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{smeared}"

                    try:
                        os.makedirs(path_output)
                    except:
                        pass

                    df1.to_csv(
                        f"{path_output}/potential_{monopole1}.csv", index=False)
