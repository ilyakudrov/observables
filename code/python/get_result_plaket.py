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


def bin_data(data, bin_size):
    data_size = len(data)
    i = 0
    bin_num = (len(data) + bin_size - 1) // bin_size
    binned = np.ndarray(bin_num)
    for i in range(bin_num):
        tmp = 0
        count = 0
        for j in range(bin_size):
            if(i * bin_size + j < data_size):
                tmp += data[i * bin_size + j]
                count += 1

        binned[i] = tmp / count

    return binned


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
def trivial_numba(x):
    return x[0]


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


# conf_type = "qc2dstag"
conf_type = "gluodynamics"
# conf_type = "su2_suzuki"
# conf_type = "SU2_dinam"
# conf_sizes = ["40^4", "32^4"]
# conf_sizes = ["24^4"]
# conf_sizes = ["32^4"]
conf_sizes = ["36^4"]
# conf_sizes = ["48^4"]
theory_type = 'su3'
# betas = ['beta2.7']
# betas = ['beta2.4', 'beta2.5', 'beta2.6']
betas = ['beta6.3']
# smeared = 'smeared'
# smeared = 'HYP1_alpha=1_1_0.5_APE400_alpha=0.5'
# smeared = 'HYP1_alpha=1_0.5_0.5_APE100_alpha=0.5'
# smeared = 'HYP0_alpha=1_1_0.5_APE100_alpha=0.5'
# smeared = 'HYP0_alpha=1_1_0.5_APE200_alpha=0.5'
smeared = '/'

for beta in betas:
    for monopole in ['/']:
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
                conf_max = 830
                mu1 = ['/']
                chains = {"/"}
            elif conf_size == '48^4':
                conf_max = 50
                mu1 = ['']
                chains = {'/'}
            elif conf_size == '36^4':
                conf_max = 430
                mu1 = ['/']
                chains = {"/"}
            elif conf_size == '24^4':
                conf_max = 5000
                mu1 = ['']
                chains = {'/'}
            elif conf_size == '28^4':
                conf_max = 3000
                mu1 = ['']
                chains = {'/'}
            for mu in mu1:
                print(monopole, conf_size, mu)
                data = []
                for chain in chains:
                    for i in range(0, conf_max + 1):
                        file_path = f"../../data/plaket/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{monopole}/{smeared}/{chain}/plaket_{i:04}"

                        if(os.path.isfile(file_path)):
                            data.append(pd.read_csv(file_path, header=0,
                                        names=["plaket"]))
                            data[-1]["conf_num"] = i

                if len(data) == 0:
                    print("no data", monopole, conf_size, mu)
                elif len(data) != 0:
                    df = pd.concat(data)

                df.to_csv(
                    f"../../data/plaket/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{monopole}/{smeared}/{chain}/plaket.csv", index=False)

                plaket = df['plaket'].to_numpy()

                for i in range(len(plaket)):
                    if math.isnan(plaket[i]):
                        print(i)

                # print(plaket)

                df1 = pd.DataFrame(
                    columns=["bin_size", "plaket_jackknife", "err_jackknife", "plaket", "err"])

                df1 = []

                for bin_size in range(1, 20):

                    binned = bin_data(plaket, bin_size)

                    # plaket_jackknife, bias, err_jackknife, conf_interval = jackknife_stats(
                    #     binned, np.mean, 0.95)
                    plaket_jackknife, err_jackknife = stat.jackknife_var_numba(
                        np.array([binned]), trivial_numba)

                    def get_error(x): return np.std(
                        x, ddof=1) / math.sqrt(np.size(x))
                    plaket_mean = np.mean(binned)
                    err_mean = get_error(binned)

                    # print(bin_size, plaket_jackknife,
                    #       err_jackknife, plaket_mean, err_mean)

                    new_row = {"bin_size": [bin_size], "plaket_jackknife": [plaket_jackknife],
                               "err_jackknife": [err_jackknife], "plaket": [plaket_mean], "err": [err_mean]}

                    df1.append(pd.DataFrame(new_row))

                df1 = pd.concat(df1)
                print(df1)

                path_output = f"../../result/plaket/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{monopole}/{smeared}/{chain}"

                try:
                    os.makedirs(path_output)
                except:
                    pass

                df1.to_csv(
                    f"{path_output}/plaket_binning_{monopole1}.csv", index=False)
