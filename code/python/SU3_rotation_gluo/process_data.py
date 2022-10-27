from numba import njit
import sys
import math
import time
import numpy as np
import os.path
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "..", ".."))
import statistics_python.src.statistics_observables as stat
import itertools
import autocorr


@njit
def trivial(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = x[0][i]
    return y


@njit
def b3(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = 12 * (x[0][i] - x[1][i] * x[1][i])
    return y


@njit
def b2(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = 12 * (x[0][i] * x[1][i] - x[2][i])
    return y


@njit
def b1(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = x[0][i] - 3 * x[1][i] * x[1][i]
    return y


@njit
def k2(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = x[0][i] - 2 * x[1][i]
    return y


@njit
def k4(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = 12 * (x[0][i] - x[1][i] * x[1][i] + x[2][i] * x[3]
                     [i] - x[4][i]) + x[5][i] - 3 * x[6][i] * x[6][i]
    return y


@njit
def k_ratio(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (12 * (x[0][i] - x[1][i] * x[1][i] + x[2][i] * x[3][i] - x[4][i]) + x[5][i] - 3 * x[6][i] * x[6][i] - (12 * (x[7][i] - x[8][i] *
                x[8][i] + x[9][i] * x[10][i] - x[11][i]) + x[12][i] - 3 * x[13][i] * x[13][i])) / (x[14][i] - 2 * x[15][i] - (x[16][i] - 2 * x[17][i]))
    return y


def estimate_composite(x, y, err_x, err_y):
    return math.log(x / y), math.sqrt((err_x / x) ** 2 + (err_y / y) ** 2)


def get_bin_borders(data_size, bin_size):
    nbins = data_size // bin_size
    bin_sizes = [bin_size for _ in range(nbins)]
    residual_size = data_size - nbins * bin_size
    idx = 0
    while residual_size > 0:
        bin_sizes[idx] += 1
        residual_size -= 1
        idx = (idx + 1) % nbins
    return np.array([0] + list(itertools.accumulate(bin_sizes)))


def make_observables(data, vol):
    data['s1^2'] = data['s1'] * data['s1']
    data['s2^2'] = data['s2'] * data['s2']
    data['s2*s1^2'] = data['s2'] * data['s1^2']
    data['s1^4'] = data['s1^2'] * data['s1^2']

    data['s1^2*vol'] = data['s1^2'] * vol
    data['s1^4*vol^2'] = data['s1^4'] * vol**2
    data['s1^2*vol'] = data['s1^2'] * vol
    data['s2*vol'] = data['s2'] * vol
    data['s2*s1^2*vol'] = data['s2*s1^2'] * vol

    return data


def get_coefficients(data, bin):
    df_0 = data[data['T'] == '0']
    df_T = data[data['T'] == 'T']

    aver_0 = make_jackknife(df_0, bin)
    aver_T = make_jackknife(df_T, bin)

    for i in range(len(aver_0) // 2):
        aver_T[i * 2] = aver_T[i * 2] - aver_0[i * 2]
        aver_T[i * 2 + 1] = math.sqrt(aver_T[i * 2 + 1] *
                                      aver_T[i * 2 + 1] + aver_0[i * 2 + 1] * aver_0[i * 2 + 1])

    # aver_ratio = jackknife_ratio(df_T, df_0, bin)

    # aver_T = aver_T + aver_ratio

    # , '<ratio>', 'err_ratio'
    return pd.DataFrame([aver_T],
                        columns=['<s1>', 'err_s1', '<a1>', 'err_a1', '<a2>', 'err_a2', '<b3>', 'err_b3', '<b2>', 'err_b2', '<b1>', 'err_b1', '<k2>', 'err_k2', '<k4>', 'err_k4'])


def make_jackknife(df, bin):
    s1 = np.array([df['s1'].to_numpy()])
    a1_arr = np.array([2 * df['s2'].to_numpy()])
    a2_arr = np.array([df['s1^2*vol'].to_numpy()])
    b3_arr = np.vstack((df['s2^2'].to_numpy(), df['s2'].to_numpy()))
    b2_arr = np.vstack(
        (df['s2*vol'].to_numpy(), df['s1^2'].to_numpy(), df['s2*s1^2*vol'].to_numpy()))
    b1_arr = np.vstack(
        (df['s1^4*vol^2'].to_numpy(), df['s1^2*vol'].to_numpy()))

    k2_arr = np.vstack((df['s1^2*vol'].to_numpy(), df['s2'].to_numpy()))
    k4_arr = np.vstack((df['s2^2'].to_numpy(), df['s2'].to_numpy(), df['s2*vol'].to_numpy(), df['s1^2'].to_numpy(),
                        df['s2*s1^2*vol'].to_numpy(), df['s1^4*vol^2'].to_numpy(), df['s1^2*vol'].to_numpy()))

    aver_s1, err_s1 = stat.jackknife_var_numba_binning(
        s1, trivial, get_bin_borders(len(s1[0]), bin))

    aver_a1, err_a1 = stat.jackknife_var_numba_binning(
        a1_arr, trivial, get_bin_borders(len(a1_arr[0]), bin))

    aver_a2, err_a2 = stat.jackknife_var_numba_binning(
        a2_arr, trivial, get_bin_borders(len(a2_arr[0]), bin))

    aver_b3, err_b3 = stat.jackknife_var_numba_binning(
        b3_arr, b3, get_bin_borders(len(b3_arr[0]), bin))

    aver_b2, err_b2 = stat.jackknife_var_numba_binning(
        b2_arr, b2, get_bin_borders(len(b2_arr[0]), bin))

    aver_b1, err_b1 = stat.jackknife_var_numba_binning(
        b1_arr, b1, get_bin_borders(len(b1_arr[0]), bin))

    aver_k2, err_k2 = stat.jackknife_var_numba_binning(
        k2_arr, k2, get_bin_borders(len(k2_arr[0]), bin))

    aver_k4, err_k4 = stat.jackknife_var_numba_binning(
        k4_arr, k4, get_bin_borders(len(k4_arr[0]), bin))

    return [aver_s1, err_s1, aver_a1, err_a1, aver_a2, err_a2, aver_b3, err_b3, aver_b2, err_b2, aver_b1, err_b1, aver_k2, err_k2, aver_k4, err_k4]


def jackknife_ratio(df_T, df_0, bin):
    # k_ratio_arr might be wrong
    k_ratio_arr = np.vstack((df_T['s2^2'].to_numpy(), df_T['s2'].to_numpy(), df_T['s2*vol'].to_numpy(), df_T['s1^2'].to_numpy(), df_T['s2*s1^2*vol'].to_numpy(),
                             df_T['s1^4*vol^2'].to_numpy(), df_T['s1^2*vol'].to_numpy(), df_0['s2^2'].to_numpy(
    ), df_0['s2'].to_numpy(), df_0['s2*vol'].to_numpy(), df_0['s1^2'].to_numpy(),
        df_0['s2*s1^2*vol'].to_numpy(), df_0['s1^4*vol^2'].to_numpy(), df_0['s1^2*vol'].to_numpy(), df_T['s2*vol'].to_numpy(), df_T['s2'].to_numpy(), df_0['s2*vol'].to_numpy(), df_0['s2'].to_numpy()))

    aver_ratio, err_ratio = stat.jackknife_var_numba_binning(
        k_ratio_arr, k_ratio, get_bin_borders(len(k_ratio_arr[0]), bin))

    return [aver_ratio, err_ratio]


def read_data(paths):
    data_full = []
    for beta, path in paths.items():
        data_T = []
        data_0 = []
        for i in range(len(path[0])):
            data_T.append(pd.read_csv(path[0][i], sep=','))
        for i in range(len(path[1])):
            data_0.append(pd.read_csv(path[1][i], sep=','))

        data_T = pd.concat(data_T)
        data_T = data_T[['S1_chair/vol', 'S2_total/vol']]
        data_T = data_T.rename(columns={'S1_chair/vol': 's1',
                                        'S2_total/vol': 's2'})
        data_T['T'] = 'T'
        data_0 = pd.concat(data_0)
        data_0 = data_0[['S1_chair/vol', 'S2_total/vol']]
        data_0 = data_0.rename(columns={'S1_chair/vol': 's1',
                                        'S2_total/vol': 's2'})
        data_0['T'] = '0'
        data_full.append(pd.concat([data_T, data_0]))
        data_full[-1]['beta'] = beta

    return pd.concat(data_full)


Nt = 6
Ns = 41
Nz = 40
vol = Nt * Nz * Ns**2
betas = ['4.04', '4.12', '4.20', '4.22', '4.24', '4.26', '4.28', '4.30', '4.32',
         '4.34', '4.36', '4.44', '4.52', '4.60', '4.68', '4.76', '4.84', '4.92', '5.00']
# betas = ['4.04']

obs1 = ['<a1>', 'err_a1', '<a2>', 'err_a2', '<k2>', 'err_k2']
obs2 = ['<b1>', 'err_b1', '<b2>', 'err_b2', '<b3>', 'err_b3', '<k4>', 'err_k4']
coef1 = 4 * Nt**4 / Ns**2
coef2 = -16 * Nt**4 * vol / Ns**4

path = '../../../data/SU3_gluodynamics'
paths = {}
chains = ['run3001000', 'run3001001', 'run3001002',
          'run3002000', 'run3002001', 'run3002002']
# chains = ['run3001000', 'run3001001']
for beta in betas:
    paths_T = []
    paths_0 = []
    for chain in chains:
        file_path = f'{path}/{Nt}x40x41^2/PBC_cV/0.00v/{beta}/{chain}/action_data'
        if os.path.isfile(file_path):
            paths_T.append(file_path)
        file_path = f'{path}/40x40x41^2/PBC_cV/0.00v/{beta}/{chain}/action_data'
        if os.path.isfile(file_path):
            paths_0.append(file_path)

    paths[beta] = [paths_T, paths_0]

# print(paths)

df = read_data(paths)


df = make_observables(df, vol)

print(df)

start = time.time()

bin = 1000
df_res = df.groupby(['beta']).apply(get_coefficients, bin).reset_index(
    level='beta').reset_index(drop=True)

# bins = autocorr.int_log_range(1, 10000, 1.05)

# data_bins = []

# for bin in bins:
#     data_bins.append(df.groupby(['beta']).apply(get_coefficients, bin).reset_index(
#         level='beta').reset_index(drop=True))
#     data_bins[-1]['bin'] = bin

# df_res = pd.concat(data_bins)

print(df_res)


df_res[obs1] = df_res[obs1] * coef1
df_res[obs2] = df_res[obs2] * coef2

# df_res['<ratio>'] = df_res[]

end = time.time()

print(df_res)

print("execution time = %s" % (end - start))

path_output = f"../../../result/SU3_gluodynamics/Nt{Nt}"
try:
    os.makedirs(path_output)
except:
    pass
df_res.to_csv(
    f"{path_output}/coefficients.csv", index=False)
