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


def make_observables(data):
    data['s1^2'] = data['s1'] * data['s1']
    data['s2^2'] = data['s2'] * data['s2']
    data['s2*s1^2'] = data['s2'] * data['s1^2']
    data['s1^4'] = data['s1^2'] * data['s1^2']

    return data


def get_stats(data):


def process(data):


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
