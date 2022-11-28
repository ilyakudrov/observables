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
    n = data.shape[0]
    data = data.agg([np.mean, np.std])
    data.loc['std'] = data.loc['std'] / math.sqrt(n)

    return data


def get_coefficients(data, vol, c1, c2):
    data['a1'] = [c1 * vol * data['s1^2'].loc['mean'],
                  c1 * vol * data['s1^2'].loc['std']]

    data['a2'] = [
        c1 * 2 * data['s2'].loc['mean'], c1 * 2 * data['s2'].loc['std']]

    data['b3'] = [c2 * vol * 12 * (data['s2^2'].loc['mean'] - data['s2'].loc['mean']**2),
                  abs(c2) * vol * 12 * math.sqrt(data['s2^2'].loc['std']**2 + 4 * data['s2'].loc['mean']**2 * data['s2'].loc['std']**2)]

    data['b2'] = [c2 * vol**2 * 12 * (data['s2'].loc['mean'] * data['s1^2'].loc['mean'] - data['s2*s1^2'].loc['mean']),
                  abs(c2) * vol**2 * 12 * math.sqrt(data['s1^2'].loc['mean']**2 * data['s2'].loc['std']**2 + data['s2'].loc['mean']**2 * data['s1^2'].loc['std']**2 + data['s2*s1^2'].loc['mean']**2)]

    data['b1'] = [c2 * vol**3 * (data['s1^4'].loc['mean'] - 3 * data['s1^2'].loc['mean']**2), abs(c2) * vol**3 * math.sqrt(
        data['s1^4'].loc['mean']**2 + 18 * data['s1^2'].loc['mean']**2 * data['s1^2'].loc['std']**2)]

    data['k2'] = [data['a2'].loc['mean'] - data['a1'].loc['mean'],
                  math.sqrt(data['a1'].loc['std']**2 + data['a2'].loc['std']**2)]

    data['k4'] = [data['b1'].loc['mean'] + data['b2'].loc['mean'] + data['b3'].loc['mean'],
                  math.sqrt(data['b1'].loc['std']**2 + data['b2'].loc['std']**2 + data['b3'].loc['std']**2)]

    return data


def process(data, Nt_T, Nt_0, Nz, Ns):

    vol_T = Nt_T * Nz * Ns**2
    vol_0 = Nt_0 * Nz * Ns**2
    c2_T = 4 * Nt_T**4 / Ns**2
    c4_T = -16 * Nt_T**4 / Ns**4
    c2_0 = 4 * Nt_0**4 / Ns**2
    c4_0 = -16 * Nt_0**4 / Ns**4

    data = data.drop('beta', axis=1)
    df_T = data[data['T'] == 'T']
    df_T = df_T.drop('T', axis=1)
    df_0 = data[data['T'] == '0']
    df_0 = df_0.drop('T', axis=1)

    df_T = make_observables(df_T)
    df_0 = make_observables(df_0)

    df_T = get_stats(df_T)
    df_0 = get_stats(df_0)

    df_T = get_coefficients(df_T, vol_T, c2_T, c4_T)
    df_0 = get_coefficients(df_0, vol_0, c2_T, c4_T)

    # df_T = df_0

    df_T.loc['mean'] = df_T.loc['mean'] - df_0.loc['mean']
    df_T.loc['std'] = np.sqrt(
        df_T.loc['std'] * df_T.loc['std'] + df_0.loc['std'] * df_0.loc['std'])

    df_T = df_T.T
    df_T = df_T.stack()
    df_T = pd.DataFrame({'obs': df_T})
    df_T = df_T.T
    df_T.columns = df_T.columns.map('_'.join).str.strip('_')

    return df_T


Nt_T = 4
Nt_0 = 40
Ns = 41
Nz = 40
betas = ['3.88', '4.04', '4.12', '4.20', '4.22', '4.24', '4.26', '4.28', '4.30', '4.32',
         '4.34', '4.36', '4.44', '4.52', '4.60', '4.68', '4.76', '4.84', '4.92', '5.00']
# betas = ['4.04']

obs1 = ['<a1>', 'err_a1', '<a2>', 'err_a2', '<k2>', 'err_k2']
obs2 = ['<b1>', 'err_b1', '<b2>', 'err_b2', '<b3>', 'err_b3', '<k4>', 'err_k4']

path = '../../../data/SU3_gluodynamics'
paths = {}
chains = ['run3001000', 'run3001001', 'run3001002',
          'run3002000', 'run3002001', 'run3002002',
          'run3003000', 'run3004000', 'run3005000',
          'run3006000', 'run3007000', 'run3008000',
          'run3009000']
# chains = ['run3001000', 'run3001001']
for beta in betas:
    paths_T = []
    paths_0 = []
    for chain in chains:
        file_path = f'{path}/{Nt_T}x40x41^2/PBC_cV/0.00v/{beta}/{chain}/action_data'
        if os.path.isfile(file_path):
            paths_T.append(file_path)
        file_path = f'{path}/{Nt_0}x40x41^2/PBC_cV/0.00v/{beta}/{chain}/action_data'
        if os.path.isfile(file_path):
            paths_0.append(file_path)

    paths[beta] = [paths_T, paths_0]

# print(paths)

df = read_data(paths)

# print(df)

start = time.time()

df_res = df.groupby(['beta']).apply(process, Nt_T, Nt_0, Nz, Ns).reset_index(
    level='beta').reset_index(drop=True)


# print(df_res)
pd.set_option('display.max_columns', None)
print(df_res.head())

end = time.time()

# print(df_res)

print("execution time = %s" % (end - start))

path_output = f"../../../result/SU3_gluodynamics/Nt{Nt_T}"
try:
    os.makedirs(path_output)
except:
    pass
df_res.to_csv(
    f"{path_output}/coefficients_naive.csv", index=False)
