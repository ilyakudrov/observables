import pandas as pd
import math
import numpy as np

# functions for reading data


def read_data_potentials_together(paths, sigma):
    data = []
    for key, value in paths.items():
        print(key)
        data.append(pd.read_csv(value[0], index_col=None))
        if len(value) >= 3:
            data[-1] = data[-1][data[-1]['smearing_step'] == value[2]]
        data[-1] = data[-1][data[-1]['T'] == value[1]]
        data[-1] = data[-1].drop(['T'], axis=1)
        data[-1].reset_index(drop=True, inplace=True)
        data[-1]['r/a'] = data[-1]['r/a'] * math.sqrt(sigma)
        data[-1][f'aV(r)'] = data[-1][f'aV(r)'] / math.sqrt(sigma)
        data[-1][f'err'] = data[-1][f'err'] / math.sqrt(sigma)
        data[-1]['type'] = key
    data = pd.concat(data)

    return data


def read_data_potential(paths):
    data = []
    for path in paths:
        data.append(pd.read_csv(path['path'], index_col=None))
        data[-1].reset_index(drop=True, inplace=True)
        if 'constraints' in path:
            for key, value in path['constraints'].items():
                data[-1] = data[-1][(data[-1][key] <= value[1])
                                    & (data[-1][key] >= value[0])]
        name = path['name']
        data[-1] = data[-1].rename(
            columns={'aV(r)': f'aV(r)_{name}', 'err': f'err_{name}'})
        data[-1]['r/a'] = data[-1]['r/a']
        data[-1][f'aV(r)_{name}'] = data[-1][f'aV(r)_{name}']
        data[-1][f'err_{name}'] = data[-1][f'err_{name}']
    data = pd.concat(data, axis=1)
    data = data.loc[:, ~data.columns.duplicated()]

    return data


def read_data_potential1(paths):
    data = []
    for type, path in paths.items():
        data.append(pd.read_csv(path['path'], index_col=None))
        data[-1].reset_index(drop=True, inplace=True)
        if 'constraints' in path:
            for key, value in path['constraints'].items():
                data[-1] = data[-1][(data[-1][key] <= value[1])
                                    & (data[-1][key] >= value[0])]
        name = path['name']
        data[-1] = data[-1].rename(
            columns={'aV(r)': f'aV(r)_{name}', 'err': f'err_{name}'})
        data[-1]['r/a'] = data[-1]['r/a']
        data[-1][f'aV(r)_{name}'] = data[-1][f'aV(r)_{name}']
        data[-1][f'err_{name}'] = data[-1][f'err_{name}']
        data[-1] = data[-1].reset_index(drop=True)
    data = pd.concat(data, axis=1)
    data = data.loc[:, ~data.columns.duplicated()]

    return data


def read_data_single(path):
    data = pd.read_csv(path['path'], index_col=None)
    data.reset_index(drop=True, inplace=True)
    if 'constraints' in path:
        for key, value in path['constraints'].items():
            data = data[(data[key] <= value[1]) & (data[key] >= value[0])]
    name = path['name']
    data = data.rename(
        columns={'aV(r)': f'aV(r)_{name}', 'err': f'err_{name}'})
    data['r/a'] = data['r/a']
    data[f'aV(r)_{name}'] = data[f'aV(r)_{name}']
    data[f'err_{name}'] = data[f'err_{name}']
    data = data.reset_index(drop=True)

    return data


# functions for reading vitaliy files
def read_viltaly_potential_mon_mls(path):
    data = pd.read_csv(f'{path}/potential_mon_mls_fit.dat', header=0,
                       names=['r/a', 'aV(r)_mon', 'err_mon',
                              'r1', 'aV(r)_mod', 'err_mod'],
                       dtype={'r/a': np.int32, "aV(r)_mon": np.float64, "err_mon": np.float64,
                              "r1": np.int32, "aV(r)_mod": np.float64, "err_mod": np.float64})
    return data[['r/a', 'aV(r)_mon', 'err_mon', 'aV(r)_mod', 'err_mod']]


def read_viltaly_potential_original(path):
    data = pd.read_csv(f'{path}/potential_original_fit.dat', header=0,
                       names=['r/a', 'aV(r)_SU(3)', 'err_SU(3)', 'err1'],
                       dtype={'r/a': np.int32, "aV(r)_SU(3)": np.float64,
                              "err_SU3": np.float64, "err1": np.float64})
    return data[['r/a', 'aV(r)_SU(3)', 'err_SU(3)']]


def read_vitaly_potential(path):
    data = []
    data.append(read_viltaly_potential_mon_mls(path))
    data.append(read_viltaly_potential_original(path))
    data = pd.concat(data, axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    return data
