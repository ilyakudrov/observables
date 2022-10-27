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


def get_flux(data):
    x = data[['correlator electric', 'wilson loop', 'plaket electric']].to_numpy()
    x = np.transpose(x)

    field_electric, err_electric = stat.jackknife_var_numba(x, field_numba)

    x = data[['correlator magnetic', 'wilson loop', 'plaket magnetic']].to_numpy()
    x = np.transpose(x)

    field_magnetic, err_magnetic = stat.jackknife_var_numba(x, field_numba)

    x = data[['correlator electric', 'correlator magnetic',
              'wilson loop', 'plaket electric', 'plaket magnetic']].to_numpy()
    x = np.transpose(x)

    field_energy, err_energy = stat.jackknife_var_numba(x, energy_numba)

    x = data[['correlator electric', 'correlator magnetic',
              'wilson loop', 'plaket electric', 'plaket magnetic']].to_numpy()
    x = np.transpose(x)

    field_action, err_action = stat.jackknife_var_numba(x, action_numba)

    return pd.Series([field_electric, err_electric, field_magnetic, err_magnetic, field_energy, err_energy, field_action, err_action],
                     index=['field_electric', 'err_electric', 'field_magnetic', 'err_magnetic', 'field_energy', 'err_energy', 'field_action', 'err_action'])


@njit
def field_numba(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = x[0][i] / x[1][i] - x[2][i]
    return y


@njit
def energy_numba(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (x[0][i] - x[1][i]) / x[2][i] - x[3][i] + x[4][i]
    return y


@njit
def action_numba(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (x[0][i] + x[1][i]) / x[2][i] - x[3][i] - x[4][i]
    return y


def trivial(x):
    a = x.mean(axis=0)
    return a[0]


def average_d(data, flux_coord):
    data[flux_coord] = data[flux_coord].apply(lambda x: abs(x))
    data = data.groupby(['R', 'T', flux_coord, 'conf']
                        ).agg(np.mean).reset_index()
    data_concat = []
    data_concat.append(data)
    data_negative = pd.DataFrame(data)
    data_negative = data_negative[data_negative[flux_coord] != 0]
    data_negative[flux_coord] = -data_negative[flux_coord]
    data_concat.append(data_negative)
    data = pd.concat(data_concat)
    return data


def fix_data_tr(x):
    if x.name[3] == 0:
        return x / 2
    return x


# conf_type = "qc2dstag"
conf_type = "su2_suzuki"
# conf_type = "SU2_dinam"
# conf_sizes = ["40^4", "32^4"]
# conf_sizes = ["32^4"]
# conf_sizes = ["40^4"]
conf_sizes = ["24^4"]
theory_type = 'su2'
betas = ['beta2.4', 'beta2.5', 'beta2.6']
# betas = ['beta2.4']
# betas = ['']
flux_coord = 'd'
# flux_coord = 'x_tr'
direction = ''
# direction = '_tr'
shift = False
# shift = True
fix_tr = False
# fix_tr = True
# smearing = 'HYP1_alpha=1_1_0.5_APE200_alpha=0.5'
# smearing = 'HYP4_alpha=1_1_0.5_APE200_alpha=0.5'
# smearing = 'HYP1_alpha=1_1_0.5_APE300_alpha=0.5'
# smearing = 'HYP1_alpha=0.75_0.6_0.3_APE200_alpha=0.5'
smearing = 'HYP1_alpha=1_1_0.5_APE100_alpha=0.5'
# smearing = '/'

for beta in betas:
    # for monopole in ['monopoless', 'su2', 'monopole']:
    for monopole in ['su2', 'monopole', 'monopoless']:
        # for monopole in ['monopole']:
        for conf_size in conf_sizes:
            if conf_size == '40^4':
                conf_max = 2000
                mu1 = ['mu0.05', 'mu0.25', 'mu0.35', 'mu0.45']
                # mu1 = ['mu0.00']
                # mu1 = ['mu0.20', 'mu0.30']
                chains = {"s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"}
                # chains = {"s0", "s1", "s2", "s3"}
                # chains = {"/"}
            elif conf_size == '32^4':
                conf_max = 2800
                mu1 = ['mu0.00']
                chains = {"/"}
            elif conf_size == '24^4':
                conf_max = 100
                mu1 = ['']
                chains = {"/"}
            for mu in mu1:
                print(monopole, conf_size, mu, beta)
                data_electric = []
                data_magnetic = []
                for chain in chains:
                    for i in range(conf_max + 1):
                        file_path_electric = f'../../data/flux_tube_wilson{direction}/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{theory_type}-{monopole}/{smearing}/{chain}/electric_{i:04}'
                        file_path_magnetic = f'../../data/flux_tube_wilson{direction}/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{theory_type}-{monopole}/{smearing}/{chain}/magnetic_{i:04}'
                        # print(file_path_electric)
                        if(os.path.isfile(file_path_electric) & os.path.isfile(file_path_magnetic)):
                            data_electric.append(
                                pd.read_csv(file_path_electric, header=0,
                                            names=['T', 'R', flux_coord, 'correlator electric', 'wilson loop', 'plaket electric']))
                            if shift:
                                data_electric[-1][flux_coord] = data_electric[-1][flux_coord] - \
                                    data_electric[-1]['R'] / 2
                            data_electric[-1]['conf'] = i
                            data_magnetic.append(
                                pd.read_csv(file_path_magnetic, header=0,
                                            names=['T', 'R', flux_coord, 'correlator magnetic', 'wilson loop', 'plaket magnetic']))
                            if shift:
                                data_magnetic[-1][flux_coord] = data_magnetic[-1][flux_coord] - \
                                    data_magnetic[-1]['R'] / 2
                            data_magnetic[-1]['conf'] = i
                df = [pd.concat(data_electric), pd.concat(data_magnetic)]
                # df = pd.concat(df)

                df = pd.concat(df, axis=1)
                df = df.loc[:, ~df.columns.duplicated()]

                if fix_tr:
                    a = np.array(df['x_tr'] == 0) + 1
                    df['correlator electric'] = df['correlator electric'] / a
                    df['correlator magnetic'] = df['correlator magnetic'] / a

                df = average_d(df, flux_coord)

                # df = df[df['T'] == 8]
                # df = df[df['R'] == 6]
                # df = df[df['d'] == -5]
                # df = df[df['conf'] < 64]

                # print(df['wilson-plaket-correlator'].to_numpy())
                # print(df['wilson-loop'].to_numpy())
                # print(df['plaket'].to_numpy())

                df = df.groupby(['T', 'R', flux_coord]).apply(
                    get_flux).reset_index()

                # print(df)

                path_output = f"../../result/flux_tube_wilson{direction}/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{smearing}"

                try:
                    os.makedirs(path_output)
                except:
                    pass

                df.to_csv(
                    f"{path_output}/flux_tube_{theory_type}-{monopole}.csv", index=False)

                # df1 = pd.DataFrame(columns=['field', 'err'])

                # df1 = df.groupby(['d', 'R', 'T']).apply(
                #     get_field, df1).reset_index()

                # df1 = df1[['d', 'R', 'T', 'field', 'err']]
