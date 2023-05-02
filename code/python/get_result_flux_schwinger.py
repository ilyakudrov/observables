import pandas as pd
import os.path
import numpy as np
import math
import sys
import os.path
import itertools
from numba import njit
sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", ".."))
import statistics_python.src.statistics_observables as stat


def get_flux(data):
    x = data[['correlator_schwinger',
              'correlator_wilson', 'wilson_loop']].to_numpy()
    # x = data['correlator_schwinger'].to_numpy()
    # print(x)
    x = np.transpose(x)
    # print(x)

    field_electric, err_electric = stat.jackknife_var_numba(x, field_numba)
    # field_electric, err_electric = stat.jackknife_var_numba(np.array([x]), trivial)

    return pd.Series([field_electric, err_electric],
                     index=['field_electric', 'err_electric'])


@njit
def field_numba(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (x[0][i] - x[1][i]) / x[2][i]
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


@njit
def trivial(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = x[0][i]
    return y


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


conf_type = "qc2dstag"
# conf_type = "gluodynamics"
# conf_type = "su2_suzuki"
# conf_type = "SU2_dinam"
theory_type = 'su2'
flux_coord = 'd'
# flux_coord = 'x_tr'
direction = 'longitudinal'
# direction = 'transversal'
# direction = '_tr'
shift = False
# shift = True
fix_tr = False
# fix_tr = True
smearing_arr = ['HYP0_alpha=1_1_0.5_APE_alpha=0.5',
                'HYP1_alpha=1_1_0.5_APE_alpha=0.5',
                'HYP3_alpha=1_1_0.5_APE_alpha=0.5']
# smearing_arr = ['HYP1_alpha=1_1_0.5_APE_alpha=0.5']
# smearing = '/'

betas = ['/']
# betas = ['beta2.5']
decomposition_plaket_arr = ['original']
decomposition_wilson_arr = ['original']
conf_sizes = ["40^4"]
# conf_sizes = ["32^4"]
mu_arr = ['mu0.00']
# mu_arr = ['mu0.05', 'mu0.20', 'mu0.25', 'mu0.30',
#           'mu0.33', 'mu0.35', 'mu0.40', 'mu0.45']
# mu_arr = ['mu0.20', 'mu0.30',
#           'mu0.35', 'mu0.40', 'mu0.45']
# mu_arr = ['mu0.40', 'mu0.45']
# mu_arr = ['/']
conf_max = 5000
additional_parameters_arr = ['/']
chains = ['/']
# chains = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]


iter_arrays = [betas, decomposition_plaket_arr, decomposition_wilson_arr,
               conf_sizes, mu_arr, additional_parameters_arr, smearing_arr]
for beta, decomposition_plaket, decomposition_wilson, conf_size, mu, additional_parameters, smearing in itertools.product(*iter_arrays):
    print(decomposition_plaket, decomposition_wilson, conf_size, mu, beta)
    data_electric = []
    data_magnetic = []
    for chain in chains:
        for i in range(conf_max + 1):
            file_path_electric = f'../../data/flux_tube_schwinger/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{decomposition_plaket}-{decomposition_wilson}/{smearing}/{chain}/{direction}/electric_{i:04}'
            # file_path_magnetic = f'../../data/flux_tube_wilson/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{decomposition_plaket}-{decomposition_wilson}/{smearing}/{chain}/{direction}/magnetic_{i:04}'
            # print(file_path_electric)
            # if(os.path.isfile(file_path_electric) & os.path.isfile(file_path_magnetic)):
            if(os.path.isfile(file_path_electric)):
                data_electric.append(pd.read_csv(file_path_electric))
                data_electric[-1]['conf'] = i
                # data_magnetic.append(pd.read_csv(file_path_magnetic, header=0, names=[
                #                      'T', 'R', flux_coord, 'correlator magnetic', 'wilson loop', 'plaket magnetic']))
                # data_magnetic[-1]['conf'] = i

    # df = [pd.concat(data_electric), pd.concat(data_magnetic)]
    df = pd.concat(data_electric)

    # df = pd.concat(df, axis=1)
    # df = df.loc[:, ~df.columns.duplicated()]

    df['correlator_wilson'] = df['correlator_wilson'] / 2

    # df = df[(df['T'] <= 16) & (df['R'] <= 16) & (df['d'] <= 16)]

    # df = average_d(df, flux_coord)

    df = df.groupby(['T', 'R', flux_coord]).apply(
        get_flux).reset_index()

    path_output = f"../../result/flux_tube_schwinger/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{smearing}/{direction}"

    try:
        os.makedirs(path_output)
    except:
        pass

    df.to_csv(
        f"{path_output}/flux_tube_{decomposition_plaket}-{decomposition_wilson}.csv", index=False)
