from astropy.stats import jackknife_resampling, jackknife_stats, bootstrap
from numba import njit
import sys
import math
import time
import numpy as np
import os.path
import pandas as pd
import itertools

sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", ".."))
import statistics_python.src.statistics_observables as stat


def get_potential(data):
    time_size_min = data["T"].min()
    time_size_max = data["T"].max()

    potential = []

    # print(data)

    for time_size in range(time_size_min, time_size_max):

        x1 = data.loc[data['T'] == time_size, 'wilson_loop'].to_numpy()
        x2 = data.loc[data['T'] == time_size + 1, 'wilson_loop'].to_numpy()

        x3 = np.vstack((x1, x2))

        field, err = stat.jackknife_var_numba(x3, potential_numba)

        potential.append([time_size, field, err])

    return pd.DataFrame(potential, columns=['T', 'aV(r)', 'err'])

def get_wilson_loop(data):
    wilson_loop, err = stat.jackknife_var_numba(data.loc[:, 'wilson_loop'].to_numpy(), potential_numba)
    return pd.DataFrame([wilson_loop, err], columns=['W', 'W_err'])

def get_potential_binning(data, bin_size):
    time_size_min = data["T"].min()
    time_size_max = data["T"].max()

    potential = []

    for time_size in range(time_size_min, time_size_max):

        x1 = data.loc[data['T'] == time_size, 'wilson_loop'].to_numpy()
        x2 = data.loc[data['T'] == time_size + 1, 'wilson_loop'].to_numpy()

        x3 = np.vstack((x1, x2))

        data_size = x3.shape[1]
        field, err = stat.jackknife_var_numba_binning(
            x3, potential_numba, get_bin_borders(data_size, bin_size))

        potential.append([time_size, field, err])

    return pd.DataFrame(potential, columns=['T', 'aV(r)', 'err'])


@njit
def potential_numba(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        fraction = x[0][i] / x[1][i]
        if (fraction >= 0):
            y[i] = math.log(fraction)
        else:
            y[i] = 0
    return y

@njit
def trivial_numba(x):
    return x[0]

def potential(x):
    a = np.mean(x, axis=1)
    fraction = a[0] / a[1]
    if (fraction >= 0):
        return math.log(fraction)
    else:
        return 0


def estimate_composite(x, y, err_x, err_y):
    return math.log(x / y), math.sqrt((err_x / x) ** 2 + (err_y / y) ** 2)


def int_log_range(_min, _max, factor):
    if _min < 1.0:
        raise ValueError(f"_min has to be not less then 1.0 "
                         f"(received _min={_min}).")
    if _max < _min:
        raise ValueError(f"_max has to be not less then _min "
                         f"(received _min={_min}, _max={_max}).")
    if factor <= 1.0:
        raise ValueError(f"factor has to be greater then 1.0 "
                         f"(received factor={factor}).")
    result = [int(_min)]
    current = float(_min)
    while current * factor < _max:
        current *= factor
        if int(current) != result[-1]:
            result.append(int(current))
    return result


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


axis = 'on-axis'
conf_type = "gluodynamics"
# conf_type = "qc2dstag"
# conf_type = "QCD/140MeV"
# conf_type = "su2_suzuki"
# conf_type = "SU2_dinam"
conf_sizes = ["24^4"]
# conf_sizes = ["32^3x64"]
# conf_sizes = ["nt16_gov", "nt14", "nt12"]
theory_type = 'su3'
betas = ['beta6.0']
# betas = ['beta2.7', 'beta2.8']
smeared_array = ['HYP0_alpha=1_1_0.5_APE_alpha=0.5']
# smeared_array = ['HYP2_alpha=1_1_0.5_APE_alpha=0.5',
#                  'HYP3_alpha=1_1_0.5_APE_alpha=0.5']
# smeared_array = ['HYP1_alpha=1_1_0.5_APE_alpha=0.5']
# smeared_array = ['HYP0_APE_alpha=0.5']
# matrix_type_array = ['monopole',
#                      'monopoless', 'photon', 'offdiagonal']
matrix_type_array = ['original']
# matrix_type_array = ['abelian']
matrix_type_array = ['monopole',
                     'monopoless', 'photon',
                     'offdiagonal', 'abelian']
operator_type = 'wilson_loop'
representation = 'fundamental'
# representation = 'adjoint'
# additional_parameters_arr = ['steps_500/copies=3/compensate_1', 'steps_1000/copies=3/compensate_1',
#  'steps_2000/copies=3/compensate_1', 'steps_4000/copies=3/compensate_1', 'steps_8000/copies=3/compensate_1']
# additional_parameters_arr = ['steps_500/copies=3', 'steps_1000/copies=3',
#                              'steps_2000/copies=3', 'steps_4000/copies=3', 'steps_8000/copies=3']
# additional_parameters_arr = ['steps_500/copies=3']
# additional_parameters_arr = ['steps_500/copies=3/compensate_1']
# additional_parameters_arr = ['T_step=0.0001', 'T_step=0.0002', 'T_step=0.0004',
#                              'T_step=0.0008', 'T_step=0.001', 'T_step=0.002',
#                              'T_step=0.004', 'T_step=0.008', 'T_step=5e-05']
# additional_parameters_arr = ['T_step=0.0002']
# additional_parameters_arr = ['T_step=0.0001', 'T_step=0.0002',
#                              'T_step=0.0004', 'T_step=0.0008', 'T_step=0.0016', 'T_step=0.0032']
additional_parameters_arr = ['steps_25/copies=4', 'steps_50/copies=4',
                             'steps_100/copies=4', 'steps_200/copies=4',
                             'steps_1000/copies=4', 'steps_2000/copies=4']
# additional_parameters_arr = ['steps_500/copies=3', 'steps_1000/copies=3',
#                              'steps_2000/copies=3', 'steps_4000/copies=3',
#                              'steps_8000/copies=3']
# additional_parameters_arr = ['steps_500/copies=3/compensate_1', 'steps_1000/copies=3/compensate_1',
#                              'steps_2000/copies=3/compensate_1', 'steps_4000/copies=3/compensate_1',
#                              'steps_8000/copies=3/compensate_1']
# additional_parameters_arr = ['steps_500/copies=3/compensate_1']
# additional_parameters_arr = ['steps_500/copies=3']
# additional_parameters_arr = ['/']

is_binning = False
bin_max = 1000
calculation_type = 'no_smearing'

if calculation_type == 'smearing':
    potential_parameters = ['smearing_step', 'r/a']
    CSV_names = ['smearing_step', "T", "r/a", "wilson_loop"]
    names_out = ['smearing_step', 'T', 'r/a', 'aV(r)', 'err']
    dtype = {'smearing_step': np.int32, "T": np.int32,
             "r/a": np.int32, "wilson_loop": np.float64}
    dir_name = 'smearing'

elif calculation_type == 'no_smearing':
    potential_parameters = ['r/a']
    CSV_names = ["T", "r/a", "wilson_loop"]
    dtype = {"T": np.int32, "r/a": np.int32, "wilson_loop": np.float64}
    names_out = ['T', 'r/a', 'aV(r)', 'err']
    dir_name = ''

conf_max = 5000
# mu1 = ['mu0.40']
mu1 = ['/']
chains = ["/"]
# mu1 = ['mu0.05',
#        'mu0.20', 'mu0.25',
#        'mu0.30', 'mu0.35', 'mu0.45']
# mu1 = ['mu0.40']
# chains = ['s1', 's2']
# chains = ['s0', 's1', 's2', 's3',
#           's4', 's5', 's6', 's7', 's8']

# adjoint_fix = True
adjoint_fix = False

iter_arrays = [matrix_type_array, smeared_array,
               betas, conf_sizes, mu1, additional_parameters_arr]
for matrix_type, smeared, beta, conf_size, mu, additional_parameters in itertools.product(*iter_arrays):
    print(matrix_type, conf_size, mu, beta, smeared)
    data = []
    for chain in chains:
        for i in range(0, conf_max + 1):
            # file_path = f'../../data/smearing/{operator_type}/{representation}/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{matrix_type}/{smeared}/{additional_parameters}/{chain}/wilson_loop_{i:04}'
            file_path = f'../../data/{operator_type}/{representation}/{axis}/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{matrix_type}/{smeared}/{additional_parameters}/{chain}/wilson_loop_{i:04}'
            # print(file_path)
            if (os.path.isfile(file_path)):
                data.append(pd.read_csv(file_path, header=0,
                                        names=CSV_names,
                                        dtype=dtype))
                data[-1]["conf_num"] = i
                if adjoint_fix:
                    data[-1]["wilson_loop"] = data[-1]["wilson_loop"] + 1
    if len(data) == 0:
        print("no data", matrix_type,
              conf_size, mu, beta, smeared)
    elif len(data) != 0:
        df = pd.concat(data)
        start = time.time()

        # print(df)

        if is_binning:
            df1 = []
            bin_sizes = int_log_range(1, bin_max, 1.05)
            for bin_size in bin_sizes:
                df1.append(df.groupby(
                    potential_parameters).apply(get_potential_binning, bin_size).reset_index(potential_parameters).reset_index())
                df1[-1]['bin_size'] = bin_size
            df1 = pd.concat(df1)
        else:
            df1 = df.groupby(
                potential_parameters).apply(get_potential).reset_index(potential_parameters).reset_index()

        end = time.time()
        print("execution time = %s" % (end - start))

        if is_binning:
            dir_name = dir_name + '/binning'
        df1 = df1[names_out]
        path_output = f"../../result/potential/wilson_loop/{representation}/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{smeared}/{additional_parameters}"
        try:
            os.makedirs(f'{path_output}/{dir_name}')
        except:
            pass
        df1.to_csv(
            f"{path_output}/{dir_name}/potential_{matrix_type}.csv", index=False)
