from astropy.stats import jackknife_resampling, jackknife_stats, bootstrap
from numba import njit
import sys
import math
import time
import numpy as np
import os.path
import pandas as pd
import itertools
import argparse
import dask.dataframe as dd
import read_data

#from tqdm import tqdm
#tqdm.pandas()
#from pandarallel import pandarallel
#pandarallel.initialize(progress_bar=False, nb_workers=4)

sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", ".."))
import statistics_python.src.statistics_observables as stat

pd.options.mode.chained_assignment = None


def get_potential(data):
    time_size_min = data["time_size"].min()
    time_size_max = data["time_size"].max()

    potential = []

    # print(data)

    for time_size in range(time_size_min, time_size_max):
        x = np.vstack((data.loc[data['time_size'] == time_size, 'wilson_loop'].to_numpy(),
                       data.loc[data['time_size'] == time_size + 1, 'wilson_loop'].to_numpy()))
        field, err = stat.jackknife_var_numba(x, potential_numba)
        potential.append([time_size, field, err])
    return pd.DataFrame(potential, columns=['time_size', 'aV(r)', 'err'])

def get_wilson_loop(data):
    wilson_loop, err = stat.jackknife_var_numba(data.loc[:, 'wilson_loop'].to_numpy(), potential_numba)
    return pd.DataFrame([wilson_loop, err], columns=['W', 'W_err'])

def get_potential_binning(data, bin_size):
    time_size_min = data["T"].min()
    time_size_max = data["T"].max()
    potential = []
    for time_size in range(time_size_min, time_size_max):
        x = np.vstack((data.loc[data['T'] == time_size, 'wilson_loop'].to_numpy(),
                        data.loc[data['T'] == time_size + 1, 'wilson_loop'].to_numpy()))
        data_size = x.shape[1]
        field, err = stat.jackknife_var_numba_binning(x,
            potential_numba, get_bin_borders(data_size, bin_size))
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

def fillup_copies(df):
    copy_num = df.groupby(['copy']).ngroups
    if copy_num > 1:
        for i in range(1, copy_num + 1):
            df1 = df[df['copy'] == i - 1]
            df2 = df[df['copy'] == i]
            df3 = df1[~df1['conf'].isin(df2['conf'])]
            df3.loc[:, 'copy'] = i
            df = pd.concat([df, df3])
    return df

parser = argparse.ArgumentParser()
parser.add_argument('--conf_type')
parser.add_argument('--theory_type')
parser.add_argument('--operator_type')
parser.add_argument('--representation')
parser.add_argument('--size', action="append")
parser.add_argument('--smearing', action="append")
parser.add_argument('--matrix_type', action="append")
parser.add_argument('--additional_parameters', action="append")
parser.add_argument('--mu', action="append")
parser.add_argument('--beta', action="append")
parser.add_argument('--copies', type=int)
parser.add_argument('--copies_each', type=bool, default=False)
args = parser.parse_args()
print('args: ', args)

axis = 'on-axis'
conf_type = args.conf_type
conf_sizes = args.size
theory_type = args.theory_type
betas = args.beta
copies = args.copies
copies_each = args.copies_each
smeared_array = args.smearing
matrix_type_array = args.matrix_type
operator_type = args.operator_type
representation = args.representation
additional_parameters_arr = args.additional_parameters

is_binning = False
bin_max = 5
bin_step = 1.3
calculation_type = 'smearing'

conf_max = 5000
# mu1 = ['mu0.00']
mu1 = args.mu
# mu1 = ['mu0.05',
#        'mu0.20', 'mu0.25',
#        'mu0.30', 'mu0.35', 'mu0.45']
# mu1 = ['mu0.40']
chains = ['/', 's0', 's1', 's2', 's3',
           's4', 's5', 's6', 's7', 's8', 's9', 's10']

base_path = "../../data"
# base_path = "/home/clusters/rrcmpi/kudrov/observables_cluster/result"

potential_parameters = ['smearing_step', 'space_size']

iter_arrays = [matrix_type_array, smeared_array,
               betas, conf_sizes, mu1, additional_parameters_arr]
for matrix_type, smeared, beta, conf_size, mu, additional_parameters in itertools.product(*iter_arrays):
    print('matrix_type: ', matrix_type, ', smeared: ', smeared, ' beta: ', beta,' conf_size: ',
          conf_size, ' mu: ', mu,' additional_parameters: ', additional_parameters)
    path = f'{base_path}/{operator_type}/{representation}/{axis}/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{matrix_type}/{smeared}/{additional_parameters}'
    print(path)
    start = time.time()
    if copies > 0:
        copy_range = range(0, copies)
        groupby_params = ['copy'] + potential_parameters
    else:
        copy_range = range(1)
        groupby_params = potential_parameters
    df1 = []
    for copy in copy_range:
        if copies > 0:
            if copies_each:
                df = read_data.read_copy_each(chains, conf_max, path,
                                              'wilson_loop', 4, copy)
            else:
                df = read_data.read_copy_best(chains, conf_max, path,
                                              'wilson_loop', 4, copy)
        else:
            df = read_data.read_no_copy(chains, conf_max, path,
                                        'wilson_loop', 4)
        if not df.empty:
            df = df[df['smearing_step1'] == df['smearing_step2']]
            df = df.rename({'smearing_step1': 'smearing_step'}, axis=1)
            df = df.drop('smearing_step2', axis=1)
            print(df)
            if is_binning:
                bin_sizes = int_log_range(1, bin_max, bin_step)
                for bin_size in bin_sizes:
                    df1.append(df.groupby(groupby_params)\
                        .apply(get_potential_binning, bin_size).reset_index(level=groupby_params))
                    df1[-1]['bin_size'] = bin_size
                df1 = pd.concat(df1)
            else:
                df1.append(df.groupby(groupby_params)\
                    .apply(get_potential, include_groups=False).reset_index(groupby_params))
    end = time.time()
    print("execution time = %s" % (end - start))
    if len(df1) != 0:
        df1 = pd.concat(df1)
    else:
        df1 = pd.DataFrame()
    print(df1)

    if not df1.empty:
        path_output = f"../../result/potential/{operator_type}/{representation}/{axis}/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{smeared}/{additional_parameters}"
        try:
            os.makedirs(f'{path_output}')
        except:
            pass
        df1.to_csv(
            f"{path_output}/potential_{matrix_type}.csv", index=False)
