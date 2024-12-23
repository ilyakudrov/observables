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

#from tqdm import tqdm
#tqdm.pandas()
#from pandarallel import pandarallel
#pandarallel.initialize(progress_bar=False, nb_workers=4)

sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", ".."))
import statistics_python.src.statistics_observables as stat

pd.options.mode.chained_assignment = None


def get_potential(data):
    time_size_min = data["T"].min()
    time_size_max = data["T"].max()

    potential = []

    # print(data)

    for time_size in range(time_size_min, time_size_max):
        x = np.vstack((data.loc[data['T'] == time_size, 'wilson_loop'].to_numpy(),
                       data.loc[data['T'] == time_size + 1, 'wilson_loop'].to_numpy()))
        field, err = stat.jackknife_var_numba(x, potential_numba)
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

def read_data(path, chains, conf_max, copy_single, copies, CSV_names, dtype, adjoint_fix):
    data = []
    for chain in chains:
        for i in range(0, conf_max + 1):
            if copy_single:
                file_path = f'{path}/{chain}/wilson_loop_{i:04}'
                if (os.path.isfile(file_path)):
                    data.append(pd.read_csv(file_path, header=0,
                                            names=CSV_names,
                                            dtype=dtype))
                    data[-1]["conf"] = i
                    data[-1]["copy"] = copies
                    if adjoint_fix:
                        data[-1]["wilson_loop"] = data[-1]["wilson_loop"] + 1
            else:
                for copy in range(1, copies + 1):
                    file_path = f'{path}/{chain}/wilson_loop_{i:04}_{copy}'
                    if (os.path.isfile(file_path)):
                        data.append(pd.read_csv(file_path, header=0,
                                                names=CSV_names,
                                                dtype=dtype))
                        data[-1]["conf"] = f'{i}-{chain}'
                        data[-1]["copy"] = copy
                        if adjoint_fix:
                            data[-1]["wilson_loop"] = data[-1]["wilson_loop"] + 1
    return data

def read_no_copy(chains, conf_max, path, CSV_names, dtype):
    data = pd.DataFrame()
    for chain in chains:
        for i in range(0, conf_max + 1):
            file_path = f'{path}/{chain}/wilson_loop_{i:04}'
            #print(file_path)
            if (os.path.isfile(file_path)):
                df = pd.read_csv(file_path, header=0,
                                        names=CSV_names,
                                        dtype=dtype)
                df["conf"] = f'{i}-{chain}'
                data = pd.concat([data, df])
    return data

def read_copy(chains, conf_max, path, copy, CSV_names, dtype):
    data = pd.DataFrame()
    for chain in chains:
        for i in range(0, conf_max + 1):
            file_path = f'{path}/{chain}/wilson_loop_{i:04}_{copy}'
            if (os.path.isfile(file_path)):
                df = pd.read_csv(file_path, header=0,
                                        names=CSV_names,
                                        dtype=dtype)
                df["conf"] = f'{i}-{chain}'
                df["copy"] = copy
                data = pd.concat([data, df])
    return data

def read_copy_last(chains, conf_max, path, copy, CSV_names, dtype):
    data = pd.DataFrame()
    for chain in chains:
        for i in range(0, conf_max + 1):
            c = copy
            while c > 0:
                file_path = f'{path}/{chain}/wilson_loop_{i:04}_{c}'
                if (os.path.isfile(file_path)):
                    df = pd.read_csv(file_path, header=0,
                                            names=CSV_names,
                                            dtype=dtype)
                    df['conf'] = f'{i}-{chain}'
                    df['copy'] = copy
                    data = pd.concat([data, df])
                    break
                c -= 1
    return data

def read_data_single_copy(path, chains, conf_max, CSV_names, dtype, copy):
    if copy == 0:
        data = read_no_copy(chains, conf_max, path, CSV_names, dtype)
    else:
        if copy == 1:
            data = read_copy(chains, conf_max, path, copy, CSV_names, dtype)
        else:
            data = read_copy_last(chains, conf_max, path, copy, CSV_names, dtype)
    return data

parser = argparse.ArgumentParser()
parser.add_argument('--size', action="append")
parser.add_argument('--smearing', action="append")
parser.add_argument('--matrix_type', action="append")
parser.add_argument('--additional_parameters', action="append")
parser.add_argument('--beta', action="append")
parser.add_argument('--copies', type=int)
args = parser.parse_args()
print('args: ', args)

axis = 'on-axis'
conf_type = "gluodynamics"
conf_sizes = args.size
theory_type = 'su2'
betas = args.beta
copies = args.copies
smeared_array = args.smearing
matrix_type_array = args.matrix_type
# operator_type = 'wilson_loop'
#operator_type = 'wilson_gevp'
operator_type = 'wilson_loop_spatial'
representation = 'fundamental'
additional_parameters_arr = args.additional_parameters

is_binning = False
bin_max = 5
bin_step = 1.3
calculation_type = 'smearing'

# if calculation_type == 'smearing':
#     potential_parameters = ['smearing_step', 'r/a']
#     CSV_names = ['smearing_step', "T", "r/a", "wilson_loop"]
#     names_out = ['smearing_step', 'T', 'r/a', 'aV(r)', 'err']
#     dtype = {'smearing_step': np.int32, "T": np.int32,
#              "r/a": np.int32, "wilson_loop": np.float64}
#     base_dir = 'smearing'
if calculation_type == 'smearing':
    potential_parameters = ['smearing_step', 'r/a']
    #CSV_names = ['smearing_step1', 'smearing_step2', "T", "r/a", "wilson_loop"]
    CSV_names = ['smearing_step', "T", "r/a", "wilson_loop"]
    #names_out = ['smearing_step1', 'smearing_step2', 'T', 'r/a', 'aV(r)', 'err']
    names_out = ['smearing_step', 'T', 'r/a', 'aV(r)', 'err']
    #dtype = {'smearing_step1': np.int32, 'smearing_step2': np.int32, "T": np.int32,
    dtype = {'smearing_step': np.int32, "T": np.int32,
             "r/a": np.int32, "wilson_loop": np.float64}
    base_dir = ''

elif calculation_type == 'no_smearing':
    potential_parameters = ['r/a']
    CSV_names = ["T", "r/a", "wilson_loop"]
    dtype = {"T": np.int32, "r/a": np.int32, "wilson_loop": np.float64}
    names_out = ['T', 'r/a', 'aV(r)', 'err']
    base_dir = ''

conf_max = 5000
# mu1 = ['mu0.40']
mu1 = ['/']
#chains = ["/"]
# mu1 = ['mu0.05',
#        'mu0.20', 'mu0.25',
#        'mu0.30', 'mu0.35', 'mu0.45']
# mu1 = ['mu0.40']
# chains = ['s1', 's2']
chains = ['/', 's0', 's1', 's2', 's3',
           's4', 's5', 's6', 's7', 's8']

# adjoint_fix = True
adjoint_fix = False

#base_path = "../../data"
base_path = "/home/clusters/rrcmpi/kudrov/observables_cluster/result"

iter_arrays = [matrix_type_array, smeared_array,
               betas, conf_sizes, mu1, additional_parameters_arr]
for matrix_type, smeared, beta, conf_size, mu, additional_parameters in itertools.product(*iter_arrays):
    print('matrix_type: ', matrix_type, ', smeared: ', smeared, ' beta: ', beta,' conf_size: ',
          conf_size, ' mu: ', mu,' additional_parameters: ', additional_parameters)
    path = f'{base_path}/{base_dir}/{operator_type}/{representation}/{axis}/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{matrix_type}/{smeared}/{additional_parameters}'
    if copies > 0:
        copy_range = range(1, args.copies + 1)
        groupby_params = ['copy'] + potential_parameters
    else:
        copy_range = range(1)
        groupby_params = potential_parameters
    df1 = []
    start = time.time()
    for copy in copy_range:
        df = read_data_single_copy(path, chains, conf_max, CSV_names, dtype, copy)
        #print(df.to_string())
        #df = df.persist()
        #df = df[df['smearing_step1'] == df['smearing_step2']]
        #df = df.rename({'smearing_step1': 'smearing_step'}, axis=1)
        #df = df.drop('smearing_step2', axis=1)
        if len(df) == 0:
            print("no data")
        else:
            print(df)
            #df.head()
            if is_binning:
                bin_sizes = int_log_range(1, bin_max, bin_step)
                for bin_size in bin_sizes:
                    df1.append(df.groupby(groupby_params)\
                        .apply(get_potential_binning, bin_size).reset_index(level=groupby_params))
                    df1[-1]['bin_size'] = bin_size
                df1 = pd.concat(df1)
            else:
                df1.append(df.groupby(groupby_params)\
                    .apply(get_potential).reset_index(groupby_params))
    end = time.time()
    print("execution time = %s" % (end - start))
    df1 = pd.concat(df1)
    print(df1)
    #df1.head()

    if is_binning:
        base_dir1 = base_dir + '/binning'
    else:
        base_dir1 = base_dir
    path_output = f"../../result/{base_dir1}/potential/{operator_type}/{representation}/{axis}/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{smeared}/{additional_parameters}"
    try:
        os.makedirs(f'{path_output}')
    except:
        pass
    df1.to_csv(
        f"{path_output}/potential_{matrix_type}.csv", index=False)
