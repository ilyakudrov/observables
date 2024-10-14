import time
import numpy as np
import os.path
import pandas as pd
import itertools
import argparse
import scipy

pd.options.mode.chained_assignment = None

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

def read_no_copy(chains, conf_max, path):
    data = pd.DataFrame()
    for chain in chains:
        for i in range(0, conf_max + 1):
            file_path = f'{path}/{chain}/wilson_loop_{i:04}'
            if (os.path.isfile(file_path)):
                df = pd.read_csv(file_path)
                df["conf"] = f'{i}-{chain}'
                data = pd.concat([data, df])
    return data

def read_copy(chains, conf_max, path, copy):
    data = pd.DataFrame()
    for chain in chains:
        for i in range(0, conf_max + 1):
            file_path = f'{path}/{chain}/wilson_loop_{i:04}_{copy}'
            if (os.path.isfile(file_path)):
                df = pd.read_csv(file_path)
                df["conf"] = f'{i}-{chain}'
                df["copy"] = copy
                data = pd.concat([data, df])
    return data

def read_copy_last(chains, conf_max, path, copy):
    data = pd.DataFrame()
    for chain in chains:
        for i in range(0, conf_max + 1):
            c = copy
            while c > 0:
                file_path = f'{path}/{chain}/wilson_loop_{i:04}_{c}'
                if (os.path.isfile(file_path)):
                    df = pd.read_csv(file_path)
                    df['conf'] = f'{i}-{chain}'
                    df['copy'] = copy
                    data = pd.concat([data, df])
                    break
                c -= 1
    return data

def read_data_single_copy(path, chains, conf_max, copy, copy_single):
    if copy_single:
        data = read_no_copy(chains, conf_max, path)
    else:
        if copy == 0:
            data = read_copy(chains, conf_max, path, copy)
        else:
            data = read_copy_last(chains, conf_max, path, copy)
    return data

# def jackknife_gevp(df):
#     wilson_loops_0 =

def fill_matrix(df):
    smearing_arr = df['smearing_step1'].unique()
    # print(smearing_arr)
    n = len(smearing_arr)
    matrix = np.zeros((n, n), dtype=np.float64)
    # print('df matrix')
    # print(df)
    for i in range(len(smearing_arr)):
        for j in range(len(smearing_arr)):
            if j >= i:
                matrix[i][j] = df.loc[(df['smearing_step1'] == smearing_arr[i]) & (df['smearing_step2'] == smearing_arr[j]), 'wilson_loop'].values[0]
                matrix[j][i] = df.loc[(df['smearing_step1'] == smearing_arr[i]) & (df['smearing_step2'] == smearing_arr[j]), 'wilson_loop'].values[0]
    # print('matrix')
    # print(matrix)
    return matrix

# def truncate(a, N_trunc):
#     w, v = scipy.linalg.eigh(a, subset_by_index=[0, N_trunc - 1])
#     matrix = np.zeros((N_trunc, N_trunc), dtype=np.float64)
#     for i in range(N_trunc):
#         for j in range(N_trunc):
#             matrix[i][j] =

def gevp_simple(df, df_0):
    a = fill_matrix(df)
    b = fill_matrix(df_0)
    # a = truncate(a, N_trunc)
    # b = truncate(b, N_trunc)
    w = scipy.linalg.eigh(a, b, eigvals_only=True)
    # w = scipy.linalg.eigh(a, eigvals_only=True)
    # w = np.sort(w)
    print('w', w)
    return pd.DataFrame({'potential': [w[-1]]})

def potenttial_gevp_simple(df, t0):
    wilson_aver = df.groupby(['smearing_step1', 'smearing_step2', 'time_size'])['wilson_loop'].apply(np.mean)\
        .reset_index(level=['smearing_step1', 'smearing_step2', 'time_size'])
    print('wilson_aver', wilson_aver)
    wilson_0 = wilson_aver[wilson_aver['time_size'] == t0]
    wilson_aver = wilson_aver[wilson_aver['time_size'] != t0]
    lambdas = wilson_aver.set_index('time_size').groupby('time_size').apply(gevp_simple, wilson_0, include_groups=False).reset_index(level='time_size').reset_index(drop=True)
    print('lambdas', lambdas)
    time_size_min = lambdas["time_size"].min()
    time_size_max = lambdas["time_size"].max()
    time_size = []
    potential = []
    for t in range(time_size_min, time_size_max):
        time_size.append(t)
        tmp = lambdas.loc[lambdas['time_size'] == t]
        lambda1 = tmp.at[tmp.index[0], 'potential']
        tmp = lambdas.loc[lambdas['time_size'] == t + 1]
        lambda2 = tmp.at[tmp.index[0], 'potential']
        if lambda1 / lambda2 > 0:
            potential.append(np.log(lambda1 / lambda2))
        else:
            potential.append(0)
    return pd.DataFrame({'time_size': time_size, 'potential': potential})

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
theory_type = 'su3'
betas = args.beta
copies = args.copies
smeared_array = args.smearing
matrix_type_array = args.matrix_type
operator_type = 'wilson_gevp'
representation = 'fundamental'
additional_parameters_arr = args.additional_parameters

is_binning = False
bin_max = 5
bin_step = 1.3

potential_parameters = ['r/a']

conf_max = 5000
mu1 = ['/']
chains = ['/', 's0', 's1', 's2', 's3',
           's4', 's5', 's6', 's7', 's8']

base_path = "../../data"
# base_path = "/home/clusters/rrcmpi/kudrov/observables_cluster/result"

iter_arrays = [matrix_type_array, smeared_array,
               betas, conf_sizes, mu1, additional_parameters_arr]
for matrix_type, smeared, beta, conf_size, mu, additional_parameters in itertools.product(*iter_arrays):
    print('matrix_type: ', matrix_type, ', smeared: ', smeared, ' beta: ', beta,' conf_size: ',
          conf_size, ' mu: ', mu,' additional_parameters: ', additional_parameters)
    path = f'{base_path}/{operator_type}/{representation}/{axis}/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{matrix_type}/{smeared}/{additional_parameters}'
    if copies > 0:
        copy_range = range(0, args.copies)
        groupby_params = ['copy'] + potential_parameters
        copy_single = False
    else:
        copy_range = range(1)
        groupby_params = potential_parameters
        copy_single = True
    df1 = []
    start = time.time()
    for copy in copy_range:
        print('copy', copy)
        df = read_data_single_copy(path, chains, conf_max, copy, copy_single)
        print(df)
        t0 = 2
        df = df[df['time_size'] >= t0]
        df = df[df['smearing_step1'].isin([0, 1, 11, 41]) & df['smearing_step2'].isin([0, 1, 11, 41])]
        # df = df[df['smearing_step1'].isin([0, 1, 11, 41, 71, 91]) & df['smearing_step2'].isin([0, 1, 11, 41, 71, 91])]
        # df = df[df['smearing_step1'].isin([41, 71, 91]) & df['smearing_step2'].isin([41, 71, 91])]
        # df = df[(df['smearing_step1'] <= 51) & (df['smearing_step2'] <= 51)]
        # print(df)
        # df = df[(df['time_size'] == 6) & (df['space_size'] == 6)]
        potential = df.set_index('space_size').groupby('space_size').apply(potenttial_gevp_simple, t0, include_groups=False).reset_index(level='space_size')
        print(potential)
        # potential = potential.reset_index(level=['time_size', 'space_size'])
        print(potential)
    #     if len(df) == 0:
    #         print("no data")
    #     else:
    #         print(df)
    #         if is_binning:
    #             bin_sizes = int_log_range(1, bin_max, bin_step)
    #             for bin_size in bin_sizes:
    #                 df1.append(df.groupby(groupby_params)\
    #                     .apply(get_potential_binning, bin_size).reset_index(level=groupby_params))
    #                 df1[-1]['bin_size'] = bin_size
    #             df1 = pd.concat(df1)
    #         else:
    #             df1.append(df.groupby(groupby_params)\
    #                 .apply(get_potential).reset_index(groupby_params))
    # end = time.time()
    # print("execution time = %s" % (end - start))
    # df1 = pd.concat(df1)
    # print(df1)

    # if is_binning:
    #     base_dir1 = '/binning'
    # else:
    #     base_dir1 = ''
    path_output = f"../../result/potential_gevp/wilson_loop/{representation}/{axis}/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{smeared}/{additional_parameters}"
    try:
        os.makedirs(f'{path_output}')
    except:
        pass
    potential.to_csv(
        f"{path_output}/potential_{matrix_type}.csv", index=False)
