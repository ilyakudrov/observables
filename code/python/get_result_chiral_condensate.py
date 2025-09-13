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


def get_chiral_condensate(data):
    x = np.array(data['chiral_condensate'].to_numpy())
    x = np.array([x])

    field, err = stat.jackknife_var_numba(x, trivial)

    return pd.Series([field, err], index=['chiral_condensate', 'err'])

def get_condensates(data):
    x_chiral = np.array(data['chiral_condensate'].to_numpy())
    x_diquark = np.array(data['diquark_condensate'].to_numpy())
    x_chiral = np.array([x_chiral])
    x_diquark = np.array([x_diquark])

    field_chiral, err_chiral = stat.jackknife_var_numba(x_chiral, trivial)
    field_diquark, err_diquark = stat.jackknife_var_numba(x_diquark, trivial)

    return pd.Series([field_chiral, err_chiral, field_diquark, err_diquark], index=['chiral_condensate', 'err_chiral', 'diquark_condensate', 'err_diquark'])

def get_chiral_condensate_binning(data, bin_size):
    x = np.array(data['chiral_condensate'].to_numpy())
    x = np.array([x])

    data_size = x.shape[0]
    field, err = stat.jackknife_var_numba_binning(
            x, trivial, get_bin_borders(data_size, bin_size))

    return pd.Series([field, err], index=['chiral_condensate', 'err'])

@njit
def potential_numba(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        # if (x[1][i] == 0):
        #     print(i)
        fraction = x[0][i] / x[1][i]
        if(fraction >= 0):
            y[i] = math.log(fraction)
        else:
            y[i] = 0
    return y


@njit
def trivial(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = x[0][i]
    return y


def potential(x):
    a = np.mean(x, axis=1)
    fraction = a[0] / a[1]
    if(fraction >= 0):
        return math.log(fraction)
    else:
        return 0


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

def read_chiral_condensate(path):
    data = []
    with open(path) as file:
        while line := file.readline():
            if(line[0] != '#'):
                data.append(float(line))
    return pd.DataFrame({'chiral_condensate': data})

def read_condensates(path):
    data_chiral = []
    data_diquark = []
    with open(path) as file:
        while line := file.readline():
            if(line[0] != '#'):
                line_splited = line.split()
                data_chiral.append(float(line_splited[0]))
                data_diquark.append(float(line_splited[1]))
    return pd.DataFrame({'chiral_condensate': data_chiral, 'diquark_condensate': data_diquark})

def make_cut(df):
    df = df.reset_index(drop=True)
    return df.loc[:99]

conf_type = "qc2dstag"
conf_sizes = ['32^3x16', '32^3x18', '32^3x20', '32^3x24', '32^3x28', '32^3x36', '32^3x40', '40^4']
# conf_sizes = ['32^3x16']
theory_type = 'su2'
betas = ['/']
copies = 0
copy_single = True
# matrix_type_array = ['monopoless', 'monopole']
matrix_type_array = ['original']
# additional_parameters_arr = ['steps_0/copies=20']
additional_parameters_arr = ['ma=0.0075']
# additional_parameters_arr = ['qc2dstag/ma=0.015', 'qc2dstag/ma=0.0075', 'qc2dstag/ma=0.0037']
# additional_parameters_arr = ['T_step=0.001/qc2dstag/ma=0.015', 'T_step=0.001/qc2dstag/ma=0.0075', 'T_step=0.001/qc2dstag/ma=0.0037']
#additional_parameters_arr = ['qc2dstag/ma=0.0075']
#additional_parameters_arr = ['ma=0.015', 'ma=0.0075', 'ma=0.0037']
#additional_parameters_arr = ['ma=0.0075']

is_binning = False
bin_max = 1000

conf_max = 5000
#mu1 = ['mu0.00', 'mu0.05', 'mu0.15']
#mu1 = ['mu0.00', 'mu0.05']
mu1 = ['mu0.15', 'mu0.20', 'mu0.25']
# mu1 = ['mu0.15']
# mu1 = ['/']
#chains = ["/"]
# mu1 = ['mu0.05',
#        'mu0.20', 'mu0.25',
#        'mu0.30', 'mu0.35', 'mu0.45']
# mu1 = ['mu0.40']
# chains = ['s1', 's2']
chains = ['/', 's0', 's1', 's2', 's3',
          's4', 's5', 's6', 's7', 's8', 's9', 's10']

# adjoint_fix = True
adjoint_fix = False

base_path = "../../data"
# base_path = '/home/clusters/rrcmpi/kudrov/observables_cluster/result'

iter_arrays = [matrix_type_array,
               betas, conf_sizes, mu1, additional_parameters_arr]
for matrix_type, beta, conf_size, mu, additional_parameters in itertools.product(*iter_arrays):
    print(matrix_type, conf_size, mu, beta)
    data = []
    for chain in chains:
        for i in range(0, conf_max + 1):
            if copy_single:
                file_path = f'{base_path}/chiral_condensate/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{matrix_type}/{additional_parameters}/{chain}/Condensates_{i:04}.txt'
                # print(file_path)
                if (os.path.isfile(file_path)):
                    # data.append(read_chiral_condensate(file_path))
                    data.append(read_condensates(file_path))
                    data[-1]["conf"] = i
                    data[-1]["copy"] = copies
            else:
                for copy in range(1, copies + 1):
                    file_path = f'{base_path}/chiral_condensate/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{matrix_type}/{additional_parameters}/{chain}/ChiralCond_{i:04}_{copy}.txt'
                    if (os.path.isfile(file_path)):
                        data.append(read_chiral_condensate(file_path))
                        data[-1]["conf"] = i
                        data[-1]["copy"] = copy
    if len(data) == 0:
        print("no data", matrix_type,
              conf_size, mu, beta)
    elif len(data) != 0:
        df = pd.concat(data)
        print(df)
        df = df.groupby(['copy', 'conf']).mean().reset_index(level=['copy', 'conf'])
        print(df)
        start = time.time()

        # df = fillup_copies(df)

        if is_binning:
            df1 = []
            bin_sizes = int_log_range(1, bin_max, 1.05)
            for bin_size in bin_sizes:
                df1.append(df.groupby('copy').apply(get_chiral_condensate_binning, bin_size).reset_index(level='copy'))
                df1[-1]['bin_size'] = bin_size
            df1 = pd.concat(df1)
        else:
            df1 = df.groupby('copy').apply(get_condensates, include_groups=False).reset_index(level='copy')
        if copy_single:
            df1 = df1.drop('copy', axis=1)
        print(df1)

        end = time.time()
        print("execution time = %s" % (end - start))

        if is_binning:
            binning = '/binning'
        else:
            binning = ''
        path_output = f"../../result/{binning}/condensates/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{additional_parameters}"
        try:
            os.makedirs(f'{path_output}')
        except:
            pass
        df1.to_csv(
            f"{path_output}/condensates_{matrix_type}.csv", index=False)
