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


def get_polyakov(data):
    x = np.array(data['polyakov_loop'].to_numpy())
    x = np.array([x])

    field, err = stat.jackknife_var_numba(x, trivial)

    return pd.Series([field, err], index=['field', 'err'])

def get_polyakov_binning(data, bin_size):
    x = np.array(data['polyakov_loop'].to_numpy())
    x = np.array([x])

    data_size = x.shape[0]
    field, err = stat.jackknife_var_numba_binning(
            x, trivial, get_bin_borders(data_size, bin_size))

    return pd.Series([field, err], index=['field', 'err'])

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


#conf_type = "gluodynamics"
# conf_type = "qc2dstag"
conf_type = "QCD/140MeV"
# conf_type = "su2_suzuki"
# conf_type = "SU2_dinam"
#conf_sizes = ["24^4"]
#conf_sizes = ["nt4", "nt6", "nt8", "nt10", "nt12", "nt14"]
# conf_sizes = ["32^3x64"]
#conf_sizes = ["nt16", "nt18", "nt20"]
conf_sizes = ["nt4", "nt6", "nt8", "nt10", "nt12", "nt14", "nt16", "nt18", "nt20"]
theory_type = 'su3'
#betas = ['beta6.0']
betas = ['/']
copies = 0
copy_single = True
# betas = ['beta2.7', 'beta2.8']
smeared_array = ['HYP10_alpha=1_1_0.5']
# smeared_array = ['HYP2_alpha=1_1_0.5_APE_alpha=0.5',
#                  'HYP3_alpha=1_1_0.5_APE_alpha=0.5']
# smeared_array = ['HYP1_alpha=1_1_0.5_APE_alpha=0.5']
# smeared_array = ['HYP0_APE_alpha=0.5']
# matrix_type_array = ['monopole',
#                      'monopoless', 'photon', 'offdiagonal']
matrix_type_array = ['original']
#matrix_type_array = ['monopole']
#matrix_type_array = ['monopole',
#                      'monopoless', 'photon',
#                      'offdiagonal', 'abelian']
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
# additional_parameters_arr = ['steps_25/copies=4', 'steps_50/copies=4',
#                              'steps_100/copies=4', 'steps_200/copies=4',
#                              'steps_1000/copies=4', 'steps_2000/copies=4']
#additional_parameters_arr = ['steps_500/copies=1']
# additional_parameters_arr = ['steps_500/copies=3', 'steps_1000/copies=3',
#                              'steps_2000/copies=3', 'steps_4000/copies=3',
#                              'steps_8000/copies=3']
# additional_parameters_arr = ['steps_500/copies=3/compensate_1', 'steps_1000/copies=3/compensate_1',
#                              'steps_2000/copies=3/compensate_1', 'steps_4000/copies=3/compensate_1',
#                              'steps_8000/copies=3/compensate_1']
# additional_parameters_arr = ['steps_500/copies=3/compensate_1']
# additional_parameters_arr = ['steps_500/copies=3']
additional_parameters_arr = ['/']

is_binning = False
bin_max = 100
bin_step = 1.3
conf_max = 5000
# mu1 = ['mu0.40']
mu1 = ['/']
#hains = ["/"]
# mu1 = ['mu0.05',
#        'mu0.20', 'mu0.25',
#        'mu0.30', 'mu0.35', 'mu0.45']
# mu1 = ['mu0.40']
# chains = ['s1', 's2']
chains = ['/', 's0', 's1', 's2', 's3',
           's4', 's5', 's6', 's7', 's8']

#base_path = "../../data"
base_path = "/home/clusters/rrcmpi/kudrov/observables_cluster/result"
potential_parameters = ['HYP_step']
base_dir = 'smearing'

iter_arrays = [matrix_type_array, smeared_array,
               betas, conf_sizes, mu1, additional_parameters_arr]
for matrix_type, smeared, beta, conf_size, mu, additional_parameters in itertools.product(*iter_arrays):
    print(matrix_type, conf_size, mu, beta, smeared)
    data = []
    for chain in chains:
        for i in range(0, conf_max + 1):
            if copy_single:
                file_path = f'{base_path}/{base_dir}/polyakov_loop/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{matrix_type}/{smeared}/{additional_parameters}/{chain}/polyakov_loop_{i:04}'
                if (os.path.isfile(file_path)):
                    data.append(pd.read_csv(file_path))
                    data[-1]["conf"] = f'{i}-{chain}'
            else:
                for copy in range(1, copies + 1):
                    file_path = f'{base_path}/{base_dir}/polyakov_loop/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{matrix_type}/{smeared}/{additional_parameters}/{chain}/polyakov_loop_{i:04}_{copy}'
                    if (os.path.isfile(file_path)):
                        data.append(pd.read_csv(file_path))
                        data[-1]["conf"] = f'{i}-{chain}'
                        data[-1]["copy"] = copy
    if len(data) == 0:
        print("no data", matrix_type,
              conf_size, mu, beta, smeared)
    elif len(data) != 0:
        df = pd.concat(data)
        start = time.time()

        if not copy_single:
            df = fillup_copies(df)
            potential_parameters += ['copy']

        if is_binning:
            df1 = []
            bin_sizes = int_log_range(1, bin_max, bin_step)
            for bin_size in bin_sizes:
                df1.append(df.groupby(
                    potential_parameters).apply(get_polyakov_binning, bin_size).reset_index(level=potential_parameters))
                df1[-1]['bin_size'] = bin_size
            df1 = pd.concat(df1)
        else:
            df1 = df.groupby(
                potential_parameters).apply(get_polyakov).reset_index(level=potential_parameters)
        print(df1)

        end = time.time()
        print("execution time = %s" % (end - start))

        if is_binning:
            base_dir1 = base_dir + '/binning'
        else:
            base_dir1 = base_dir
        # df1 = df1[names_out]
        path_output = f"../../result/{base_dir1}/polyakov_loop/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{smeared}/{additional_parameters}"
        try:
            os.makedirs(f'{path_output}')
        except:
            pass
        df1.to_csv(
            f"{path_output}/polyakov_loop_{matrix_type}.csv", index=False)
