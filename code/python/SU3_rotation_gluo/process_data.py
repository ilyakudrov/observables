from numba import njit
import sys
import math
import time
import numpy as np
import os.path
import pandas as pd
import itertools

sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "..", ".."))
import statistics_python.src.statistics_observables as stat
import autocorr


class DataRotation:

    def __init__(self, paths, Nt_T, Nt_0, Nz, Ns, version):
        self.Nt_T = Nt_T
        self.Nt_0 = Nt_0
        self.Nz = Nz
        self.Ns = Ns

        data_full = []
        for beta, path in paths.items():
            data_T = []
            data_0 = []

            data_T = DataRotation.read_beta(path[0], version, 'T')
            data_0 = DataRotation.read_beta(path[1], version, '0')

            vol_T, vol_0 = self.get_volumes()
            data_T = DataRotation.make_observables(data_T, vol_T, version)
            data_0 = DataRotation.make_observables(data_0, vol_0, version)

            data_full.append(pd.concat([data_T, data_0]))
            data_full[-1]['beta'] = beta

        self.data = pd.concat(data_full)

    def read_beta(paths, version, T_label):
        data = []
        for path in paths:
            data.append(pd.read_csv(path, sep=','))
            if version == 'old':
                data[-1] = data[-1][['S1_chair/vol', 'S2_total/vol']]
                data[-1] = data[-1].rename(columns={'S1_chair/vol': 's1',
                                                    'S2_total/vol': 's2'})
                data[-1]['T'] = T_label
            elif version == 'new':
                data[-1] = data[-1][['<S2>_center/vol',
                                    '<S1^2>_center/vol', '<S2^2>_center/vol',
                                     '<S1^4>_center/vol', '<S2 S1^2>_center/vol']]
                data[-1] = data[-1].rename(columns={'<S2>_center/vol': 's2'})
                data[-1]['s1'] = 0.0
                data[-1]['T'] = T_label
            else:
                print('wrong version')

        return pd.concat(data)

    @njit
    def trivial(x):
        n = x.shape[1]
        y = np.zeros(n)
        for i in range(n):
            y[i] = x[0][i]
        return y

    @njit
    def b3(x):
        n = x.shape[1]
        y = np.zeros(n)
        for i in range(n):
            y[i] = 12 * (x[0][i] - x[1][i] * x[1][i])
        return y

    @njit
    def b2(x):
        n = x.shape[1]
        y = np.zeros(n)
        for i in range(n):
            y[i] = 12 * (x[0][i] * x[1][i] - x[2][i])
        return y

    @njit
    def b1(x):
        n = x.shape[1]
        y = np.zeros(n)
        for i in range(n):
            y[i] = x[0][i] - 3 * x[1][i] * x[1][i]
        return y

    @njit
    def k2(x):
        n = x.shape[1]
        y = np.zeros(n)
        for i in range(n):
            y[i] = x[0][i] - 2 * x[1][i]
        return y

    @njit
    def k4(x):
        n = x.shape[1]
        y = np.zeros(n)
        for i in range(n):
            y[i] = 12 * (x[0][i] - x[1][i] * x[1][i] + x[2][i] * x[3]
                         [i] - x[4][i]) + x[5][i] - 3 * x[6][i] * x[6][i]
        return y

    def do_jackknife(df, bin, c2, c4):
        s1 = np.array([df['s1'].to_numpy(dtype=np.float64)])
        a1_arr = np.array([df['s1^2*vol'].to_numpy(dtype=np.float64)])
        a2_arr = np.array([2 * df['s2'].to_numpy(dtype=np.float64)])
        b3_arr = np.stack((df['s2^2*vol'].to_numpy(dtype=np.float64),
                           df['s2*sqrt_vol'].to_numpy(dtype=np.float64))
                          )
        b2_arr = np.stack((df['s2*vol^2'].to_numpy(dtype=np.float64),
                           df['s1^2'].to_numpy(dtype=np.float64),
                           df['s2*s1^2*vol^2'].to_numpy(dtype=np.float64)))
        b1_arr = np.stack((df['s1^4*vol^3'].to_numpy(dtype=np.float64),
                           df['s1^2*sqrt_vol^3'].to_numpy(dtype=np.float64)))

        k2_arr = np.stack((df['s1^2*vol'].to_numpy(dtype=np.float64),
                           df['s2'].to_numpy(dtype=np.float64)))
        k4_arr = np.stack((df['s2^2*vol'].to_numpy(dtype=np.float64),
                           df['s2*sqrt_vol'].to_numpy(dtype=np.float64),
                           df['s2*vol^2'].to_numpy(dtype=np.float64),
                           df['s1^2'].to_numpy(dtype=np.float64),
                           df['s2*s1^2*vol^2'].to_numpy(dtype=np.float64),
                           df['s1^4*vol^3'].to_numpy(dtype=np.float64),
                           df['s1^2*sqrt_vol^3'].to_numpy(dtype=np.float64)))

        aver_s1, err_s1 = stat.jackknife_var_numba_binning(
            s1, DataRotation.trivial, get_bin_borders(len(s1[0]), bin))
        aver_a1, err_a1 = stat.jackknife_var_numba_binning(
            a1_arr, DataRotation.trivial, get_bin_borders(len(a1_arr[0]), bin))
        aver_a2, err_a2 = stat.jackknife_var_numba_binning(
            a2_arr, DataRotation.trivial, get_bin_borders(len(a2_arr[0]), bin))
        aver_b3, err_b3 = stat.jackknife_var_numba_binning(
            b3_arr, DataRotation.b3, get_bin_borders(len(b3_arr[0]), bin))
        aver_b2, err_b2 = stat.jackknife_var_numba_binning(
            b2_arr, DataRotation.b2, get_bin_borders(len(b2_arr[0]), bin))
        aver_b1, err_b1 = stat.jackknife_var_numba_binning(
            b1_arr, DataRotation.b1, get_bin_borders(len(b1_arr[0]), bin))
        aver_k2, err_k2 = stat.jackknife_var_numba_binning(
            k2_arr, DataRotation.k2, get_bin_borders(len(k2_arr[0]), bin))
        aver_k4, err_k4 = stat.jackknife_var_numba_binning(
            k4_arr, DataRotation.k4, get_bin_borders(len(k4_arr[0]), bin))

        return ([aver_s1, aver_a1 * c2 / 2, aver_a2 * c2 / 2,
                aver_b3 * c4 / 4, aver_b2 * c4 / 4, aver_b1 * c4 / 4,
                aver_k2 * c2 / 2, aver_k4 * c4 / 4],
                [err_s1, err_a1 * c2 / 2, err_a2 * c2 / 2,
                err_b3 * abs(c4) / 4, err_b2 * abs(c4) / 4,
                err_b1 * abs(c4) / 4, err_k2 * c2 / 2,
                err_k4 * abs(c4) / 4])

    def get_volumes(self):
        vol_T = self.Nt_T * self.Nz * self.Ns**2
        vol_0 = self.Nt_0 * self.Nz * self.Ns**2
        return (vol_T, vol_0)

    def get_coefficients2(self):
        c1_T = 4 * self.Nt_T**4 / self.Ns**2
        c1_0 = 4 * self.Nt_0**4 / self.Ns**2
        return (c1_T, c1_0)

    def get_coefficients4(self):
        c2_T = -16 * self.Nt_T**4 / self.Ns**4
        c2_0 = -16 * self.Nt_0**4 / self.Ns**4
        return (c2_T, c2_0)

    def make_observables(data, vol, version):
        if version == 'old':
            data['s1^2'] = data['s1'] * data['s1']
            data['s2^2'] = data['s2'] * data['s2']
            data['s2*s1^2'] = data['s2'] * data['s1^2']
            data['s1^4'] = data['s1^2'] * data['s1^2']

            data['s1^2*vol'] = data['s1^2'] * vol
            data['s2^2*vol'] = data['s2^2'] * vol
            data['s2*sqrt_vol'] = data['s2'] * math.sqrt(vol)
            data['s2*vol^2'] = data['s2'] * vol**2
            data['s2*s1^2*vol^2'] = data['s2*s1^2'] * vol**2
            data['s1^4*vol^3'] = data['s1^4'] * vol**3
            data['s1^2*sqrt_vol^3'] = data['s1^2'] * math.sqrt(vol**3)
        elif version == 'new':
            data['s1^2'] = data['<S1^2>_center/vol'] / vol
            data['s1^2*vol'] = data['<S1^2>_center/vol']
            data['s2^2*vol'] = data['<S2^2>_center/vol']
            data['s2*sqrt_vol'] = data['s2'] * math.sqrt(vol)
            data['s2*vol^2'] = data['s2'] * vol**2
            data['s2*s1^2*vol^2'] = data['<S2 S1^2>_center/vol']
            data['s1^4*vol^3'] = data['<S1^4>_center/vol']
            data['s1^2*sqrt_vol^3'] = data['<S1^2>_center/vol'] * \
                math.sqrt(vol)
        else:
            print('wrong version')

        return data

    def muliply_by_coefs(data, obs2, c2, obs4, obs4_err, c4):
        data[obs2] = data[obs2] * c2
        data[obs4] = data[obs4] * c4
        data[obs4_err] = data[obs4_err] * abs(c4)

    def get_observables(self, data, bin, mode):
        df_T = data[data['T'] == 'T']
        df_0 = data[data['T'] == '0']

        c2_T, c2_0 = self.get_coefficients2()
        c4_T, c4_0 = self.get_coefficients4()

        if mode == 'non-zero_temperature' or mode == 'common':
            aver_T, err_T = DataRotation.do_jackknife(df_T, bin, c2_T, c4_T)
        if mode == 'zero_temperature' or mode == 'common':
            aver_0, err_0 = DataRotation.do_jackknife(df_0, bin, c2_T, c4_T)

        result_aver = []
        result_err = []
        if mode == 'common':
            for i in range(len(aver_0)):
                result_aver.append(aver_T[i] - aver_0[i])
                result_err.append(math.sqrt(err_T[i]**2 +
                                            err_0[i]**2))
        elif mode == 'non-zero_temperature':
            for i in range(len(aver_T)):
                result_aver.append(aver_T[i])
                result_err.append(err_T[i])
        elif mode == 'zero_temperature':
            for i in range(len(aver_0)):
                result_aver.append(aver_0[i])
                result_err.append(err_0[i])
        else:
            print('wrong mode')

        result = []
        for i in range(len(result_aver)):
            result.append(result_aver[i])
            result.append(result_err[i])

        return pd.DataFrame([result],
                            columns=['<s1>', 'err_s1', '<a1>', 'err_a1',
                                     '<a2>', 'err_a2', '<b3>', 'err_b3',
                                     '<b2>', 'err_b2', '<b1>', 'err_b1',
                                     '<k2>', 'err_k2', '<k4>', 'err_k4'])

    def process_beta(self, bin, mode):
        return self.data.groupby(['beta']).apply(self.get_observables, bin, mode).reset_index(
            level='beta').reset_index(drop=True)


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


def filter_bins(data, beta, bin):
    df1 = data[(data['beta'] != beta) & (data['bin'] != bin)]
    df2 = data[(data['beta'] == beta) & (data['bin'] == bin)]

    return pd.concat([df1, df2])


def make_paths(path, betas, chains_T, chains_0, file_name):
    paths = {}
    for beta in betas:
        paths_T = []
        paths_0 = []
        for chain_T in chains_T:
            # file_path = f'{path}/{Nt_T}x{Nz}x{Ns}^2/PBC_cV/0.00v/{beta}/{chain_T}/{file_name}'
            file_path = f'{path}/{Nt_T}x{Nz}x{Ns}^2/{beta}/{chain_T}/{file_name}'
            if os.path.isfile(file_path):
                paths_T.append(file_path)
        for chain_0 in chains_0:
            # file_path = f'{path}/{Nt_0}x{Nz}x{Ns}^2/PBC_cV/0.00v/{beta}/{chain_0}/{file_name}'
            file_path = f'{path}/{Nt_0}x{Nz}x{Ns}^2/{beta}/{chain_0}/{file_name}'
            if os.path.isfile(file_path):
                paths_0.append(file_path)

        if paths_T != [] and paths_0 != []:
            paths[beta] = [paths_T, paths_0]

    return paths


Nt_T = int(sys.argv[1])
Nt_0 = int(sys.argv[2])
Ns = int(sys.argv[3])
Nz = int(sys.argv[4])

betas = []
if Nt_T == 3:
    betas = ['3.961']
elif Nt_T == 4:
    betas = ['3.88', '4.04', '4.12', '4.20', '4.26',
             '4.36', '4.52', '4.68', '4.84', '5.00']
elif Nt_T == 5:
    betas = ['4.00', '4.166', '4.253', '4.341',
             '4.407', '4.517', '4.69', '4.859']
elif Nt_T == 6:
    betas = ['4.10', '4.279', '4.374', '4.468',
             '4.539', '4.656', '4.838', '5.0']

data_version = 'new'
# path = f'/home/clusters/rrcmpi/kudrov/SU3_gluodynamics_rotation/results/{data_version}/logs'
path = f'/home/ilya/soft/lattice/observables/data/SU3_gluodynamics/{data_version}'
chains = []
for i in range(17):
    for j in range(17):
        chains.append('run3' + '{:03d}'.format(i) + '{:03d}'.format(j))
chains_T = chains
chains_0 = chains

paths = make_paths(path, betas, chains_T, chains_0, 'action_data.csv')

data_columns = 'new'
data = DataRotation(paths, Nt_T, Nt_0, Nz, Ns, data_columns)
print(data.data)

start = time.time()

bins = autocorr.int_log_range(10, 10000, 1.2)
# bins = [1000]

# modes = ['common', 'non-zero_temperature', 'zero_temperature']
modes = ['common']
# modes = ['common', 'non-zero_temperature']
binning = True
# binning = False

for mode in modes:
    data_bins = []
    for bin in bins:
        data_bins.append(data.process_beta(bin, mode))
        data_bins[-1]['bin'] = bin

    df_res = pd.concat(data_bins)

    # df_res = filter_bins(df_res, '4.20', 5000)

    end = time.time()

    print(df_res)

    print("execution time = %s" % (end - start))

    path_output = f"../../../result/SU3_gluodynamics/{data_version}/{Nt_T}({Nt_0})x{Nz}x{Ns}^2/columns_{data_columns}"
    try:
        os.makedirs(path_output)
    except:
        pass

    file_name = 'coefficients_'

    if binning:
        file_name = file_name + 'binning_'

    if mode == 'common':
        file_name = file_name + f'common_Nt{Nt_T}.csv'
    elif mode == 'non-zero_temperature':
        file_name = file_name + f'Nt{Nt_T}.csv'
    elif mode == 'zero_temperature':
        file_name = file_name + f'Nt{Nt_0}.csv'
    else:
        print('wrong mode')

    df_res.to_csv(f"{path_output}/{file_name}", index=False)
