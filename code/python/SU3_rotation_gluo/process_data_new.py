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


def usage():
    process = psutil.Process(os.getpid())
    values = psutil.virtual_memory()
    return (process.memory_info().rss / (1024.0 ** 2),
            values.available / (1024.0 ** 2),
            values.percent)


class DataRotation:

    def __init__(self, paths, Nt, Nz, Ns):
        self.Nt = Nt
        self.Nz = Nz
        self.Ns = Ns

        vol = self.get_volumes()
        data = DataRotation.read_beta(paths)
        data = DataRotation.make_observables(data, vol)

        self.data = data

    def read_beta(paths):
        data = []
        for path in paths:
            data.append(pd.read_csv(path, sep=','))
            data[-1] = data[-1].rename(columns={'<S1^2>_center/vol': 's1^2*vol',
                                                '<S2>_center/vol': 's2',
                                                '<S2^2>_center/vol': 's2^2*vol',
                                                '<S1^4>_center/vol': 's1^4*vol^3',
                                                '<S2 S1^2>_center/vol': 's2*s1^2*vol^2'})
            print("reading memory", usage())
            data_final = pd.concat([data_final, data])

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
            # if i < 5:
            #     print(x[0][i])
            # y[i] = x[5][i]
        return y

    def do_jackknife(self, bin, c2, c4):
        # s1 = np.array([df['s1'].to_numpy(dtype=np.float64)])
        # a1_arr = np.array([df['s1^2*vol'].to_numpy(dtype=np.float64)])
        # a2_arr = np.array([2 * df['s2'].to_numpy(dtype=np.float64)])
        # b3_arr = np.stack((df['s2^2*vol'].to_numpy(dtype=np.float64),
        #                    df['s2*sqrt_vol'].to_numpy(dtype=np.float64))
        #                   )
        # b2_arr = np.stack((df['s2*vol^2'].to_numpy(dtype=np.float64),
        #                    df['s1^2'].to_numpy(dtype=np.float64),
        #                    df['s2*s1^2*vol^2'].to_numpy(dtype=np.float64)))
        # b1_arr = np.stack((df['s1^4*vol^3'].to_numpy(dtype=np.float64),
        #                    df['s1^2*sqrt_vol^3'].to_numpy(dtype=np.float64)))

        # k2_arr = np.stack((df['s1^2*vol'].to_numpy(dtype=np.float64),
        #                    df['s2'].to_numpy(dtype=np.float64)))

        print("before to_numpy", usage())

        k4_arr = np.array([self.data['s2^2*vol'].to_numpy(dtype=np.float64)])
        self.data = self.data.drop('s2^2*vol', axis=1)
        k4_arr = np.vstack(
            [k4_arr, self.data['s2*sqrt_vol'].to_numpy(dtype=np.float64)])
        self.data = self.data.drop('s2*sqrt_vol', axis=1)
        k4_arr = np.vstack(
            [k4_arr, self.data['s2*vol^2'].to_numpy(dtype=np.float64)])
        self.data = self.data.drop('s2*vol^2', axis=1)
        k4_arr = np.vstack(
            [k4_arr, self.data['s1^2'].to_numpy(dtype=np.float64)])
        self.data = self.data.drop('s1^2', axis=1)
        k4_arr = np.vstack(
            [k4_arr, self.data['s2*s1^2*vol^2'].to_numpy(dtype=np.float64)])
        self.data = self.data.drop('s2*s1^2*vol^2', axis=1)
        k4_arr = np.vstack(
            [k4_arr, self.data['s1^4*vol^3'].to_numpy(dtype=np.float64)])
        self.data = self.data.drop('s1^4*vol^3', axis=1)
        k4_arr = np.vstack(
            [k4_arr, self.data['s1^2*sqrt_vol^3'].to_numpy(dtype=np.float64)])
        self.data = self.data.drop('s1^2*sqrt_vol^3', axis=1)

        print("after to_numpy", usage())

        del self.data

        print("after del data", usage())

        aver_k4, err_k4 = stat.jackknife_var_numba_binning(
            k4_arr, DataRotation.k4, get_bin_borders(len(k4_arr[0]), bin))

        # return ([aver_s1, aver_a1 * c2 / 2, aver_a2 * c2 / 2,
        #         aver_b3 * c4 / 4, aver_b2 * c4 / 4, aver_b1 * c4 / 4,
        #         aver_k2 * c2 / 2, aver_k4 * c4 / 4],
        #         [err_s1, err_a1 * c2 / 2, err_a2 * c2 / 2,
        #         err_b3 * abs(c4) / 4, err_b2 * abs(c4) / 4,
        #         err_b1 * abs(c4) / 4, err_k2 * c2 / 2,
        #         err_k4 * abs(c4) / 4])
        return ([aver_k4 * c4 / 4],
                [err_k4 * abs(c4) / 4])

    def get_volumes(self):
        return self.Nt * self.Nz * self.Ns**2

    def get_coefficients2(self):
        return 4 * self.Nt**4 / self.Ns**2

    def get_coefficients4(self):
        return -16 * self.Nt**4 / self.Ns**4

    def make_observables(data, vol):
        data['s1^2'] = data['s1^2*vol'] / vol
        data['s1^2*sqrt_vol^3'] = data['s1^2*vol'] * \
            math.sqrt(vol)
        data = data.drop('s1^2*vol', axis=1)
        data['s2*sqrt_vol'] = data['s2'] * math.sqrt(vol)
        data['s2*vol^2'] = data['s2'] * vol**2
        data = data.drop('s2', axis=1)

        return data

    def get_observables(self, bin, c2, c4):
        aver, err = self.do_jackknife(bin, c2, c4)

        # return pd.DataFrame([result],
        #                     columns=['<s1>', 'err_s1', '<a1>', 'err_a1',
        #                              '<a2>', 'err_a2', '<b3>', 'err_b3',
        #                              '<b2>', 'err_b2', '<b1>', 'err_b1',
        #                              '<k2>', 'err_k2', '<k4>', 'err_k4'])
        return aver, err

    def process_beta(self, bin, mode):
        return self.data.groupby(['beta']).apply(self.get_observables, bin, mode).reset_index(
            level='beta').reset_index(drop=True)


def process_data(paths, bin, Nt_T, Nt_0, Nz, Ns):
    data_T = DataRotation(paths[0], Nt_T, Nz, Ns)
    print(data_T.data.head())
    data_T.data.info()
    c2 = data_T.get_coefficients2()
    c4 = data_T.get_coefficients4()
    aver_T, err_T = data_T.get_observables(bin, c2, c4)
    data_0 = DataRotation(paths[1], Nt_0, Nz, Ns)
    data_0.data.info()
    aver_0, err_0 = data_0.get_observables(bin, c2, c4)

    print(aver_T, err_T)
    print(aver_0, err_0)

    result_aver = []
    result_err = []
    for i in range(len(aver_T)):
        result_aver.append(aver_T[i] - aver_0[i])
        result_err.append(math.sqrt(err_T[i]**2 +
                                    err_0[i]**2))
    result = []
    for i in range(len(result_aver)):
        result.append(result_aver[i])
        result.append(result_err[i])

    print(result)


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


Nt_T = 3
Nt_0 = 8
Ns = 11
Nz = 10
# Nt3
betas = ['3.961']
# Nt4
# betas = ['3.88', '4.04', '4.12', '4.20', '4.26',
#         '4.36', '4.52', '4.68', '4.84', '5.00']
# betas = ['3.88']
# Nt5
#betas = ['4.00', '4.166', '4.253', '4.341', '4.407', '4.517', '4.69', '4.859']
# betas = ['4.00']
# Nt6
#betas = ['4.10', '4.279', '4.374', '4.468', '4.539', '4.656', '4.838', '5.0']

data_version = 'new'
# path = f'/home/clusters/rrcmpi/kudrov/SU3_gluodynamics_rotation/results/{data_version}/logs'
path = f'/home/ilya/soft/lattice/observables/data/SU3_gluodynamics/{data_version}'
chains = []
for i in range(1, 3):
    chains.append('run3001' + '{:03d}'.format(i))
chains_T = chains
chains_0 = chains

paths = make_paths(path, betas, chains_T, chains_0, 'action_data.csv')

start = time.time()

# bins = autocorr.int_log_range(1, 10000, 1.05)
bins = [1000]

# modes = ['common', 'non-zero_temperature', 'zero_temperature']
modes = ['common']
# modes = ['common', 'non-zero_temperature']
# binning = True
binning = False

for mode in modes:
    data_bins = []
    for bin in bins:
        for beta, files in paths.items():
            process_data(files, bin, Nt_T, Nt_0, Nz, Ns)

    # df_res = pd.concat(data_bins)

    # df_res = filter_bins(df_res, '4.20', 5000)

    # end = time.time()

    # print(df_res)

    # print("execution time = %s" % (end - start))

    # path_output = f"../../../result/SU3_gluodynamics/{data_version}/{Nt_T}x{Nz}x{Ns}^2/columns_new"
    # try:
    #     os.makedirs(path_output)
    # except:
    #     pass

    # file_name = 'coefficients_'

    # if binning:
    #     file_name = file_name + 'binning_'

    # if mode == 'common':
    #     file_name = file_name + f'common_Nt{Nt_T}.csv'
    # elif mode == 'non-zero_temperature':
    #     file_name = file_name + f'Nt{Nt_T}.csv'
    # elif mode == 'zero_temperature':
    #     file_name = file_name + f'Nt{Nt_0}.csv'
    # else:
    #     print('wrong mode')

    # df_res.to_csv(f"{path_output}/{file_name}", index=False)
