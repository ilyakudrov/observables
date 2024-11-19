from numba import njit
import sys
import numpy as np
import os.path
import pandas as pd
import dask.dataframe as dd
import argparse
from scipy.stats import norm, binned_statistic
import itertools
import math
from typing import List, Tuple, Optional

sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "..", ".."))
import statistics_python.src.statistics_observables as stat

def int_log_range(_min: int, _max: int, factor: float) -> List[int]:
    """Make a geometric progression of integer values in range (_min, _max) with
    ratio factor.

    Args:
        _min: initial value of the progression
        _max: maximal value of the progression
        factor: ratio of the progression

    Returns: list of integer values of the geometric progression
    """
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

def get_bin_borders(data_size: int, bin_size: int) -> np.ndarray:
    """Make values of borders.

    Args:
        data_size: size of data
        bin_size: size of bin

    Returns: list of integer values for borders of bins
    """
    nbins = data_size // bin_size
    bin_sizes = [bin_size for _ in range(nbins)]
    residual_size = data_size - nbins * bin_size
    idx = 0
    while residual_size > 0:
        bin_sizes[idx] += 1
        residual_size -= 1
        idx = (idx + 1) % nbins
    return np.array([0] + list(itertools.accumulate(bin_sizes)))

def read_blocks(path: str) -> dd.DataFrame:
    """Read csv file of eos_rotation_imaginary data.

    Agrs: path to the csv file

    Returns: pandas DataFrame of the data
    """
    return dd.read_csv(path, header=None, delimiter=' ', names=['x', 'y', '2', '3', '4', '5',
                                                 '6', '7', '8', '9', '10', '11', '12',
                                                 '13', '14', '15', '16', '17', '18', 'S', '20', '21'])

def get_block_size(f: str) -> int:
    """Extract block size of eos data from it's name.

    Args:
        f: name of the file

    Returns: integer value of the block size
    """
    return int(f[5:f.find('-')])

def get_lattice_sizes(lattice_name: str) -> Tuple[int]:
    """Extract lattice sizes of eos data from name of a lattice size.

    Args:
        f: name of a lattice size

    Returns: tuple of integer values [t, z, s]
    """
    a = lattice_name.split('x')
    a[2] = a[2][:a[2].find('sq')]
    return [int(i) for i in a]

def get_conf_range(f: str) -> int:
    """Extract range of configurations in a block of eos data from it's name.

    Args:
        f: name of the file

    Returns: tuple of two integer configuration numbers
    """
    arr = f[f.find('_') + 1:f.find('.')].split('_')
    return int(arr[0]), int(arr[1])

def get_file_names(path: str) -> List[str]:
    """Get names of files with blocked eos data in directory path.

    Args:
        path: path of eos data

    Returns: list of file names
    """
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        files.extend(filenames)
        break
    return list((f for f in files if f[f.find('-') + 1:].startswith('SEBxy') and f.startswith(f'block')))

def get_dir_names(path: str) -> List[str]:
    """Get names of subdirectories in path.

    Args:
        path: path where to find subdirectories

    Returns: list of names of the subdirectories
    """
    directories = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        directories.extend(dirnames)
        break
    return directories

def thermalization_length(therm_path: str, additional_path: str, beta: str) -> int:
    """Get thermlization size for data
    with particular beta from corresponding csv file.

    Args:
        therm_path: path to csv file with data for thermlization size
        beta: string of beta directory name

    Returns: thermlization size
    """
    try:
        df_therm = pd.read_csv(therm_path, header=None, delimiter=' ', names=['beta', 'therm_length'])
    except:
        try:
            df_therm = pd.read_csv(additional_path, header=None, delimiter=' ', names=['beta', 'therm_length'])
        except:
            raise Exception('could not read spec_therm')
    if float(beta) in df_therm['beta'].unique():
        therm_length = df_therm.loc[df_therm['beta'] == float(beta), 'therm_length'].values[0]
    else:
        therm_length = df_therm.loc[df_therm['beta'] == 0.00, 'therm_length'].values[0]
    return therm_length

def bin_length(bins_path: str, additional_path: str, beta: str) -> int:
    """Get bin size for data
    with particular beta from corresponding csv file.

    Args:
        bins_path: path to csv file with data for bin size
        beta: string of beta directory name

    Returns: bin size
    """
    try:
        df_bins = pd.read_csv(bins_path, header=None, delimiter=' ', names=['beta', 'bin_size'])
    except:
        try:
            df_bins = pd.read_csv(additional_path, header=None, delimiter=' ', names=['beta', 'bin_size'])
        except:
            raise Exception('could not read spec_bins')
    if float(beta) in df_bins['beta'].unique():
        bin_size = df_bins.loc[df_bins['beta'] == float(beta), 'bin_size'].values[0]
    else:
        print(df_bins)
        print(df_bins.loc[df_bins['beta'] == 0, 'bin_size'])
        bin_size = df_bins.loc[df_bins['beta'] == 0, 'bin_size'].values[0]
    return bin_size

def get_data(base_path: str, args: argparse.Namespace, therm_length: int, bin_size: Optional[int] = None) -> dd.DataFrame:
    """Read and concatenate eos data for particlular parameters
    (lattice_size, boundary, velocity, beta).

    Args:
        base_path: path to a directory with lattice_size directory
        args: command line arguments with data parameters
        therm_length: thermalization length to cut from data
        bin_size: bin size to add to a resulting DataFrame

    Returns: DataFrame with the data to process
    """
    #df = dd.DataFrame()
    df = []
    print('therm_length: ', therm_length)
    chain_dirs = get_dir_names(f'{base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/{args.beta}')
    chain_dirs.sort()
    print('chain_dirs: ', chain_dirs)
    confs_to_skip = therm_length
    conf_last = 0
    for chain in chain_dirs:
        filenames = get_file_names(f'{base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/{args.beta}/{chain}')
        filenames.sort()
        for f in filenames:
            conf_start, conf_end = get_conf_range(f)
            if conf_start > confs_to_skip:
#                print(f'{base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/{args.beta}/{chain}/{f}')
                data = read_blocks(f'{base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/{args.beta}/{chain}/{f}')
                t, z, s = get_lattice_sizes(args.lattice_size)
#                print(t, z, s)
                if len(data.index) == s**2:
                    #data = data[['x', 'y', 'S']]
                    data = data.drop(labels=['20', '21'], axis=1)
                    data['block_size'] = get_block_size(f)
                    data['conf_start'] = conf_start + conf_last
                    data['conf_end'] = conf_end + conf_last
                    if bin_size is not None:
                        data['bin_size'] = bin_size
                    #data = data.astype({'x': 'int32', 'y': 'int32', 'block_size': 'int32',
                    #                'conf_start': 'int32', 'conf_end': 'int32', 'bin_size': 'int32','S': 'float64'})
                    #df = dd.concat([df, data])
                    df.append(data)
                else:
                    print(f'{base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/{args.beta}/{chain}/{f}', ' not complete')
        if len(filenames) != 0:
            _, conf_tmp = get_conf_range(filenames[-1])
            conf_last += conf_tmp
            confs_to_skip -= conf_tmp
    return dd.concat(df)

def make_jackknife(df: pd.DataFrame, bin_size: Optional[int] = None) -> pd.DataFrame:
    """Make blocked jackknife for data from df.

    Args:
        df: DataFrame with data to do jackknife on
        bin_size: size of bin in kackknife. If None, derive it from df.

    Returns: DataFrame with jackknifed values of mean and error
    """
    df = df.reset_index()
    block_size = df.loc[0, 'block_size']
    if bin_size == None:
        bin_size = df.loc[0, 'bin_size']
    else:
        bin_size = bin_size * block_size
    bin_size = (bin_size + block_size -1)//block_size
    data_arr = df[['col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', 'col11', 'col12',
                'col13', 'col14', 'col15', 'col16', 'col17', 'col18', 'S']].to_numpy().T
    S_mean, S_err = stat.jackknife_var_numba_binning(data_arr, trivial, get_bin_borders(data_arr.shape[1], bin_size))
    Jv_mean, Jv_err = stat.jackknife_var_numba_binning(data_arr, Jv, get_bin_borders(data_arr.shape[1], bin_size))
    Jv1_mean, Jv1_err = stat.jackknife_var_numba_binning(data_arr, Jv1, get_bin_borders(data_arr.shape[1], bin_size))
    Jv2_mean, Jv2_err = stat.jackknife_var_numba_binning(data_arr, Jv2, get_bin_borders(data_arr.shape[1], bin_size))
    Blab_mean, Blab_err = stat.jackknife_var_numba_binning(data_arr, Blab, get_bin_borders(data_arr.shape[1], bin_size))
    E_mean, E_err = stat.jackknife_var_numba_binning(data_arr, E, get_bin_borders(data_arr.shape[1], bin_size))
    Elab_mean, Elab_err = stat.jackknife_var_numba_binning(data_arr, Elab, get_bin_borders(data_arr.shape[1], bin_size))
    Bz_mean, Bz_err = stat.jackknife_var_numba_binning(data_arr, Bz, get_bin_borders(data_arr.shape[1], bin_size))
    Bxy_mean, Bxy_err = stat.jackknife_var_numba_binning(data_arr, Bxy, get_bin_borders(data_arr.shape[1], bin_size))
    Ez_mean, Ez_err = stat.jackknife_var_numba_binning(data_arr, Ez, get_bin_borders(data_arr.shape[1], bin_size))
    Exy_mean, Exy_err = stat.jackknife_var_numba_binning(data_arr, Exy, get_bin_borders(data_arr.shape[1], bin_size))
    ElabzT_mean, ElabzT_err = stat.jackknife_var_numba_binning(data_arr, ElabzT, get_bin_borders(data_arr.shape[1], bin_size))
    ElabxyT_mean, ElabxyT_err = stat.jackknife_var_numba_binning(data_arr, ElabxyT, get_bin_borders(data_arr.shape[1], bin_size))
    Ae_mean, Ae_err = stat.jackknife_var_numba_binning(data_arr, Ae, get_bin_borders(data_arr.shape[1], bin_size))
    Am_mean, Am_err = stat.jackknife_var_numba_binning(data_arr, Am, get_bin_borders(data_arr.shape[1], bin_size))
    AlabeT_mean, AlabeT_err = stat.jackknife_var_numba_binning(data_arr, AlabeT, get_bin_borders(data_arr.shape[1], bin_size))
    return pd.DataFrame({'S': [S_mean], 'S_err': [S_err], 'Jv': [Jv_mean], 'Jv_err': [Jv_err], 'Jv1': [Jv1_mean],\
                        'Jv1_err': [Jv1_err], 'Jv2': [Jv2_mean], 'Jv2_err': [Jv2_err], 'Blab': [Blab_mean],\
                        'Blab_err': [Blab_err], 'E': [E_mean], 'E_err': [E_err], 'Elab': [Elab_mean],\
                        'Elab_err': [Elab_err], 'Bz': [Bz_mean], 'Bz_err': [Bz_err], 'Bxy': [Bxy_mean],\
                        'Bxy_err': [Bxy_err], 'Ez': [Ez_mean], 'Ez_err': [Ez_err], 'Exy': [Exy_mean],\
                        'Exy_err': [Exy_err], 'ElabzT': [ElabzT_mean], 'ElabzT_err': [ElabzT_err],\
                        'ElabxyT': [ElabxyT_mean], 'ElabxyT_err': [ElabxyT_err], 'Ae': [Ae_mean],\
                        'Ae_err': [Ae_err], 'Am': [Am_mean], 'Am_err': [Am_err], 'AlabeT': [AlabeT_mean],\
                        'AlabeT_err': [AlabeT_err], 'bin_size': [bin_size * block_size]})

def get_radii_sq(square_size: int) -> List[int]:
    """Make squares of radii of circles cutting lattice points inside a square.

    Args:
        square_size: distance from a center of the square
            to it's border in lattice units

    Returns: sortes list of squares of the radii in descending order
    """
    radii = set()
    for i in range(2):
        y = square_size
        x = square_size - i
        while x ** 2 + y ** 2 > (square_size - 1) ** 2:
            radii.add(x ** 2 + y ** 2)
            x -= 1
            y -= 1
    radii.add(square_size ** 2)
    return sorted(list(radii), key=float, reverse=True)

@njit
def trivial(x: np.ndarray) -> np.ndarray:
    """Function for trivial observable jackknife.

    Args:
        x: 2D numpy array of a data with shape (18, n)

    Returns: 1D numpy array of the data
    """
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = x[17][i]
    return y

@njit
def Jv(x: np.ndarray) -> np.ndarray:
    """Function for Jv observable jackknife.

    Args:
        x: 2D numpy array of a data with shape (18, n)

    Returns: 1D numpy array of the data
    """
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = -x[15][i] - 2 * (x[16][i] + x[12][i] + x[13][i] + x[14][i] - x[3][i] - x[4][i] - x[5][i])
    return y

@njit
def Jv1(x: np.ndarray) -> np.ndarray:
    """Function for Jv1 observable jackknife.

    Args:
        x: 2D numpy array of a data with shape (18, n)

    Returns: 1D numpy array of the data
    """
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = -x[15][i]
    return y

@njit
def Jv2(x: np.ndarray) -> np.ndarray:
    """Function for Jv2 observable jackknife.

    Args:
        x: 2D numpy array of a data with shape (18, n)

    Returns: 1D numpy array of the data
    """
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = - 2 * (x[16][i] + x[12][i] + x[13][i] + x[14][i] - x[3][i] - x[4][i] - x[5][i])
    return y

@njit
def Blab(x: np.ndarray) -> np.ndarray:
    """Function for Blab observable jackknife.

    Args:
        x: 2D numpy array of a data with shape (18, n)

    Returns: 1D numpy array of the data
    """
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = x[3][i] + x[4][i] + x[5][i] + x[9][i] + x[10][i] + x[11][i]
    return y

@njit
def E(x: np.ndarray) -> np.ndarray:
    """Function for E observable jackknife.

    Args:
        x: 2D numpy array of a data with shape (18, n)

    Returns: 1D numpy array of the data
    """
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = x[0][i] + x[1][i] + x[2][i] + x[6][i] + x[7][i] + x[8][i]
    return y

@njit
def Elab(x: np.ndarray) -> np.ndarray:
    """Function for Elab observable jackknife.

    Args:
        x: 2D numpy array of a data with shape (18, n)

    Returns: 1D numpy array of the data
    """
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = x[0][i] + x[1][i] + x[2][i] + x[6][i] + x[7][i] + x[8][i]\
        + x[15][i] + x[16][i] + x[12][i] + x[13][i] + x[14][i] - x[3][i] - x[4][i] - x[5][i]
    return y


@njit
def Bz(x: np.ndarray) -> np.ndarray:
    """Function for Bz observable jackknife.

    Args:
        x: 2D numpy array of a data with shape (18, n)

    Returns: 1D numpy array of the data
    """
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = x[5][i] + x[11][i]
    return y

@njit
def Bxy(x: np.ndarray) -> np.ndarray:
    """Function for Bxy observable jackknife.

    Args:
        x: 2D numpy array of a data with shape (18, n)

    Returns: 1D numpy array of the data
    """
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (x[3][i] + x[4][i] + x[9][i] + x[10][i]) / 2
    return y

@njit
def Ez(x: np.ndarray) -> np.ndarray:
    """Function for Ez observable jackknife.

    Args:
        x: 2D numpy array of a data with shape (18, n)

    Returns: 1D numpy array of the data
    """
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = x[2][i] + x[8][i]
    return y

@njit
def Exy(x: np.ndarray) -> np.ndarray:
    """Function for Bxy observable jackknife.

    Args:
        x: 2D numpy array of a data with shape (18, n)

    Returns: 1D numpy array of the data
    """
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (x[0][i] + x[1][i] + x[6][i] + x[7][i]) / 2
    return y

@njit
def ElabzT(x: np.ndarray) -> np.ndarray:
    """Function for ElabzT observable jackknife.

    Args:
        x: 2D numpy array of a data with shape (18, n)

    Returns: 1D numpy array of the data
    """
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = x[2][i] + x[8][i] + x[12][i] + x[13][i] - x[3][i] - x[4][i] + x[16][i]
    return y

@njit
def ElabxyT(x: np.ndarray) -> np.ndarray:
    """Function for ElabxyT observable jackknife.

    Args:
        x: 2D numpy array of a data with shape (18, n)

    Returns: 1D numpy array of the data
    """
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (x[0][i] + x[1][i] + x[6][i] + x[7][i] + x[14][i] - x[5][i]) / 2
    return y

@njit
def Ae(x: np.ndarray) -> np.ndarray:
    """Function for Ae observable jackknife.

    Args:
        x: 2D numpy array of a data with shape (18, n)

    Returns: 1D numpy array of the data
    """
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (x[0][i] + x[1][i] + x[6][i] + x[7][i]) / 2 - x[2][i] - x[8][i]
    return y

@njit
def Am(x: np.ndarray) -> np.ndarray:
    """Function for Am observable jackknife.

    Args:
        x: 2D numpy array of a data with shape (18, n)

    Returns: 1D numpy array of the data
    """
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (x[3][i] + x[4][i] + x[9][i] + x[10][i]) / 2 - x[5][i] - x[11][i]
    return y

@njit
def AlabeT(x: np.ndarray) -> np.ndarray:
    """Function for AlabeT observable jackknife.

    Args:
        x: 2D numpy array of a data with shape (18, n)

    Returns: 1D numpy array of the data
    """
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (x[0][i] + x[1][i] + x[6][i] + x[7][i] + x[14][i] - x[5][i]) / 2 - (x[2][i] + x[8][i] + x[12][i] + x[13][i] - x[3][i] - x[4][i] + x[16][i])
    return y

def main():
    """Read and process eos data for particular parameters (lattice_size, boundary, velocity, beta).
    Result of calculation is saved inside beta directory.

    Command line args:
        --base_path: path to a directory where is lattice_size directory
        --beta: name of beta directory
        --velocity: name of velocity directory
        --lattice_size: name of lattice_size directory
        --boundary: name of boundary directory
        --bin_test: if passed, then dependence on size of a bin is calculated,
            otherwise dependence on a cutting with predetermined bin size is calculated
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path')
    parser.add_argument('--beta')
    parser.add_argument('--velocity')
    parser.add_argument('--lattice_size')
    parser.add_argument('--boundary')
    parser.add_argument('--result_path', default=None)
    parser.add_argument('--spec_additional_path')
    parser.add_argument('--bin_test', action='store_true')
    args = parser.parse_args()
    print('args: ', args)
    if args.result_path is None:
        result_path = f'{args.base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/{args.beta}'
    else:
        result_path = args.result_path
        try:
            os.makedirs(f'{result_path}')
        except:
            pass

    if args.bin_test:
        therm_length = thermalization_length(f'{args.base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/spec_therm.log',
                                             f'{args.spec_additional_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/spec_therm.log', args.beta)
        df = get_data(args.base_path, args, therm_length)
        if not df.empty:
            bin_max = df['conf_end'].max() // df.loc[0, 'block_size'] // 4
            bin_sizes = int_log_range(1, bin_max, 1.05)
            print('bin_sizes', bin_sizes)
            df_result = []
            for bin in bin_sizes:
                df_result.append(make_jackknife(df, bin_size=bin))
            df_result = pd.concat(df_result)
            df_result.to_csv(f'{result_path}/S_binning.csv', sep=' ', index=False)
        else:
            raise Exception('No data found')
    else:
        therm_length = thermalization_length(f'{args.base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/spec_therm.log',
                                             f'{args.spec_additional_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/spec_therm.log', args.beta)
        bin_size = bin_length(f'{args.base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/spec_bin_S.log',
                              f'{args.spec_additional_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/spec_bin_S.log', args.beta)
        df = get_data(args.base_path, args, therm_length, bin_size)
        if not len(df.index) == 0:
            print('data_size', df['conf_end'].max().compute() - df['conf_start'].min().compute())
            print('bin_size', bin_size)
            if df['conf_end'].max().compute() - df['conf_start'].min().compute() >= 3 * bin_size:
                coord_max = df['x'].max().compute()//2
                df['x'] = df['x'] - coord_max
                df['y'] = df['y'] - coord_max
                df['rad_sq'] = df['x'] ** 2 + df['y'] ** 2
                df_result = []
                for thickness in [1, 2, 3, 4, 5]:
                    rad_inner = 1
                    while rad_inner < coord_max * np.sqrt(2):
                        df1 = df.loc[(df['rad_sq'] >= rad_inner ** 2) & (df['rad_sq'] < (rad_inner + thickness) ** 2)]
                        rad_aver = np.sqrt(df['rad_sq']).mean()
                        df1 = df1.groupby(['conf_start', 'conf_end', 'block_size', 'bin_size'], observed=False)\
                            .agg(S=('S', 'mean'), col2=('2', 'mean'), col3=('3', 'mean'),\
                            col4=('4', 'mean'), col5=('5', 'mean'), col6=('6', 'mean'),\
                            col7=('7', 'mean'), col8=('8', 'mean'), col9=('9', 'mean'),\
                            col10=('10', 'mean'), col11=('11', 'mean'), col12=('12', 'mean'),\
                            col13=('13', 'mean'), col14=('14', 'mean'), col15=('15', 'mean'),\
                            col16=('16', 'mean'), col17=('17', 'mean'), col18=('18', 'mean'))\
                            .reset_index(level=['conf_start', 'conf_end', 'block_size', 'bin_size'])
                        df_result.append(make_jackknife(df1))
                        df_result[-1]['rad_aver'] = rad_aver
                        df_result[-1]['thickness'] = thickness
                        rad_inner = rad_inner + thickness
                df_result = pd.concat(df_result)
                df_result.to_csv(f'{result_path}/observables_ring_result.csv', sep=' ', index=False)
        else:
            #raise Exception('No data found')
            print('No data found')

if __name__ == "__main__":
    main()
