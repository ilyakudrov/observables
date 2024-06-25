from numba import njit
import sys
import numpy as np
import os.path
import pandas as pd
import argparse
from scipy.stats import norm, binned_statistic
import itertools
import math
from typing import Sequence, Iterable, Tuple, Optional

sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "..", ".."))
import statistics_python.src.statistics_observables as stat

def int_log_range(_min: int, _max: int, factor: float) -> list[int]:
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

def read_blocks(path: str) -> pd.DataFrame:
    """Read csv file of eos_rotation_imaginary data.

    Agrs: path to the csv file

    Returns: pandas DataFrame of the data
    """
    return pd.read_csv(path, header=None,delimiter=' ', names=['x', 'y', 'q', 'r', 'n', 'u',
                                                 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                                                 'h', 'i', 'j', 'k', 'l', 'm', 'S', 'o', 'p'])

def get_block_size(f: str) -> int:
    """Extract block size of eos data from it's name.

    Args:
        f: name of the file

    Returns: integer value of the block size
    """
    return int(f[5:f.find('-')])

def get_conf_range(f: str) -> int:
    """Extract range of configurations in a block of eos data from it's name.

    Args:
        f: name of the file

    Returns: tuple of two integer configuration numbers
    """
    arr = f[f.find('_') + 1:f.find('.')].split('_')
    return int(arr[0]), int(arr[1])

def get_file_names(path: str) -> list[str]:
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

def get_dir_names(path: str) -> list[str]:
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

def therm_bin(df_therm: pd.DataFrame, df_bins: pd.DataFrame, beta: str) -> Tuple[int, int]:
    """Get thermlization size and bin size for data
    with particular beta from corresponding DataFrames.

    Args:
        df_therm: DataFrame with data for thermlization size
        df_bins: DataFrame with data for bin size
        beta: string of beta directory name

    Returns: tuple of integer values of thermlization size and bin size
    """
    if float(beta) in df_therm['beta'].unique():
        therm_length = df_therm.loc[df_therm['beta'] == float(beta), 'therm_length'].values[0]
    else:
        therm_length = df_therm.loc[df_therm['beta'] == 0.00, 'therm_length'].values[0]
    if float(beta) in df_bins['beta'].unique():
        bin_size = df_bins.loc[df_bins['beta'] == float(beta), 'bin_size'].values[0]
    else:
        bin_size = df_bins.loc[df_bins['beta'] == 0.00, 'bin_size'].values[0]
    return therm_length, bin_size

def get_data(base_path: str, args: argparse.Namespace, therm_length: int, bin_size: int) -> pd.DataFrame:
    """Read and concatenate eos data for particlular parameters
    (lattice_size, boundary, velocity, beta).

    Args:
        base_path: path to a directory with lattice_size directory
        args: command line arguments with data parameters
        therm_length: thermalization length to cut from data
        bin_size: bin size to add to a resulting DataFrame

    Returns: DataFrame with the data to process
    """
    df = pd.DataFrame()
    print('therm_length: ', therm_length)
    chain_dirs = get_dir_names(f'{base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/{args.beta}')
    chain_dirs.sort()
    print('chain_dirs: ', chain_dirs)
    for chain in chain_dirs:
        filenames = get_file_names(f'{base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/{args.beta}/{chain}')
        filenames.sort()
        for f in filenames:
            conf_start, conf_end = get_conf_range(f)
            if conf_start > therm_length:
                data = read_blocks(f'{base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/{args.beta}/{chain}/{f}')
                data = data[['x', 'y', 'S']]
                data['block_size'] = get_block_size(f)
                data['conf_start'] = conf_start
                data['conf_end'] = conf_end
                data['bin_size'] = bin_size
                data = data.astype({'x': 'int32', 'y': 'int32', 'block_size': 'int32',
                                    'conf_start': 'int32', 'conf_end': 'int32', 'bin_size': 'int32','S': 'float64'})
                df = pd.concat([df, data])
    return df

def make_jackknife(df: pd.DataFrame, bin_size: Optional[int] = None) -> pd.DataFrame:
    """Make blocked jackknife for data from S column from df.

    Args:
        df: DataFrame with S column to do jackknife
        bin_size: size of bin in kackknife. If None, derive it from df.

    Returns: DataFrame with jackknifed value of mean and error
    """
    df = df.reset_index()
    block_size = df.loc[0, 'block_size']
    if bin_size == None:
        bin_size = df.loc[0, 'bin_size']
    else:
        bin_size = bin_size * block_size
    bin_size = (bin_size + block_size -1)//block_size
    S_arr = np.array([df['S'].to_numpy()])
    mean, err = stat.jackknife_var_numba_binning(S_arr, trivial, get_bin_borders(S_arr.shape[1], bin_size))
    return pd.DataFrame({'S': [mean], 'err': [err], 'bin_size': [bin_size * block_size]})

def get_radii_sq(square_size: int) -> list[int]:
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
    return sorted(list(radii), key=float, reverse=True)[1:]

@njit
def trivial(x: np.ndarray) -> np.ndarray:
    """Function for trivial observable jackknife.

    Args:
        x: 2D numpy array of a data with shape (1, n)

    Returns: 1D numpy array of the data
    """
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = x[0][i]
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
    parser.add_argument('--bin_test', action='store_true')
    args = parser.parse_args()
    print(type(args))
    print('args: ', args)

    df_therm = pd.read_csv(f'{args.base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/spec_therm.log',
                           header=None, delimiter=' ', names=['beta', 'therm_length'])
    df_bins = pd.read_csv(f'{args.base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/spec_therm.log',
                           header=None, delimiter=' ', names=['beta', 'bin_size'])
    therm_length, bin_size = therm_bin(df_therm, df_bins, args.beta)
    df = get_data(args.base_path, args, therm_length, bin_size)
    print(df)
    if args.bin_test:
        bin_max = df['conf_end'].max() // df.loc[0, 'block_size'] // 4
        bin_sizes = int_log_range(1, bin_max, 1.05)
        print(bin_sizes)
        df_result = []
        for bin in bin_sizes:
            df_result.append(make_jackknife(df, bin_size=bin))
        df_result = pd.concat(df_result)
        df_result.to_csv(f'{args.base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/{args.beta}/S_binning.csv', sep=' ', index=False)
    else:
        coord_max = df['x'].max()//2
        df['x'] = df['x'] - coord_max
        df['y'] = df['y'] - coord_max
        df['rad_sq'] = df['x'] ** 2 + df['y'] ** 2
        Nt = int(args.lattice_size[0])
        df_result = []
        print(coord_max)
        for cut in range(0, coord_max - 10):
            print(cut)
            df1 = df.loc[(df['x'] <= coord_max - cut) & (df['x'] >= cut - coord_max) & (df['y'] <= coord_max - cut) & (df['y'] >= cut - coord_max)]
            for radius_sq in get_radii_sq(df1['x'].max()):
                df1 = df1.loc[df1['rad_sq'] <= radius_sq]
                df_result.append(make_jackknife(df1))
                df_result[-1]['box_size'] = coord_max - cut
                df_result[-1]['radius'] = math.sqrt(radius_sq)
        df_result = pd.concat(df_result)
        df_result.to_csv(f'{args.base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/{args.beta}/S_result.csv', sep=' ', index=False)

    print(df_result)

if __name__ == "__main__":
    main()
