import os
import pandas as pd
import numpy as np
from typing import List
import argparse

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


def find_lattices(path, lattice_names):
    directories = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for dir_name in dirnames:
            if dir_name in lattice_names:
                directories.append(f'{dirpath}/{dir_name}')
    return directories

def get_lengths(lattice_name):
    if lattice_name == '5x30x121sq':
        return 1500, 150
    elif lattice_name == '30x30x121sq':
        return 200, 20
    elif lattice_name == '6x36x145sq':
        return 1500, 150
    elif lattice_name == '36x36x145sq':
        return 300, 20
    elif lattice_name == '7x42x169sq':
        return 2000, 200
    elif lattice_name == '42x42x169sq':
        return 350, 20

def create_df(df, path, name):
    try:
        os.makedirs(path)
    except:
        pass
    df.to_csv(f'{path}/{name}', index=False, header=False, sep=' ')

def make_spec(data_path, lattice_size, boundary, velocity):
    therm_length, bin_length = get_lengths(lattice_size)
    try: 
        df = pd.read_csv(f'{data_path}/spec_bin_Pl.log')
    #if not os.path.isfile(f'{data_path}/spec_bin_Pl.log'):
    except:
        df = pd.DataFrame({'beta': [0], 'length': [bin_length]})
        create_df(df, f'../../../data/eos_rotation_imaginary/{lattice_size}/{boundary}/{velocity}', 'spec_bin_Pl.log')
    try:
        df = pd.read_csv(f'{data_path}/spec_bin_S.log')
    #if not os.path.isfile(f'{data_path}/spec_bin_S.log'):
    except:
        df = pd.DataFrame({'beta': [0], 'length': [bin_length]})
        create_df(df, f'../../../data/eos_rotation_imaginary/{lattice_size}/{boundary}/{velocity}', 'spec_bin_S.log')
    try:
        df = pd.read_csv(f'{data_path}/spec_therm.log')
    except:
    #if not os.path.isfile(f'{data_path}/spec_therm.log'):
        df = pd.DataFrame({'beta': [0], 'length': [therm_length]})
        create_df(df, f'../../../data/eos_rotation_imaginary/{lattice_size}/{boundary}/{velocity}', 'spec_therm.log')

parser = argparse.ArgumentParser()
parser.add_argument('--data_paths', nargs='+')
args = parser.parse_args()
print('args: ', args)
lattice_names = ['30x30x121sq', '36x36x145sq', '42x42x169sq',
                '5x30x121sq', '6x36x145sq', '7x42x169sq']
df = pd.DataFrame()
for base_path in args.data_paths:
    lattice_directories = find_lattices(base_path, lattice_names)
    for lattice_dir in lattice_directories:
        lattice_size = lattice_dir[lattice_dir.rfind('/') + 1:]
        boundary_dirs = get_dir_names(lattice_dir)
        for boundary in boundary_dirs:
            velocity_dirs = get_dir_names(f'{lattice_dir}/{boundary}')
            for velocity in velocity_dirs:
                make_spec(f'{lattice_dir}/{boundary}/{velocity}', lattice_size, boundary, velocity)
print('done')
