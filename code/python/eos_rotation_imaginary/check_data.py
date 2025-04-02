import os
import pandas as pd
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

def get_file_names(path: str, obs_name) -> List[str]:
    """Get names of files with blocked eos data in directory path.

    Args:
        path: path of eos data

    Returns: list of file names
    """
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        files.extend(filenames)
        break
    return list(f for f in files if f[f.find('-') + 1:].startswith(obs_name) and f.startswith(f'block'))

def get_unblocked_files(path: str, obs_name) -> List[str]:
    """Get names of initial eos files directory path.

    Args:
        path: path of eos data

    Returns: list of file names
    """
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        files.extend(filenames)
        break
    return list(f for f in files if f.startswith(obs_name))

def get_block_size(f: str) -> int:
    """Extract block size of eos data from it's name.

    Args:
        f: name of the file

    Returns: integer value of the block size
    """
    return int(f[5:f.find('-')])

def find_lattices(path, lattice_names):
    directories = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for dir_name in dirnames:
            if dir_name in lattice_names:
                directories.append(f'{dirpath}/{dir_name}')
    return directories

def check_spec(path):
    spec_info = {}
    for spec_name in ['spec_bin_Pl.log', 'spec_bin_S.log', 'spec_therm.log']:
        tmp = ''
        if os.path.isfile(f'{path}/{spec_name}'):
            tmp = tmp + 'exists'
            if os.stat(f'{path}/{spec_name}').st_size != 0:
                tmp = tmp + ',not empty'
                try:
                    pd.read_csv(f'{path}/{spec_name}')
                    tmp = tmp + ',readable'
                except:
                    tmp = tmp + ',unreadable'
            else:
                tmp = tmp + ',empty'
        else:
            tmp = tmp + 'doesnt exist'
        spec_info[spec_name] = tmp
    return spec_info

def get_data_info(beta_path, obs_name):
    chain_dirs = get_dir_names(beta_path)
    file_number = 0
    observation_number = 0
    unblocked_data = 0
    for chain in chain_dirs:
#        print('chain', chain)
        filenames = get_file_names(f'{beta_path}/{chain}', obs_name)
        for file in filenames:
#            print('file', file)
            if os.stat(f'{beta_path}/{chain}/{file}').st_size != 0:
                try:
                    #pd.read_csv(f'{beta_path}/{chain}/{file}')
                    file_number += 1
                    observation_number += get_block_size(file)
                except:
                    pass
        unblocked_files = get_unblocked_files(f'{beta_path}/{chain}', obs_name)
        for file in unblocked_files:
            if os.stat(f'{beta_path}/{chain}/{file}').st_size != 0:
                unblocked_data += 1
    return file_number, observation_number, unblocked_data

parser = argparse.ArgumentParser()
parser.add_argument('--data_paths', nargs='+')
parser.add_argument('--result_path')
parser.add_argument('--obs_name')
args = parser.parse_args()
print('args: ', args)
lattice_names = ['24x24x97sq', '30x30x121sq', '30x30x181sq', '30x30x81sq', '36x36x145sq', '42x42x169sq',
                '4x24x97sq', '5x30x121sq', '5x30x181sq', '5x30x81sq', '6x36x145sq', '7x42x169sq']
df = pd.DataFrame()
for base_path in args.data_paths:
#    print('base_path', base_path)
    lattice_directories = find_lattices(base_path, lattice_names)
    for lattice_dir in lattice_directories:
#        print('lattice_dir', lattice_dir)
        lattice_size = lattice_dir[lattice_dir.rfind('/') + 1:]
        boundary_dirs = get_dir_names(lattice_dir)
        for boundary in boundary_dirs:
#            print('boundary', boundary)
            velocity_dirs = get_dir_names(f'{lattice_dir}/{boundary}')
            for velocity in velocity_dirs:
#                print('velocity', velocity)
#                spec_info = check_spec(f'{lattice_dir}/{boundary}/{velocity}')
#                print('spec_info', spec_info)
                beta_dirs = get_dir_names(f'{lattice_dir}/{boundary}/{velocity}')
                for beta in beta_dirs:
#                    print('beta', beta)
                    file_number, observation_number, unblocked_data = get_data_info(f'{lattice_dir}/{boundary}/{velocity}/{beta}', args.obs_name)
                    info_data = {'lattice_dir': [lattice_dir], 'lattice_size': [lattice_size], 'boundary': [boundary],
                                'velocity': [velocity], 'beta': [beta], 'file_number': [file_number],
                                'observation_number': [observation_number], 'unblocked_data': [unblocked_data]}
#                    for key, value in spec_info.items():
#                        info_data[key] = [value]
#                    print('info_data', info_data)
                    df = pd.concat([df, pd.DataFrame(info_data)])
if args.obs_name == 'PLxy':
    df.to_csv(f'{args.result_path}/data_polyakov_summary.csv', index=False)
elif args.obs_name == 'SEBxy':
    df.to_csv(f'{args.result_path}/data_summary.csv', index=False)