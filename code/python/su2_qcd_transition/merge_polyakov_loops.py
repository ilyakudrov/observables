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

def parse_lattice_sizes(lattice_name):
    if lattice_name.find('x') != -1:
        b = lattice_name.split('x')
        space_size = int(b[0][:b[0].find('^')])
        time_size = int(b[1])
    else:
        space_size = int(lattice_name[:lattice_name.find('^')])
        time_size = space_size
    return time_size, space_size

parser = argparse.ArgumentParser()
parser.add_argument('--data_path')
args = parser.parse_args()
print('args: ', args)

chains = ['/', 's0', 's1', 's2', 's3',
           's4', 's5', 's6', 's7', 's8', 's9', 's10']
conf_max = 5000

lattice_dirs = get_dir_names(args.data_path)
df = []
for lattice in lattice_dirs:
    Nt, Ns = parse_lattice_sizes(lattice)
    mu_dirs = get_dir_names(f'{args.data_path}/{lattice}')
    for mu_name in mu_dirs:
        mu = float(mu_name[2:])
        for chain in chains:
            if chain == '/':
                chain_name = 's0'
            else:
                chain_name = chain
            for i in range(0, conf_max + 1):
                data_file = f'{args.data_path}/{lattice}/{mu_name}/original/HYP10_alpha=1_1_0.5/{chain}/polyakov_loop_{i:04}'
                if (os.path.isfile(data_file)):
                    df_tmp = pd.read_csv(data_file)
                    df_tmp['Nt'] = Nt
                    df_tmp['Ns'] = Ns
                    df_tmp['mu'] = mu
                    df_tmp['chain'] = chain_name
                    df_tmp['conf'] = i
                    df_tmp = df_tmp.rename(columns={'polyakov_loop': 'PL', 'HYP_step': 'HYP'})
                    df.append(df_tmp)
df = pd.concat(df, ignore_index=True)
print(df)
path_output = '/home/ilya/soft/lattice/observables/result/merged/polyakov_loop'
try:
    os.makedirs(f'{path_output}')
except:
    pass
df.to_csv(f"{path_output}/polyakov_loops.csv", index=False)