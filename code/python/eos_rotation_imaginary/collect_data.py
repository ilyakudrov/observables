import argparse
import os
import pandas as pd

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

parser = argparse.ArgumentParser()
parser.add_argument('--base_path')
parser.add_argument('--lattice_size')
parser.add_argument('--boundary')
parser.add_argument('--file_name')
args = parser.parse_args()
print('args: ', args)

df = pd.DataFrame()
velocity_dirs = get_dir_names(f'{args.base_path}/{args.lattice_size}/{args.boundary}')
for velocity in velocity_dirs:
    beta_dirs = get_dir_names(f'{args.base_path}/{args.lattice_size}/{args.boundary}/{velocity}')
    for beta in beta_dirs:
        print('velocity: ', velocity, ', beta: ', beta) 
        file_path = f'{args.base_path}/{args.lattice_size}/{args.boundary}/{velocity}/{beta}/{args.file_name}'
        if (os.path.isfile(file_path)):
            df_tmp = pd.read_csv(file_path, sep=' ')
            df_tmp['velocity'] = float(velocity[:-1])
            df_tmp['beta'] = float(beta)
            df = pd.concat([df, df_tmp])

if not df.empty:
    path_output = f'../../../result/eos_rotation_imaginary/{args.lattice_size}/{args.boundary}'
    try:
        os.makedirs(f'{path_output}')
    except:
        pass
    df.to_csv(f'{path_output}/{args.file_name}', index=False)
