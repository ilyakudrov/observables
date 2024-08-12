import os
import pandas as pd
from typing import List

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

df = pd.DataFrame()
base_path = "/home/clusters/rrcmpi/kudrov/observables/result/eos_rotation_imaginary"
lattice_dirs = get_dir_names(base_path)
for lattice_size in lattice_dirs:
    boundary_dirs = get_dir_names(f'{base_path}/{lattice_size}')
    for boundary in boundary_dirs:
        velocity_dirs = get_dir_names(f'{base_path}/{lattice_size}/{boundary}')
        for velocity in velocity_dirs:
            beta_dirs = get_dir_names(f'{base_path}/{lattice_size}/{boundary}/{velocity}')
            for beta in beta_dirs:
                log_path = f'{base_path}/{lattice_size}/{boundary}/{velocity}/{beta}/S_result.csv'
                if os.path.isfile(log_path) and os.stat(log_path).st_size != 0:
                    df = pd.concat([df, pd.DataFrame({'lattice_size': [lattice_size], 'boundary': [boundary],
                                                      'velocity': [velocity], 'beta': [beta]})])
df.to_csv(f'{base_path}/result_summary.csv', index=False)
