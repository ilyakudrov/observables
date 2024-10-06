import sys
import math
import time
import numpy as np
import os.path
import pandas as pd
import itertools
import argparse

sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "..", "..", "python_jupyter", "common"))
import scaler

def get_densities(df_wrapped, df_unwrapped, percolating_threshold, r0, lattice_volume):
    df_wrapped['density'] = df_wrapped['length'] / lattice_volume / r0 ** 3
    df_unwrapped['density'] = df_unwrapped['length'] * df_unwrapped['number'] / lattice_volume / r0 ** 3
    if df_wrapped.empty:
        df = df_unwrapped[['color', 'density']]
    else:
        df = pd.concat([df_wrapped[['color', 'density']], df_unwrapped[['color', 'density']]])
    df = df.groupby('color')['density'].agg([('density', 'sum')]).reset_index(level='color')
    density_mean = df['density'].mean()
    df_unwrapped.loc[df_unwrapped['length'] < percolating_threshold, 'density'] = 0
    df_wrapped = df_wrapped[df_wrapped['percolating_group'] == 'percolating']
    if df_wrapped.empty:
        df = df_unwrapped[['color', 'density']]
    else:
        df = pd.concat([df_wrapped[['color', 'density']], df_unwrapped[['color', 'density']]])
    df = df.groupby('color')['density'].agg([('density', 'sum')]).reset_index(level='color')
    density_percolating_mean = df['density'].mean()
    return density_mean, density_percolating_mean


parser = argparse.ArgumentParser()
parser.add_argument('--size')
parser.add_argument('--additional_parameters')
parser.add_argument('--beta')
parser.add_argument('--copies', type=int)
parser.add_argument('--percolating_threshold', type=int)
args = parser.parse_args()
print('args: ', args)

conf_type = "gluodynamics"
theory_type = 'su3'

conf_max = 6000
mu1 = ['/']
#chains = ["/"]
# mu1 = ['mu0.05',
#        'mu0.20', 'mu0.25',
#        'mu0.30', 'mu0.35', 'mu0.45']
chains = ['/', 's0', 's1', 's2', 's3',
           's4', 's5', 's6', 's7', 's8', 's9', 's10']

#base_path = "../../../data"
base_path = "/home/clusters/rrcmpi/kudrov/observables_cluster/result"

start = time.time()
print('beta: ', args.beta,' conf_size: ', args.size, ' additional_parameters: ', args.additional_parameters)
path = f'{base_path}/monopoles_su3/{conf_type}/{args.size}/{args.beta}/{args.additional_parameters}'
r0 = scaler.get_r0(float(args.beta[4:]))
lattice_volume = 4 * eval(args.size.replace('x', '*').replace('^', '**'))
if args.copies > 0:
    copy_range = range(args.copies)
else:
    copy_range = range(1)
data = []
for chain in chains:
    for conf in range(1, conf_max + 1):
        print(conf)
        densities = []
        densities_percolating = []
        copies = []
        for copy in copy_range:
            path_wrapped = f'{path}/{chain}/clusters_wrapped/clusters_wrapped_{conf:04}_{copy}'
            path_unwrapped = f'{path}/{chain}/clusters_unwrapped/clusters_unwrapped_{conf:04}_{copy}'
            if (os.path.isfile(path_wrapped)) and (os.path.isfile(path_unwrapped)):
                df_wrapped = pd.read_csv(path_wrapped)
                df_unwrapped = pd.read_csv(path_unwrapped)
                density_mean, density_percolating_mean = get_densities(df_wrapped, df_unwrapped, args.percolating_threshold, r0, lattice_volume)
                densities.append(density_mean)
                densities_percolating.append(density_percolating_mean)
                copies.append(copy)
        # df_result = pd.DataFrame({'density': densities, 'densitiy_percolating': densities_percolating, 'copy': copies})
        df = pd.DataFrame({'density': densities, 'densitiy_percolating': densities_percolating, 'copy': copies, 'conf': chain + f'-{conf}'})
        df = df.astype({'copy': 'int32'})
        data.append(df)
data = pd.concat(data)
path_output = f"../../../result/monopoles_su3/{conf_type}/{args.size}/{args.beta}/{args.additional_parameters}"
try:
    os.makedirs(f'{path_output}')
except:
    pass
data.to_csv(
    f"{path_output}/density", index=False)

end = time.time()
print("execution time = %s" % (end - start))
