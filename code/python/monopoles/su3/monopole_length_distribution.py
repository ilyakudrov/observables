import sys
import time
import math
import numpy as np
import os.path
import pandas as pd
import argparse

sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "..", "..", "..", "python_jupyter", "common"))
import scaler

def functional_bin(df, bins='auto'):
    df_binned = pd.DataFrame()
    if type(bins) == float:
        bins = math.ceil((df['functional'].max() - df['functional'].min()) / bins)
    bins = np.histogram_bin_edges(df['functional'], bins=bins)
    df['bin'] = 0
    for i in range(0, len(bins) - 2):
        # df_tmp = df[(df['functional'] >= bins[i]) & (df['functional'] < bins[i + 1])]
        # df_tmp.loc[:, 'bin'] = int(i)
        # df_binned = pd.concat([df_binned, df_tmp])
        df.loc[(df['functional'] >= bins[i]) & (df['functional'] < bins[i + 1]), 'bin'] = int(i)
    # df_tmp = df[(df['functional'] >= bins[len(bins) - 2]) & (df['functional'] <= bins[len(bins) - 1])]
    # df_tmp.loc[:, 'bin'] = int(len(bins) - 2)
    df.loc[(df['functional'] >= bins[len(bins) - 2]) & (df['functional'] <= bins[len(bins) - 1]), 'bin'] = int(len(bins) - 2)
    # df_binned = pd.concat([df_binned, df_tmp])
    # return df_binned.reset_index(drop=True)
    return df.reset_index(drop=True)

parser = argparse.ArgumentParser()
parser.add_argument('--size')
parser.add_argument('--additional_parameters')
parser.add_argument('--beta')
parser.add_argument('--functional_padding')
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

# base_path = "../../../../data"
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
data_functional = []
for chain in chains:
    for conf in range(1, conf_max + 1):
        path_functional = f'{base_path}/mag/functional/su3/{conf_type}/{args.size}/{args.beta}/{args.additional_parameters}/{chain}/functional_{conf:0{args.functional_padding}}'
        if os.path.isfile(path_functional):
            data_functional.append(pd.read_csv(path_functional))
            data_functional[-1]['conf'] = chain + '-' + str(conf)
            for copy in copy_range:
                path_unwrapped = f'{path}/{chain}/clusters_unwrapped/clusters_unwrapped_{conf:04}_{copy}'
                # print(path_unwrapped)
                if os.path.isfile(path_unwrapped):
                    data.append(pd.read_csv(path_unwrapped))
                    data[-1]['conf'] = chain + '-' + str(conf)
                    data[-1]['copy'] = copy
data = pd.concat(data)
data = data[data['length'] <= args.percolating_threshold]
data['density'] = data['length'] * data['number'] / lattice_volume / r0 ** 3
data = data.drop(['number'], axis=1)
data = data.groupby(['copy', 'conf', 'length']).agg(density=('density', 'mean')).reset_index(level=['copy', 'conf', 'length'])
data_functional = pd.concat(data_functional)
data_functional['functional'] = (1 - data_functional['functional']) * 3/2
data = data.merge(data_functional, how='inner', on=['copy', 'conf'])
data = data.set_index(['length', 'copy']).groupby(['length', 'copy']).apply(functional_bin, include_groups=False).reset_index(level=['length', 'copy'])
data = data.groupby(['bin', 'length']).agg(density=('density', 'mean'), std_density=('density', 'sem'), functional=('functional', 'mean')).reset_index(level=['length', 'bin'])
data = data.drop('bin', axis=1)
path_output = f"../../../../result/monopoles_su3/{conf_type}/{args.size}/{args.beta}/{args.additional_parameters}"
try:
    os.makedirs(f'{path_output}')
except:
    pass
data.to_csv(
    f"{path_output}/length_distribution", index=False)

end = time.time()
print("execution time = %s" % (end - start))
