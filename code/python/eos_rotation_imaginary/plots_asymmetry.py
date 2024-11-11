import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import os
import sys

import common.plots as plots
import common.scale_setters as scale_setters

def get_dir_names(path):
    directories = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        directories.extend(dirnames)
        break
    return directories

def concat_observables(df, observables):
    data = []
    for obs in observables:
        data.append(df.loc[:, ['box_size', 'radius', obs, f'{obs}_err']])
        data[-1] = data[-1].rename({obs: 'y', f'{obs}_err': 'err'}, axis=1)
        data[-1]['observable'] = obs
    return pd.concat(data)

def read_data(path, observables):
    df = pd.DataFrame()
    lattice_dirs = get_dir_names(path)
    for lattice in lattice_dirs:
        border_dirs = get_dir_names(f'{path}/{lattice}')
        for border in border_dirs:
            velocity_dirs = get_dir_names(f'{path}/{lattice}/{border}')
            for velocity in velocity_dirs:
                beta_dirs = get_dir_names(f'{path}/{lattice}/{border}/{velocity}')
                for beta in beta_dirs:
                    file_path = f'{path}/{lattice}/{border}/{velocity}/{beta}/observables_result.csv'
                    if os.path.isfile(file_path):
                        df_tmp = pd.read_csv(file_path, delimiter=' ')
                        df_tmp = concat_observables(df_tmp, observables)
                        df_tmp['velocity'] = float(velocity[:-1])
                        df_tmp['beta'] = float(beta)
                        df_tmp['border'] = border
                        df_tmp['lattice'] = lattice[:-2]
                        df = pd.concat([df, df_tmp])
    return df

def find_closest(val, arr, threshold):
    closest = None
    index = None
    diff_old = abs(arr.max() - arr.min())
    for i in range(len(arr)):
        diff_new = abs(val - arr[i])
        if diff_new < diff_old and diff_new / abs(val) < threshold:
            diff_old = diff_new
            closest = arr[i]
            index = i
    return closest, index

def group_T(T_arr, threshold):
    a = {}
    for i in range(len(T_arr)):
        while len(T_arr[i]) > 0:
            T_group = []
            T = T_arr[i][0]
            T_group.append(T)
            T_arr[i] = np.delete(T_arr[i], 0)
            for j in range(i + 1, len(T_arr)):
                closest, index = find_closest(T, T_arr[j], threshold)
                if closest is not None:
                    T = closest
                    T_group.append(closest)
                    T_arr[j] = np.delete(T_arr[j], index)
            T_mean = sum(T_group) / len(T_group)
            for t in T_group:
                a[t] = T_mean
    return a

def plot_asymmetry(df):
    T = df.name[1]
    border = df.name[0]
    plots.make_plot(df, 'velocity', 'y', 'observable', r'$\Omega$ (GeV)', r'A ($GeV^{4}$)', 'asymmetry' + r', $T/T_{c}$ = ' + f'{T:.2f}, ' + border, '../../images/eos_rotation_imaginary/asymmetry', f'asymmetry_{border}_T={T:.2f}', True, err='err', dashed_line_y=[0], save_figure=True)

def group_df(df):
    # print(df)
    T_arr = []
    T_arr.append(df[df['nt'] == 7]['T'].to_numpy())
    T_arr.append(df[df['nt'] == 6]['T'].to_numpy())
    T_arr.append(df[df['nt'] == 5]['T'].to_numpy())
    T_arr.append(df[df['nt'] == 4]['T'].to_numpy())
    a = group_T(T_arr, 0.03)
    df['T'] = df['T'].apply(lambda x: a[x])
    return df

beta_critical = {'nt4o': 4.088,
                 'nt5o': 4.225,
                 'nt6o': 4.350,
                 'nt7o': 4.470,
                 'nt4p': 4.073,
                 'nt5p': 4.202,
                 'nt6p': 4.318,
                 'nt7p': 4.417,
                 }

path = '../../result/eos_rotation_imaginary'
fm_to_GeV = 1/0.197327 # 1 fm = 1/0.197327 GeV ** -1
df = read_data(path, ['Ae', 'Am'])
df['nt'] = df['lattice'].apply(lambda x: int(x[:x.find('x')]))
df['ns'] = df['lattice'].apply(lambda x: int(x[x.rfind('x') + 1:]))
df = df[(df['box_size'] == df['ns'] // 2 - df['nt']) & (df['radius'] == (df['ns'] // 2 - df['nt']) * np.sqrt(2))]
scale_setter = scale_setters.ExtendedSymanzikScaleSetter()
df['a_GeV'] = scale_setter.get_spacing_in_fm(df['beta']) * fm_to_GeV
df['a_fm'] = scale_setter.get_spacing_in_fm(df['beta'])
df['border_brief'] = df['border'].apply(lambda x: x[0].lower())
df['beta_crit'] = df.apply(lambda x: beta_critical[f'nt{x['nt']}{x['border_brief']}'], axis=1)
df['a_crit'] = scale_setter.get_spacing_in_fm(df['beta_crit']) * fm_to_GeV
df['T'] = df['a_crit'] / df['a_GeV']
df['y'] = df['y'] * df['beta'] / 6 / df['a_GeV'] ** 4
df['err'] = df['err'] * df['beta'] / 6 / df['a_GeV'] ** 4
df['velocity'] = df['velocity'] / df['a_GeV']
df['y'] = -df['y']
# print(df)
# df = df.groupby(['border', 'velocity', 'observable']).apply(group_df, include_groups=False)
df['a_str'] = df['a_fm'].apply(lambda x: "%.2f" % x)
df['observable'] = df['observable'] + ', a = ' + df['a_str'] + ' fm'
print(df)
df.set_index(['border', 'T']).groupby(['border', 'T']).apply(plot_asymmetry, include_groups=False)