import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn
import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(''))))
import python_jupyter.common.plots as plots

def get_dir_names(path):
    directories = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        directories.extend(dirnames)
        break
    return directories

def read_data(path, observable):
    df = pd.DataFrame()
    velocity_dirs = get_dir_names(path)
    for velocity in velocity_dirs:
        beta_dirs = get_dir_names(f'{path}/{velocity}')
        for beta in beta_dirs:
            df_tmp = pd.read_csv(f'{path}/{velocity}/{beta}/observables_distribution_aver.csv', delimiter=' ')
            df_tmp = df_tmp[['x', 'y', observable, f'{observable}_err']]
            df_tmp['velocity'] = velocity
            df_tmp['beta'] = float(beta)
            df = pd.concat([df, df_tmp])
    return df

def gaussian_smearing(df, observable):
    data = []
    for x in df['x'].unique():
        data.append(df[df['x'] == x][observable].to_numpy())
    data = np.array(data)
    data = scipy.ndimage.gaussian_filter(data, sigma=1)
    df = pd.DataFrame()
    for x in range(len(data)):
        df = pd.concat([df, pd.DataFrame({'x': [x - len(data) // 2] * len(data), 'y': list(range(-(len(data) // 2), len(data) // 2 + 1)), observable: data[x]})])
    return df

def heatmap_errorbar(df, subplot_ratio, observable, image_path, image_name):
    betas = np.sort(df['beta'].unique())
    velocities = np.sort(df['velocity'].unique())
    vmax = max(abs(df[observable].max()), abs(df[observable].min()))
    vmin = -vmax
    gs_kw = dict(height_ratios=list([subplot_ratio if i % 2 == 0 else 1 for i in range(len(velocities) * 2)]))
    fig, ax = plt.subplots(nrows=len(velocities) * 2, ncols=len(betas), gridspec_kw=gs_kw)
    fig.set_size_inches(50, 18)
    fig.suptitle(observable, fontsize=16)
    df['x'] = df['x'] - df['x'].max()//2
    df['y'] = df['y'] - df['y'].max()//2
    df['rad'] = np.sqrt(df['x'] ** 2 + df['y'] ** 2)
    for i in range(len(velocities)):
        for j in range(len(betas)):
            print(betas[j])
            print(velocities[i])
            df_tmp = df[(df['beta'] == betas[j]) & (df['velocity'] == velocities[i])]
            # df1 = df_tmp.loc[(df_tmp['y'] == 0) | (df_tmp['y'] == df_tmp['x'])]
            df1 = df_tmp.loc[(df_tmp['y'] == 0)]
            df_tmp = df_tmp.loc[:, ['x', 'y', observable]]
            df_tmp = gaussian_smearing(df_tmp, observable)
            df_tmp = df_tmp.pivot(columns='x', index='y', values=observable)
            seaborn.heatmap(df_tmp, ax=ax[i * 2, j], vmin=vmin, vmax=vmax, square=True, annot=False, cmap='RdBu')
            ax[i * 2 + 1, j].errorbar(df1['x'], df1[observable], df1[f'{observable}_err'], ls='none', capsize=1, fmt='o', ms=1)
            ax[i * 2, j].set_title(f'beta = {betas[j]} | velocity = {velocities[i]}')
    plots.save_image_plt(image_path, image_name, dpi=400)
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--base_path')
parser.add_argument('--lattice_size_0')
parser.add_argument('--lattice_size_T')
parser.add_argument('--boundary')
parser.add_argument('--images_path')
args = parser.parse_args()
print('args: ', args)

observable = 'Jv'
df_0 = read_data(f'{args.base_path}/{args.lattice_size_0}/{args.boundary}', observable)
df_T = read_data(f'{args.base_path}/{args.lattice_size_T}/{args.boundary}', observable)
df_T[observable] = df_T[observable] - df_0[observable]
df_T[f'{observable}_err'] = np.sqrt(df_T[f'{observable}_err'] ** 2 + df_0[f'{observable}_err'] ** 2)
df_T = df_T[['x', 'y', observable, f'{observable}_err', 'beta', 'velocity']]
heatmap_errorbar(df_T, 3, observable, f'{args.images_path}/{args.lattice_size_T}/{args.boundary}', f'distribution_common_{observable}')