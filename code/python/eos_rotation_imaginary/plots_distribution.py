import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn
import os
import sys
import argparse
import multiprocessing
import time
from typing import List

sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "..", ".."))
import python_jupyter.common.plots as plots

def get_dir_names(path: str) -> List[str]:
    directories = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        directories.extend(dirnames)
        break
    return directories

def read_data(path, observable):
    df = pd.DataFrame()
    velocity_dirs = get_dir_names(path)
    velocity_dirs.sort()
    for velocity in velocity_dirs:
        print(velocity)
        beta_dirs = get_dir_names(f'{path}/{velocity}')
        beta_dirs.sort()
        print(beta_dirs)
        for beta in beta_dirs:
            if os.path.isfile(f'{path}/{velocity}/{beta}/observables_distribution_aver.csv'):
                df_tmp = pd.read_csv(f'{path}/{velocity}/{beta}/observables_distribution_aver.csv', delimiter=' ')
                df_tmp = df_tmp[['x', 'y', observable, f'{observable}_err']]
                df_tmp['velocity'] = velocity
                df_tmp['beta'] = float(beta)
                df = pd.concat([df, df_tmp])
    return df

def gaussian_smearing(df, observable):
    #data = df[observable].to_numpy()
    #n = df['x'].max() * 2 + 1
    data = []
    for x in df['x'].unique():
        data.append(df[df['x'] == x][observable].to_numpy())
    data = np.array(data)
    data = scipy.ndimage.gaussian_filter(data, sigma=1)
    df = pd.DataFrame()
    for x in range(len(data)):
        df = pd.concat([df, pd.DataFrame({'x': [x - len(data) // 2] * len(data), 'y': list(range(-(len(data) // 2), len(data) // 2 + 1)), observable: data[x]})])
    return df

def get_unique_pair(df):
    df = df.loc[:, ['velocity', 'beta']]
    df = df[~(df.duplicated(subset=['velocity', 'beta'], keep='first'))]
    df['C'] = False
    df = df.set_index(['velocity', 'beta']).unstack('beta', fill_value=True).stack()['C'].reset_index(level=['velocity', 'beta'])
    df = df.groupby('beta').filter(lambda x: not x['C'].any())
    return df['beta'].unique(), df['velocity'].unique()

def heatmap_errorbar(df, subplot_ratio, observable, image_path, image_name):
    #betas = np.sort(df['beta'].unique())
    #velocities = np.sort(df['velocity'].unique())
    betas, velocities = get_unique_pair(df)
    vmax = max(abs(df[observable].max()), abs(df[observable].min()))
    vmin = -vmax
    gs_kw = dict(height_ratios=list([subplot_ratio if i % 2 == 0 else 1 for i in range(len(velocities) * 2)]))
    fig, ax = plt.subplots(nrows=len(velocities) * 2, ncols=len(betas), gridspec_kw=gs_kw)
    fig.set_size_inches(8 * len(betas), 6 * len(velocities))
    fig.suptitle(observable, fontsize=16)
    df['x'] = df['x'] - df['x'].max()//2
    df['y'] = df['y'] - df['y'].max()//2
    df['rad'] = np.sqrt(df['x'] ** 2 + df['y'] ** 2)
    for i in range(len(velocities)):
        for j in range(len(betas)):
            print(betas[j])
            print(velocities[i])
            start = time.time()
            df_tmp = df[(df['beta'] == betas[j]) & (df['velocity'] == velocities[i])]
            # df1 = df_tmp.loc[(df_tmp['y'] == 0) | (df_tmp['y'] == df_tmp['x'])]
            df1 = df_tmp.loc[(df_tmp['y'] == 0)]
            df_tmp = df_tmp.loc[:, ['x', 'y', observable]]
            df_tmp = gaussian_smearing(df_tmp, observable)
            print(df_tmp)
            end = time.time()
            print("time1 = %s" % (end - start))
            start = time.time()
            x_max = df_tmp['x'].max()
            n = x_max * 2 + 1
            df_tmp = df_tmp[observable].to_numpy()
            df_tmp = np.reshape(df_tmp, (n, n))
            fig = ax[i * 2, j].imshow(df_tmp, cmap='RdBu', vmin=vmin, vmax=vmax, extent=[-x_max, x_max, -x_max, x_max])
            cbar = ax[i * 2, j].figure.colorbar(fig, ax=ax[i * 2, j])
            cbar.ax.set_ylabel('', rotation=-90, va="bottom")
            end = time.time()
            print("heatmap time = %s" % (end - start))
            start = time.time()
            ax[i * 2 + 1, j].errorbar(df1['x'], df1[observable], df1[f'{observable}_err'], ls='none', capsize=1, fmt='o', ms=1)
            ax[i * 2, j].set_title(f'beta = {betas[j]} | velocity = {velocities[i]}')
            end = time.time()
            print("errorbar time = %s" % (end - start))
    dpi = 100
    #print(8 * 6 * len(betas) * len(velocities) * dpi**2)
    #print(
    #if 8 * len(betas) * dpi > 2**16:
    #     dpi = 2**16 // (8 * len(betas))
    print(dpi)
    plots.save_image_plt(image_path, image_name, dpi=dpi)

parser = argparse.ArgumentParser()
parser.add_argument('--observable')
parser.add_argument('--base_path')
parser.add_argument('--lattice_size_0')
parser.add_argument('--lattice_size_T')
parser.add_argument('--boundary')
parser.add_argument('--images_path')
args = parser.parse_args()
print('args: ', args)

observable = args.observable
df_0 = read_data(f'{args.base_path}/{args.lattice_size_0}/{args.boundary}', observable)
df_T = read_data(f'{args.base_path}/{args.lattice_size_T}/{args.boundary}', observable)
df_tmp = pd.concat([df_0[['velocity', 'beta', 'x', 'y']], df_T[['velocity', 'beta', 'x', 'y']]])
df_tmp = df_tmp[df_tmp.duplicated(subset=['velocity', 'beta', 'x', 'y'], keep='first')]
df_0 = df_tmp.merge(df_0, how='left', on=['velocity', 'beta', 'x', 'y'])
df_T = df_tmp.merge(df_T, how='left', on=['velocity', 'beta', 'x', 'y'])
df_T[observable] = df_T[observable] - df_0[observable]
df_T[f'{observable}_err'] = np.sqrt(df_T[f'{observable}_err'] ** 2 + df_0[f'{observable}_err'] ** 2)
heatmap_errorbar(df_T, 3, observable, f'{args.images_path}/{args.lattice_size_T}/{args.boundary}', f'distribution_common_{observable}.png')
