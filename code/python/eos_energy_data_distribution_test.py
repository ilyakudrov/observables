from numba import njit
import sys
import numpy as np
import os.path
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import argparse

sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", ".."))
import statistics_python.src.statistics_observables as stat

def read_blocks(path):
    return pd.read_csv(path, header=None,delimiter=' ', names=['x', 'y', 'q', 'r', 'n', 'u',
                                                 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                                                 'h', 'i', 'j', 'k', 'l', 'm', 'S', 'o', 'p'])

def get_file_names(path, args):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        files.extend(filenames)
        break
    return list((f for f in files if f.startswith(f'block{args.block_size}-SEBxy_')))

def get_dir_names(path):
    directories = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        directories.extend(dirnames)
        break
    return directories

def get_data(base_path, args, df_therm):
    df = []
    beta_dirs = get_dir_names(f'{base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}')
    print('beta_dirs: ', beta_dirs)
    for beta in beta_dirs:
        if float(beta) in df_therm['beta'].unique():
            therm_length = df_therm.loc[df_therm['beta'] == float(beta), 'therm_length'].values[0]
        else:
            therm_length = df_therm.loc[df_therm['beta'] == 0.00, 'therm_length'].values[0]
        print('therm_length', therm_length)
        chain_dirs = get_dir_names(f'{base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/{beta}')
        print('chain_dirs: ', chain_dirs)
        i = 0
        df1 = []
        for chain in chain_dirs:
            filenames = get_file_names(f'{base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/{beta}/{chain}', args)
            for f in filenames:
                df1.append(read_blocks(f'{base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/{beta}/{chain}/{f}'))
                df1[-1] = df1[-1][['S']]
                df1[-1]['conf'] = i + 1
                i += 1
        df.append(pd.concat(df1))
        df[-1]['beta'] = beta
        # df[-1] = df[-1][df[-1]['conf'] > therm_length/args.block_size]
    return pd.concat(df)

def cut_artifacts(df, n):
    aver = df['S'].agg('mean')
    std = df['S'].agg('std') / np.sqrt(len(df.index))
    df = df.loc[df['S'] >= aver - n * std]
    return df

parser = argparse.ArgumentParser()
parser.add_argument('--velocity')
parser.add_argument('--lattice_size')
parser.add_argument('--boundary')
parser.add_argument('--block_size', type=int)
args = parser.parse_args()
print('args: ', args)

# base_path = '/home/clusters/rrcmpi/kudrov/eos_high_precision/result/logs'
base_path = '../../data/eos_high_precision'

df_therm = pd.read_csv(f'{base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/spec_therm.log',
                       header=None, delimiter=' ', names=['beta', 'therm_length'])
print(df_therm)

df = get_data(base_path, args, df_therm)
print(df)
df = df.groupby(['conf', 'beta']).agg('mean')
print(df)
df1 = df.groubpy('beta').apply(cut_artifacts, 3)
print(df1)
aver = df.groupby('beta').agg('mean')
aver1 = df1.groupby('beta').agg('mean')
print('aver before a cut', aver)
print('aver after a cut', aver1)
print('min: ', df.groupby('beta').apply(min))
image_path = f'../../images/eos_high_precision/{args.lattice_size}/{args.boundary}/{args.velocity}'
try:
    os.makedirs(image_path)
except:
    pass

for beta in df['beta'].unique():
    plot = seaborn.histplot(df.loc[df['beta'] == beta, 'S'])
    plot.set_xlim(aver - aver/1000, aver + aver/1000)
    fig = plot.get_figure()
    fig.savefig(f'{image_path}/S_{beta}')
    plt.close()
    plot = seaborn.histplot(df1.loc[df1['beta'] == beta, 'S'])
    fig = plot.get_figure()
    fig.savefig(f'{image_path}/S1_{beta}')
    plt.close()