from numba import njit
import sys
import numpy as np
import os.path
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import argparse
import scipy
from scipy.stats import norm
import scipy.stats

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
    #beta_dirs = beta_dirs[:1]
    #beta_dirs = ['4.2200']
    print('beta_dirs: ', beta_dirs)
    for beta in beta_dirs:
        if float(beta) in df_therm['beta'].unique():
            therm_length = df_therm.loc[df_therm['beta'] == float(beta), 'therm_length'].values[0]
        else:
            therm_length = df_therm.loc[df_therm['beta'] == 0.00, 'therm_length'].values[0]
        print('therm_length', therm_length)
        chain_dirs = get_dir_names(f'{base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/{beta}')
        chain_dirs.sort()
        print('chain_dirs: ', chain_dirs)
        i = 0
        df1 = []
        for chain in chain_dirs:
            filenames = get_file_names(f'{base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/{beta}/{chain}', args)
            filenames.sort()
            for f in filenames:
                df1.append(read_blocks(f'{base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/{beta}/{chain}/{f}'))
                df1[-1] = df1[-1][['S']]
                df1[-1]['conf'] = i + 1
                i += 1
        df.append(pd.concat(df1))
        df[-1]['beta'] = beta
        df[-1] = df[-1][df[-1]['conf'] > therm_length/args.block_size]
    return pd.concat(df)

def cut_artifacts(df, n):
    aver = df['S'].agg('median')
    std = df['S'].agg('std')
    df = df.loc[(df['S'] >= aver - n * std) & (df['S'] <= aver + n * std)]
    return df

def chi_square(obs, exp):
    return np.sum((obs - exp)**2/exp)/(len(obs)-3)

def get_statistics(df):
    normality, _ = scipy.stats.normaltest(df)
    skewness, _ = scipy.stats.skewtest(df)
    kurtosis, _ = scipy.stats.kurtosistest(df)
    mu, std_fit = norm.fit(df)
    hist, bins = np.histogram(df, bins='auto', density=True)
    h = (bins[1] - bins[0])/2
    bins = np.array([x + h for x in bins[:-1]])
    expected = norm.pdf(bins, mu, std_fit)
    chi_sq = chi_square(hist, expected)
    mean = df.mean()
    std = df.std()
    err = std / np.sqrt(len(df))
    min_val = df.min()
    return pd.DataFrame({'normality': [normality], 'skewness': [skewness], 
                         'kurtosis': [kurtosis], 'chi_square': [chi_sq], 
                         'mean': [mean], 'std': [std], 'err': [err], 
                         'min': [min_val], 'mu_fit': [mu], 'std_fit': [std_fit]})

def save_image_plt(image_path, image_name):
    try:
        os.makedirs(image_path)
    except:
        pass

    output_path = f'{image_path}/{image_name}'
    print(output_path)
    plt.savefig(output_path, dpi=800)

def make_plots_statistics(df, image_path):
    for obs in ['normality', 'skewness', 'kurtosis', 'chi_square', 'mean',
                'std', 'err', 'min', 'mu_fit', 'std_fit']:
        ax = seaborn.scatterplot(data=df, x='beta', y=obs, hue='cut', palette=seaborn.color_palette('bright'))
        plt.tight_layout()
        save_image_plt(image_path, obs)
        plt.close()

parser = argparse.ArgumentParser()
parser.add_argument('--base_path')
parser.add_argument('--velocity')
parser.add_argument('--lattice_size')
parser.add_argument('--boundary')
parser.add_argument('--block_size', type=int)
args = parser.parse_args()
print('args: ', args)

#base_path = '/home/clusters/rrcmpi/kudrov/eos_high_precision/result/logs'
base_path = args.base_path
#base_path = '../../data/eos_high_precision'

df_therm = pd.read_csv(f'{base_path}/{args.lattice_size}/{args.boundary}/{args.velocity}/spec_therm.log',
                       header=None, delimiter=' ', names=['beta', 'therm_length'])
print(df_therm)
df = get_data(base_path, args, df_therm)
print(df)
df = df.groupby(['conf', 'beta']).agg('mean').reset_index(level=['beta', 'conf'])
print(df)
for beta in df['beta'].unique():
    seaborn.scatterplot(df[df['beta'] == beta], x='conf', y='S', s=1)
    name_beta = beta.replace('.', 'p')
    try:
        os.makedirs(f'../../images/eos_high_precision/{args.lattice_size}/{args.boundary}/{args.velocity}')
    except:
        pass
    plt.savefig(f'../../images/eos_high_precision/{args.lattice_size}/{args.boundary}/{args.velocity}/history_{name_beta}', dpi=800)
    plt.close()

df_statistics = []
df_statistics.append(df.groupby('beta')['S'].apply(get_statistics).reset_index(level='beta'))
df_statistics[-1]['cut'] = 0
for cut in [10, 5, 3]:
    df1 = df.groupby('beta').apply(cut_artifacts, cut).reset_index(drop=True)
    df_statistics.append(df1.groupby('beta')['S'].apply(get_statistics).reset_index(level='beta'))
    df_statistics[-1]['cut'] = cut
df_statistics = pd.concat(df_statistics)
print(df_statistics)
make_plots_statistics(df_statistics, f'../../images/eos_high_precision/{args.lattice_size}/{args.boundary}/{args.velocity}')

for beta in df['beta'].unique():
    name_beta = beta.replace('.', 'p')
    plot = seaborn.histplot(df.loc[df['beta'] == beta, 'S'], stat='density')
    mu, std = norm.fit(df.loc[df['beta'] == beta, 'S'])
    x = np.linspace(mu - 5*std, mu + 5*std, 1000)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title("Fit results: mu = %.4f,  std = %.7f" % (mu, std))
    save_image_plt(f'../../images/eos_high_precision/{args.lattice_size}/{args.boundary}/{args.velocity}', f'S_{name_beta}')
    plt.close()
