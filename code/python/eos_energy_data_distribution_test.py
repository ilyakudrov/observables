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

def get_file_names(path, name_start):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(base_path):
        files.extend(filenames)
        break
    return list((f for f in files if f.startswith(name_start)))

def get_dir_names(path):
    directories = []
    for (dirpath, dirnames, filenames) in os.walk(base_path):
        directories.extend(filenames)
        break
    return directories

parser = argparse.ArgumentParser()
parser.add_argument('--velocity')
parser.add_argument('--lattice_size')
parser.add_argument('--boundary')
parser.add_argument('--name_start')
args = parser.parse_args()
print('args: ', args)

df_therm = pd.read_csv(f'/home/clusters/rrcmpi/kudrov/eos_high_precision/result/logs/{args.lattice_size}/{args.boundary}/{args.velocity}/spec_therm.log', 
                       header=None, delimiter=' ', names=['beta', 'therm_length'])
print('listdir: ', os.listdir
if float(args.beta) in df_therm['beta'].unique():
    therm_length = df_therm.loc[df_therm['beta'] == float(args.beta), 'therm_length'].values[0]
    print('therm_length', therm_length)
else:
    therm_length = df_therm.loc[df_therm['beta'] == 0.00, 'therm_length'].values[0]

base_path = f'/home/clusters/rrcmpi/kudrov/eos_high_precision/result/logs/7x42x169sq/OBCb_cV/{args.velocity}/{args.beta}/run1001001'
filenames = get_file_names(base_path, args.name_start)
df = []
for i in range(len(filenames)):
    df.append(read_blocks(f'{base_path}/{filenames[i]}'))
    df[-1] = df[-1][['S']]
    df[-1]['conf'] = i + 1
df = pd.concat(df)
print(df)
df = df[df['conf'] > therm_length/10]
print(df)
df = df.groupby('conf').agg('mean')
print(df)
aver = df['S'].agg('mean')
std = df['S'].agg('std') / np.sqrt(len(df.index))
print('aver before a cut', aver, std)
#df1 = df.loc[(df['S'] >= aver - 3 * std) & (df['S'] <= aver + 3 * std)]
df1 = df.loc[df['S'] >= aver - 3 * std]
aver1 = df1['S'].agg('mean')
std1 = df1['S'].agg('std') / np.sqrt(len(df1.index))
print('aver after a cut', aver1, std1)
print(df['S'].min())
#df = df.sort_values(by='S')
print(df)
image_path = f'../../images/eos_high_precision/7x42x169sq/OBCb_cV/{args.velocity}/{args.beta}'
try:
    os.makedirs(image_path)
except:
    pass

plot = seaborn.histplot(df.loc[:, 'S'])
plot.set_xlim(aver - aver/1000, aver + aver/1000)
fig = plot.get_figure()
fig.savefig(f'{image_path}/S')
plt.close()
plot = seaborn.histplot(df1.loc[:, 'S'])
#plot.set_xlim(aver1 - aver1/1000, aver + aver/1000)
fig = plot.get_figure()
fig.savefig(f'{image_path}/S1')
plt.close()

df = df.agg('mean')
print(df)
