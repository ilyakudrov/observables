from numba import njit
import sys
import numpy as np
import os.path
import pandas as pd
import seaborn

sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", ".."))
import statistics_python.src.statistics_observables as stat

def read_blocks(path):
    return pd.read_csv(path, header=None,delimiter=' ', names=['x', 'y', 'S1', 'S2', 'S3', 'S4',
                                                 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                                                 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'])
    # return pd.read_csv(path, header=None, delimiter=' ')

base_path = '/home/clusters/rrcmpi/kudrov/eos_high_precision/result/logs/5x30x121sq/OBCb_cV/0.000000v/3.8800/run3001001'
# base_path = '../../data/eos_high_precision/5x30x121sq/OBCb_cV/0.000000v/3.8800/run3001001'
files = []
for (dirpath, dirnames, filenames) in os.walk(base_path):
    files.extend(filenames)
    break
print(filenames)
df = []
for i in range(len(filenames)):
    df.append(read_blocks(f'{base_path}/{filenames[i]}'))
    df[-1]['conf'] = i
df = pd.concat(df)
print(df)
df = df.groupby('conf').agg('mean')
print(df)
image_path = '../../images/eos_high_precision/5x30x121sq/OBCb_cV/0.000000v/3.8800/run3001001'
try:
    os.makedirs(image_path)
except:
    pass

plot = seaborn.histplot(df['S1'])
fig = plot.get_figure()
fig.savefig(f'{image_path}/S1')
plot = seaborn.histplot(df['S2'])
fig = plot.get_figure()
fig.savefig(f'{image_path}/S2')
plot = seaborn.histplot(df['S3'])
fig = plot.get_figure()
fig.savefig(f'{image_path}/S3')
plot = seaborn.histplot(df['S4'])
fig = plot.get_figure()
fig.savefig(f'{image_path}/S4')

df = df.agg('mean')
print(df)