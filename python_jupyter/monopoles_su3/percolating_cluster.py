import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import monopole_data

def pair_wrapped(data, threshold):
    """Find percolating cluster pairing clusters above threshold wrapped in the same direction"""
    # print(data)
    data = data.sort_values(by='length', ascending=False)
    length = (data['length'] >= threshold).sum()
    length = length + length % 2
    if length >= 2:
        data = data.iloc[0:length]
        return pd.DataFrame({'length': [data['length'].sum()]})
    else:
        return pd.DataFrame({'length': [0]})

def find_percolating_wrapped_sum(data, threshold):
    data = data.groupby(['direction', 'color']).apply(pair_wrapped, threshold).reset_index(level=['direction', 'color']).reset_index(drop=True)
    return data.groupby('color')['length'].sum().reset_index()

def clusters_divide(data_unwrapped, data_wrapped, size_threshold):
    data_large_unwrapped = data_unwrapped.copy()
    data_large_unwrapped.loc[data_large_unwrapped['length'] < size_threshold, 'length'] = 0
    data_large_unwrapped['length'] = data_large_unwrapped['length'] * data_large_unwrapped['number']
    data_large_unwrapped = data_large_unwrapped.drop('number', axis=1)
    data_large_unwrapped = data_large_unwrapped.groupby(['conf', 'color'])['length'].agg([('length', np.sum)]).reset_index(level=['conf', 'color'])
    data_large_unwrapped = data_large_unwrapped.set_index(['conf', 'color']).unstack('color', fill_value=0).stack().reset_index()
    data_large_wrapped = data_wrapped.copy()
    data_large_wrapped = data_large_wrapped.groupby(['conf']).apply(find_percolating_wrapped_sum, size_threshold).reset_index(level='conf').reset_index(drop=True)
    data_large = pd.concat([data_large_unwrapped, data_large_wrapped])
    data_large = data_large.groupby(['conf', 'color'])['length'].agg([('length', np.sum)]).reset_index(level=['conf', 'color'])
    # print(data_large)
    data_large = data_large.groupby(['conf'])['length'].mean()
    # print('mean', data_large.mean())
    return data_large.mean()

def cluster_gap_unwrapped(data_unwrapped, data_wrapped, thresholds):
    perc_cluster_aver = []
    for threshold in thresholds:
        perc_cluster_aver.append(clusters_divide(data_unwrapped, data_wrapped, threshold))
    return perc_cluster_aver

def percolating_clusters(start, end, paths_unwrapped, paths_wrapped, label_name, min_threshold, max_threshold, threshold_step, image_path, imag_name):
    data_unwrapped = monopole_data.read_data((start, end), paths_unwrapped, 'clusters_unwrapped', label_name).reset_index(drop=True)
    data_unwrapped = data_unwrapped.drop(label_name, axis = 1)
    data_wrapped = monopole_data.read_data((start, end), paths_wrapped, 'clusters_wrapped', label_name).reset_index(drop=True)
    data_wrapped = data_wrapped.drop(label_name, axis = 1)

    size_thresholds = list(range(min_threshold, max_threshold, threshold_step))
    perc_cluster_aver = cluster_gap_unwrapped(data_unwrapped, data_wrapped, size_thresholds)
    plt.plot(size_thresholds, perc_cluster_aver);

    try:
        os.makedirs(image_path)
    except:
        pass
    plt.savefig(f'{image_path}/{imag_name}', dpi=400, facecolor='white')