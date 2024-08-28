import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import monopole_data

def group_percolating(df):
    wrappings = df[0, ['x0_wrap', 'x1_wrap', 'x2_wrap', 'x3_wrap']].to_tumpy()
    df2 = df.iloc[1:]
    for i in range(4):
        if wrappings[i] != 0:
            df2 = df2[(df2[f'x{i}_wrap'] * wrappings[i]/abs(wrappings[i])) <= abs(wrappings[i])]

def pair_wrapped(data, threshold):
    """Find percolating cluster pairing clusters above threshold wrapped in the same direction"""
    # print(data)
    data = data.sort_values(by='length', ascending=False)

    length = (data['length'] >= threshold / 2).sum()
    length = length + length % 2
    if length >= 2:
        data = data.iloc[0:length]
        return pd.DataFrame({'length': [data['length'].sum()]})
    else:
        return pd.DataFrame({'length': [0]})

def find_percolating_wrapped_sum(data, threshold=None):
    return pd.DataFrame({'length': [data[data['percolating_group'] == 'percolating']['length'].sum()]}).reset_index(drop=True)

def clusters_divide(data_unwrapped, data_wrapped, size_threshold, groupby_keys):
    df_unwrapped = data_unwrapped.copy()
    df_unwrapped.loc[df_unwrapped['length'] < size_threshold, 'length'] = 0
    df_unwrapped['length'] = df_unwrapped['length'] * df_unwrapped['number']
    df_unwrapped = df_unwrapped.drop('number', axis=1)
    df_unwrapped = df_unwrapped.groupby(groupby_keys + ['color'])['length'].agg([('length', 'sum')]).reset_index(level=groupby_keys + ['color'])
    df_unwrapped = df_unwrapped.set_index(groupby_keys + ['color']).unstack('color', fill_value=0).stack(future_stack=True).reset_index()
    df_wrapped = data_wrapped.groupby(groupby_keys + ['color']).apply(find_percolating_wrapped_sum, size_threshold).reset_index(level=groupby_keys + ['color']).reset_index(drop=True)
    data_large = pd.concat([df_unwrapped, df_wrapped])
    # data_large = data_unwrapped
    data_large = data_large.groupby(groupby_keys + ['color'])['length'].agg([('length', 'sum')]).reset_index(level=groupby_keys + ['color'])
    data_large = data_large.groupby(groupby_keys)['length'].mean().reset_index(level=groupby_keys)
    return data_large

def cluster_gap_unwrapped(data_unwrapped, data_wrapped, thresholds, groupby_keys):
    perc_cluster_aver = []
    for threshold in thresholds:
        perc_cluster_aver.append(clusters_divide(data_unwrapped, data_wrapped, threshold, groupby_keys)['length'].mean())
    return perc_cluster_aver

def percolating_clusters(paths_unwrapped, paths_wrapped, min_threshold, max_threshold, threshold_step, groupby_keys, image_path, imag_name):
    data_unwrapped = monopole_data.read_data_unwrapped_copies(paths_unwrapped).reset_index(drop=True)
    data_wrapped = monopole_data.read_data_wrapped_copies(paths_wrapped).reset_index(drop=True)
    size_thresholds = list(range(min_threshold, max_threshold, threshold_step))
    perc_cluster_aver = cluster_gap_unwrapped(data_unwrapped, data_wrapped, size_thresholds, groupby_keys)
    plt.plot(size_thresholds, perc_cluster_aver);

    try:
        os.makedirs(image_path)
    except:
        pass
    plt.savefig(f'{image_path}/{imag_name}', dpi=400, facecolor='white')