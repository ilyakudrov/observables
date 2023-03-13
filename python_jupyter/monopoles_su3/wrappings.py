import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn

import monopole_data
import fit_func
import plots

# functions for wrappings processing


def data_process_wrappings(data, hue_name):
    return data.groupby(['winding_number', hue_name, 'conf'])['cluster_number']\
        .agg([('cluster_number', np.mean)]).reset_index()\
        .groupby(['winding_number', hue_name])['cluster_number']\
        .agg([('cluster_number', np.mean), ('std', lambda x: np.std(x, ddof=1) / math.sqrt(np.size(x)))]).reset_index()

# fill with zeros missing values


def fill_windings(data):
    return data.set_index(['conf', 'color', 'winding_number']).unstack('winding_number', fill_value=0).stack()


def wrappings_time_dependence1(conf_range, paths, hue_name):
    print(conf_range)


def wrappings_time_dependence(conf_range, paths, hue_name, path_csv):
    data = monopole_data.read_data(conf_range, paths, 'windings', hue_name)

    data = data[data['direction'] == 'time']
    data = data.drop(columns=['direction'])
    data = data.set_index([hue_name, 'conf', 'color', 'winding_number']).unstack(
        'winding_number', fill_value=0).stack()

    # data = data.groupby(['time size'])[['color', 'cluster_number', 'winding_number', 'conf']].apply(fill_windings).reset_index()

    data = data_process_wrappings(data, hue_name)
    monopole_data.save_data_windings_separate_size(data, path_csv)

    data.groupby(['winding_number']).apply(
        plots.make_plot_time_wrappings, hue_name)

    # data.to_csv('../../data/wrappings_common', sep = ' ', index=False)


def wrappings_number_dependence(start, end, paths):
    data = monopole_data.read_data(
        start, end, paths, 'windings')

    data = data[data['direction'] == 'time']
    data = data.drop(columns=['direction'])
    data = data.set_index(['time size', 'conf', 'color', 'winding_number']).unstack(
        'winding_number', fill_value=0).stack().reset_index()

    data = data_process_wrappings(data)

    fg = seaborn.FacetGrid(data=data, hue='time size', height=5, aspect=1.61)
    plt.yscale('log')
    fg.map(plt.errorbar, 'winding_number', 'cluster_number',
           'std', marker="o", fmt='', linestyle='').add_legend()

    # save_image_time_wrappings('../../images/common', f'number-wrappings', fg)

    # data.to_csv('../../data/wrappings_common', sep = ' ', index=False)


def fit_wrappings(start, end, paths, ranges, func, plotting_func, name):
    data = monopole_data.read_data(
        start, end, paths, 'windings')

    data = data[data['direction'] == 'time']
    data = data.drop(columns=['direction'])
    data = data.set_index(['time size', 'conf', 'color', 'winding_number']).unstack(
        'winding_number', fill_value=0).stack()

    data = data_process_wrappings(data)

    # print(data)
    # data.to_csv('../../data/wrappings_common', sep = ',', index=False)

    for d in ranges:
        data = data[(data['winding_number'] <= ranges[d][1]) |
                    np.logical_not((data['time size'] == d))]
        data = data[(data['winding_number'] >= ranges[d][0]) |
                    np.logical_not((data['time size'] == d))]

    df = data.groupby(['time size']).apply(
        fit_func.fit_data, func).reset_index()
    # print(df)

    # df1 = df.groupby(['time size']).apply(plot_fit_func_polinomial, data['winding_number'].min(), data['winding_number'].max())
    df1 = df.groupby(['time size']).apply(
        plotting_func, data['winding_number'].min(), data['winding_number'].max())

    df1.index = df1.index.get_level_values('time size')
    df1 = df1.reset_index()

    fg = seaborn.FacetGrid(data=data, hue='time size', height=5, aspect=1.61)
    plt.yscale('log')
    fg.map(plt.errorbar, 'winding_number', 'cluster_number',
           'std', marker="o", fmt='', linestyle='').add_legend()

    df1.groupby(['time size']).apply(plot_test)

    # save_image_time_wrappings('../../images/common/fit', f'{name}', fg)
    # plt.plot(df1['winding_number'], df1['monopole number'])
    return df
