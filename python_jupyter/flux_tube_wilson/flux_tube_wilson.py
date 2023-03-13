import pandas as pd
import math

import flux_tube_wilson_data as flux_data
import fit_func


def join_back(data, flux_coord, matrix_types, columns):
    data1 = []
    for matrix_type in matrix_types:
        for field_type in ['electric', 'magnetic', 'energy', 'action']:
            data1.append(
                data[[flux_coord, 'R', f'field_{field_type}_{matrix_type}', f'err_{field_type}_{matrix_type}'] + columns])
            data1[-1] = data1[-1].rename(columns={
                                         f'field_{field_type}_{matrix_type}': 'field', f'err_{field_type}_{matrix_type}': 'err'})
            data1[-1]['matrix_type'] = matrix_type
            data1[-1]['field_type'] = field_type

    return pd.concat(data1)


def find_sum(data):
    for flux_type in ['electric', 'magnetic', 'energy', 'action']:
        data[f'err_{flux_type}_mon+nomon'] = data.apply(lambda x: math.sqrt(
            x[f'err_{flux_type}_monopole'] ** 2 + x[f'err_{flux_type}_monopoless'] ** 2), axis=1)
        data[f'field_{flux_type}_mon+nomon'] = data.apply(
            lambda x: x[f'field_{flux_type}_monopole'] + x[f'field_{flux_type}_monopoless'], axis=1)

    return data


def flux(paths, flux_coord, image_path, plot_function, sigma):
    data = flux_data.get_flux_data(paths, flux_coord, sigma)

    # if flux_coord == 'x_tr':
    #     data = data[data['x_tr'] <= 10 * math.sqrt(sigma)]
    #     data = data[data['x_tr'] >= -10 * math.sqrt(sigma)]

    # if flux_coord == 'd':
    #     data = data[data['d'] <= 3 * math.sqrt(sigma)]
    #     data = data[data['d'] >= -3 * math.sqrt(sigma)]

    # data = data[data['R'] == 8]

    data.groupby(['R', 'field_type']).apply(
        plot_function, flux_coord, image_path)


def flux_R(paths, flux_coord, image_path, plot_function, fit_func, plotting_func, sigma):
    data = flux_data.get_flux_data(paths, flux_coord, sigma)

    data = data[data['R'] <= 16]
    data = data[data['d'] == 0]

    df_fit = data.groupby(['type', 'field_type']).apply(
        fit_func.fit_data, fit_func).reset_index()
    df1 = df_fit.groupby(['type', 'field_type']).apply(
        plotting_func, data['R'].min(), data['R'].max())

    # df1.index = df1.index.get_level_values(['field_type', 'type'])
    df1 = df1.reset_index()
    df1 = df1.drop(['level_2'], axis=1)

    # print(df1)
    # print(data)

    # data1 = [df1, data[['field_type', 'R', 'field', 'err', 'type']]]
    # data = pd.concat(data1)

    # print(data)

    data.groupby(['field_type']).apply(plot_function, flux_coord, image_path)


def relative_variation(betas, type, func):
    data = []
    data.append(flux_data.concat_betas(
        betas, f'act-lng-prof0{type}.dat', func))
    data[-1] = data[-1].rename(columns={'field': 'field_su2',
                               'err': 'err_su2'})
    data.append(flux_data.concat_betas(
        betas, f'act-lng-prof0{type}_mon.dat', func))
    data[-1] = data[-1].rename(columns={'field': 'field_mon',
                               'err': 'err_mon'})
    data.append(flux_data.concat_betas(
        betas, f'act-lng-prof0{type}-off.dat', func))
    data[-1] = data[-1].rename(
        columns={'field': 'field_nomon', 'err': 'err_nomon'})

    data = pd.concat(data, axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    data['field_diff'] = data.apply(
        lambda x: x['field_su2'] - x['field_mon'] - x['field_nomon'], axis=1)
    data['err_diff'] = data.apply(lambda x: math.sqrt(
        x['err_su2'] ** 2 + x['err_mon'] ** 2 + x['err_nomon'] ** 2), axis=1)
    data['err_diff'] = data.apply(lambda x: math.sqrt(x['err_diff'] ** 2 / x['field_su2']
                                  ** 2 + x['err_su2'] ** 2 * x['field_diff'] ** 2 / x['field_su2'] ** 4), axis=1)
    data['field_diff'] = data.apply(
        lambda x: x['field_diff'] / x['field_su2'], axis=1)

    return data[['R', 'd', 'beta', 'field_diff', 'err_diff']]


def flux_decomposition(paths, flux_coord, image_path, plot_function, sigma):
    data = flux_data.get_flux_data_decomposition(paths, flux_coord, sigma)

    data.groupby(['R', 'field_type']).apply(
        plot_function, flux_coord, image_path)
