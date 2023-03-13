import matplotlib.pyplot as plt
import seaborn
import math
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

import plots
import potential_data


def plot_relative_variation_potential(data):
    # T = data['T'].iloc[0]
    # sigma = data['sigma'].iloc[0]
    fg = seaborn.FacetGrid(data=data, hue='beta', height=5, aspect=1.2)
    fg.fig.suptitle(f'relative variation')
    fg.map(plt.errorbar, 'R', 'potential_diff', 'err_diff', marker="o", fmt='', linestyle=''
           ).add_legend()
    fg.ax.set_xlabel(r"r$\sqrt{\sigma}$")
    fg.ax.set_ylabel(r"$\Delta$")
    plt.xlim((0, 2.5))
    plt.ylim((-0.3, 0.3))
    fg.ax.spines['right'].set_visible(True)
    fg.ax.spines['top'].set_visible(True)
    fg.ax.minorticks_on()
    fg.ax.tick_params(which='both', bottom=True,
                      top=True, left=True, right=True)
    plt.axhline(y=0, color='k', linestyle='-')
    # plt.show()

    plots.save_image(
        f'../images/potential/relative_variation/vitaliy', f'relative_variation', fg)


def relative_variation_potential(paths):
    data = potential_data.read_data_potential(paths)

    data = data[data['potential_su2'] != 0.0]

    data['potential_diff'] = data.apply(
        lambda x: x['potential_su2'] - x['potential_monopole'] - x['potential_monopoless'], axis=1)
    data['err_diff'] = data.apply(lambda x: math.sqrt(
        x['err_su2'] ** 2 + x['err_monopole'] ** 2 + x['err_monopoless'] ** 2), axis=1)
    data['err_diff'] = data.apply(lambda x: math.sqrt(x['err_diff'] ** 2 / x['potential_su2']
                                  ** 2 + x['err_su2'] ** 2 * x['potential_diff'] ** 2 / x['potential_su2'] ** 4), axis=1)
    data['potential_diff'] = data.apply(
        lambda x: x['potential_diff'] / x['potential_su2'], axis=1)

    data['R'] = data.apply(lambda x: x['R'] * x['sigma'], axis=1)

    plot_relative_variation_potential(data)


def plot_potential_decomposition(data, y_lims, ls_arr, marker_arr, fillstyle_arr, colors, image_path, image_name):
    fg = seaborn.FacetGrid(data=data, hue='matrix_type', height=5, aspect=1.4, legend_out=False,
                           hue_kws={"ls": ls_arr, "marker": marker_arr,
                                    "fillstyle": fillstyle_arr, "color": colors})
    map = fg.map(plt.errorbar, 'r/a', 'aV(r)', 'err', ms=8,
                 capsize=8, lw=0.5).add_legend(title='')
    fg.ax.set_xlabel(r"R$/r_{0}$")
    fg.ax.set_ylabel(r"$r_{0}V(R)$")
    fg.ax.spines['right'].set_visible(True)
    fg.ax.spines['top'].set_visible(True)
    fg.ax.minorticks_on()
    fg.ax.tick_params(which='both', bottom=True,
                      top=True, left=True, right=True)
    fg.ax.set_ylim(y_lims[0], y_lims[1])

    return fg


def join_back(data, matrix_types):
    data1 = []
    for matrix_type in matrix_types:
        data1.append(
            data[['r/a', f'aV(r)_{matrix_type}', f'err_{matrix_type}']])
        data1[-1] = data1[-1].rename(
            columns={f'aV(r)_{matrix_type}': 'aV(r)', f'err_{matrix_type}': 'err'})
        data1[-1]['matrix_type'] = matrix_type

    return pd.concat(data1)


def find_sum(data, term1, term2, sum):
    data[f'err_' + sum] = data.apply(lambda x: math.sqrt(
        x[f'err_' + term1] ** 2 + x[f'err_' + term2] ** 2), axis=1)
    data[f'aV(r)_' + sum] = data.apply(lambda x: x[f'aV(r)_' +
                                                   term1] + x[f'aV(r)_' + term2], axis=1)

    return data


def func_exponent(x, a, b, c):
    return a + b * np.exp(-x * c)


def func_quark_potential(x, c, alpha, sigma):
    return c + alpha * np.power(x, -1) + sigma * x


def func_coloumb(x, c, alpha):
    return c + alpha * np.power(x, -1)


def func_linear(x, c, sigma):
    return c + sigma * x


def fit_potential(data, fit_function, fit_range, fit_name):
    r = data['r/a'].iloc[0]
    data = data[(data['T'] >= fit_range[0]) & (data['T'] <= fit_range[1])]
    y = data['aV(r)_' + fit_name]
    y_err = data['err_' + fit_name]
    x = data['T']
    try:
        popt, pcov = curve_fit(fit_function, x, y, sigma=y_err)
        val = popt[0]
        err = np.sqrt(np.diag(pcov)[0])
    except:
        print('potential fit did not converge at r =', r)
        val = data.loc[data['T'] == 5, 'aV(r)_' + fit_name].iloc[0]
        err = data.loc[data['T'] == 5, 'err_' + fit_name].iloc[0]
    return pd.DataFrame([[val, err]], columns=['aV(r)_' + fit_name, 'err_' + fit_name])


def get_potential_fit(data, fit_func, fit_range, fit_name):
    return data.groupby(['r/a']).apply(fit_potential, fit_func, fit_range, fit_name).reset_index('r/a').reset_index()


def fit_string(data, fit_range, fit_name):
    data = data[(data['r/a'] >= fit_range[0]) & (data['r/a'] <= fit_range[1])]
    y = data['aV(r)_' + fit_name]
    y_err = data['err_' + fit_name]
    x = data['r/a'].to_numpy(dtype=np.float64)
    popt, pcov = curve_fit(func_quark_potential, x, y, sigma=y_err)
    return popt


def fit_consts(data, coloumb_name, string_name, alpha, sigma):
    y_coloumb = data['aV(r)_' + coloumb_name]
    y_string = data['aV(r)_' + string_name]
    x = data['r/a'].to_numpy(dtype=np.float64)
    c_coloumb, err_coloumb = curve_fit(
        lambda x, c: c + alpha * np.power(x, -1), x, y_coloumb)
    c_string, err_string = curve_fit(lambda x, c: c + sigma * x, x, y_string)
    return c_coloumb, c_string


def make_fit_original(data, orig_pot_name, coloumb_name, string_name, fit_range):
    c, alpha, sigma = fit_string(data, fit_range, orig_pot_name)
    c_coloumb, c_string = fit_consts(
        data, coloumb_name, string_name, alpha, sigma)
    x_fit = np.arange(data['r/a'].min(), data['r/a'].max(), 0.01)
    y_coloumb = func_coloumb(x_fit, c_coloumb, alpha)
    y_string = func_linear(x_fit, c_string, sigma)
    y_original = func_quark_potential(x_fit, c, alpha, sigma)
    return pd.DataFrame(np.array([x_fit, y_coloumb, y_string, y_original]).T,
                        columns=['r/a', 'aV(r)_' + coloumb_name,
                                 'aV(r)_' + string_name, 'aV(r)_' + orig_pot_name])


def make_fit_separate(data, terms, fit_range, r0):
    x_fit = np.arange(data['r/a'].min(), data['r/a'].max(), 0.01)
    data = data[(data['r/a'] >= fit_range[0]) &
                (data['r/a'] <= fit_range[1])].reset_index()
    data_fits = []
    columns = []
    x = data['r/a'].to_numpy(dtype=np.float64)
    data_fits.append(x_fit)
    columns.append('r/a')
    for term in terms:
        y = data['aV(r)_' + term]
        y_err = data['err_' + term]
        popt, pcov = curve_fit(func_quark_potential, x, y, sigma=y_err)
        perr = np.sqrt(np.diag(pcov))
        chi_sq = chi_square(x, y, popt[0], popt[1], popt[2])
        # print('aV(r)_' + term, popt[0] / r0, perr[0] / r0,
        #                         popt[1], perr[1], popt[2] / r0**2,
        #                         perr[2] / r0**2, 'chi_sq =', chi_sq)
        data_fits.append(func_quark_potential(x_fit, *popt))
        columns.append(f'aV(r)_' + term)
    return pd.DataFrame(np.array(data_fits).T, columns=columns)


def get_terms(paths):
    terms = []
    for key, value in paths.items():
        terms.append(value['name'])
    if 'monopole' in paths and 'monopoless' in paths:
        monopole_name = paths['monopole']['name']
        monopoless_name = paths['monopoless']['name']
        terms.append(f'{monopole_name}+{monopoless_name}')
    if 'abelian' in paths and 'offdiagonal' in paths:
        abelian_name = paths['abelian']['name']
        offdiagonal_name = paths['offdiagonal']['name']
        terms.append(f'{abelian_name}+{offdiagonal_name}')
    return terms


def chi_square(x, y, c, alpha, sigma):
    chi_sq = 0
    for i in range(len(x)):
        expected = func_quark_potential(x[i], c, alpha, sigma)
        chi_sq += (expected - y[i])**2 / expected
    return chi_sq

# find a/r_0 for 5.7 <= beta <= 6.92


def get_r0(beta):
    return math.exp(-1.6804 - 1.7331 * (beta - 6) + 0.7849 * (beta - 6)**2 - 0.4428 * (beta - 6)**3)


def potential_decomposition(paths, image_path, image_name, beta, y_lims, fit_original, r0, fit_range, remove_from_plot, black_colors):
    data = potential_data.read_data_potential1(paths)
    data1 = []
    for type, path in paths.items():
        if 'T' in path:
            data1.append(data[data['T'] == path['T']].reset_index()[
                         ['r/a', 'aV(r)_' + path['name'], 'err_' + path['name']]])
        else:
            data1.append(get_potential_fit(
                data, func_exponent, (2, 8), path['name']))

    data = pd.concat(data1, axis=1)
    data = data.loc[:, ~data.columns.duplicated()]

    terms = get_terms(paths)
    terms_fit = list(terms)

    data_fits = []
    if fit_original:
        if 'original' in paths and 'monopole' in paths and 'monopoless' in paths:
            data_fits.append(make_fit_original(
                data, paths['original']['name'], paths['monopoless']['name'], paths['monopole']['name'], fit_range))
            for term in [paths['original']['name'], paths['monopoless']['name'], paths['monopole']['name']]:
                try:
                    terms_fit.remove(term)
                except:
                    pass
        if 'original' in paths and 'abelian' in paths and 'offdiagonal' in paths:
            data_fits.append(make_fit_original(
                data, paths['original']['name'], paths['offdiagonal']['name'], paths['abelian']['name'], fit_range))
            for term in [paths['original']['name'], paths['offdiagonal']['name'], paths['abelian']['name']]:
                try:
                    terms_fit.remove(term)
                except:
                    pass

    if 'monopole' in paths and 'monopoless' in paths:
        monopole_name = paths['monopole']['name']
        monopoless_name = paths['monopoless']['name']
        sum_name = f'{monopole_name}+{monopoless_name}'
        data = find_sum(data, paths['monopole']['name'],
                        paths['monopoless']['name'], sum_name)

    if 'abelian' in paths and 'offdiagonal' in paths:
        abelian_name = paths['abelian']['name']
        offdiagonal_name = paths['offdiagonal']['name']
        sum_name = f'{abelian_name}+{offdiagonal_name}'
        data = find_sum(data, paths['abelian']['name'],
                        paths['offdiagonal']['name'], sum_name)

    data_fits.append(make_fit_separate(data, terms_fit, fit_range, r0))

    data_fits = pd.concat(data_fits, axis=1)
    data_fits = data_fits.loc[:, ~data_fits.columns.duplicated()]

    data['r/a'] = data['r/a'] * r0
    for term in terms:
        data[f'aV(r)_' + term] = data[f'aV(r)_' + term] / r0
        data[f'err_' + term] = data[f'err_' + term] / r0

    data_fits['r/a'] = data_fits['r/a'] * r0
    for term in terms:
        data_fits['aV(r)_' + term] = data_fits['aV(r)_' + term] / r0

    for term in remove_from_plot:
        terms.remove(paths[term]['name'])
        data = data.drop(f'aV(r)_' + paths[term]['name'], axis=1)
        data = data.drop(f'err_' + paths[term]['name'], axis=1)

    data = join_back(data, terms)

    ls_arr = ['', '', '', '', '', '', '']
    marker_arr = ['o', 'v', 'o', '^', 's', 's', 'D']
    fillstyle_arr = ['full', 'full', 'none', 'full', 'full', 'none', 'none']
    if black_colors:
        colors = ['black', 'black', 'black',
                  'black', 'black', 'black', 'black']
    else:
        colors = ['mediumblue', 'orange', 'g', 'r',
                  'rebeccapurple', 'saddlebrown', 'olive']
    fg = plot_potential_decomposition(
        data, y_lims, ls_arr, marker_arr, fillstyle_arr, colors, image_path, image_name)

    for i in range(len(terms)):
        seaborn.lineplot(data=data_fits, x='r/a',
                         y='aV(r)_' + terms[i], color=colors[i])

    plt.show()
    plots.save_image(image_path, image_name, fg)


def plot_together(data):
    fg = seaborn.FacetGrid(data=data, hue='type', height=5,
                           aspect=1.4, legend_out=False)
    fg.fig.suptitle(f'potentials together')
    fg.map(plt.errorbar, 'r/a', 'aV(r)', 'err', mfc=None, fmt='o', ms=3, capsize=5, lw=0.5, ls='-'
           ).add_legend()
    # plt.legend(loc='upper left')
    fg.ax.set_xlabel(r"R$\sqrt{\sigma}$")
    fg.ax.set_ylabel(r"V(r)/$\sigma$")
    fg.ax.spines['right'].set_visible(True)
    fg.ax.spines['top'].set_visible(True)
    fg.ax.minorticks_on()
    fg.ax.tick_params(which='both', bottom=True,
                      top=True, left=True, right=True)
    plt.grid(dash_capstyle='round')

    plt.show()


def potentials_together(paths, sigma, r_max):
    data = potential_data.read_data_potentials_together(paths, sigma)

    data = data[data['r/a'] <= r_max]

    plot_together(data)


def potential_decomposition_vitaly(path, image_path, image_name, coefs, terms, y_lims, beta):
    data = potential_data.read_vitaly_potential(path)

    terms.append(f'{terms[1]}+{terms[2]}')
    data = find_sum(data, terms[1], terms[2], terms[3])

    r0 = get_r0(beta)
    data['r/a'] = data['r/a'] * r0
    for term in terms:
        data[f'aV(r)_' + term] = data[f'aV(r)_' + term] / r0
        data[f'err_' + term] = data[f'err_' + term] / r0

    data = join_back(data, [terms[0], terms[1], terms[2], terms[3]])

    ls_arr = ['', '', '', '']
    marker_arr = ['o', 'v', 'o', '^']
    fillstyle_arr = ['full', 'full', 'none', 'full']
    colors = ['mediumblue', 'orange', 'g', 'r']
    fg = plot_potential_decomposition(
        data, y_lims, ls_arr, marker_arr, fillstyle_arr, colors, image_path, image_name)

    data_fits = []
    columns = []
    x_fit = np.arange(data['r/a'].min(), data['r/a'].max(), 0.01)
    data_fits.append(x_fit)
    columns.append('r/a')
    for term in terms:
        data_fits.append(func_quark_potential(x_fit, *coefs[term]))
        columns.append(f'aV(r)_' + term)
    data_fits = pd.DataFrame(np.array(data_fits).T, columns=columns)

    colors = ['mediumblue', 'orange', 'g', 'r']
    for i in range(len(terms)):
        seaborn.lineplot(data=data_fits, x='r/a',
                         y='aV(r)_' + terms[i], color=colors[i])

    plt.show()
    plots.save_image(image_path, image_name, fg)


def potential_decomposition_vitaly1(paths, path, image_path, image_name, coefs, terms, y_lims, beta, r_max):
    data = []
    data.append(potential_data.read_vitaly_potential(path))
    data1 = potential_data.read_data_potential(paths)
    for dict in paths:
        if 'T' in dict:
            data.append(data1[(data1['T'] == dict['T']) & (data1['r/a'] <= r_max)
                              ].reset_index()[['r/a', 'aV(r)_' + dict['name'], 'err_' + dict['name']]])
    data = pd.concat(data, axis=1)
    data = data.loc[:, ~data.columns.duplicated()]

    terms.append(f'{terms[1]}+{terms[2]}')
    for i in paths:
        terms.append(i['name'])
    terms.append(f'{terms[4]}+{terms[5]}')
    data = find_sum(data, terms[1], terms[2], terms[3])
    data = find_sum(data, terms[4], terms[5], terms[6])

    r0 = get_r0(beta)
    data['r/a'] = data['r/a'] * r0
    for term in terms:
        data[f'aV(r)_' + term] = data[f'aV(r)_' + term] / r0
        data[f'err_' + term] = data[f'err_' + term] / r0

    data = data.drop(f'aV(r)_{terms[4]}', axis=1)
    data = data.drop(f'err_{terms[4]}', axis=1)
    data = data.drop(f'aV(r)_{terms[5]}', axis=1)
    data = data.drop(f'err_{terms[5]}', axis=1)
    terms = [terms[0], terms[1], terms[2], terms[3], terms[6]]
    data = join_back(data, terms)

    ls_arr = ['', '', '', '', '']
    marker_arr = ['o', 'v', 'o', '^', 's']
    fillstyle_arr = ['full', 'full', 'none', 'full', 'full']
    colors = ['mediumblue', 'orange', 'g', 'r', 'rebeccapurple']
    fg = plot_potential_decomposition(
        data, y_lims, ls_arr, marker_arr, fillstyle_arr, colors, image_path, image_name)

    data_fits = []
    columns = []
    x_fit = np.arange(data['r/a'].min(), data['r/a'].max(), 0.01)
    data_fits.append(x_fit)
    columns.append('r/a')
    for term in terms[:-1]:
        data_fits.append(func_quark_potential(x_fit, *coefs[term]))
        columns.append(f'aV(r)_' + term)
    data_fits = pd.DataFrame(np.array(data_fits).T, columns=columns)

    colors = ['mediumblue', 'orange', 'g', 'r']
    for i in range(len(terms[:-1])):
        seaborn.lineplot(data=data_fits, x='r/a',
                         y='aV(r)_' + terms[i], color=colors[i])

    plt.show()
    plots.save_image(image_path, image_name, fg)


def potential_decomposition_general(paths, path, image_path, image_name, fit_coefs, to_fit, to_remove, terms, y_lims, beta, r_max):
    data = []
    data.append(potential_data.read_vitaly_potential(path))
    data1 = potential_data.read_data_potential(paths)
    for dict in paths:
        if 'T' in dict:
            data.append(data1[(data1['T'] == dict['T']) & (data1['r/a'] <= r_max)
                              ].reset_index()[['r/a', 'aV(r)_' + dict['name'], 'err_' + dict['name']]])
    data = pd.concat(data, axis=1)
    data = data.loc[:, ~data.columns.duplicated()]

    terms.append(f'{terms[1]}+{terms[2]}')
    for i in paths:
        terms.append(i['name'])
    terms.append(f'{terms[4]}+{terms[5]}')
    data = find_sum(data, terms[1], terms[2], terms[3])
    data = find_sum(data, terms[4], terms[5], terms[6])

    r0 = get_r0(beta)
    data['r/a'] = data['r/a'] * r0
    for term in terms:
        data[f'aV(r)_' + term] = data[f'aV(r)_' + term] / r0
        data[f'err_' + term] = data[f'err_' + term] / r0

    data_fits = []
    columns = []
    x_fit = np.arange(data['r/a'].min(), data['r/a'].max(), 0.01)
    data_fits.append(x_fit)
    columns.append('r/a')
    for term in fit_coefs:
        data_fits.append(func_quark_potential(x_fit, *fit_coefs[term]))
        columns.append(f'aV(r)_' + term)
    x = x = data['r/a']
    for term in to_fit:
        y = data['aV(r)_' + term]
        y_err = data['err_' + term]
        popt, pcov = curve_fit(func_quark_potential, x, y, sigma=y_err)
        data_fits.append(func_quark_potential(x_fit, *popt))
        columns.append(f'aV(r)_' + term)
    data_fits = pd.DataFrame(np.array(data_fits).T, columns=columns)

    for term in to_remove:
        data = data.drop(f'aV(r)_{term}', axis=1)
        data = data.drop(f'err_{term}', axis=1)
        terms.remove(term)
    data = join_back(data, terms)

    ls_arr = ['', '', '', '', '', '', '']
    marker_arr = ['o', 'v', 'o', '^', 's', 's', 'D']
    fillstyle_arr = ['full', 'full', 'none', 'full', 'full', 'none', 'none']
    colors = ['mediumblue', 'orange', 'g', 'r',
              'rebeccapurple', 'saddlebrown', 'olive']
    fg = plot_potential_decomposition(
        data, y_lims, ls_arr, marker_arr, fillstyle_arr, colors, image_path, image_name)

    colors = ['mediumblue', 'orange', 'g', 'r',
              'rebeccapurple', 'saddlebrown', 'olive']
    color_num = 0
    for term in fit_coefs:
        seaborn.lineplot(data=data_fits, x='r/a', y='aV(r)_' +
                         term, color=colors[color_num])
        color_num += 1
    for term in to_fit:
        seaborn.lineplot(data=data_fits, x='r/a', y='aV(r)_' +
                         term, color=colors[color_num])
        color_num += 1

    plt.show()
    plots.save_image(image_path, image_name, fg)


def plot_potential_fitted_single(data, y_lims, term, image_path, image_name):
    fg = seaborn.FacetGrid(data=data, height=5, aspect=1.4, legend_out=False)
    map = fg.map(plt.errorbar, 'r/a', 'aV(r)_' + term, 'err_' +
                 term, ms=8, capsize=8, lw=0.5).add_legend(title='potential')
    fg.ax.set_xlabel(r"R$/r_{0}$")
    fg.ax.set_ylabel(r"$r_{0}V(R)$")
    fg.ax.spines['right'].set_visible(True)
    fg.ax.spines['top'].set_visible(True)
    fg.ax.minorticks_on()
    fg.ax.tick_params(which='both', bottom=True,
                      top=True, left=True, right=True)
    fg.ax.set_ylim(y_lims[0], y_lims[1])

    return fg


def fit_single(data, term, fit_range):
    data = data[(data['r/a'] >= fit_range[0]) &
                (data['r/a'] <= fit_range[1])].reset_index()
    x = data['r/a'].to_numpy(dtype=np.float64)
    y = data['aV(r)_' + term]
    y_err = data['err_' + term]
    popt, pcov = curve_fit(func_quark_potential, x, y, sigma=y_err)
    return popt, pcov


def potential_fit_single(path, term, fit_range, r0, y_lims, image_path, image_name):
    data = potential_data.read_data_single(path)
    if 'T' in path:
        data = data[data['T'] == path['T']].reset_index(
        )[['r/a', 'aV(r)_' + path['name'], 'err_' + path['name']]]
    else:
        data = get_potential_fit(data, func_exponent, (3, 14), path['name'])

    popt, pcov = fit_single(data, term, fit_range)
    perr = np.sqrt(np.diag(pcov))
    x = data['r/a'].to_numpy(dtype=np.float64)
    y = data['aV(r)_' + term]
    chi_sq = chi_square(x, y, popt[0], popt[1], popt[2])
    print('aV(r)_' + term, popt[0], perr[0],
          popt[1], perr[1], popt[2],
          perr[2], 'chi_sq =', chi_sq)
    x_fit = np.arange(data['r/a'].min(), data['r/a'].max(), 0.01)
    y_fit = func_quark_potential(x_fit, *popt)

    data_fits = pd.DataFrame(np.array([x_fit, y_fit]).T, columns=[
                             'r/a', f'aV(r)_' + term])

    fg = plot_potential_fitted_single(
        data, y_lims, term, image_path, image_name)
    seaborn.lineplot(data=data_fits, x='r/a', y='aV(r)_' + term)

    plt.show()
    plots.save_image(image_path, image_name, fg)


def potentials_fit(paths, term, fit_range, r0, y_lims, image_path, image_name):
    for path in paths:
        potential_fit_single(path, term, fit_range, r0,
                             y_lims, image_path, image_name)
