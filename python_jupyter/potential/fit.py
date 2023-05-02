import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import math


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
        print(f'potential {fit_name} fit did not converge at r =', r)
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
    return popt, pcov


def fit_consts(data, coloumb_name, string_name, alpha, sigma):
    y_coloumb = data['aV(r)_' + coloumb_name]
    y_string = data['aV(r)_' + string_name]
    x = data['r/a'].to_numpy(dtype=np.float64)
    c_coloumb, err_coloumb = curve_fit(
        lambda x, c: c + alpha * np.power(x, -1), x, y_coloumb)
    c_string, err_string = curve_fit(lambda x, c: c + sigma * x, x, y_string)
    return c_coloumb, math.sqrt(err_coloumb[0][0]), c_string, math.sqrt(err_string[0][0])


def make_fit_original(data, orig_pot_name, coloumb_name, string_name, fit_range):
    print(data)
    popt, pcov = fit_string(data, fit_range, orig_pot_name)
    perr = np.sqrt(np.diag(pcov))
    c_coloumb, err_coloumb, c_string, err_string = fit_consts(
        data, coloumb_name, string_name, popt[1], popt[2])
    x_fit = np.arange(data['r/a'].min(), data['r/a'].max(), 0.01)
    y_coloumb = func_coloumb(x_fit, c_coloumb, popt[1])
    y_string = func_linear(x_fit, c_string, popt[2])
    y_original = func_quark_potential(x_fit, *popt)
    return pd.DataFrame(np.array([x_fit, y_coloumb, y_string, y_original]).T,
                        columns=['r/a', 'aV(r)_' + coloumb_name,
                                 'aV(r)_' + string_name, 'aV(r)_' + orig_pot_name]), \
        popt, perr, c_coloumb, err_coloumb, c_string, err_string


def make_fit_separate(data, terms, fit_range):
    x_fit = np.arange(data['r/a'].min(), data['r/a'].max(), 0.01)
    data = data[(data['r/a'] >= fit_range[0]) &
                (data['r/a'] <= fit_range[1])].reset_index()
    data_fits = []
    columns = []
    x = data['r/a'].to_numpy(dtype=np.float64)
    data_fits.append(x_fit)
    columns.append('r/a')
    fit_params = {}
    for term in terms:
        y = data['aV(r)_' + term]
        y_err = data['err_' + term]
        popt, pcov = curve_fit(func_quark_potential, x, y, sigma=y_err)
        perr = np.sqrt(np.diag(pcov))
        fit_params['aV(r)_' + term] = popt
        fit_params['err_' + term] = perr
        chi_sq = chi_square(x, y, popt[0], popt[1], popt[2])
        # print('aV(r)_' + term, popt[0] / r0, perr[0] / r0,
        #                         popt[1], perr[1], popt[2] / r0**2,
        #                         perr[2] / r0**2, 'chi_sq =', chi_sq)
        data_fits.append(func_quark_potential(x_fit, *popt))
        columns.append(f'aV(r)_' + term)
    return pd.DataFrame(np.array(data_fits).T, columns=columns), fit_params


def chi_square(x, y, c, alpha, sigma):
    chi_sq = 0
    for i in range(len(x)):
        expected = func_quark_potential(x[i], c, alpha, sigma)
        chi_sq += (expected - y[i])**2 / expected
    return chi_sq


def fit_single(data, term, fit_range):
    data = data[(data['r/a'] >= fit_range[0]) &
                (data['r/a'] <= fit_range[1])].reset_index()
    x = data['r/a'].to_numpy(dtype=np.float64)
    y = data['aV(r)_' + term]
    y_err = data['err_' + term]
    popt, pcov = curve_fit(func_quark_potential, x, y, sigma=y_err)
    return popt, pcov
