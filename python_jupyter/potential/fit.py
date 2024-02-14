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
    print(data)
    data = data[(data['r/a'] >= fit_range[0]) & (data['r/a'] <= fit_range[1])]
    y = data['aV(r)_' + fit_name]
    y_err = data['err_' + fit_name]
    x = data['r/a'].to_numpy(dtype=np.float64)
    print('x', x)
    print('y', y)
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
        chi_sq = chi_square(x, y, popt, func_quark_potential)
        # print('aV(r)_' + term, popt[0] / r0, perr[0] / r0,
        #                         popt[1], perr[1], popt[2] / r0**2,
        #                         perr[2] / r0**2, 'chi_sq =', chi_sq)
        data_fits.append(func_quark_potential(x_fit, *popt))
        columns.append(f'aV(r)_' + term)
    return pd.DataFrame(np.array(data_fits).T, columns=columns), fit_params


def chi_square(x, y, popt, func_fit):
    chi_sq = 0
    for i in range(len(x)):
        expected = func_fit(x[i], *popt)
        chi_sq += (expected - y[i])**2 / expected
    return chi_sq


def fit_single(data, fit_range, fit_func):
    data = data[(data['r/a'] >= fit_range[0]) &
                (data['r/a'] <= fit_range[1])].reset_index()
    x = data['r/a'].to_numpy(dtype=np.float64)
    y = data['aV(r)']
    y_err = data['err']
    popt, pcov = curve_fit(fit_func, x, y, sigma=y_err)
    return popt, pcov

def make_fit(data, fit_range, fit_func, param_names, x_col, y_col, err_col=None):
    data = data[(data[x_col] >= fit_range[0]) &
                (data[x_col] <= fit_range[1])].reset_index()
    x = data[x_col].to_numpy(dtype=np.float64)
    y = data[y_col]
    if err_col is not None:
        y_err = data[err_col]
    else:
        y_err = None
    popt, pcov = curve_fit(fit_func, x, y, sigma=y_err)
    chi_sq = chi_square(x, y, popt, fit_func)
    fits = {}
    err_diag = np.sqrt(np.diag(pcov))
    for i in range(len(popt)):
        fits[param_names[i]] = [popt[i]]
        fits[param_names[i] + '_err'] = [err_diag[i]]
    fits['chi_sq'] = chi_sq
    return pd.DataFrame(data=fits)

def potential_fit_data(data, fit_range, fit_func, param_names, x_col, y_col, err_col=None):
    """takes DataFrame with potential data to fit and returns Dataframe with x and y of fit curve"""
    fit_params = make_fit(data, fit_range, fit_func, param_names, x_col, y_col, err_col=err_col)
    fit_params = fit_params.iloc[0][param_names].values
    x_max = data[x_col].max()
    x_min = data[x_col].min()
    x = np.linspace(x_min, x_max, 1000)
    y = fit_func(x, *fit_params)
    return pd.DataFrame({x_col: x, y_col: y})

def make_fit_curve(fit_params, fit_func, x_min, x_max, x_col, y_col, param_names):
    fit_params = fit_params.iloc[0][param_names].values
    x = np.linspace(x_min, x_max, 1000)
    y = fit_func(x, *fit_params)
    return pd.DataFrame({x_col: x, y_col: y})

def potential_fit_T(data, fit_range):
    data = data[(data['T'] >= fit_range[0]) &
                (data['T'] <= fit_range[1])].reset_index()
    x = data['T'].to_numpy(dtype=np.float64)
    y = data['aV(r)']
    y_err = data['err']
    popt, pcov = curve_fit(func_exponent, x, y, sigma=y_err)
    return pd.DataFrame({'T': [None], 'aV(r)': [popt[0]], 'err': [np.sqrt(np.diag(pcov))[0]]})

def fit_curve_shift(data_potential, fit_parameters, fit_func):
    """finds shift constant so that fit curve fits data_potential
    with other parameters fixed"""
    x = data_potential['r/a'].to_numpy(dtype=np.float64)
    y = data_potential['aV(r)'].to_numpy()
    popt, pcov = curve_fit(lambda x, c: fit_func(x, c, *fit_parameters), x, y)
    return popt[0], np.sqrt(np.diag(pcov))[0]

def fit_potentials(data, potentials):
    """Fits potentials and returns fitting DataFrame with fitting curves
    Args:
        data: data for potentials
        potentials: dictionary of dictionaries with fitting info
    """
    df_fits = []
    for key, value in potentials.items():
        df = potential_fit_data(data.loc[data['potential_type'] == key, ['r/a','aV(r)', 'err']], value['fit_range'], value['fit_func'], value['fit_parameters'], 'r/a', 'aV(r)', 'err')
        df['potential_type'] = key
        df_fits.append(df)
    return pd.concat(df_fits)

def fit_from_original(data, potential_type, fit_func, fit_parameters):
    """fits potential of potential_type with parameters from original potential and shifts fit to it"""
    c, c_err = fit_curve_shift(data.loc[data['potential_type'] == potential_type, ['r/a','aV(r)', 'err']], fit_parameters, fit_func)
    fit_shifted = make_fit_curve(pd.DataFrame({'V0': [c], 'sigma': fit_parameters}), fit_func, data.loc[data['potential_type'] == potential_type, 'r/a'].min(),
                                 data.loc[data['potential_type'] == potential_type, 'r/a'].max(), 'r/a', 'aV(r)', ['V0', 'sigma'])
    fit_shifted['potential_type'] = potential_type
    return fit_shifted