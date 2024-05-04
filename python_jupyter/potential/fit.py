import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import math


def func_exponent(x, a, b, c):
    return a + b * np.exp(-x * c)


def func_cornell(x, c, alpha, sigma):
    return c + alpha * np.power(x, -1) + sigma * x


def func_coloumb(x, c, alpha):
    return c + alpha * np.power(x, -1)


def func_linear(x, c, sigma):
    return c + sigma * x

def chi_square(x, y, popt, func_fit):
    """
    Calculate chi_square of a fit with func_fit
    """
    y_exp = func_fit(x, *popt)
    return np.sum((y - y_exp) ** 2 / y_exp)

def chi_square_reduced(x, y, sigma, popt, func_fit):
    """
    Calculate reduced chi_square of a fit with func_fit
    """
    y_exp = func_fit(x, *popt)
    return np.sum(((y - y_exp)/sigma) ** 2)

def make_fit(data, fit_range, fit_func, param_names, x_col, y_col, err_col=None):
    """Fits data from x_col and y_col with fit functions
    and returns DataFrame with fit parameters"""
    data1 = data[(data[x_col] >= fit_range[0]) &
                (data[x_col] <= fit_range[1])]
    if err_col is not None:
        y_err = data1[err_col]
    else:
        y_err = None
    try:
        popt, pcov, *output = curve_fit(fit_func, data1[x_col], data1[y_col], sigma=y_err, absolute_sigma=True, full_output=True)
    except RuntimeError:
        return pd.DataFrame()
    fits = {}
    err_diag = np.sqrt(np.diag(pcov))
    for i in range(len(popt)):
        fits[param_names[i]] = [popt[i]]
        fits[param_names[i] + '_err'] = [err_diag[i]]
    dof = data1[x_col].size - len(param_names) - 1
    fits['chi_square'] = np.sum(output[0]['fvec']**2)/dof
    return pd.DataFrame(data=fits)

def potential_fit_data(data, fit_range, fit_func, param_names, x_col, y_col, err_col=None):
    """Takes DataFrame with potential data to fit and returns Dataframe with x and y of fit curve"""
    fit_df = make_fit(data, fit_range, fit_func, param_names, x_col, y_col, err_col=err_col)
    fit_params = fit_df.iloc[0][param_names].values
    x_max = data[x_col].max()
    x_min = data[x_col].min()
    x = np.linspace(x_min, x_max, 1000)
    y = fit_func(x, *fit_params)
    return pd.DataFrame({x_col: x, y_col: y, 'chi_square': fit_df.at[0, 'chi_sq']})

def make_fit_curve(fit_params, fit_func, x_range, x_col, y_col, param_names):
    """Takes fit parameters and makes fit curve"""
    fit_params = fit_params.iloc[0][param_names].values
    x = np.linspace(*x_range, 1000)
    y = fit_func(x, *fit_params)
    return pd.DataFrame({x_col: x, y_col: y})

def potential_fit_T(data, fit_range):
    data1 = data[(data['T'] >= fit_range[0]) &
                (data['T'] <= fit_range[1])]
    popt, pcov = curve_fit(func_exponent, data1['T'], data1['aV(r)'], sigma=data1['err'], absolute_sigma=True)
    return pd.DataFrame({'T': [None], 'aV(r)': [popt[0]], 'err': [np.sqrt(np.diag(pcov))[0]]})

def generate_ranges(min, max):
    ranges = []
    for i in range(min, max-3):
        for j in range(i + 3, max-1):
            ranges.append((i, j))
    return ranges

def potential_fit_T_range(data, fit_range):
    ranges = generate_ranges(*fit_range)
    fits = []
    for i in range(len(ranges)):
        try:
            df_tmp = potential_fit_T(data, ranges[i])
        except:
            pass
        else:
            fits.append(df_tmp)
            fits[-1]['range'] = i
    fits = pd.concat(fits)
    if fits.empty:
        return pd.DataFrame({'T': [None], 'aV(r)': [0], 'err': [0]})
    fits = fits[fits['err'] != math.inf]
    popt, pcov = curve_fit(lambda x, c: c, fits['range'], fits['aV(r)'], sigma=fits['err'], absolute_sigma=True)
    return pd.DataFrame({'T': [None], 'aV(r)': [popt[0]], 'err': [np.sqrt(np.diag(pcov))[0]]})

def potential_fit_T_range_best(data, fit_range):
    ranges = generate_ranges(*fit_range)
    fits = []
    for i in range(len(ranges)):
        try:
            df_tmp = potential_fit_T(data, ranges[i])
        except:
            pass
        else:
            fits.append(df_tmp)
            fits[-1]['range'] = i
    if not fits:
        return pd.DataFrame({'T': [None], 'aV(r)': [0], 'err': [0]})
    fits = pd.concat(fits)
    fits = fits[fits['err'] != math.inf]
    fits = fits.sort_values(by='chi_square').reset_index()
    return pd.DataFrame({'T': [None], 'aV(r)': [fits.at[0, 'aV(r)']], 'err': [fits.at[0, 'err']]})

def fit_curve_shift(data_potential, fit_parameters, fit_func):
    """finds shift constant so that fit curve fits data_potential
    with other parameters fixed"""
    x = data_potential['r/a'].to_numpy(dtype=np.float64)
    y = data_potential['aV(r)'].to_numpy()
    popt, pcov = curve_fit(lambda x, c: fit_func(x, c, *fit_parameters), x, y, absolute_sigma=True)
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
    c, c_err = fit_curve_shift(data.loc[data['potential_type'] == potential_type, ['r/a','aV(r)', 'err']], fit_parameters, fit_func, absolute_sigma=True)
    fit_shifted = make_fit_curve(pd.DataFrame({'V0': [c], 'sigma': fit_parameters}), fit_func, data.loc[data['potential_type'] == potential_type, 'r/a'].min(),
                                 data.loc[data['potential_type'] == potential_type, 'r/a'].max(), 'r/a', 'aV(r)', ['V0', 'sigma'])
    fit_shifted['potential_type'] = potential_type
    return fit_shifted