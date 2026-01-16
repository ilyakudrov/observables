import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2
import pandas as pd
import math


def func_exponent(x, a, b, c):
    return a + b * np.exp(-x * c)

def func_double_exponent(x, a, b, c, d, e):
    return a + b * np.exp(-x * c) + d * np.exp(-x * e)

def func_cornell(x, c, alpha, sigma):
    return c + alpha / x + sigma * x

def func_cornell_force(x, alpha, sigma):
    return - alpha / x**2 + sigma

def func_screened(x, c, alpha, mass):
    return c + alpha / x * math.exp(-mass * x)

def func_coloumb(x, c, alpha):
    return c + alpha / x

def func_linear(x, c, sigma):
    return c + sigma * x

def func_constant(x, c):
    return c

def func_polynomial3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

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
    else:
        fits = {}
        err_diag = np.sqrt(np.diag(pcov))
        for i in range(len(popt)):
            fits[param_names[i]] = [popt[i]]
            fits[param_names[i] + '_err'] = [err_diag[i]]
        dof = data1[x_col].size - len(param_names) - 1
        fits['chi_square'] = np.sum(output[0]['fvec']**2)/dof
        fits['p_value'] = chi2.sf(fits['chi_square'], dof)
        df = pd.DataFrame(data=fits)
        df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
        return df

def average_fit_p_value(df_fit, param_names, x_col):
    data = {}
    for name in param_names:
        aver = (df_fit[name] * df_fit[f'w_{name}']).sum()
        err_stat = (df_fit[f'{name}_err'] ** 2 * df_fit[f'w_{name}']).sum()
        df_fit[f'w_{name}'] * (df_fit[name] - aver) ** 2
        err_syst = (df_fit[f'w_{name}'] * (df_fit[name] - aver) ** 2).sum()
        data[name] = [aver]
        data[f'{name}_err'] = [np.sqrt(err_syst + err_stat)]
    data[f'{x_col}_min'] = df_fit[f'{x_col}_min'].min()
    data[f'{x_col}_max'] = df_fit[f'{x_col}_max'].max()
    return pd.DataFrame(data)

def make_fit_range(df, fit_func, param_names, x_col, y_col, err_col, range_min_len=None):
    if range_min_len is None:
        range_min_len = df.reset_index(level='range_min_r').reset_index(drop=True).loc[0, 'range_min_r']
    fit_range = (df[x_col].min(), df[x_col].max())
    ranges = generate_ranges(*fit_range, range_min_len)
    fit_df = []
    for r in ranges:
        fit_df.append(make_fit(df, r, fit_func, param_names, x_col, y_col, err_col=err_col))
        fit_df[-1][f'{x_col}_min'] = r[0]
        fit_df[-1][f'{x_col}_max'] = r[1]
    fit_df = pd.concat(fit_df)
    fit_df = fit_df[~fit_df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
    if fit_df.empty:
        return pd.DataFrame()
    for name in param_names:
        fit_df[f'w_{name}'] = fit_df['p_value'] / fit_df[f'{name}_err'] ** 2
        norm = fit_df[f'w_{name}'].sum()
        fit_df[f'w_{name}'] = fit_df[f'w_{name}'] / norm
    return fit_df

def make_fit_curve(fit_params, fit_func, x_col, y_col, param_names):
    """Takes fit parameters and makes fit curve"""
    x_min = fit_params.reset_index().loc[0, f'{x_col}_min']
    x_max = fit_params.reset_index().loc[0, f'{x_col}_max']
    fit_params = fit_params.iloc[0][param_names].values
    x = np.linspace(x_min, x_max, 1000)
    y = fit_func(x, *fit_params)
    return pd.DataFrame({x_col: x, y_col: y})

def generate_ranges(min, max, range_min_len):
    ranges = []
    for i in range(min, max-range_min_len+2):
        for j in range(i + range_min_len - 1, max+1):
            ranges.append((i, j))
    return ranges

def potential_fit_T_range(df, fit_func, param_list, range_min_len=None):
    if range_min_len is None:
        range_min_len = df.reset_index(level='range_min_T').reset_index(drop=True).loc[0, 'range_min_T']
    # df_fit = make_fit_range(df, func_exponent, ['V', 'a', 'b'], 'T', 'aV(r)', 'err', range_min_len)
    df_fit = make_fit_range(df, fit_func, param_list, 'T', 'aV(r)', 'err', range_min_len)
    # df_fit = make_fit_range(df, func_constant, ['V'], 'T', 'aV(r)', 'err', range_min_len)
    if df_fit.empty:
        return pd.DataFrame()
    V_aver = (df_fit['V'] * df_fit['w_V']).sum()
    dV_stat = (df_fit['V_err'] ** 2 * df_fit['w_V']).sum()
    dV_syst = (df_fit['w_V'] * (df_fit['V'] - V_aver) ** 2).sum()
    return pd.DataFrame({'T': [None], 'aV(r)': [V_aver], 'err': [math.sqrt(dV_syst + dV_stat)]})

def fit_curve_shift(df, fit_parameters, x_col, y_col, err_col, fit_func):
    """finds shift constant so that fit curve fits potential from df
    with other parameters fixed"""
    df_fit = make_fit_range(df, lambda x, c: fit_func(x, *fit_parameters) + c, 6, ['c'], 'r/a', 'aV(r)', err_col='err').reset_index(level=-1, drop=True)
    df_fit = average_fit_p_value(df_fit, ['c'], 'r/a').reset_index(level=-1, drop=True)
    df = df.reset_index(drop=True)
    return df_fit.at[0, 'c'], df_fit.at[0, 'c_err']

def fit_from_original(data, potential_type, fit_func, fit_parameters):
    """fits potential of potential_type with parameters from original potential and shifts fit to it"""
    c, c_err = fit_curve_shift(data.loc[data['potential_type'] == potential_type, ['r/a','aV(r)', 'err']], fit_parameters, fit_func, absolute_sigma=True)
    fit_shifted = make_fit_curve(pd.DataFrame({'V0': [c], 'sigma': fit_parameters}), fit_func, data.loc[data['potential_type'] == potential_type, 'r/a'].min(),
                                 data.loc[data['potential_type'] == potential_type, 'r/a'].max(), 'r/a', 'aV(r)', ['V0', 'sigma'])
    fit_shifted['potential_type'] = potential_type
    return fit_shifted