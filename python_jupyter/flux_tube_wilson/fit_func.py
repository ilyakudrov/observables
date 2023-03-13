import pandas as pd
from scipy.optimize import curve_fit


def inverse_c(x, c, b):
    return c + b / x


def inverse(x, b):
    return b / x


def fit_data(data, func):
    popt, pcov = curve_fit(
        func, data['R'].to_numpy(), data['field'].to_numpy())
    return pd.Series({'params': popt})
