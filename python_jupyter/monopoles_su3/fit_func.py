import numpy as np
from scipy.optimize import curve_fit

# fit functions


def exponent_polinomial_5_2(x, a, mu):
    return a * np.exp(-mu * x) * np.power(x, -5 / 2)


def exponent_polinomial(x, a, mu, b):
    return a * np.exp(-mu * x) * np.power(x, -b)


def inverse_polinomial_5_2(x, a):
    return a * np.power(x, -5 / 2)


def inverse_polinomial(x, a, b):
    return a * np.power(x, -b)
