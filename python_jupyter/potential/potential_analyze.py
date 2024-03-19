import matplotlib.pyplot as plt
import seaborn
import math
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

import plots
import scaler
import fit

def shift_fit(df1, df2, fit_range1, fit_range2, fit_func):
        """potential from df1 gets shifted towards potential from df2
            so that fit shift constants coincide coinside. Shift constant
            in fit function must be the first"""
        popt1, pcov1 = fit.fit_single(df1, fit_range1, fit_func)
        popt2, pcov2 = fit.fit_single(df2, fit_range2, fit_func)
        df1['aV(r)'] = df1['aV(r)'] + popt2[0] - popt1[0]
        return df1
