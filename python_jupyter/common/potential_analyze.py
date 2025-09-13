import matplotlib.pyplot as plt
import seaborn
import math
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

import common.plots
import scaler
import common.fit as fit

def shift_fit(df1, df2, fit_range1, fit_range2, fit_func):
        """potential from df1 gets shifted towards potential from df2
            so that fit shift constants coincide coinside. Shift constant
            in fit function must be the first"""
        popt1, pcov1 = fit.fit_single(df1, fit_range1, fit_func)
        popt2, pcov2 = fit.fit_single(df2, fit_range2, fit_func)
        df1['aV(r)'] = df1['aV(r)'] + popt2[0] - popt1[0]
        return df1

def slice_smearing(df, df_smearing):
    df = df.reset_index(level='smearing_step')
    df_smearing = df_smearing.set_index('r/a')
    for r in df['r/a'].unique():
        df = df[~((df['r/a'] == r) & ~(df['smearing_step'] == df_smearing.loc[r, 'smearing_step']))]
    df['smearing_step'] = 'custom'
    return df.set_index('smearing_step', append=True)

