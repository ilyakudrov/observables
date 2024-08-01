import matplotlib.pyplot as plt
import pandas as pd
import os
from random import randrange
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
from scipy.stats import chi2
import warnings
import sys
import multiprocessing

sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(''))))
import common.plots as plots
import potential_data
import fit

warnings.simplefilter("ignore", OptimizeWarning)
warnings.simplefilter("ignore", RuntimeWarning)

def plot_T_fit(df, df_pot, decomposition_type, image_path):
    r = df.name[0]
    smearing_step = df.name[1]
    fg = plots.make_plot(df[(df['r/a'] == r) & (df['smearing_step'] == smearing_step)], 'T', 'aV(r)', 'range', 'T', 'aV(r)', f'potential {decomposition_type} smearing_step={smearing_step}', f'{image_path}/r={r}', \
                         f'potential_monopole_smearing_step={smearing_step}', False)
    df1 = df_pot[(df_pot['r/a'] == r) & (df_pot['smearing_step'] == smearing_step)]
    fg.ax.errorbar(df1['T'], df1['aV(r)'], df1['err'], mfc=None, fmt='o', ms=6, capsize=5, lw=0.5, ls=None)
    plots.save_image(f'{image_path}/r={r}', f'potential_monopole_smearing_step={smearing_step}', fg)
    plt.close()

def potential_T_fit(lattice_size, L, beta, smearing, additional_params, decomposition_type, copy=None):
    paths = [{'path': f'../../result/smearing/potential/wilson_loop/fundamental/on-axis/su3/gluodynamics/{lattice_size}/beta{beta}/{smearing}/{additional_params}/potential_{decomposition_type}.csv',
            'parameters': {'beta': f'beta={beta}'}, 'constraints': {'r/a': (1, L // 2), 'T': (1, L // 2 - 1)}}]
    if copy is not None:
        paths[0]['constraints'] = {'copy': (copy, copy)}
    image_path = f'../../images/smearing/potential/su3/gluodynamics/T_fit/{lattice_size}/beta{beta}/{smearing}/{additional_params}/{decomposition_type}'
    print(paths[0]['path'])
    df = potential_data.get_potantial_df(paths, coluns_to_multiindex=['smearing_step'])
    # df_fit = df.groupby(df.index.names + ['r/a']).apply(fit.make_fit_range, fit.func_exponent, ['V', 'a', 'b'], 'T', 'aV(r)', 'err', L // 2 - 4).reset_index(level=-1, drop=True).reset_index(level=df.index.names + ['r/a'])
    # df_curve = df_fit.groupby(df.index.names + ['r/a', 'T_min', 'T_max']).apply(fit.make_fit_curve, fit.func_exponent, 'T', 'aV(r)', ['V', 'a', 'b']).reset_index(level=-1, drop=True).reset_index(level=df.index.names + ['r/a', 'T_min', 'T_max'])
    df_fit = df.groupby(df.index.names + ['r/a']).apply(fit.make_fit_range, fit.func_double_exponent, ['V', 'a', 'b', 'c', 'd'], 'T', 'aV(r)', 'err', L // 2 - 4).reset_index(level=-1, drop=True).reset_index(level=df.index.names + ['r/a'])
    df_curve = df_fit.groupby(df.index.names + ['r/a', 'T_min', 'T_max']).apply(fit.make_fit_curve, fit.func_double_exponent, 'T', 'aV(r)', ['V', 'a', 'b', 'c', 'd']).reset_index(level=-1, drop=True).reset_index(level=df.index.names + ['r/a', 'T_min', 'T_max'])
    df_curve['range'] = '(' + df_curve['T_min'].astype(str) + ', ' + df_curve['T_max'].astype(str) + ')'
    df = df.reset_index(level=['smearing_step'])
    df_curve.groupby(['r/a', 'smearing_step']).apply(plot_T_fit, df, decomposition_type, image_path)

def potential_T_fit_wrapping(args):
    potential_T_fit(*args)


# args = [['16^4', 16, '6.0', 'HYP0_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'monopole', 1],
#         ['24^4', 24, '6.0', 'HYP0_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'monopole', 1],
#         ['32^4', 32, '6.0', 'HYP0_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'monopole', 1],
#         ['28^4', 28, '6.1', 'HYP0_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'monopole', 1],
#         ['32^4', 32, '6.2', 'HYP0_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'monopole', 1],
#         ['36^4', 36, '6.3', 'HYP0_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'monopole', 1],
#         ['40^4', 40, '6.4', 'HYP0_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'monopole', 1],
#         ['16^4', 16, '6.0', 'HYP0_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'original'],
#         ['24^4', 24, '6.0', 'HYP0_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'original'],
#         ['32^4', 32, '6.0', 'HYP0_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'original'],
#         ['28^4', 28, '6.1', 'HYP0_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'original'],
#         ['32^4', 32, '6.2', 'HYP0_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'original'],
#         ['36^4', 36, '6.3', 'HYP0_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'original'],
#         ['40^4', 40, '6.4', 'HYP0_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'original'],
#         ['16^4', 16, '6.0', 'HYP1_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'original'],
#         ['24^4', 24, '6.0', 'HYP1_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'original'],
#         ['32^4', 32, '6.0', 'HYP1_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'original'],
#         ['28^4', 28, '6.1', 'HYP1_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'original'],
#         ['32^4', 32, '6.2', 'HYP1_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'original'],
#         ['36^4', 36, '6.3', 'HYP1_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'original'],
#         ['40^4', 40, '6.4', 'HYP1_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'original']]
args = [['24^4', 24, '6.0', 'HYP0_alpha=1_1_0.5_APE_alpha=0.3', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP0_alpha=1_1_0.5_APE_alpha=0.4', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP0_alpha=1_1_0.5_APE_alpha=0.5', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP0_alpha=1_1_0.5_APE_alpha=0.6', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP0_alpha=1_1_0.5_APE_alpha=0.7', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP0_alpha=1_1_0.5_APE_alpha=0.8', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP0_alpha=1_1_0.5_APE_alpha=0.9', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP0_alpha=1_1_0.5_APE_alpha=1', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP1_alpha=1_1_0.5_APE_alpha=0.3', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP1_alpha=1_1_0.5_APE_alpha=0.4', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP1_alpha=1_1_0.5_APE_alpha=0.5', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP1_alpha=1_1_0.5_APE_alpha=0.6', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP1_alpha=1_1_0.5_APE_alpha=0.7', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP1_alpha=1_1_0.5_APE_alpha=0.8', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP1_alpha=1_1_0.5_APE_alpha=0.9', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP1_alpha=1_1_0.5_APE_alpha=1', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP3_alpha=1_1_0.5_APE_alpha=0.3', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP3_alpha=1_1_0.5_APE_alpha=0.4', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP3_alpha=1_1_0.5_APE_alpha=0.5', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP3_alpha=1_1_0.5_APE_alpha=0.6', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP3_alpha=1_1_0.5_APE_alpha=0.7', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP3_alpha=1_1_0.5_APE_alpha=0.8', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP3_alpha=1_1_0.5_APE_alpha=0.9', 'steps_500/copies=4', 'monopole'],
        ['24^4', 24, '6.0', 'HYP3_alpha=1_1_0.5_APE_alpha=1', 'steps_500/copies=4', 'monopole']]
pool = multiprocessing.Pool(4)
pool.map(potential_T_fit_wrapping, args)

# potential_T_fit('16^4', 16, '6.0', 'HYP0_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'monopole')
# potential_T_fit('24^4', 24, '6.0', 'HYP0_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'monopole')
# potential_T_fit('32^4', 32, '6.0', 'HYP0_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'monopole')
# potential_T_fit('28^4', 28, '6.1', 'HYP0_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'monopole')
# potential_T_fit('32^4', 32, '6.2', 'HYP0_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'monopole')
# potential_T_fit('36^4', 36, '6.3', 'HYP0_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'monopole')
# potential_T_fit('40^4', 40, '6.4', 'HYP0_alpha=1_1_0.5_APE_alpha=0.6', 'steps_0/copies=20', 'monopole')