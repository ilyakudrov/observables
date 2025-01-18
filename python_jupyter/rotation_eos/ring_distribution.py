import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import math
import seaborn
import numpy as np
import multiprocessing
import matplotlib
import palettable
from PIL import Image
matplotlib.use('Agg')

sys.path.append(os.path.abspath(os.path.join('..', '..', 'code', 'python')))
sys.path.append(os.path.abspath(os.path.join('..', '..', 'code', 'python', 'common')))
# import common.plots as plots
import scale_setters as scale_setters

def dbeta_da_derivative(beta):
    _mev_in_tension_units = 1 / 440
    _fm_in_MeV = 1 / 197.327
    def f(_beta: float) -> float:
        b0 = 11 / (4 * np.pi)**2
        b1 = 102 / (4 * np.pi)**4
        return ((6 * b0 / _beta) ** (-b1 / (2 * b0**2))
                * np.exp(-_beta / (6 * 2 * b0)))
    def f_deriv(_beta: float) -> float:
        b0 = 11 / (4 * np.pi)**2
        b1 = 102 / (4 * np.pi)**4
        return ((6 * b0) ** (-b1 / (2 * b0**2)) * (b1 / (2 * b0**2)) * _beta ** (b1 / (2 * b0**2) - 1)
                * np.exp(-_beta / (6 * 2 * b0)) + (6 * b0 / _beta) ** (-b1 / (2 * b0**2))
                * np.exp(-_beta / (6 * 2 * b0)) * (-1./(6 * 2 * b0)))
    frac = f(beta) / f(4.0)
    # return _mev_in_tension_units / _fm_in_MeV * f(beta) * (
    #         1 + 0.54850073 * frac**2 - 0.06597973 * frac**4
    #         + 0.02551499 * frac**6) / 0.07296355

    return _mev_in_tension_units / _fm_in_MeV * f_deriv(beta) * (
            1 + 0.54850073 * frac**2 - 0.06597973 * frac**4
            + 0.02551499 * frac**6) / 0.07296355 + _mev_in_tension_units / _fm_in_MeV * f(beta) * (
            1 + 0.54850073 * 2 * frac - 0.06597973 * 4 * frac**3
            + 0.02551499 * 6 * frac**5) / 0.07296355 * f_deriv(beta) / f(4.0)

def get_dir_names(path):
    directories = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        directories.extend(dirnames)
        break
    return directories

def concat_observables(df, observables, add_columns):
    data = []
    for obs in observables:
        data.append(df.loc[:, add_columns + [obs, f'{obs}_err']])
        # data[-1] = data[-1].rename({obs: 'y', f'{obs}_err': 'err'}, axis=1)
        # data[-1]['observable'] = obs
    return pd.concat(data)

def read_data(path, name, observables, add_columns, lattice, border):
    df = pd.DataFrame()
    velocity_dirs = get_dir_names(f'{path}/{lattice}/{border}')
    for velocity in velocity_dirs:
        beta_dirs = get_dir_names(f'{path}/{lattice}/{border}/{velocity}')
        for beta in beta_dirs:
            file_path = f'{path}/{lattice}/{border}/{velocity}/{beta}/{name}'
            # print(file_path)
            if os.path.isfile(file_path):
                df_tmp = pd.read_csv(file_path, delimiter=' ')
                df_tmp = concat_observables(df_tmp, observables, add_columns)
                df_tmp['velocity'] = float(velocity[:-1])
                df_tmp['beta'] = float(beta)
                df_tmp['border'] = border
                df_tmp['lattice'] = lattice[:-2]
                df = pd.concat([df, df_tmp])
    return df

def save_image(image_path, image_name, fg, format='png', save_black=False):
    try:
        os.makedirs(image_path)
    except:
        pass

    output_path = f'{image_path}/{image_name}.{format}'
    fg.savefig(output_path, dpi=800, facecolor='white', format=format)
    if save_black:
        Image.open(f'{output_path}.png').convert('L').save(f'{output_path}_bw.png')

def make_plot(data, x, y, hue, x_label, y_label, title, image_path, image_name, show_plot, err=None, df_fits=None, black_line_y=None, dashed_line_y=None, markers_different=False, color_palette='bright', save_figure=False, x_log=False, y_log=False):
    markers_arr = ['o', '^', 's', 'D', 'P', 'X', 'v', '*', '<', '>']
    # markers_arr = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
    if hue is not None:
        hues = data[hue].unique()
        n_colors = hues.shape[0]
        color_palette = seaborn.color_palette(palette=color_palette, n_colors=n_colors)
        potential_type_hue = dict(zip(data[hue].unique(), hues))
        color_palette = dict(zip(hues, color_palette))
        markers_hue = markers_arr[:len(hues)]
    else:
        color_palette = None
        markers_hue = None
    # hue_kws=dict(marker=markers_hue)
    fg_kws = {}
    if markers_different:
        fg_kws['hue_kws'] = dict(marker=markers_hue)
    fg = seaborn.FacetGrid(data=data, hue=hue, height=5,
                           aspect=1.33, legend_out=False, palette=color_palette, **fg_kws)
    if err is not None:
        fg.map(plt.errorbar, x, y, err, mfc=None, ms=5, capsize=5, lw=0.5, ls=''
           ).add_legend(fontsize=10)
    else:
        fg.map(plt.errorbar, x, y, mfc=None, ms=3, capsize=5, lw=0.5, ls=''
           ).add_legend(fontsize=10)
    if x_log:
        fg.ax.set_xscale('log')
    if y_log:
        fg.ax.set_yscale('log')
    # plt.rc('text', usetex=True)
    fg.ax.set_xlabel(x_label, fontsize=16)
    fg.ax.set_ylabel(y_label, fontsize=16)
    fg.ax.set_title(title)
    fg.ax.spines['right'].set_visible(True)
    fg.ax.spines['top'].set_visible(True)
    fg.ax.minorticks_on()
    fg.ax.tick_params(which='both', bottom=True,
                      top=True, left=True, right=True)
    # plt.rcParams.update({"text.usetex": True})
    # plt.grid(dash_capstyle='round')
    if black_line_y is not None:
        plt.axhline(y=black_line_y, color='k', linestyle='-')

    if dashed_line_y is not None:
        for coord in dashed_line_y:
            plt.axhline(y=coord, color='k', linestyle='--')

    if df_fits is not None:
        for key in df_fits[hue].unique():
            plt.plot(df_fits[df_fits[hue] == key][x], df_fits[df_fits[hue] == key][y],
                     color=color_palette[potential_type_hue[key]], linewidth=1)

    if show_plot:
        plt.show()
    if save_figure:
        save_image(f'{image_path}',
            f'{image_name}', fg)
    return fg

def get_hue_dict(hues, color_palette, markers_arr):
    temps = []
    borders = []
    for h in hues:
        temps.append(h.split(',')[0])
        # borders.append(h.split(',')[1])
    temps_unique = list(set(temps))
    # borders_unique = list(set(borders))
    color_dict = {}
    marker_dict = {}
    for i in range(len(temps_unique)):
        color_dict[temps_unique[i]] = color_palette[i]
        marker_dict[temps_unique[i]] = markers_arr[i]
    color_hue = {}
    marker_hue = {}
    for h in hues:
        color_hue[h] = color_dict[h.split(',')[0]]
        marker_hue[h] = marker_dict[h.split(',')[0]]
    return color_hue, marker_hue

def make_plot_temperature(data, x, y, hue, x_label, y_label, title, image_path, image_name, show_plot, err=None, df_fits=None, black_line_y=None, dashed_line_y=None, markers_different=False, color_palette='bright', save_figure=False, x_log=False, y_log=False):
    # markers_arr = ['o', '^', 's', 'D', 'P', 'X', 'v', '*', '<', '>']
    markers_arr = ['o', '^', 's', 'D', 'P', 'X', 'v', '*', 'o', '^', 's', 'D', 'P', 'X', 'v', '*']
    # markers_arr = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
    if hue is not None:
        hues = data[hue].unique()
        n_colors = hues.shape[0] // 2
        color_palette = seaborn.color_palette(palette=color_palette, n_colors=n_colors)
        color_hue, marker_hue = get_hue_dict(hues, color_palette, markers_arr)
        # potential_type_hue = dict(zip(data[hue].unique(), hues))
        # color_palette = dict(zip(hues, color_palette))
        markers_hue = markers_arr[:len(hues)]
    else:
        color_palette = None
        markers_hue = None
    #hue_kws=dict(marker=markers_hue)
    mfc_arr = [None, None, None, None, None, None, None, None, 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none']
    fg_kws = {}
    if markers_different:
        fg_kws['hue_kws'] = dict(marker=markers_hue, mfc=mfc_arr)
    fg = seaborn.FacetGrid(data=data, hue=hue, height=5,
                           aspect=1.33, legend_out=False, palette=color_hue, **fg_kws)
    if err is not None:
        fg.map(plt.errorbar, x, y, err, mfc='none', ms=5, capsize=5, lw=0.5, ls=''
           ).add_legend(fontsize=10, ncol=2)
    else:
        fg.map(plt.errorbar, x, y, mfc='none', ms=3, capsize=5, lw=0.5, ls=''
           ).add_legend(fontsize=10, ncol=2)
    if x_log:
        fg.ax.set_xscale('log')
    if y_log:
        fg.ax.set_yscale('log')
    # plt.rc('text', usetex=True)
    fg.ax.set_xlabel(x_label, fontsize=16)
    fg.ax.set_ylabel(y_label, fontsize=16)
    fg.ax.set_title(title)
    fg.ax.spines['right'].set_visible(True)
    fg.ax.spines['top'].set_visible(True)
    fg.ax.minorticks_on()
    fg.ax.tick_params(which='both', bottom=True,
                      top=True, left=True, right=True)
    # plt.rcParams.update({"text.usetex": True})
    # plt.grid(dash_capstyle='round')
    if black_line_y is not None:
        plt.axhline(y=black_line_y, color='k', linestyle='-')

    if dashed_line_y is not None:
        for coord in dashed_line_y:
            plt.axhline(y=coord, color='k', linestyle='--')

    if df_fits is not None:
        for key in df_fits[hue].unique():
            plt.plot(df_fits[df_fits[hue] == key][x], df_fits[df_fits[hue] == key][y],
                     color=color_palette[potential_type_hue[key]], linewidth=1)

    if show_plot:
        plt.show()
    if save_figure:
        save_image(f'{image_path}',
            f'{image_name}', fg, format='pdf')
    return fg

beta_critical = {'nt4o': 4.088,
                 'nt5o': 4.225,
                 'nt6o': 4.350,
                 'nt7o': 4.470,
                 'nt4p': 4.073,
                 'nt5p': 4.202,
                 'nt6p': 4.318,
                 'nt7p': 4.417,
                 'nt24o': 4.088,
                 'nt30o': 4.225,
                 'nt36o': 4.350,
                 'nt42o': 4.470,
                 }

beta_critical_aver = {'nt4o': (4.088 + 4.073)/2,
                 'nt5o': (4.225+4.202)/2,
                 'nt6o': (4.350+4.318)/2,
                 'nt7o': (4.470+4.417)/2,
                 'nt4p': (4.088 + 4.073)/2,
                 'nt5p': (4.225+4.202)/2,
                 'nt6p': (4.350+4.318)/2,
                 'nt7p': (4.470+4.417)/2,
                 'nt24o': (4.088 + 4.073)/2,
                 'nt30o': (4.225+4.202)/2,
                 'nt36o': (4.350+4.318)/2,
                 'nt42o': (4.470+4.417)/2,
                 }

def plot_distribution_boundary_cut(df, observable):
    T = df.name[0]
    velocity = df.name[1]
    thickness = df.name[2]
    # print(observable, T, velocity, thickness)
    make_plot(df, 'rad_aver', observable, 'hue', r'$\text{r/R}$', observable, f'distribution {observable} T={T:.4f} velocity={velocity:.7f} thickness={thickness}', f'../../images/eos_rotation_imaginary/distribution_boundary_cut_dependence/5x30x121sq/{observable}/T={T:.4f}', f'distribution_thickness={thickness}', False, err=f'{observable}_err', color_palette=palettable.cartocolors.qualitative.Prism_10.mpl_colors, save_figure=True, dashed_line_y=[0])
    plt.close()

# Dependence on boundary cut
def distribution_boundary(observable):
    R = 60
    path = '../../result/eos_rotation_imaginary'
    df_T = []
    df_0 = []
    df_T.append(read_data(path, 'observables_ring_result.csv', [observable], ['rad_aver', 'thickness', 'cut'], '5x30x121sq', 'OBCb_cV'))
    df_T[-1]['border'] = 'nt5o'
    df_T[-1]['boundary'] = 'OBCb_cV'
    df_0.append(read_data(path, 'observables_ring_result.csv', [observable], ['rad_aver', 'thickness', 'cut'], '30x30x121sq', 'OBCb_cV'))
    df_0[-1]['border'] = 'nt5o'
    df_0[-1]['boundary'] = 'OBCb_cV'
    df_T.append(read_data(path, 'observables_ring_result.csv', [observable], ['rad_aver', 'thickness', 'cut'], '5x30x121sq', 'PBC_cV'))
    df_T[-1]['border'] = 'nt5p'
    df_T[-1]['boundary'] = 'PBC_cV'
    df_0.append(read_data(path, 'observables_ring_result.csv', [observable], ['rad_aver', 'thickness', 'cut'], '30x30x121sq', 'PBC_cV'))
    df_0[-1]['border'] = 'nt5p'
    df_0[-1]['boundary'] = 'PBC_cV'
    df_T = pd.concat(df_T)
    df_0 = pd.concat(df_0)
    df_T['rad_aver'] = df_T['rad_aver'].apply(lambda x: round(x, 4))
    df_0['rad_aver'] = df_0['rad_aver'].apply(lambda x: round(x, 4))
    # df_T = df_T[df_T['beta'] == 3.860]
    # df_0 = df_0[df_0['beta'] == 3.860]
    df_T = df_T[df_T['velocity'].isin([0.692821, 0.489898])]
    df_0 = df_0[df_0['velocity'].isin([0.692821, 0.489898])]
    duplicate_cols = ['cut', 'thickness', 'rad_aver', 'border', 'boundary', 'beta', 'velocity']
    df_tmp = pd.concat([df_0[duplicate_cols], df_T[duplicate_cols]])
    df_tmp = df_tmp[df_tmp.duplicated(subset=duplicate_cols, keep='first')]
    df_0 = df_tmp.merge(df_0, how='left', on=duplicate_cols)
    df_T = df_tmp.merge(df_T, how='left', on=duplicate_cols)
    df_T[observable] = df_T[observable] - df_0[observable]
    df_T[f'{observable}_err'] = np.sqrt(df_T[f'{observable}_err'] ** 2 + df_0[f'{observable}_err'] ** 2)
    df_T = df_T[df_T['beta'] >= 3.85]
    fm_to_GeV = 1/0.197327
    scale_setter = scale_setters.ExtendedSymanzikScaleSetter()
    if observable in ['E', 'Elab', 'Blab']:
        df_T[observable] = -df_T[observable]
    df_T['a_GeV'] = scale_setter.get_spacing_in_fm(df_T['beta']) * fm_to_GeV
    df_T['beta_crit'] = df_T.apply(lambda x: beta_critical_aver[x['border']], axis=1)
    df_T['a_crit'] = scale_setter.get_spacing_in_fm(df_T['beta_crit']) * fm_to_GeV
    df_T['T'] = df_T['a_crit'] / df_T['a_GeV']
    df_T['rad_aver'] = df_T['rad_aver'] / R
    df_T['hue'] = df_T['boundary'] + ', cut = ' +  df_T['cut'].astype(str)
    df_T.set_index(['T', 'velocity', 'thickness']).groupby(['T', 'velocity', 'thickness']).apply(plot_distribution_boundary_cut, observable, include_groups=False)

def plot_distribution_thickness(df, observable):
    T = df.name[0]
    velocity = df.name[1]
    cut = df.name[2]
    # print(observable, T, velocity, cut)
    make_plot(df, 'rad_aver', observable, 'hue', r'$\text{r/R}$', observable, f'distribution {observable} T={T:.4f} velocity={velocity:.7f} cut={cut}', f'../../images/eos_rotation_imaginary/distribution_thickness_dependence/5x30x121sq/{observable}/T={T:.4f}', f'distribution_cut={cut}', False, err=f'{observable}_err', color_palette=palettable.cartocolors.qualitative.Prism_10.mpl_colors, save_figure=True, dashed_line_y=[0])
    plt.close()

# Dependence on ring thickness
def distribution_thickness(observable):
    R = 60
    path = '../../result/eos_rotation_imaginary'
    df_T = []
    df_0 = []
    df_T.append(read_data(path, 'observables_ring_result.csv', [observable], ['rad_aver', 'thickness', 'cut'], '5x30x121sq', 'OBCb_cV'))
    df_T[-1]['border'] = 'nt5o'
    df_T[-1]['boundary'] = 'OBCb_cV'
    df_0.append(read_data(path, 'observables_ring_result.csv', [observable], ['rad_aver', 'thickness', 'cut'], '30x30x121sq', 'OBCb_cV'))
    df_0[-1]['border'] = 'nt5o'
    df_0[-1]['boundary'] = 'OBCb_cV'
    df_T.append(read_data(path, 'observables_ring_result.csv', [observable], ['rad_aver', 'thickness', 'cut'], '5x30x121sq', 'PBC_cV'))
    df_T[-1]['border'] = 'nt5p'
    df_T[-1]['boundary'] = 'PBC_cV'
    df_0.append(read_data(path, 'observables_ring_result.csv', [observable], ['rad_aver', 'thickness', 'cut'], '30x30x121sq', 'PBC_cV'))
    df_0[-1]['border'] = 'nt5p'
    df_0[-1]['boundary'] = 'PBC_cV'
    df_T = pd.concat(df_T)
    df_0 = pd.concat(df_0)
    df_T['rad_aver'] = df_T['rad_aver'].apply(lambda x: round(x, 4))
    df_0['rad_aver'] = df_0['rad_aver'].apply(lambda x: round(x, 4))
    # df_T = df_T[df_T['beta'] == 4.4400]
    # df_0 = df_0[df_0['beta'] == 4.4400]
    df_T = df_T[df_T['velocity'].isin([0.692821, 0.489898])]
    df_0 = df_0[df_0['velocity'].isin([0.692821, 0.489898])]
    duplicate_cols = ['cut', 'thickness', 'rad_aver', 'border', 'boundary', 'beta', 'velocity']
    df_tmp = pd.concat([df_0[duplicate_cols], df_T[duplicate_cols]])
    df_tmp = df_tmp[df_tmp.duplicated(subset=duplicate_cols, keep='first')]
    df_0 = df_tmp.merge(df_0, how='left', on=duplicate_cols)
    df_T = df_tmp.merge(df_T, how='left', on=duplicate_cols)
    df_T[observable] = df_T[observable] - df_0[observable]
    df_T[f'{observable}_err'] = np.sqrt(df_T[f'{observable}_err'] ** 2 + df_0[f'{observable}_err'] ** 2)
    df_T = df_T[df_T['beta'] >= 3.85]
    fm_to_GeV = 1/0.197327
    scale_setter = scale_setters.ExtendedSymanzikScaleSetter()
    if observable in ['E', 'Elab', 'Blab']:
        df_T[observable] = -df_T[observable]
    df_T['a_GeV'] = scale_setter.get_spacing_in_fm(df_T['beta']) * fm_to_GeV
    df_T['beta_crit'] = df_T.apply(lambda x: beta_critical_aver[x['border']], axis=1)
    df_T['a_crit'] = scale_setter.get_spacing_in_fm(df_T['beta_crit']) * fm_to_GeV
    df_T['T'] = df_T['a_crit'] / df_T['a_GeV']
    df_T['rad_aver'] = df_T['rad_aver'] / R
    df_T['hue'] = df_T['boundary'] + ', thickness = ' +  df_T['thickness'].astype(str)
    df_T.set_index(['T', 'velocity', 'cut']).groupby(['T', 'velocity', 'cut']).apply(plot_distribution_thickness, observable, include_groups=False)

def plot_distribution_temperature(df, observable, observable_label):
    velocity = df.name[0]
    cut = df.name[1]
    thickness = df.name[2]
    make_plot_temperature(df, 'rad_aver', observable, r'$T/T_{c}$', 'r/R', observable_label, f'distribution {observable}',
              f'../../images/eos_rotation_imaginary/distribution_temperature_dependence/5x30x121sq/{observable}/velocity={velocity}',
              f'distribution_cut={cut}_thickness={thickness}', False, err=f'{observable}' + '_err',
              color_palette=palettable.cartocolors.qualitative.Prism_10.mpl_colors, save_figure=True, dashed_line_y=[0], markers_different=True)
    plt.close()

def rescale_observable(df, observable, Nt):
    if observable == 'Jv':
        observable_label = r'$j/RT^{4}$'
        df[observable] = - df[observable] * df['beta'] * Nt**4 / df['velocity']
        df[observable + '_err'] = df[observable + '_err'] * df['beta'] * Nt**4 / df['velocity']
    if observable == 'Jv1':
        observable_label = r'$j_{1}/RT^{4}$'
        df[observable] = - df[observable] * df['beta'] * Nt**4 / df['velocity']
        df[observable + '_err'] = df[observable + '_err'] * df['beta'] * Nt**4 / df['velocity']
    if observable == 'Jv2':
        observable_label = r'$j_{2}/RT^{4}$'
        df[observable] = - df[observable] * df['beta'] * Nt**4 / df['velocity']
        df[observable + '_err'] = df[observable + '_err'] * df['beta'] * Nt**4 / df['velocity']
    if observable == 'S':
        observable_label = r'$G^{2}/T^{4}$'
        df[observable] = - df[observable] * df['a'] * Nt**4 / df['beta_deriv']
        df[observable + '_err'] = np.abs(df[observable + '_err'] * df['a'] * Nt**4 / df['beta_deriv'])
    if observable == 'Blab':
        observable_label = r'$B_{lab}^{2}/T^{4}$'
        df[observable] = - df[observable] * df['a'] * Nt**4 / df['beta_deriv']
        df[observable + '_err'] = np.abs(df[observable + '_err'] * df['a'] * Nt**4 / df['beta_deriv'])
    if observable == 'Elab':
        observable_label = r'$E_{lab}^{2}/T^{4}$'
        df[observable] = - df[observable] * df['a'] * Nt**4 / df['beta_deriv']
        df[observable + '_err'] = np.abs(df[observable + '_err'] * df['a'] * Nt**4 / df['beta_deriv'])
    return df, observable_label

# Dependence of ring distribution on temperature
def distribution_temperature(observable):
    R = 60
    Nt = 5
    path = '../../result/eos_rotation_imaginary'
    df_T = []
    df_0 = []
    df_T.append(read_data(path, 'observables_ring_result.csv', [observable], ['rad_aver', 'thickness', 'cut'], '5x30x121sq', 'OBCb_cV'))
    df_T[-1]['border'] = 'nt5o'
    df_T[-1]['boundary'] = 'OBCb_cV'
    df_0.append(read_data(path, 'observables_ring_result.csv', [observable], ['rad_aver', 'thickness', 'cut'], '30x30x121sq', 'OBCb_cV'))
    df_0[-1]['border'] = 'nt5o'
    df_0[-1]['boundary'] = 'OBCb_cV'
    df_T.append(read_data(path, 'observables_ring_result.csv', [observable], ['rad_aver', 'thickness', 'cut'], '5x30x121sq', 'PBC_cV'))
    df_T[-1]['border'] = 'nt5p'
    df_T[-1]['boundary'] = 'PBC_cV'
    df_0.append(read_data(path, 'observables_ring_result.csv', [observable], ['rad_aver', 'thickness', 'cut'], '30x30x121sq', 'PBC_cV'))
    df_0[-1]['border'] = 'nt5p'
    df_0[-1]['boundary'] = 'PBC_cV'
    df_T = pd.concat(df_T)
    df_0 = pd.concat(df_0)
    df_T['rad_aver'] = df_T['rad_aver'].apply(lambda x: round(x, 4))
    df_0['rad_aver'] = df_0['rad_aver'].apply(lambda x: round(x, 4))
    df_T = df_T[df_T['cut'] == 2*Nt]
    df_0 = df_0[df_0['cut'] == 2*Nt]
    df_T = df_T[df_T['velocity'].isin([0.692821, 0.489898, 0.34641])]
    df_0 = df_0[df_0['velocity'].isin([0.692821, 0.489898, 0.34641])]
    # df_T = df_T[df_T['velocity'].isin([0.489898])]
    # df_0 = df_0[df_0['velocity'].isin([0.489898])]
    duplicate_cols = ['cut', 'thickness', 'rad_aver', 'border', 'beta', 'velocity']
    df_tmp = pd.concat([df_0[duplicate_cols], df_T[duplicate_cols]])
    df_tmp = df_tmp[df_tmp.duplicated(subset=duplicate_cols, keep='first')]
    df_0 = df_tmp.merge(df_0, how='left', on=duplicate_cols)
    df_T = df_tmp.merge(df_T, how='left', on=duplicate_cols)
    df_T[observable] = df_T[observable] - df_0[observable]
    df_T[f'{observable}_err'] = np.sqrt(df_T[f'{observable}_err'] ** 2 + df_0[f'{observable}_err'] ** 2)
    df_T = df_T[df_T['beta'] >= 3.85]
    df_T = df_T[df_T['beta'].isin([3.860, 4.020, 4.120, 4.192, 4.280, 4.440, 4.600, 4.760, 5.160])]
    # df_T = df_T[df_T['beta'].isin([4.440, 4.600, 4.760, 5.160])]
    fm_to_GeV = 1/0.197327
    scale_setter = scale_setters.ExtendedSymanzikScaleSetter()
    df_T['a_GeV'] = scale_setter.get_spacing_in_fm(df_T['beta']) * fm_to_GeV
    df_T['a'] = scale_setter.get_spacing_in_fm(df_T['beta'])
    df_T['beta_crit'] = df_T.apply(lambda x: beta_critical_aver[x['border']], axis=1)
    df_T['a_crit'] = scale_setter.get_spacing_in_fm(df_T['beta_crit']) * fm_to_GeV
    df_T['beta_deriv'] = dbeta_da_derivative(df_T['beta'])
    df_T['u'] = df_T['velocity'] * df_T['rad_aver'] / R
    df_T['T'] = df_T['a_crit'] / df_T['a_GeV']
    # df_T[r'$T/T_{c}$'] = df_T['T'].apply(lambda x: "%.2f" % x)
    df_T[r'$T/T_{c}$'] = df_T['T'].round(2)
    # print(df_T)
    df_T, observable_label = rescale_observable(df_T, observable, Nt)
    # print(df_T)
    df_T['rad_aver'] = df_T['rad_aver'] / R
    df_T = df_T[df_T['rad_aver'] <= 1]
    df_0 = read_data(path, 'observables_result.csv', [observable], ['box_size', 'radius'], '30x30x121sq', 'OBCb_cV')
    df_0 = df_0[df_0['beta'].isin([3.860, 4.020, 4.120, 4.192, 4.280, 4.440, 4.600, 4.760, 5.160])]
    df_0 = df_0[df_0['velocity'].isin([0])]
    df_0 = df_0[df_0['box_size'] == R]
    df_0 = df_0[df_0['radius'] == R * math.sqrt(2)]
    df_T0 = read_data(path, 'observables_result.csv', [observable], ['box_size', 'radius'], '5x30x121sq', 'OBCb_cV')
    df_T0 = df_T0[df_T0['beta'].isin([3.860, 4.020, 4.120, 4.192, 4.280, 4.440, 4.600, 4.760, 5.160])]
    df_T0 = df_T0[df_T0['velocity'].isin([0])]
    df_T0 = df_T0[df_T0['box_size'] == R]
    df_T0 = df_T0[df_T0['radius'] == R * math.sqrt(2)]
    df_0[observable] = df_T0[observable] - df_0[observable]
    df_0[f'{observable}_err'] = np.sqrt(df_T0[f'{observable}_err'] ** 2 + df_0[f'{observable}_err'] ** 2)
    df_0['border'] = 'nt30o'
    df_0['beta_crit'] = df_0.apply(lambda x: beta_critical_aver[x['border']], axis=1)
    df_0[r'$T/T_{c}$'] = (scale_setter.get_spacing_in_fm(df_0['beta_crit'])/scale_setter.get_spacing_in_fm(df_0['beta'])).round(2)
    df_0['rad_aver'] = 0
    df_0['beta_deriv'] = dbeta_da_derivative(df_0['beta'])
    df_0['a'] = scale_setter.get_spacing_in_fm(df_0['beta'])
    unique = df_T[['velocity', 'cut', 'thickness']].drop_duplicates()
    df_tmp = []
    for index in unique.index:
        df_0['boundary'] = 'OBCb_cV'
        df_0['cut'] = unique.loc[index, 'cut']
        df_0['velocity'] = unique.loc[index, 'velocity']
        df_0['thickness'] = unique.loc[index, 'thickness']
        df_tmp.append(df_0.copy())
    df_tmp = pd.concat(df_tmp)
    df_tmp = df_tmp[df_tmp['beta'].isin([3.860, 4.020, 4.120, 4.192, 4.280, 4.440, 4.600, 4.760, 5.160])]
    # print(df_tmp)
    df_tmp, _ = rescale_observable(df_tmp, observable, Nt)
    df_tmp = df_tmp[[observable, observable + '_err', r'$T/T_{c}$', 'rad_aver', 'boundary', 'cut', 'thickness', 'velocity']]
    # print(df_tmp)
    df_T = df_T[[observable, observable + '_err', r'$T/T_{c}$', 'rad_aver', 'boundary', 'cut', 'thickness', 'velocity']]
    # print(df_T)
    df_T = pd.concat([df_T, df_tmp])
    df_T[r'$T/T_{c}$'] = df_T[r'$T/T_{c}$'].astype(str) + ', ' +  df_T['boundary'].astype(str)
    # print(df_T.to_string())
    df_T.set_index(['velocity', 'cut', 'thickness']).groupby(['velocity', 'cut', 'thickness']).apply(plot_distribution_temperature, observable, observable_label, include_groups=False)

def distribution_boundary_wrapping(args):
    distribution_boundary(*args)

def distribution_thickness_wrapping(args):
    distribution_thickness(*args)

def distribution_temperature_wrapping(args):
    distribution_temperature(*args)

# distribution_temperature('S')

args = [['Jv'], ['Jv1'], ['Jv2'], ['S'], ['Elab'], ['Blab']]

# pool = multiprocessing.Pool(6)
# pool.map(distribution_boundary_wrapping, args)
# pool = multiprocessing.Pool(6)
# pool.map(distribution_thickness_wrapping, args)
pool = multiprocessing.Pool(6)
pool.map(distribution_temperature_wrapping, args)
pool.close()
pool.join()