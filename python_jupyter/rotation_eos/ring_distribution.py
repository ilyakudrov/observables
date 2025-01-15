import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
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

def get_dir_names(path):
    directories = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        directories.extend(dirnames)
        break
    return directories

def concat_observables(df, observables):
    data = []
    for obs in observables:
        data.append(df.loc[:, ['rad_aver', 'thickness', 'cut', obs, f'{obs}_err']])
        # data[-1] = data[-1].rename({obs: 'y', f'{obs}_err': 'err'}, axis=1)
        # data[-1]['observable'] = obs
    return pd.concat(data)

def read_data(path, observables, lattice, border):
    df = pd.DataFrame()
    velocity_dirs = get_dir_names(f'{path}/{lattice}/{border}')
    for velocity in velocity_dirs:
        beta_dirs = get_dir_names(f'{path}/{lattice}/{border}/{velocity}')
        for beta in beta_dirs:
            file_path = f'{path}/{lattice}/{border}/{velocity}/{beta}/observables_ring_result.csv'
            # print(file_path)
            if os.path.isfile(file_path):
                df_tmp = pd.read_csv(file_path, delimiter=' ')
                df_tmp = concat_observables(df_tmp, observables)
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
    markers_arr = ['o', '^', 's', 'D', 'P', 'X', 'v', '*']
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
    #hue_kws=dict(marker=markers_hue)
    fg_kws = {}
    if markers_different:
        fg_kws['hue_kws'] = dict(marker=markers_hue)
    fg = seaborn.FacetGrid(data=data, hue=hue, height=5,
                           aspect=1.33, legend_out=False, palette=color_palette, **fg_kws)
    if err is not None:
        fg.map(plt.errorbar, x, y, err, mfc=None, fmt='o', ms=5, capsize=5, lw=0.5, ls=None
           ).add_legend(fontsize=10)
    else:
        fg.map(plt.errorbar, x, y, mfc=None, fmt='o', ms=3, capsize=5, lw=0.5, ls=None
           ).add_legend(fontsize=10)
    if x_log:
        fg.ax.set_xscale('log')
    if y_log:
        fg.ax.set_yscale('log')
    fg.ax.set_xlabel(x_label, fontsize=16)
    fg.ax.set_ylabel(y_label, fontsize=16)
    fg.ax.set_title(title)
    fg.ax.spines['right'].set_visible(True)
    fg.ax.spines['top'].set_visible(True)
    fg.ax.minorticks_on()
    fg.ax.tick_params(which='both', bottom=True,
                      top=True, left=True, right=True)
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

beta_critical = {'nt4o': 4.088,
                 'nt5o': 4.225,
                 'nt6o': 4.350,
                 'nt7o': 4.470,
                 'nt4p': 4.073,
                 'nt5p': 4.202,
                 'nt6p': 4.318,
                 'nt7p': 4.417,
                 }

beta_critical_aver = {'nt4o': (4.088 + 4.073)/2,
                 'nt5o': (4.225+4.202)/2,
                 'nt6o': (4.350+4.318)/2,
                 'nt7o': (4.470+4.417)/2,
                 'nt4p': (4.088 + 4.073)/2,
                 'nt5p': (4.225+4.202)/2,
                 'nt6p': (4.350+4.318)/2,
                 'nt7p': (4.470+4.417)/2,
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
    df_T.append(read_data(path, [observable], '5x30x121sq', 'OBCb_cV'))
    df_T[-1]['border'] = 'nt5o'
    df_T[-1]['boundary'] = 'OBCb_cV'
    df_0.append(read_data(path, [observable], '30x30x121sq', 'OBCb_cV'))
    df_0[-1]['border'] = 'nt5o'
    df_0[-1]['boundary'] = 'OBCb_cV'
    df_T.append(read_data(path, [observable], '5x30x121sq', 'PBC_cV'))
    df_T[-1]['border'] = 'nt5p'
    df_T[-1]['boundary'] = 'PBC_cV'
    df_0.append(read_data(path, [observable], '30x30x121sq', 'PBC_cV'))
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
    df_T.append(read_data(path, [observable], '5x30x121sq', 'OBCb_cV'))
    df_T[-1]['border'] = 'nt5o'
    df_T[-1]['boundary'] = 'OBCb_cV'
    df_0.append(read_data(path, [observable], '30x30x121sq', 'OBCb_cV'))
    df_0[-1]['border'] = 'nt5o'
    df_0[-1]['boundary'] = 'OBCb_cV'
    df_T.append(read_data(path, [observable], '5x30x121sq', 'PBC_cV'))
    df_T[-1]['border'] = 'nt5p'
    df_T[-1]['boundary'] = 'PBC_cV'
    df_0.append(read_data(path, [observable], '30x30x121sq', 'PBC_cV'))
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

def plot_distribution_temperature(df, observable):
    boundary = df.name[0]
    velocity = df.name[1]
    cut = df.name[2]
    thickness = df.name[3]
    make_plot(df, 'u', observable, 'T', 'u', observable + r'/R$T^{4}$', f'distribution {observable} boundary={boundary} velocity={velocity:.7f} cut={cut} thickness={thickness}', f'../../images/eos_rotation_imaginary/distribution_temperature_dependence/5x30x121sq/{observable}/velocity={velocity}', f'distribution_boundary={boundary}_cut={cut}_thickness={thickness}', False, err=f'{observable}_err', color_palette=palettable.cartocolors.qualitative.Prism_10.mpl_colors, save_figure=True, dashed_line_y=[0])
    plt.close()

# Dependence of ring distribution on temperature
def distribution_temperature(observable):
    R = 60
    Nt = 5
    path = '../../result/eos_rotation_imaginary'
    df_T = []
    df_0 = []
    df_T.append(read_data(path, [observable], '5x30x121sq', 'OBCb_cV'))
    df_T[-1]['border'] = 'nt5o'
    df_T[-1]['boundary'] = 'OBCb_cV'
    df_0.append(read_data(path, [observable], '30x30x121sq', 'OBCb_cV'))
    df_0[-1]['border'] = 'nt5o'
    df_0[-1]['boundary'] = 'OBCb_cV'
    df_T.append(read_data(path, [observable], '5x30x121sq', 'PBC_cV'))
    df_T[-1]['border'] = 'nt5p'
    df_T[-1]['boundary'] = 'PBC_cV'
    df_0.append(read_data(path, [observable], '30x30x121sq', 'PBC_cV'))
    df_0[-1]['border'] = 'nt5p'
    df_0[-1]['boundary'] = 'PBC_cV'
    df_T = pd.concat(df_T)
    df_0 = pd.concat(df_0)
    df_T['rad_aver'] = df_T['rad_aver'].apply(lambda x: round(x, 4))
    df_0['rad_aver'] = df_0['rad_aver'].apply(lambda x: round(x, 4))
    df_T = df_T[df_T['cut'] == 2*Nt]
    df_0 = df_0[df_0['cut'] == 2*Nt]
    # df_T = df_T[df_T['thickness'] == 2 * Nt]
    # df_0 = df_0[df_0['thickness'] == 2 * Nt]
    df_T = df_T[df_T['velocity'].isin([0.692821, 0.489898, 0.34641])]
    df_0 = df_0[df_0['velocity'].isin([0.692821, 0.489898, 0.34641])]
    # df_T = df_T[df_T['velocity'] != 0]
    # df_0 = df_0[df_0['velocity'] != 0]
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
    df_T['beta_crit'] = df_T.apply(lambda x: beta_critical[x['border']], axis=1)
    df_T['a_crit'] = scale_setter.get_spacing_in_fm(df_T['beta_crit']) * fm_to_GeV
    df_T['u'] = df_T['velocity'] * df_T['rad_aver'] / R
    df_T['T'] = df_T['a_crit'] / df_T['a_GeV']
    if observable in ['E', 'Elab', 'Blab']:
        df_T[observable] = -df_T[observable]
    df_T[observable] = df_T[observable] / Nt**4
    df_T[f'{observable}_err'] = df_T[f'{observable}_err'] / Nt**4
    df_T['rad_aver'] = df_T['rad_aver'] / R
    df_T['hue'] = df_T['boundary'] + ', thickness = ' +  df_T['thickness'].astype(str)
    df_T.set_index(['boundary', 'velocity', 'cut', 'thickness']).groupby(['boundary', 'velocity', 'cut', 'thickness']).apply(plot_distribution_temperature, observable, include_groups=False)

def distribution_boundary_wrapping(args):
    distribution_boundary(*args)

def distribution_thickness_wrapping(args):
    distribution_thickness(*args)

def distribution_temperature_wrapping(args):
    distribution_temperature(*args)

args = [['Jv'], ['Jv1'], ['Jv2'], ['E'], ['Elab'], ['Blab']]

# pool = multiprocessing.Pool(6)
# pool.map(distribution_boundary_wrapping, args)
# pool = multiprocessing.Pool(6)
# pool.map(distribution_thickness_wrapping, args)
pool = multiprocessing.Pool(6)
pool.map(distribution_temperature_wrapping, args)
pool.close()
pool.join()