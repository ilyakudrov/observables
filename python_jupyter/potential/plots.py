import os
import seaborn
import matplotlib.pyplot as plt
from PIL import Image

import potential_data


def save_image(image_path, image_name, fg):
    try:
        os.makedirs(image_path)
    except:
        pass

    output_path = f'{image_path}/{image_name}'
    fg.savefig(output_path, dpi=800, facecolor='white')

def save_image_plt(image_path, image_name):
    try:
        os.makedirs(image_path)
    except:
        pass

    output_path = f'{image_path}/{image_name}'
    print(output_path)
    plt.savefig(output_path, dpi=800)

def plot_relative_variation_potential(data):
    # T = data['T'].iloc[0]
    # sigma = data['sigma'].iloc[0]
    fg = seaborn.FacetGrid(data=data, hue='beta', height=5, aspect=1.2)
    fg.figure.suptitle(f'relative variation')
    fg.map(plt.errorbar, 'R', 'potential_diff', 'err_diff', marker="o", fmt='', linestyle=''
           ).add_legend()
    fg.ax.set_xlabel(r"r$\sqrt{\sigma}$")
    fg.ax.set_ylabel(r"$\Delta$")
    plt.xlim((0, 2.5))
    plt.ylim((-0.3, 0.3))
    fg.ax.spines['right'].set_visible(True)
    fg.ax.spines['top'].set_visible(True)
    fg.ax.minorticks_on()
    fg.ax.tick_params(which='both', bottom=True,
                      top=True, left=True, right=True)
    plt.axhline(y=0, color='k', linestyle='-')
    # plt.show()

    save_image(f'../images/potential/relative_variation/vitaliy',
               f'relative_variation', fg)


def plot_potential_decomposition(data, y_lims, ls_arr, marker_arr, fillstyle_arr, colors, image_path, image_name, title):
    fg = seaborn.FacetGrid(data=data, hue='matrix_type', height=5, aspect=1.4, legend_out=False,
                           hue_kws={"ls": ls_arr, "marker": marker_arr,
                                    "fillstyle": fillstyle_arr, "color": colors})
    map = fg.map(plt.errorbar, 'r/a', 'aV(r)', 'err', ms=8,
                 capsize=8, lw=0.5).add_legend(title='')
    # fg.figure.suptitle(title)
    # fg.ax.set_title(title, loc='center')
    fg.ax.set_xlabel(r"r$/r_{0}$", fontsize=16)
    fg.ax.set_ylabel(r"$r_{0}V(r)$", fontsize=16)
    fg.ax.tick_params(axis='both', which='major', labelsize=14)
    fg.ax.spines['right'].set_visible(True)
    fg.ax.spines['top'].set_visible(True)
    # plt.xlabel('xlabel', fontsize=18)
    # plt.ylabel('ylabel', fontsize=16)
    fg.ax.minorticks_on()
    fg.ax.tick_params(which='both', bottom=True,
                      top=True, left=True, right=True)
    fg.ax.set_ylim(y_lims[0], y_lims[1])

    return fg


def plot_together(data):
    fg = seaborn.FacetGrid(data=data, hue='type', height=5,
                           aspect=1.4, legend_out=False)
    fg.figure.suptitle(f'potentials together')
    fg.map(plt.errorbar, 'r/a', 'aV(r)', 'err', mfc=None, fmt='o', ms=3, capsize=5, lw=0.5, ls='-'
           ).add_legend()
    # plt.legend(loc='upper left')
    fg.ax.set_xlabel(r"R$\sqrt{\sigma}$")
    fg.ax.set_ylabel(r"V(r)/$\sigma$")
    fg.ax.spines['right'].set_visible(True)
    fg.ax.spines['top'].set_visible(True)
    fg.ax.minorticks_on()
    fg.ax.tick_params(which='both', bottom=True,
                      top=True, left=True, right=True)
    plt.grid(dash_capstyle='round')

    plt.show()


def plot_potential_fitted_single(data, y_lims, term, image_path, image_name):
    fg = seaborn.FacetGrid(data=data, height=5, aspect=1.4, legend_out=False)
    map = fg.map(plt.errorbar, 'r/a', 'aV(r)_' + term, 'err_' +
                 term, ms=8, capsize=8, lw=0.5).add_legend(title='potential')
    fg.ax.set_xlabel(r"R$/r_{0}$")
    fg.ax.set_ylabel(r"$r_{0}V(R)$")
    fg.ax.spines['right'].set_visible(True)
    fg.ax.spines['top'].set_visible(True)
    fg.ax.minorticks_on()
    fg.ax.tick_params(which='both', bottom=True,
                      top=True, left=True, right=True)
    fg.ax.set_ylim(y_lims[0], y_lims[1])

    return fg

def plot_potential_single(data, x, y, hue, x_label, y_label, title, image_path, image_name, show_plot, err=None, df_fits=None, black_line_y=None, dashed_line_y=None):
    if hue is not None:
        hues = data[hue].unique()
        n_colors = hues.shape[0]
        color_palette = seaborn.color_palette(palette='bright', n_colors=n_colors)
        potential_type_hue = dict(zip(data[hue].unique(), hues))
        color_palette = dict(zip(hues, color_palette))
    else:
        color_palette = None
    fg = seaborn.FacetGrid(data=data, hue=hue, height=5,
                           aspect=1.61, palette=color_palette, legend_out=True)
    fg.figure.suptitle(f'potential')
    if err is not None:
        fg.map(plt.errorbar, x, y, err, mfc=None, fmt='o', ms=3, capsize=5, lw=0.5, ls=None
           ).add_legend()
    else:
        fg.map(plt.errorbar, x, y, mfc=None, fmt='o', ms=3, capsize=5, lw=0.5, ls=None
           ).add_legend()

    fg.ax.set_xlabel(x_label)
    fg.ax.set_ylabel(y_label)
    fg.figure.suptitle(title)
    fg.ax.spines['right'].set_visible(True)
    fg.ax.spines['top'].set_visible(True)
    fg.ax.minorticks_on()
    fg.ax.tick_params(which='both', bottom=True,
                      top=True, left=True, right=True)
    plt.grid(dash_capstyle='round')
    if black_line_y is not None:
        plt.axhline(y=black_line_y, color='r', linestyle='-')

    if dashed_line_y is not None:
        for coord in dashed_line_y:
            plt.axhline(y=coord, color='r', linestyle='--')

    if df_fits is not None:
        for key in df_fits[hue].unique():
            plt.plot(df_fits[df_fits[hue] == key][x], df_fits[df_fits[hue] == key][y],
                     color=color_palette[potential_type_hue[key]], linewidth=1)

    if show_plot:
        plt.show()
    save_image(f'{image_path}',
               f'{image_name}', fg)
    return fg

def make_plots_single(data, x, y, hue, groupby, x_label, y_label, title, image_path, image_name, show_plot, err=None, black_line_y=None, dashed_line_y=None):
    if groupby:
        data.groupby(groupby).apply(
            plot_potential_single, x, y, err, hue, x_label, y_label, title, image_path, image_name, show_plot, black_line_y=black_line_y, dashed_line_y=dashed_line_y)
    else:
        plot_potential_single(data, x, y, err, hue, x_label, y_label, title, image_path, image_name, show_plot, black_line_y=black_line_y, dashed_line_y=dashed_line_y)

def plot_errorbar(data, args, kwargs, ax):
    data = data.reset_index()
    marker = data.at[0, 'marker']
    err_container = plt.errorbar(x=data[args[0]], y=data[args[1]], yerr=data[args[2]], fmt=marker, **kwargs)
    ax.legend((err_container.lines), ('beta'))

def my_plot_func(*args, **kwargs):
        ax = kwargs['ax']
        kwargs1 = dict(kwargs)
        del kwargs1['data']
        del kwargs1['ax']
        kwargs['data'].groupby('beta').apply(plot_errorbar, args, kwargs1, ax)

def plot_potential_err_markers(data, x, y, err, hue, x_label, y_label, title, image_path, image_name, show_plot, df_fits=None, black_line_y=None):
    color_palette = None
    fg = seaborn.FacetGrid(data=data, hue='name', height=5,
                           aspect=1.61, palette=color_palette, legend_out=True)
    fg.map_dataframe(my_plot_func, x, y, err, ms=6, capsize=5, ls=None, lw=1, markerfacecolor='none', ax=fg.ax)

    fg.ax.set_xlabel(x_label)
    fg.ax.set_ylabel(y_label)
    # figure.suptitle(title)
    fg.ax.spines['right'].set_visible(True)
    fg.ax.spines['top'].set_visible(True)
    fg.ax.minorticks_on()
    fg.ax.tick_params(which='both', bottom=True,
                      top=True, left=True, right=True)
    plt.grid(dash_capstyle='round')

    if show_plot:
        plt.show()
    save_image(f'{image_path}',
               f'{image_name}', fg)
    if not show_plot:
        plt.close()

def plot_potential_markers(data, x, y, err, hue, x_label, y_label, title, image_path, image_name, show_plot, df_fits=None, black_line_y=None):
    # if hue is not None:
    #     hues = data[hue].unique()
    #     n_colors = hues.shape[0]
    #     color_palette = seaborn.color_palette(palette='bright', n_colors=n_colors)
    #     potential_type_hue = dict(zip(data[hue].unique(), hues))
    #     color_palette = dict(zip(hues, color_palette))
    # else:
    #     color_palette = None
    color_palette = None
    # fg = seaborn.FacetGrid(data=data, hue='name', style='beta', height=5,
    #                        aspect=1.61, palette=color_palette, legend_out=True)
    # fg.figure.suptitle(f'potential')
    # fg.map(plt.errorbar, x, y, err, mfc=None, fmt='o', ms=3, capsize=5, lw=0.5, ls=None
    #        ).add_legend()
    # fg.map(plt.scatterplot, x, y, mfc=None, fmt='o', ms=3, capsize=5, lw=0.5, ls=None
    #        ).add_legend()
    # ax = seaborn.scatterplot(data=data, x=x, y=y, hue='name', style='beta', height=7)
    fg = seaborn.relplot(data=data, x=x, y=y, hue='name', style='beta', height=7, kind='scatter', aspect=1.6)
    # plt.show()

    fg.ax.set_xlabel(x_label)
    fg.ax.set_ylabel(y_label)
    # figure.suptitle(title)
    fg.ax.spines['right'].set_visible(True)
    fg.ax.spines['top'].set_visible(True)
    fg.ax.minorticks_on()
    fg.ax.tick_params(which='both', bottom=True,
                      top=True, left=True, right=True)
    plt.grid(dash_capstyle='round')
    plt.savefig(f'{image_path}/{image_name}', dpi=800, facecolor='white')
    # if black_line_y is not None:
    #     plt.axhline(y=black_line_y, color='r', linestyle='-')

    # if df_fits is not None:
    #     for key in df_fits[hue].unique():
    #         plt.plot(df_fits[df_fits[hue] == key][x], df_fits[df_fits[hue] == key][y],
    #                  color=color_palette[potential_type_hue[key]], linewidth=1)


    if show_plot:
        plt.show()
    # save_image(f'{image_path}',
    #            f'{image_name}', fg)
    # save_image_plt(f'{image_path}',
    #            f'{image_name}')
    if not show_plot:
        plt.close()
