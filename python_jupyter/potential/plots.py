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
    print(output_path)
    fg.savefig(output_path, dpi=400, facecolor='white')
    # output_path_grey = f'{image_path}/{image_name}_grey'
    # Image.open(f'{output_path}.png').convert('L').save(f'{output_path_grey}.png')


def plot_relative_variation_potential(data):
    # T = data['T'].iloc[0]
    # sigma = data['sigma'].iloc[0]
    fg = seaborn.FacetGrid(data=data, hue='beta', height=5, aspect=1.2)
    fg.fig.suptitle(f'relative variation')
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
    fg.ax.set_title(title, loc='center')
    fg.ax.set_xlabel(r"R$/r_{0}$")
    fg.ax.set_ylabel(r"$r_{0}V(R)$")
    fg.ax.spines['right'].set_visible(True)
    fg.ax.spines['top'].set_visible(True)
    fg.ax.minorticks_on()
    fg.ax.tick_params(which='both', bottom=True,
                      top=True, left=True, right=True)
    fg.ax.set_ylim(y_lims[0], y_lims[1])

    return fg


def plot_together(data):
    fg = seaborn.FacetGrid(data=data, hue='type', height=5,
                           aspect=1.4, legend_out=False)
    fg.fig.suptitle(f'potentials together')
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


def plot_potential_single(data, hue, image_path, image_name, show_plot):
    # R = data['r/a'].iloc[0]
    # field_type = data['field_type'].iloc[0]
    # print('R = ', R)
    fg = seaborn.FacetGrid(data=data, hue=hue, height=5,
                           aspect=1.61, legend_out=False)
    fg.fig.suptitle(f'potential')
    fg.map(plt.errorbar, f'r/a', f'aV(r)', 'err', mfc=None, fmt='o', ms=3, capsize=5, lw=0.5, ls='-'
           ).add_legend()

    # fg.ax.set_xlabel(r"R$\sqrt{\sigma}$")
    # fg.ax.set_ylabel(r"V(r)/$\sigma$")
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


def make_plots_single(paths, hue, groupby, image_path, image_name, show_plot):
    data = potential_data.get_potantial_data(paths)
    if groupby:
        data.groupby(groupby).apply(
            plot_potential_single, hue, image_path, image_name, show_plot)
    else:
        plot_potential_single(data, hue, image_path, image_name, show_plot)
