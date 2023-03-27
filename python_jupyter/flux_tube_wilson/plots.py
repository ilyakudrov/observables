import os
import matplotlib.pyplot as plt
import seaborn

import flux_tube_wilson


def save_image(image_path, image_name, fg):
    try:
        os.makedirs(image_path)
    except:
        pass

    output_path = f'{image_path}/{image_name}'
    fg.savefig(output_path, dpi=400, facecolor='white')


def plot_action(data, flux_coord, image_path):
    R = data['R'].iloc[0]
    field_type = data['field_type'].iloc[0]
    print('R = ', R)
    fg = seaborn.FacetGrid(data=data, hue='matrix_type', height=5,
                           aspect=1.61, hue_kws={"ls": ['-', '', '', '-']})
    fg.fig.suptitle(f'{field_type}, R = {R}')
    fg.map(plt.errorbar, flux_coord, 'field', 'err', mfc=None, fmt='o', ms=3, capsize=5, lw=0.5
           ).add_legend()
    fg.ax.set_xlabel(r"d$\sqrt{\sigma}$")
    fg.ax.set_ylabel(r"A/$\sigma^{2}$")

    plt.show()
    save_image(f'{image_path}', f'flux_tube_{field_type}_R={R}', fg)


def plot_action_long(direction, path, params, flux_coord, image_path, sigma):
    data = flux_tube_wilson.read_data_qc2dstag_decomposition(
        path, params, flux_coord, sigma)

    # data = pd.concat(data, axis = 1)
    # data = data.loc[:,~data.columns.duplicated()]
    # data = data.drop(['index'], axis = 1)

    # print(data)

    data = flux_tube_wilson.find_sum(data)

    # print(data)

    data = flux_tube_wilson.join_back(data, flux_coord, [
        'su2', 'monopole', 'monopoless', 'mon+nomon'], [])

    # print(data)

    data.groupby(['R', 'field_type']).apply(
        plot_action, flux_coord, image_path)


def plot_flux(data, flux_coord, hue, image_path, field_type, show_plot):
    R = data['R'].iloc[0]
    # field_type = data['field_type'].iloc[0]
    print('R = ', R)
    fg = seaborn.FacetGrid(data=data, hue=hue, height=5,
                           aspect=1.61, legend_out=False)
    fg.fig.suptitle(f'{field_type}, R = {R}')
    fg.map(plt.errorbar, flux_coord, f'field_{field_type}', f'err_{field_type}', mfc=None, fmt='o', ms=3, capsize=5, lw=0.5, ls='-'
           ).add_legend()

    # fg.ax.set_xlabel(r"R$\sqrt{\sigma}$")
    # fg.ax.set_ylabel(r"V(r)/$\sigma$")
    fg.ax.spines['right'].set_visible(True)
    fg.ax.spines['top'].set_visible(True)
    fg.ax.minorticks_on()
    fg.ax.tick_params(which='both', bottom=True,
                      top=True, left=True, right=True)
    plt.grid(dash_capstyle='round')

    if field_type == 'action':
        type_sign = r"$\mathcal{L}$"
    elif field_type == 'energy':
        type_sign = r"$\epsilon$"
    else:
        type_sign = r"A"

    if flux_coord == 'x_tr':
        coord_sign = r"$x_{\perp}$"
    elif flux_coord == 'd':
        coord_sign = r"$x_{\parallel}$"
    else:
        coord_sign = "wrong flux_coord"
    fg.ax.set_xlabel(coord_sign + r"$\sqrt{\sigma}$")
    fg.ax.set_ylabel(type_sign + r"/$\sigma^{2}$")
    # plt.ylim((-1, 1))
    # fg.ax.set_xlabel(r"$x_{\perp}$")
    # fg.ax.set_ylabel("A")

    if show_plot:
        plt.show()
    save_image(f'{image_path}',
               f'flux_tube_{field_type}_R={R}', fg)
    if not show_plot:
        plt.close()


def plot_flux_R(data, flux_coord, image_path):
    field_type = data['field_type'].iloc[0]
    fg = seaborn.FacetGrid(data=data, hue='type', height=5, aspect=1.61)
    fg.fig.suptitle(f'{field_type} density')
    fg.map(plt.errorbar, 'R', 'field', 'err', mfc=None, fmt='o', ms=3, capsize=5, lw=0.5, ls='-'
           ).add_legend()
    fg.ax.set_xlabel("R")
    fg.ax.set_ylabel(r"$\mathcal{L}$/$\sigma^{2}$")

    plt.show()
    save_image(f'{image_path}', f'flux_tube_{field_type}', fg)


def plot_diff(data):
    R = data['R'].iloc[0]
    fg = seaborn.FacetGrid(data=data, hue='beta', height=5, aspect=1.61)
    fg.map(plt.errorbar, 'd', 'field_diff', 'err_diff', mfc=None, fmt='o', ms=5, capsize=5
           ).add_legend()

    fg.ax.set_ylabel("relative variation")
    fg.ax.set_title(f'R = {R}')

    save_image(f'../images/flux_tube_wilson/action/',
               f'relative_variation_R={R}', fg)


def plot_action_long_difference(betas, sigma):
    data = flux_tube_wilson.relative_variation(betas)

    # data = data[data['R'] == 6]

    # data[['field', 'err']] = data[['field', 'err']]
    # data['d'] = data['d'] * math.sqrt(sigma)

    data = data[data['d'] >= -3]
    data = data[data['d'] <= 3]

    data.groupby('R').apply(plot_diff)

    # data.groupby(['R']).apply(plot_action, beta)


def plot_flux_decomposition(data, flux_coord, image_path):
    R = data['R'].iloc[0]
    field_type = data['field_type'].iloc[0]
    print('R = ', R)
    fg = seaborn.FacetGrid(data=data, hue='matrix_type',
                           height=5, aspect=1.61, legend_out=False)
    fg.fig.suptitle(f'{field_type}, R = {R}')
    fg.map(plt.errorbar, flux_coord, 'field', 'err', mfc=None, fmt='o', ms=3, capsize=5, lw=0.5, ls='-'
           ).add_legend()

    fg.ax.set_xlabel(r"R$\sqrt{\sigma}$")
    fg.ax.set_ylabel(r"V(r)/$\sigma$")
    fg.ax.spines['right'].set_visible(True)
    fg.ax.spines['top'].set_visible(True)
    fg.ax.minorticks_on()
    fg.ax.tick_params(which='both', bottom=True,
                      top=True, left=True, right=True)
    plt.grid(dash_capstyle='round')

    if field_type == 'action':
        type_sign = r"$\mathcal{L}$"
    elif field_type == 'energy':
        type_sign = r"$\epsilon$"
    else:
        type_sign = r"A"

    if flux_coord == 'x_tr':
        coord_sign = r"$x_{\perp}$"
    elif flux_coord == 'd':
        coord_sign = r"$x_{\parallel}$"
    else:
        coord_sign = "wrong flux_coord"
    fg.ax.set_xlabel(coord_sign + r"$\sqrt{\sigma}$")
    fg.ax.set_ylabel(type_sign + r"/$\sigma^{2}$")
    # plt.ylim((-1, 1))
    # fg.ax.set_xlabel(r"$x_{\perp}$")
    # fg.ax.set_ylabel("A")

    plt.show()
    save_image(f'{image_path}',
               f'flux_tube_{field_type}_{flux_coord}_R={R}', fg)
