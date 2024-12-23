import os
import seaborn
import matplotlib.pyplot as plt
from PIL import Image

def save_image(image_path, image_name, fg, format='png', save_black=False):
    try:
        os.makedirs(image_path)
    except:
        pass

    output_path = f'{image_path}/{image_name}.{format}'
    fg.savefig(output_path, dpi=800, facecolor='white', format=format)
    if save_black:
        Image.open(f'{output_path}.png').convert('L').save(f'{output_path}_bw.png')

def save_image_plt(image_path, image_name, dpi=800):
    try:
        os.makedirs(image_path)
    except:
        pass

    output_path = f'{image_path}/{image_name}'
    print(output_path)
    plt.savefig(output_path, dpi=dpi)

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
                           aspect=1.6, palette=color_palette, legend_out=True, **fg_kws)
    if err is not None:
        fg.map(plt.errorbar, x, y, err, mfc=None, fmt='o', ms=5, capsize=5, lw=0.5, ls=None
           ).add_legend()
    else:
        fg.map(plt.errorbar, x, y, mfc=None, fmt='o', ms=3, capsize=5, lw=0.5, ls=None
           ).add_legend()
    if x_log:
        fg.ax.set_xscale('log')
    if y_log:
        fg.ax.set_yscale('log')
    fg.ax.set_xlabel(x_label, fontsize=14)
    fg.ax.set_ylabel(y_label, fontsize=14)
    fg.ax.set_title(title)
    fg.ax.spines['right'].set_visible(True)
    fg.ax.spines['top'].set_visible(True)
    fg.ax.minorticks_on()
    fg.ax.tick_params(which='both', bottom=True,
                      top=True, left=True, right=True)
    # plt.grid(dash_capstyle='round')
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
    if save_figure:
        save_image(f'{image_path}',
            f'{image_name}', fg)
    return fg
