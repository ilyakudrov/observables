import os
import seaborn
import matplotlib.pyplot as plt


def save_image(image_path, image_name, fg):
    try:
        os.makedirs(image_path)
    except:
        pass

    output_path = f'{image_path}/{image_name}'
    fg.savefig(output_path, dpi=400, facecolor='white')


def make_plot_time_wrappings(data, hue_name):
    wrapping_number = data['winding_number'].iloc[0]
    fg = seaborn.FacetGrid(data=data, height=5, aspect=1.61)
    fg.figure.suptitle(f'winding_number = {wrapping_number}')
    fg.map(plt.errorbar, hue_name, 'cluster_number', 'std',
           marker="o", fmt='', linestyle='').add_legend()

    # save_image_time_wrappings('../../images/common', f'time-wrappings_wrapping_number={wrapping_number}', fg)
