import os


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
