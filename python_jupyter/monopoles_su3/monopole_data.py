import pandas as pd
import os

# functions for reading files


def read_data(conf_range, paths, file_name, label_name):
    data = []
    for path in paths:
        for i in range(conf_range[0], conf_range[1] + 1):
            data_path = f'{path[0]}/{file_name}_{i:04}'
            if(os.path.isfile(data_path)):
                data.append(pd.read_csv(data_path))
                data[-1]['conf'] = i
                data[-1][label_name] = path[1]

    return pd.concat(data)


def save_data_windings(data, path):
    size = data['size'].iloc[0]
    data.to_csv(f'{path}/windings_{size}.csv', index=False)


def save_data_windings_separate_size(data, path):
    try:
        os.makedirs(path)
    except:
        pass
    data.groupby(['size']).apply(save_data_windings, path)
