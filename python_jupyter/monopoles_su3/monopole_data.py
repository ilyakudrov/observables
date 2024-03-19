import pandas as pd
import os

# functions for reading files

def fill_wrapped(df):
    for color in range(1, 4):
        if df[df['color'] == color].empty:
            df = pd.concat([df, pd.DataFrame({'color': [color], 'length': [0], 'x0_wrap': [0], 'x1_wrap': [0], 'x2_wrap': [0], 'x3_wrap': [0], 'percolating_group': ['percolating']})])
            df = pd.concat([df, pd.DataFrame({'color': [color], 'length': [0], 'x0_wrap': [0], 'x1_wrap': [0], 'x2_wrap': [0], 'x3_wrap': [0], 'percolating_group': ['non-percolating']})])
    return df

def read_data_unwrapped(paths):
    df = []
    for path in paths:
        for i in range(path['conf_range'][0], path['conf_range'][1] + 1):
            data_path = path['path'] + f'_{i:04}'
            if(os.path.isfile(data_path)):
                df1 = pd.read_csv(data_path)
                if not df1.empty:
                    df.append(df1)
                    df[-1]['conf'] = i
                    if 'parameters' in path:
                            for key, val in path['parameters'].items():
                                df[-1][key] = val

    return pd.concat(df)

def read_data_wrapped(paths):
    df = []
    for path in paths:
        for i in range(path['conf_range'][0], path['conf_range'][1] + 1):
            data_path = path['path'] + f'_{i:04}'
            if(os.path.isfile(data_path)):
                df1 = pd.read_csv(data_path)
                df1 = fill_wrapped(df1)
                df.append(df1)
                df[-1]['conf'] = i
                if 'parameters' in path:
                        for key, val in path['parameters'].items():
                            df[-1][key] = val

    return pd.concat(df)

def read_data_unwrapped_copies(paths):
    df = []
    for path in paths:
        for i in range(path['conf_range'][0], path['conf_range'][1] + 1):
            for j in range(path['copies_num']):
                data_path = path['path'] + f'_{i:04}_{j}'
                if(os.path.isfile(data_path)):
                    df1 = pd.read_csv(data_path)
                    if not df1.empty:
                        df.append(df1)
                        df[-1]['conf'] = i
                        df[-1]['copy'] = j
                        if 'parameters' in path:
                                for key, val in path['parameters'].items():
                                    df[-1][key] = val

    return pd.concat(df)

def read_data_wrapped_copies(paths):
    df = []
    for path in paths:
        for i in range(path['conf_range'][0], path['conf_range'][1] + 1):
            for j in range(path['copies_num']):
                data_path = path['path'] + f'_{i:04}_{j}'
                if(os.path.isfile(data_path)):
                    df1 = pd.read_csv(data_path)
                    df1 = fill_wrapped(df1)
                    df.append(df1)
                    df[-1]['conf'] = i
                    df[-1]['copy'] = j
                    if 'parameters' in path:
                            for key, val in path['parameters'].items():
                                df[-1][key] = val

    return pd.concat(df)


def save_data_windings(data, path):
    size = data['size'].iloc[0]
    data.to_csv(f'{path}/windings_{size}.csv', index=False)


def save_data_windings_separate_size(data, path):
    try:
        os.makedirs(path)
    except:
        pass
    data.groupby(['size']).apply(save_data_windings, path)
