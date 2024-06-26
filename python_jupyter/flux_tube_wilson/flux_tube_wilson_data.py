import pandas as pd
import os
import io
import math

import flux_tube_wilson

# functions for reading data files


def read_data_decomposition(path, params, flux_coord, sigma):
    data = []
    for key, value in params.items():
        # print(value)
        data.append(pd.read_csv(
            f"{path}/flux_tube_original-{key}.csv", index_col=None))
        data[-1] = data[-1].rename(columns={'field_electric': f'field_electric_{key}', 'err_electric': f'err_electric_{key}',
                                            'field_magnetic': f'field_magnetic_{key}', 'err_magnetic': f'err_magnetic_{key}',
                                            'field_energy': f'field_energy_{key}', 'err_energy': f'err_energy_{key}',
                                            'field_action': f'field_action_{key}', 'err_action': f'err_action_{key}'})
        data[-1] = data[-1][data[-1]['T'] == value]
        data[-1] = data[-1].drop(['T'], axis=1).reset_index()
        data[-1][flux_coord] = data[-1][flux_coord] * math.sqrt(sigma)
        data[-1][f"field_electric_{key}"] = data[-1][f"field_electric_{key}"] / sigma**2
        data[-1][f"field_magnetic_{key}"] = data[-1][f"field_magnetic_{key}"] / sigma**2
        data[-1][f"field_energy_{key}"] = data[-1][f"field_energy_{key}"] / sigma**2
        data[-1][f"field_action_{key}"] = data[-1][f"field_action_{key}"] / sigma**2
        data[-1][f"err_electric_{key}"] = data[-1][f"err_electric_{key}"] / sigma**2
        data[-1][f"err_magnetic_{key}"] = data[-1][f"err_magnetic_{key}"] / sigma**2
        data[-1][f"err_energy_{key}"] = data[-1][f"err_energy_{key}"] / sigma**2
        data[-1][f"err_action_{key}"] = data[-1][f"err_action_{key}"] / sigma**2
    data = pd.concat(data, axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    data = data.drop(['index'], axis=1)

    return data


def get_flux_data(paths):
    data = []
    for path in paths:
        data.append(pd.read_csv(path['path']))
        if 'parameters' in path:
            for key, val in path['parameters'].items():
                data[-1][key] = val
        if 'constraints' in path:
            for key, val in path['constraints'].items():
                data[-1] = data[-1][(data[-1][key] >= val[0])
                                    & (data[-1][key] <= val[1])]

    data = pd.concat(data)
    return data


def get_flux_data_decomposition(paths, flux_coord, sigma):
    data = []
    for conf_info in paths:
        # print(conf_info)
        data.append(read_data_decomposition(
            conf_info[0], conf_info[1], flux_coord, sigma))
        data[-1]['type'] = conf_info[2]

    # print(data)
    data = pd.concat(data)
    data = flux_tube_wilson.find_sum(data)

    data = flux_tube_wilson.join_back(data, flux_coord, [
        'su2', 'monopole', 'monopoless', 'mon+nomon'], ['type'])

    return data
