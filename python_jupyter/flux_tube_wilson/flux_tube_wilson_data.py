import pandas as pd
import os
import io
import math

import flux_tube_wilson

# functions for reading data files


def get_size(str):
    str_tmp = str[str.find('=') + 1:].strip()
    time_size = int(str_tmp[0])
    str_tmp = str_tmp[str_tmp.find('=') + 1:].strip()
    space_size = int(str_tmp[0])

    return time_size, space_size


def remove_spaces(line):
    line_tmp = line.strip()
    result = ''
    while line_tmp.find(' ') != -1:
        result += line_tmp[0:line_tmp.find(' ')] + ','
        line_tmp = line_tmp[line_tmp.find(' '):]
        line_tmp = line_tmp.lstrip()

    result += line_tmp

    return result


def remove_spaces_and_first_number(line):
    line_tmp = line.strip()
    line_tmp = line_tmp[1:]
    line_tmp = line_tmp.lstrip()
    result = ''
    while line_tmp.find(' ') != -1:
        result += line_tmp[0:line_tmp.find(' ')] + ','
        line_tmp = line_tmp[line_tmp.find(' '):]
        line_tmp = line_tmp.lstrip()

    result += line_tmp

    return result


def read_file_vitaliy(file_path, func, field_type, field_coord):
    if(os.path.isfile(file_path)):
        df = []
        with open(file_path, 'r') as f:
            lines = f.readlines()

            data = []
            for i in range(len(lines)):
                line = lines[i].strip()
                if len(line) != 0:
                    if (line)[0] == '#':
                        if len(data) > 0:
                            df.append(pd.read_csv(io.StringIO('\n'.join(data)), names=[
                                      field_coord, f"field_action_{field_type}", f"err_action_{field_type}"], engine='python'))
                            # df[-1]['T'] = sizes[0]
                            df[-1]['R'] = sizes[1]
                            df[-1][field_coord] = df[-1][field_coord] - \
                                (4 + sizes[1] / 2)
                        data = []
                        sizes = get_size(line)

                    else:
                        data.append(func(lines[i]))

            df.append(pd.read_csv(io.StringIO('\n'.join(data)), names=[
                      field_coord, f"field_action_{field_type}", f"err_action_{field_type}"], engine='python'))
            # df[-1]['T'] = sizes[0]
            df[-1]['R'] = sizes[1]
            df[-1][field_coord] = df[-1][field_coord] - (4 + sizes[1] / 2)

            df = pd.concat(df)

            return df


def concat_betas(betas, name, func):
    data = []
    for beta in betas:
        data.append(read_file_vitaliy(beta[0], name, func))
        data[-1]['beta'] = beta[0]
        data[-1]['field'] = data[-1]['field'] / beta[1]**2
        data[-1]['err'] = data[-1]['err'] / beta[1]**2
        data[-1]['d'] = data[-1]['d'] * math.sqrt(beta[1])

    return pd.concat(data)


def read_data_qc2dstag_decomposition(path, params, flux_coord, sigma):
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


def read_data_beta():
    data = []
    # data.append(concat_betas(betas, f'act-lng-prof0{direction}.dat', func))
    data.append(pd.read_csv(
        f"../result/flux_tube_wilson/su2_suzuki/24^4/flux_tube_electric_mu=0.00.csv", index_col=None))
    data[-1] = data[-1].rename(columns={'field': 'field_su2',
                               'err': 'err_su2'})
    data[-1]['beta'] = 2.4
    data[-1] = data[-1][data[-1]['T'] == 4]
    data[-1] = data[-1][data[-1]['R'] == 6]
    data[-1] = data[-1].drop(['T'], axis=1).reset_index()
    # data.append(concat_betas(betas, f'act-lng-prof0{direction}_mon.dat', func))
    data.append(pd.read_csv(
        f"../result/flux_tube_wilson/monopole/su2_suzuki/24^4/flux_tube_electric_mu=0.00.csv", index_col=None))
    data[-1] = data[-1].rename(columns={'field': 'field_mon',
                               'err': 'err_mon'})
    data[-1]['beta'] = 2.4
    data[-1] = data[-1][data[-1]['T'] == 6]
    data[-1] = data[-1][data[-1]['R'] == 6]
    data[-1] = data[-1].drop(['T'], axis=1).reset_index()
    # data.append(concat_betas(betas, f'act-lng-prof0{direction}-off.dat', func))
    data.append(pd.read_csv(
        f"../result/flux_tube_wilson/monopoless/su2_suzuki/24^4/flux_tube_electric_mu=0.00.csv", index_col=None))
    data[-1] = data[-1].rename(
        columns={'field': 'field_nomon', 'err': 'err_nomon'})
    data[-1]['beta'] = 2.4
    data[-1] = data[-1][data[-1]['T'] == 4]
    data[-1] = data[-1][data[-1]['R'] == 6]
    data[-1] = data[-1].drop(['T'], axis=1).reset_index()


def get_flux_data(paths):
    data = []
    for path in paths:
        data.append(pd.read_csv(path['path']))
        data[-1]['label'] = path['label']
        for key, val in path['constraints'].items():
            data[-1][key] = val

    data = pd.concat(data)
    return data


def get_flux_data_decomposition(paths, flux_coord, sigma):
    data = []
    for conf_info in paths:
        # print(conf_info)
        data.append(read_data_qc2dstag_decomposition(
            conf_info[0], conf_info[1], flux_coord, sigma))
        data[-1]['type'] = conf_info[2]

    # print(data)
    data = pd.concat(data)
    data = flux_tube_wilson.find_sum(data)

    data = flux_tube_wilson.join_back(data, flux_coord, [
        'su2', 'monopole', 'monopoless', 'mon+nomon'], ['type'])

    return data
