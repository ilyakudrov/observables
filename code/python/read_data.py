import pandas as pd
import os

def read_no_copy(chains, conf_max, path, file_name, padding):
    data = []
    for chain in chains:
        for i in range(0, conf_max + 1):
            file_path = f'{path}/{chain}/{file_name}_{i:0{padding}}'
            if (os.path.isfile(file_path)):
                try:
                    df_file = pd.read_csv(file_path)
                    df_file["conf"] = f'{i}-{chain}'
                except:
                    df_file = pd.DataFrame()
                data.append(df_file)
    try:
        result_df = pd.concat(data)
    except:
        result_df = pd.DataFrame()
    return result_df

def read_copy_each(chains, conf_max, path, file_name, padding, copy):
    data = []
    for chain in chains:
        for i in range(0, conf_max + 1):
            file_path = f'{path}/{chain}/{file_name}_{i:0{padding}}_{copy}'
            if (os.path.isfile(file_path)):
                try:
                    df_file = pd.read_csv(file_path)
                    df_file["conf"] = f'{i}-{chain}'
                    df_file["copy"] = copy
                except:
                    df_file = pd.DataFrame()
                data.append(df_file)

    try:
        result_df = pd.concat(data)
    except:
        result_df = pd.DataFrame()
    return result_df

def read_copy_best(chains, conf_max, path, file_name, padding, copy):
    data = []
    for chain in chains:
        for i in range(0, conf_max + 1):
            c = copy
            while c >= 0:
                file_path = f'{path}/{chain}/{file_name}_{i:0{padding}}_{c}'
                if (os.path.isfile(file_path)):
                    try:
                        df_file = pd.read_csv(file_path)
                        df_file['conf'] = f'{i}-{chain}'
                        df_file['copy'] = copy
                    except:
                        df_file = pd.DataFrame()
                    data.append(df_file)
                    break
                c -= 1
    try:
        result_df = pd.concat(data)
    except:
        result_df = pd.DataFrame()
    return result_df