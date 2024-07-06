import os
import pandas as pd
import numpy as np
import math

def read_functional(paths):
    df = []
    for path in paths:
        if 'chain' in path:
            chain = path['chain']
        else:
            chain = 0
        if 'padding' in path:
            padding = path['padding']
        else:
            padding = 4
        for i in range(path['conf_range'][0], path['conf_range'][1] + 1):
            data_path = path['path'] + f'_{i:0{padding}}'
            if(os.path.isfile(data_path)):
                df1 = pd.read_csv(data_path)
                if not df1.empty:
                    df.append(df1)
                    df[-1]['conf'] = f'{i}-{chain}'
                    if 'copy' in df[-1]:
                        if df[-1].loc[0, 'copy'] == 0:
                            df[-1]['copy'] = df[-1]['copy'] + 1
                    if 'parameters' in path:
                            for key, val in path['parameters'].items():
                                df[-1][key] = val
    return pd.concat(df)

def fill_funcational_max(df, groupby_keys):
    df2 = []
    copy_num = df.groupby(['copy']).ngroups
    for copy_max in range(1, copy_num + 1):
        df1 = df[df['copy'] <= copy_max]
        df1 = df1.groupby(groupby_keys + ['conf'])['functional'].max().reset_index(level=groupby_keys + ['conf'])
        df2.append(df1.groupby(groupby_keys)['functional']\
                   .agg([('functional', 'mean'), ('std', lambda x: np.std(x, ddof=1)/math.sqrt(np.size(x)))]).reset_index(level=groupby_keys))
        df2[-1]['copy'] = copy_max
    return pd.concat(df2)
