import os
import pandas as pd
import numpy as np
import math

def read_functional(paths, start, end, increment, padding):
    data = []
    for path in paths:
        for i in range(start, end + 1):
            data_path = f'{path[0]}/functional_{i:0{padding}}'
            if(os.path.isfile(data_path)):
                data.append(pd.read_csv(data_path))
                data[-1]['num'] = i
                data[-1]['type'] = path[1]
                if 'copy' in data[-1]:
                    if increment:
                        data[-1]['copy'] = data[-1]['copy'] + 1
                else:
                    data[-1]['copy'] = 1
                data[-1]['time'] = path[2]

    return pd.concat(data)

def functional_average_max(df):
    df = df.groupby(['num', 'type']).agg({'functional': 'max'})['functional'].reset_index()
    print(df)
    df = df.groupby(['type'])['functional']\
        .agg([('functional', np.mean), ('std', lambda x: np.std(x, ddof=1)/math.sqrt(np.size(x)))]).reset_index()
    return df