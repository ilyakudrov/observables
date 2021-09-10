import pandas as pd
import os.path
import numpy as np
import math

conf_type = "qc2dstag"


def get_field(data, df1):
    x = data[['wilson-plaket-correlator',
              'wilson-loop', 'plaket']].to_numpy()

    field, err = jackknife_var(x, field_electric)

    new_row = {'field': field, 'err': math.sqrt(err)}

    df1 = df1.append(new_row, ignore_index=True)

    return df1


def field_electric(x):
    a = x.mean(axis=0)
    return a[0] / a[1] - a[2]


def jackknife(x, func):
    n = len(x)
    idx = np.arange(n)
    return sum(func(x[idx != i]) for i in range(n))/float(n)


def jackknife_var(x, func):
    n = len(x)
    idx = np.arange(n)
    j_est = jackknife(x, func)
    return j_est, (n-1)/(n + 0.0) * sum((func(x[idx != i]) - j_est)**2.0
                                        for i in range(n))


T1 = [8, 10, 12]
R1 = [8, 10, 12, 14, 16, 18]

mu1 = ['0.00', '0.25', '0.40']
conf_size = "32^4"
chains = 1

for mu in mu1:
    data = []
    for chain in range(chains):
        for T in T1:
            for R in R1:
                for i in range(1, 2900):
                    # file_path = f"../../data/flux_tube/qc2dstag/{conf_size}/HYP_APE/mu{mu}/s{chain}/T={T}/R={R}/electric_{i:04}"
                    file_path = f"../../data/flux_tube/qc2dstag/{conf_size}/HYP_APE/mu{mu}/T={T}/R={R}/electric_{i:04}"
                    if(os.path.isfile(file_path)):
                        data.append(pd.read_csv(file_path))
                        data[-1]["R"] = R
                        data[-1]["T"] = T
                        data[-1]['d'] = data[-1]['d'].transform(
                            lambda x: x - R/2)

    df = pd.concat(data)

    df1 = pd.DataFrame(columns=['field', 'err'])

    df1 = df.groupby(['d', 'R', 'T']).apply(get_field, df1).reset_index()

    df1 = df1[['d', 'R', 'T', 'field', 'err']]

    # print(df1)

    path_output = f"../../result/flux_tube/qc2dstag/{conf_size}/HYP_APE"

    try:
        os.makedirs(path_output)
    except:
        pass

    df1.to_csv(f"{path_output}/flux_tube_mu={mu}.csv", index=False)