import pandas as pd
import os.path
import numpy as np
import math

conf_type = "qc2dstag"


def get_field(data, df1):
    x = data[['wilson-plaket-correlator',
              'wilson-loop', 'plaket']].to_numpy()

    field, err = jackknife_var(x, field_electric)
    # a1, a2, a3 = jackknife_var(x, field_electric)

    # print(data['d'].iloc[0], a1, a2, a3)

    # field = 0
    # err = 0

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
    # a = x.mean(axis=0)
    return j_est, (n-1)/(n + 0.0) * sum((func(x[idx != i]) - j_est)**2.0
                                        for i in range(n))
    # return a[0] / a[1] - a[2], 0
    # return a[0], a[1], a[2]


# T1 = [8, 10]
# R1 = [10, 14, 18]

T1 = [8, 10, 12]
R1 = [8, 10, 12, 14, 16, 18]

mu1 = ['0.00', '0.25', '0.40']
conf_size = "32^4"
chains = 1

# test_electric = []
# test_wilson = []
# test_plaket = []

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
                        # data.append(pd.read_csv(file_path, header=None, names=[
                        #             'd', 'wilson-plaket-correlator', 'wilson-loop', 'plaket']))
                        # data[-1]["conf_num"] = i
                        # print(data)
                        data[-1]["R"] = R
                        data[-1]["T"] = T
                        data[-1]['d'] = data[-1]['d'].transform(
                            lambda x: x - R/2)
                        # test_electric.append(data[-1].loc[(data[-1]['T'] == 8) & (data[-1]
                        #                                                           ['R'] == 14) & (data[-1]['d'] == 0), 'wilson-plaket-correlator'].values[0])
                        # test_wilson.append(data[-1].loc[(data[-1]['T'] == 8) & (data[-1]
                        #                                                         ['R'] == 14) & (data[-1]['d'] == 0), 'wilson-loop'].values[0])
                        # test_plaket.append(data[-1].loc[(data[-1]['T'] == 8) & (data[-1]
                        #                                                         ['R'] == 14) & (data[-1]['d'] == 0), 'plaket'].values[0])
                        # print(data)
                        # data[-1]["chain"] = chain

    # electric = 0
    # wilson = 0
    # plaket = 0

    # for i in range(len(test_electric)):
    #     if test_electric[i] == 0:
    #         print("zero")
    #     electric += test_electric[i]
    #     wilson += test_wilson[i]
    #     plaket += test_plaket[i]

    # electric = electric / len(test_electric)
    # wilson = wilson / len(test_wilson)
    # plaket = plaket / len(test_plaket)

    # print(electric / wilson - plaket)

    df = pd.concat(data)

    df1 = pd.DataFrame(columns=['field', 'err'])

    df1 = df.groupby(['d', 'R', 'T']).apply(get_field, df1).reset_index()

    df1 = df1[['d', 'R', 'T', 'field', 'err']]

    # print(df1)

    df1.to_csv(
        f"../../result/flux_tube/qc2dstag/{conf_size}/HYP_APE/flux_tube_mu={mu}.csv", index=False)
