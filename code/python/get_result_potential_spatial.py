import pandas as pd
import os.path
import numpy as np
import math

def get_field(data, df1, df, r1_max):

    time_size = data["r1/a"].iloc[0]
    space_size = data["r2/a"].iloc[0]

    if time_size < r1_max:

        x1 = data[['wilson_loop']].to_numpy()

        x2 = df[(df["r1/a"] == time_size + 1) & (df["r2/a"] == space_size)][["wilson_loop"]].to_numpy()

        x3 = np.vstack((x1.T, x2.T)).T

        field, err = jackknife_var(x3, potential)

        field1 = average(x3)

        new_row = {'aV(r)': field, 'err': math.sqrt(err)}

        df1 = df1.append(new_row, ignore_index=True)

        return df1


def potential(x):
    a = x.mean(axis=0)
    fraction = a[0] / a[1]
    if(fraction >= 0):
        return math.log(fraction)
    else:
        return 0


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

def average(x):
    a = x.mean(axis=0)
    fraction = a[0] / a[1]
    if(fraction >= 0):
        return math.log(fraction)
    else:
        return 0

def add_transposed(data):
    for index, row in data.iterrows():
        if(row['r1/a'] != row['r2/a']):
            print(row['conf_num'])
            # print(row['r1/a'], row['r2/a'])
            new_row = {'r2/a': row['r1/a'], 'r1/a': row['r2/a'], 'wilson_loop': row['wilson_loop'], 'conf_num': row['conf_num']}
            data = data.append(new_row, ignore_index=True)

conf_type = "qc2dstag"
# monopole = "monopoless"
monopole = ""
mu1 = ['0.05']
conf_size = "40^4"
chains = {"s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"}
# chains = {"s0"}
# chains = {""}
# axis = 'on-axis'
smearing = "HYP2_APE120"
# smearing = "unsmeared"
test = ""

for mu in mu1:
    data = []
    for chain in chains:
        for i in range(0, 300):
            file_path = f"../../data/wilson_loop_spatial/{test}/{monopole}/qc2dstag/{conf_size}/mu{mu}/{chain}/wilson_loop_spatial_{i:04}"
            # file_path = f"../../data/wilson_loop/{axis}/su2_dynam/{monopole}/{conf_size}/{smearing}/wilson_loop_{i:04}"

            # print(file_path)
            if(os.path.isfile(file_path)):
                data.append(pd.read_csv(file_path, header = 0, names=["r1/a", "r2/a", "wilson_loop"]))
                data[-1]["conf_num"] = i

    df = pd.concat(data)

    # add_transposed(df)
    # print(df)

    # df = df[df['T'] <= 16]
    # df = df[df['r/a'] <= 16]

    # df_test = df[np.isnan(df['wilson_loop']) ]

    # print(df_test)

    # print(df)

    # wilson = df[['wilson_loop']].to_numpy()
    # conf_num = df[['conf_num']].to_numpy()

    # for i in range(len(wilson)):
    #     if(math.isnan(wilson[i])):
    #         print(conf_num[i])

    df1 = pd.DataFrame(columns=["aV(r)", "err"])

    r1_max = df["r1/a"].max()

    df1 = df.groupby(['r1/a', 'r2/a']).apply(get_field, df1, df, r1_max).reset_index()

    df1 = df1[['r1/a', 'r2/a', 'aV(r)', 'err']]

    path_output = f"../../result/potential_spatial/{test}/{monopole}/qc2dstag/{conf_size}"
    # path_output = f"../../result/potential/{test}/{axis}/{monopole}/su2_dynam/{conf_size}/{smearing}"

    try:
        os.makedirs(path_output)
    except:
        pass

    df1.to_csv(f"{path_output}/potential_spatial_mu={mu}.csv", index=False)