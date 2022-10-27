from numba import njit
import sys
import math
import time
import numpy as np
import os.path
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", ".."))
import statistics_python.src.statistics_observables as stat
from astropy.stats import jackknife_resampling, jackknife_stats, bootstrap


def get_correlator(data):

    x = np.array(data['correlator'].to_numpy())
    x = np.array([x])

    correlator, err = stat.jackknife_var_numba(x, trivial)

    return pd.Series([correlator, math.sqrt(err)], index=['correlator', 'err'])


@njit
def trivial(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = -math.log(x[0][i])
    return y


# conf_type = "gluodynamics"
conf_type = 'QCD/140MeV'
conf_sizes = ['nt4', 'nt6', 'nt8', 'nt10',
              'nt12', 'nt14', 'nt16_gov', 'nt18_gov']
theory_type = 'su3'
betas = ['/']
smeared_array = ['HYP1_alpha=1_1_0.5', 'HYP2_alpha=1_1_0.5']
conf_max = 2000
mu1 = ['/']
chains = ['/']

for smeared in smeared_array:
    for beta in betas:
        for conf_size in conf_sizes:
            for mu in mu1:
                print(conf_size, mu, beta, smeared)
                data = []
                for chain in chains:
                    for i in range(0, conf_max + 1):
                        file_path = f'../../data/polyakov_loop_correlator/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{smeared}/{chain}/correlator_{i:04}'
                        # print(file_path)
                        if(os.path.isfile(file_path)):
                            data.append(pd.read_csv(file_path))
                            data[-1]["conf_num"] = i

                if len(data) == 0:
                    print("no data",
                          conf_size, mu, beta, smeared)
                elif len(data) != 0:
                    df = pd.concat(data)

                    start = time.time()

                    df1 = df.groupby(
                        ['distance']).apply(get_correlator).reset_index()

                    end = time.time()

                    print("execution time = %s" % (end - start))

                    path_output = f"../../result/polyakov_loop_correlator/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{smeared}"
                    try:
                        os.makedirs(path_output)
                    except:
                        pass
                    df1.to_csv(
                        f"{path_output}/polyakov_correlator.csv", index=False)
