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


def get_potential(data):
    time_size_min = data["T"].min()
    time_size_max = data["T"].max()

    potential = []

    # print(data)

    for time_size in range(time_size_min, time_size_max):

        x1 = data.loc[data['T'] == time_size, 'wilson_loop'].to_numpy()
        x2 = data.loc[data['T'] == time_size + 1, 'wilson_loop'].to_numpy()

        # print(x1)
        # print(x2)

        x3 = np.vstack((x1, x2))

        # field, err = stat.jackknife_var(x3, potential)
        field, err = stat.jackknife_var_numba(x3, potential_numba)

        potential.append([time_size, field, err])

    return pd.DataFrame(potential, columns=['T', 'aV(r)', 'err'])


@njit
def potential_numba(x):
    n = x.shape[1]
    y = np.zeros(n)
    for i in range(n):
        fraction = x[0][i] / x[1][i]
        if(fraction >= 0):
            y[i] = math.log(fraction)
        else:
            y[i] = 0
    return y


def potential(x):
    a = np.mean(x, axis=1)
    fraction = a[0] / a[1]
    if(fraction >= 0):
        return math.log(fraction)
    else:
        return 0


def estimate_composite(x, y, err_x, err_y):
    return math.log(x / y), math.sqrt((err_x / x) ** 2 + (err_y / y) ** 2)


axis = 'on-axis'
conf_type = "gluodynamics"
# conf_type = "QCD/140MeV"
# conf_type = "su2_suzuki"
# conf_type = "SU2_dinam"
# conf_sizes = ["40^4", "32^4"]
# conf_sizes = ["36^4"]
# conf_sizes = ["32^4"]
# conf_sizes = ["48^4"]
conf_sizes = ["24^4"]
# conf_sizes = ["nt16_gov", "nt14", "nt12"]
theory_type = 'su3'
betas = ['beta6.0']
# betas = ['beta2.7', 'beta2.8']
# betas = ['beta2.5', 'beta2.6']
# betas = ['/']
# smeared = 'smeared'
smeared_array = ['HYP0_alpha=1_1_0.5_APE_alpha=0.5',
                 'HYP1_alpha=1_1_0.5_APE_alpha=0.5']
# smeared_array = ['HYP0_alpha=1_1_0.5_APE_alpha=0.5']
# smeared_array = ['HYP0_APE_alpha=0.5']
# smeared = 'HYP1_alpha=1_0.5_0.5_APE_alpha=0.5'
# matrix_type_wilson_array = ['monopole',
#                             'monopoless', 'photon', 'offdiagonal']
# matrix_type_wilson_array = ['original']
matrix_type_wilson_array = ['abelian']
# matrix_type_wilson_array = ['original', 'monopole', 'monopoless']
wilson_loop_type = 'wilson_loop'
potential_type = 'potential'
# additional_parameters = 'steps_500/copies=3/compensate_1'
additional_parameters = 'steps_500/copies=3'
# additional_parameters = '/'
# additional_parameters = 'compensate_1'
# wilson_loop_type = 'wilson_loop_adjoint'
# potential_type = 'potential_adjoint'

calculation_type = 'no_smearing'

conf_max = 5000
mu1 = ['/']
chains = ["/"]

# adjoint_fix = True
adjoint_fix = False

for matrix_type_wilson in matrix_type_wilson_array:
    for smeared in smeared_array:
        for beta in betas:
            for conf_size in conf_sizes:
                for mu in mu1:
                    print(matrix_type_wilson, conf_size, mu, beta, smeared)
                    data = []
                    for chain in chains:
                        for i in range(0, conf_max + 1):
                            # file_path = f'../../data/smearing/{wilson_loop_type}/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{matrix_type_wilson}/{smeared}/{additional_parameters}/{chain}/wilson_loop_{i:04}'
                            file_path = f'../../data/{wilson_loop_type}/{axis}/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{matrix_type_wilson}/{smeared}/{additional_parameters}/{chain}/wilson_loop_{i:04}'
                            # print(file_path)
                            if(os.path.isfile(file_path)):
                                if calculation_type == 'smearing':
                                    data.append(pd.read_csv(file_path, header=0,
                                                            names=[
                                                                'smearing_step', "T", "r/a", "wilson_loop"],
                                                            dtype={'smearing_step': np.int32, "T": np.int32, "r/a": np.int32, "wilson_loop": np.float64}))
                                elif calculation_type == 'no_smearing':
                                    data.append(pd.read_csv(file_path, header=0,
                                                            names=[
                                                                "T", "r/a", "wilson_loop"],
                                                            dtype={"T": np.int32, "r/a": np.int32, "wilson_loop": np.float64}))
                                data[-1]["conf_num"] = i
                                if adjoint_fix:
                                    data[-1]["wilson_loop"] = data[-1]["wilson_loop"] + 1
                    if len(data) == 0:
                        print("no data", matrix_type_wilson,
                              conf_size, mu, beta, smeared)
                    elif len(data) != 0:
                        df = pd.concat(data)

                        start = time.time()

                        if calculation_type == 'smearing':
                            df1 = df.groupby(
                                ['smearing_step', 'r/a']).apply(get_potential).reset_index(['smearing_step', 'r/a']).reset_index()
                        elif calculation_type == 'no_smearing':
                            df1 = df.groupby(
                                ['r/a']).apply(get_potential).reset_index('r/a').reset_index()

                        # print(df1)

                        end = time.time()
                        print("execution time = %s" % (end - start))

                        if calculation_type == 'smearing':
                            df1 = df1[['smearing_step',
                                       'T', 'r/a', 'aV(r)', 'err']]
                            path_output = f"../../result/smearing/{potential_type}/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{smeared}/{additional_parameters}"
                        elif calculation_type == 'no_smearing':
                            df1 = df1[['T', 'r/a', 'aV(r)', 'err']]
                            path_output = f"../../result/potential/wilson_loop/{potential_type}/{theory_type}/{conf_type}/{conf_size}/{beta}/{mu}/{smeared}/{additional_parameters}"
                        try:
                            os.makedirs(path_output)
                        except:
                            pass
                        df1.to_csv(
                            f"{path_output}/potential_{matrix_type_wilson}.csv", index=False)
