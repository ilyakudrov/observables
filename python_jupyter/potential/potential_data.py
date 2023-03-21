import pandas as pd
import math
import numpy as np
import seaborn
import matplotlib.pyplot as plt

import fit
import plots

# class for data for decomposition of potential


class DataDecomposition:

    def __init__(self, paths):
        data = []
        for type, path in paths.items():
            data.append(pd.read_csv(path['path'], index_col=None))
            data[-1].reset_index(drop=True, inplace=True)
            if 'constraints' in path:
                for key, value in path['constraints'].items():
                    data[-1] = data[-1][(data[-1][key] <= value[1])
                                        & (data[-1][key] >= value[0])]
            name = path['name']
            data[-1] = data[-1].rename(
                columns={'aV(r)': f'aV(r)_{name}', 'err': f'err_{name}'})
            data[-1]['r/a'] = data[-1]['r/a']
            data[-1][f'aV(r)_{name}'] = data[-1][f'aV(r)_{name}']
            data[-1][f'err_{name}'] = data[-1][f'err_{name}']
            data[-1] = data[-1].reset_index(drop=True)
        data = pd.concat(data, axis=1)
        data = data.loc[:, ~data.columns.duplicated()]

        self.data = data
        self.paths = paths

    def get_single_T(self):
        data1 = []
        for type, path in self.paths.items():
            if 'T' in path:
                data1.append(self.data[self.data['T'] == path['T']].reset_index()[
                             ['r/a', 'aV(r)_' + path['name'], 'err_' + path['name']]])
            else:
                data1.append(fit.get_potential_fit(
                    self.data, fit.func_exponent, (2, 8), path['name']))

        self.data = pd.concat(data1, axis=1)
        self.data = self.data.loc[:, ~self.data.columns.duplicated()]

    def make_fits(self, fit_original, fit_range):
        self.terms = self.get_terms(self.paths)
        self.terms_fit = list(self.terms)

        self.data_fits = []
        params = {}
        if fit_original:
            if 'original' in self.paths and 'monopole' in self.paths and 'monopoless' in self.paths:
                original_name = self.paths['original']['name']
                monopole_name = self.paths['monopole']['name']
                monopoless_name = self.paths['monopoless']['name']
                fit_data, params_original, params_original_err, c_coloumb, c_coloumb_err, c_string, c_string_err = fit.make_fit_original(
                    self.data, original_name, monopoless_name, monopole_name, fit_range)
                self.data_fits.append(fit_data)
                params[f'aV(r)_' + original_name] = params_original
                params[f'err_' + original_name] = params_original_err
                params[f'aV(r)_' + monopole_name] = (c_string,
                                                     0, params_original[2])
                params[f'err_' + monopole_name] = (
                    c_string_err, 0, params_original_err[2])
                params[f'aV(r)_' + monopoless_name] = (c_coloumb,
                                                       params_original[1], 0)
                params[f'err_' + monopoless_name] = (
                    c_coloumb_err, params_original_err[1], 0)
                for term in [original_name, monopoless_name, monopole_name]:
                    try:
                        self.terms_fit.remove(term)
                    except:
                        pass
            if 'original' in self.paths and 'abelian' in self.paths and 'offdiagonal' in self.paths:
                original_name = self.paths['original']['name']
                abelian_name = self.paths['abelian']['name']
                offdiagonal_name = self.paths['offdiagonal']['name']
                fit_data, params_original, params_original_err, c_coloumb, c_coloumb_err, c_string, c_string_err = fit.make_fit_original(
                    self.data, original_name, offdiagonal_name, abelian_name, fit_range)
                self.data_fits.append(fit_data)
                params[f'aV(r)_' + original_name] = params_original
                params[f'err_' + original_name] = params_original_err
                params[f'aV(r)_' + abelian_name] = (c_string,
                                                    0, params_original[2])
                params[f'err_' + abelian_name] = (
                    c_string_err, 0, params_original_err[2])
                params[f'aV(r)_' + offdiagonal_name] = (c_coloumb,
                                                        params_original[1], 0)
                params[f'err_' + offdiagonal_name] = (
                    c_coloumb_err, params_original_err[1], 0)
                for term in [original_name, offdiagonal_name, abelian_name]:
                    try:
                        self.terms_fit.remove(term)
                    except:
                        pass

        if 'monopole' in self.paths and 'monopoless' in self.paths:
            monopole_name = self.paths['monopole']['name']
            monopoless_name = self.paths['monopoless']['name']
            sum_name = f'{monopole_name}+{monopoless_name}'
            self.data = self.find_sum(self.data, self.paths['monopole']['name'],
                                      self.paths['monopoless']['name'], sum_name)

        if 'abelian' in self.paths and 'offdiagonal' in self.paths:
            abelian_name = self.paths['abelian']['name']
            offdiagonal_name = self.paths['offdiagonal']['name']
            sum_name = f'{abelian_name}+{offdiagonal_name}'
            self.data = self.find_sum(self.data, self.paths['abelian']['name'],
                                      self.paths['offdiagonal']['name'], sum_name)

        fit_data, fit_params = fit.make_fit_separate(
            self.data, self.terms_fit, fit_range)
        self.data_fits.append(fit_data)
        params = {**params, **fit_params}

        self.data_fits = pd.concat(self.data_fits, axis=1)
        self.data_fits = self.data_fits.loc[:,
                                            ~self.data_fits.columns.duplicated()]
        return params

    def scale_by_r0(self, r0):
        self.data['r/a'] = self.data['r/a'] * r0
        for key, value in self.paths.items():
            self.data[f'aV(r)_' + value['name']
                      ] = self.data[f'aV(r)_' + value['name']] / r0
            self.data[f'err_' + value['name']
                      ] = self.data[f'err_' + value['name']] / r0
        # for term in self.terms:
        #     self.data[f'aV(r)_' + term] = self.data[f'aV(r)_' + term] / r0
        #     self.data[f'err_' + term] = self.data[f'err_' + term] / r0

        # self.data_fits['r/a'] = self.data_fits['r/a'] * r0
        # for term in self.terms:
        #     self.data_fits['aV(r)_' +
        #                    term] = self.data_fits['aV(r)_' + term] / r0

    def remove_from_plot(self, remove_from_plot):
        for term in remove_from_plot:
            self.terms.remove(self.paths[term]['name'])
            self.data = self.data.drop(
                f'aV(r)_' + self.paths[term]['name'], axis=1)
            self.data = self.data.drop(
                f'err_' + self.paths[term]['name'], axis=1)

        self.data = self.join_back(self.data, self.terms)

    def plot(self, black_colors, y_lims, image_path, image_name):
        ls_arr = ['', '', '', '', '', '', '']
        marker_arr = ['o', 'v', 'o', '^', 's', 's', 'D']
        fillstyle_arr = ['full', 'full', 'none',
                         'full', 'full', 'none', 'none']
        if black_colors:
            colors = ['black', 'black', 'black',
                      'black', 'black', 'black', 'black']
        else:
            colors = ['mediumblue', 'orange', 'g', 'r',
                      'rebeccapurple', 'saddlebrown', 'olive']
        fg = plots.plot_potential_decomposition(
            self.data, y_lims, ls_arr, marker_arr, fillstyle_arr, colors, image_path, image_name, 'potential')

        for i in range(len(self.terms)):
            seaborn.lineplot(data=self.data_fits, x='r/a',
                             y='aV(r)_' + self.terms[i], color=colors[i])

        plt.show()
        plots.save_image(image_path, image_name, fg)

    def get_terms(self, paths):
        terms = []
        for key, value in paths.items():
            terms.append(value['name'])
        if 'monopole' in paths and 'monopoless' in paths:
            monopole_name = paths['monopole']['name']
            monopoless_name = paths['monopoless']['name']
            terms.append(f'{monopole_name}+{monopoless_name}')
        if 'abelian' in paths and 'offdiagonal' in paths:
            abelian_name = paths['abelian']['name']
            offdiagonal_name = paths['offdiagonal']['name']
            terms.append(f'{abelian_name}+{offdiagonal_name}')
        return terms

    def join_back(self, data, matrix_types):
        data1 = []
        for matrix_type in matrix_types:
            data1.append(
                data[['r/a', f'aV(r)_{matrix_type}', f'err_{matrix_type}']])
            data1[-1] = data1[-1].rename(
                columns={f'aV(r)_{matrix_type}': 'aV(r)', f'err_{matrix_type}': 'err'})
            data1[-1]['matrix_type'] = matrix_type

        return pd.concat(data1)

    def find_sum(self, data, term1, term2, sum):
        data[f'err_' + sum] = data.apply(lambda x: math.sqrt(
            x[f'err_' + term1] ** 2 + x[f'err_' + term2] ** 2), axis=1)
        data[f'aV(r)_' + sum] = data.apply(lambda x: x[f'aV(r)_' +
                                                       term1] + x[f'aV(r)_' + term2], axis=1)

        return data
# functions for reading data


def read_data_potentials_together(paths, sigma):
    data = []
    for key, value in paths.items():
        print(key)
        data.append(pd.read_csv(value[0], index_col=None))
        if len(value) >= 3:
            data[-1] = data[-1][data[-1]['smearing_step'] == value[2]]
        data[-1] = data[-1][data[-1]['T'] == value[1]]
        data[-1] = data[-1].drop(['T'], axis=1)
        data[-1].reset_index(drop=True, inplace=True)
        data[-1]['r/a'] = data[-1]['r/a'] * math.sqrt(sigma)
        data[-1][f'aV(r)'] = data[-1][f'aV(r)'] / math.sqrt(sigma)
        data[-1][f'err'] = data[-1][f'err'] / math.sqrt(sigma)
        data[-1]['type'] = key
    data = pd.concat(data)

    return data


def read_data_potential(paths):
    data = []
    for path in paths:
        data.append(pd.read_csv(path['path'], index_col=None))
        data[-1].reset_index(drop=True, inplace=True)
        if 'constraints' in path:
            for key, value in path['constraints'].items():
                data[-1] = data[-1][(data[-1][key] <= value[1])
                                    & (data[-1][key] >= value[0])]
        name = path['name']
        data[-1] = data[-1].rename(
            columns={'aV(r)': f'aV(r)_{name}', 'err': f'err_{name}'})
        data[-1]['r/a'] = data[-1]['r/a']
        data[-1][f'aV(r)_{name}'] = data[-1][f'aV(r)_{name}']
        data[-1][f'err_{name}'] = data[-1][f'err_{name}']
    data = pd.concat(data, axis=1)
    data = data.loc[:, ~data.columns.duplicated()]

    return data


def read_data_potential1(paths):
    data = []
    for type, path in paths.items():
        data.append(pd.read_csv(path['path'], index_col=None))
        data[-1].reset_index(drop=True, inplace=True)
        if 'constraints' in path:
            for key, value in path['constraints'].items():
                print(key, value)
                data[-1] = data[-1][(data[-1][key] <= value[1])
                                    & (data[-1][key] >= value[0])]
        name = path['name']
        data[-1] = data[-1].rename(
            columns={'aV(r)': f'aV(r)_{name}', 'err': f'err_{name}'})
        data[-1]['r/a'] = data[-1]['r/a']
        data[-1][f'aV(r)_{name}'] = data[-1][f'aV(r)_{name}']
        data[-1][f'err_{name}'] = data[-1][f'err_{name}']
        data[-1] = data[-1].reset_index(drop=True)
    data = pd.concat(data, axis=1)
    data = data.loc[:, ~data.columns.duplicated()]

    return data


def read_data_single(path):
    data = pd.read_csv(path['path'], index_col=None)
    data.reset_index(drop=True, inplace=True)
    if 'constraints' in path:
        for key, value in path['constraints'].items():
            data = data[(data[key] <= value[1]) & (data[key] >= value[0])]
    name = path['name']
    data = data.rename(
        columns={'aV(r)': f'aV(r)_{name}', 'err': f'err_{name}'})
    data['r/a'] = data['r/a']
    data[f'aV(r)_{name}'] = data[f'aV(r)_{name}']
    data[f'err_{name}'] = data[f'err_{name}']
    data = data.reset_index(drop=True)

    return data


# functions for reading vitaliy files
def read_viltaly_potential_mon_mls(path):
    data = pd.read_csv(f'{path}/potential_mon_mls_fit.dat', header=0,
                       names=['r/a', 'aV(r)_mon', 'err_mon',
                              'r1', 'aV(r)_mod', 'err_mod'],
                       dtype={'r/a': np.int32, "aV(r)_mon": np.float64, "err_mon": np.float64,
                              "r1": np.int32, "aV(r)_mod": np.float64, "err_mod": np.float64})
    return data[['r/a', 'aV(r)_mon', 'err_mon', 'aV(r)_mod', 'err_mod']]


def read_viltaly_potential_original(path):
    data = pd.read_csv(f'{path}/potential_original_fit.dat', header=0,
                       names=['r/a', 'aV(r)_SU(3)', 'err_SU(3)', 'err1'],
                       dtype={'r/a': np.int32, "aV(r)_SU(3)": np.float64,
                              "err_SU3": np.float64, "err1": np.float64})
    return data[['r/a', 'aV(r)_SU(3)', 'err_SU(3)']]


def read_vitaly_potential(path):
    data = []
    data.append(read_viltaly_potential_mon_mls(path))
    data.append(read_viltaly_potential_original(path))
    data = pd.concat(data, axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    return data
