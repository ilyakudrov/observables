import pandas as pd
import math
import numpy as np
import seaborn
import matplotlib.pyplot as plt

import fit
import plots

class DataDecomposition:
    """Reads and processes data for a static potential."""

    def __init__(self, paths):
        """Reads data for potentials and initiates data for potentials.

        Args:
            paths:
                double dictionary with properties for each
                potential data set. For example:

                {'original': {'name' : 'SU2', 'path' : f'{path}
                /HYP1_alpha=1_1_0.5_APE_alpha=0.5/potential
                /potential_original.csv', 'T' : 8, 'constraints': constraints},
                'monopole': {'name' : 'monopole', 'path' :
                f'{path}/HYP1_alpha=1_1_0.5_APE_alpha=0.5/{params}
                /{compensate}/potential/potential_monopole.csv',
                'T' : 8, 'constraints': {'r/a': (1, 16)}}}

                path: path to potential data .csv file
                parameters: dictironary of paramenters to be included as coumns
                constraints: (optional) dictionary of ranges of values

                optionaly may contain other key-values to be used by methods
        """
        df = []
        for decomposition_type, path in paths.items():
            df.append(get_potantial_df(path))
            df[-1] = append_multiindex(df[-1], [decomposition_type])
        self.df = pd.concat(df)

    def scale_potentials(self, r0):
        """multiplies V(r) by r0 and divides r by r0."""
        self.df['r/a'] = self.df['r/a'] * r0
        self.df['aV(r)'] = self.df['aV(r)'] / r0
        self.df['err'] = self.df['err'] / r0

    def remove_from_plot(self, types_to_remove):
        self.df = self.df.loc[~self.df['decomposition_type'].isin(types_to_remove)]

    def find_sum(self, term1, term2):
        df1 = self.df.loc[self.df['decomposition_type'] == term1]
        df1.loc[:, 'decomposition_type'] = self.df.loc[self.df['decomposition_type'] == term1, 'decomposition_type'].array + "+" + self.df.loc[self.df['decomposition_type'] == term2, 'decomposition_type'].array
        df1.loc[:, 'name'] = self.df.loc[self.df['decomposition_type'] == term1, 'name'].array + "+" + self.df.loc[self.df['decomposition_type'] == term2, 'name'].array
        df1.loc[:, 'err'] = np.sqrt(self.df.loc[self.df['decomposition_type'] == term1, 'err'].array**2 + self.df.loc[self.df['decomposition_type'] == term2, 'err'].array**2)
        df1.loc[:, 'aV(r)'] = self.df.loc[self.df['decomposition_type'] == term1, 'aV(r)'].array + self.df.loc[self.df['decomposition_type'] == term2, 'aV(r)'].array
        self.df = pd.concat([self.df, df1])

    def find_difference(self, term1, term2):
        df1 = self.df.loc[self.df['decomposition_type'] == term1]
        df1.loc[:, 'decomposition_type'] = self.df.loc[self.df['decomposition_type'] == term1, 'decomposition_type'].array + "-" + self.df.loc[self.df['decomposition_type'] == term2, 'decomposition_type'].array
        df1.loc[:, 'name'] = self.df.loc[self.df['decomposition_type'] == term1, 'name'].array + "-" + self.df.loc[self.df['decomposition_type'] == term2, 'name'].array
        df1.loc[:, 'err'] = np.sqrt(self.df.loc[self.df['decomposition_type'] == term1, 'err'].array**2 + self.df.loc[self.df['decomposition_type'] == term2, 'err'].array**2)
        df1.loc[:, 'aV(r)'] = self.df.loc[self.df['decomposition_type'] == term1, 'aV(r)'].array - self.df.loc[self.df['decomposition_type'] == term2, 'aV(r)'].array
        self.df = pd.concat([self.df, df1])

    def find_ratio(self, term1, term2):
        df1 = self.df.loc[self.df['decomposition_type'] == term1]
        df1.loc[:, 'decomposition_type'] = self.df.loc[self.df['decomposition_type'] == term1, 'decomposition_type'].array + "/" + self.df.loc[self.df['decomposition_type'] == term2, 'decomposition_type'].array
        df1.loc[:, 'name'] = self.df.loc[self.df['decomposition_type'] == term1, 'name'].array + "/" + self.df.loc[self.df['decomposition_type'] == term2, 'name'].array
        df1.loc[:, 'err'] = np.sqrt(self.df.loc[self.df['decomposition_type'] == term1, 'err'].array**2 / self.df.loc[self.df['decomposition_type'] == term2, 'aV(r)'].array**2
                                     + self.df.loc[self.df['decomposition_type'] == term2, 'err'].array**2 * self.df.loc[self.df['decomposition_type'] == term1, 'aV(r)'].array**2
                                     /self.df.loc[self.df['decomposition_type'] == term2, 'aV(r)'].array**4)
        df1.loc[:, 'aV(r)'] = self.df.loc[self.df['decomposition_type'] == term1, 'aV(r)'].array / self.df.loc[self.df['decomposition_type'] == term2, 'aV(r)'].array
        self.df = pd.concat([self.df, df1])

    def shift_fit_linear(self, type1, type2, fit_range1, fit_range2):
        """potential of type1 gets shifted towards potential of type2
            so that linear fits coinside."""
        popt1, pcov1 = fit.fit_single(self.df.loc[self.df['decomposition_type'] == type1], fit_range1, fit.func_linear)
        popt2, pcov2 = fit.fit_single(self.df.loc[self.df['decomposition_type'] == type2], fit_range2, fit.func_linear)
        self.df.loc[self.df['decomposition_type'] == type1, 'aV(r)'] = popt2[0] - popt1[0]

# functions for reading df

def find_sum(df, term1, term2):
    df1 = df.loc[df['decomposition_type'] == term1]
    df1.loc[:, 'decomposition_type'] = df.loc[df['decomposition_type'] == term1, 'decomposition_type'].array + "+" + df.loc[df['decomposition_type'] == term2, 'decomposition_type'].array
    df1.loc[:, 'err'] = np.sqrt(df.loc[df['decomposition_type'] == term1, 'err'].array**2 + df.loc[df['decomposition_type'] == term2, 'err'].array**2)
    df1.loc[:, 'aV(r)'] = df.loc[df['decomposition_type'] == term1, 'aV(r)'].array + df.loc[df['decomposition_type'] == term2, 'aV(r)'].array
    return pd.concat([df, df1])

def read_df_potential(paths):
    df = []
    for type, path in paths.items():
        df.append(pd.read_csv(path['path'], index_col=None))
        df[-1].reset_index(drop=True, inplace=True)
        if 'constraints' in path:
            for key, value in path['constraints'].items():
                df[-1] = df[-1][(df[-1][key] <= value[1])
                                    & (df[-1][key] >= value[0])]
        name = path['name']
        df[-1] = df[-1].rename(
            columns={'aV(r)': f'aV(r)_{name}', 'err': f'err_{name}'})
        df[-1]['r/a'] = df[-1]['r/a']
        df[-1][f'aV(r)_{name}'] = df[-1][f'aV(r)_{name}']
        df[-1][f'err_{name}'] = df[-1][f'err_{name}']
        df[-1] = df[-1].reset_index(drop=True)
    df = pd.concat(df, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    return df


def read_df_single(path):
    df = pd.read_csv(path['path'], index_col=None)
    df.reset_index(drop=True, inplace=True)
    if 'constraints' in path:
        for key, value in path['constraints'].items():
            df = df[(df[key] <= value[1]) & (df[key] >= value[0])]
    name = path['name']
    df = df.rename(
        columns={'aV(r)': f'aV(r)_{name}', 'err': f'err_{name}'})
    df['r/a'] = df['r/a']
    df[f'aV(r)_{name}'] = df[f'aV(r)_{name}']
    df[f'err_{name}'] = df[f'err_{name}']
    df = df.reset_index(drop=True)

    return df

def append_multiindex(df, values):
    """
    Add parameters to DataFrame as multiindex
    """
    for key, value in values.items():
        df[key] = value
    return df.set_index(list(values.keys()), append=True)

def get_potantial_df(paths, coluns_to_multiindex = []):
    """
    Read csv files for potential data and make DataFrame

    Parameters:
        paths: list of dictionaries with parameters
        coluns_to_multiindex: columns from csv file to turn to multiindex
    """
    df = []
    for path in paths:
        df.append(pd.read_csv(path['path']))
        for key, value in path['parameters'].items():
            df[-1][key] = value
        if 'constraints' in path:
            for key, val in path['constraints'].items():
                df[-1] = df[-1][(df[-1][key] >= val[0])
                                    & (df[-1][key] <= val[1])]
        if coluns_to_multiindex:
            df[-1] = df[-1].set_index(coluns_to_multiindex, append=True)
    df = pd.concat(df).reset_index(level=0, drop=True)
    return df[df['aV(r)'] != 0]
