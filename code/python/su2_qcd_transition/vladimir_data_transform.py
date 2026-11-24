import os
import pandas as pd
from typing import List
import argparse

def get_dir_names(path: str) -> List[str]:
    """Get names of subdirectories in path.

    Args:
        path: path where to find subdirectories

    Returns: list of names of the subdirectories
    """
    directories = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        directories.extend(dirnames)
        break
    return directories

def parse_lattice_sizes(lattice_name):
    splited = lattice_name.split('_')
    return int(splited[0][2:]), int(splited[1][2:])

def parse_mu_value(mu_name):
    splited = mu_name.split('_')
    if len(splited) > 2:
        chain = splited[2]
    else:
        chain = 's0'
    return float(splited[1][2:]), chain

def parse_add_params(add_params):
    splited = add_params.split('_')
    if len(splited) == 3:
        return float(splited[0][3:]), 2, float(splited[1][2:]), float(splited[2][6:]), 'calc'
    else:
        return float(splited[0][3:]), int(splited[1]), float(splited[2][2:]), float(splited[3][6:]), 'test'

def parse_HYP(line):
    splited = line.split()
    return int(line[line.find('nbs') + 3:])

def parse_conf(line):
    return int(line[line.find('iconf') + 5:line.find('mode') - 1])

def read_data_PL(filename):
    data = []
    HYP_arr = []
    conf_arr = []
    HYP = 0
    conf = 0
    with open(filename) as file:
        while line := file.readline():
            if(line[:8] == '# CPARAM'):
                HYP = parse_HYP(line)
            elif(line[:8] == '# IPARAM'):
                conf = parse_conf(line)
            elif(line[0] != '#'):
                data.append(float(line))
                HYP_arr.append(HYP)
                conf_arr.append(conf)
    df = pd.DataFrame({'PL': data, 'HYP': HYP_arr, 'conf': conf_arr})
    df = df.astype({'conf': int, 'PL': float, 'HYP': int})
    return df

def read_data_condensates(filename):
    data_chiral = []
    data_diquark = []
    conf = []
    stoch_vec = []
    count = 1
    with open(filename) as file:
        while line := file.readline():
            if line.startswith('# IPARAM'):
                conf_num = int(line.split(' ')[3][5:])
                count = 1
            if line[0] != '#':
                conf.append(conf_num)
                data_chiral.append(float(line.split('\t')[0]))
                data_diquark.append(float(line.split('\t')[1]))
                stoch_vec.append(count)
                count += 1
    df = pd.DataFrame({'conf': conf, 'chiral_condensate': data_chiral, 'diquark_condensate': data_diquark, 'stoch_vec': stoch_vec})
    df = df.astype({'conf': int, 'chiral_condensate': float, 'diquark_condensate': float, 'stoch_vec': int})
    return df

def read_data_quark_density(filename):
    data = []
    conf = []
    stoch_vec = []
    count = 1
    with open(filename) as file:
        while line := file.readline():
            if line.startswith('# IPARAM'):
                conf_num = int(line.split(' ')[3][5:])
                count = 1
            if line[0] != '#':
                conf.append(conf_num)
                data.append(float(line))
                stoch_vec.append(int(count))
                count += 1
    df = pd.DataFrame({'conf': conf, 'quark_density': data, 'stoch_vec': stoch_vec})
    df.astype({'conf': int, 'quark_density': float, 'stoch_vec': int})
    return df

parser = argparse.ArgumentParser()
parser.add_argument('--data_path')
parser.add_argument('--observable')
args = parser.parse_args()
print('args: ', args)

lattice_dirs = get_dir_names(args.data_path)
df = []
for lattice in lattice_dirs:
    if lattice.startswith('Nt'):
        Nt, Ns = parse_lattice_sizes(lattice)
        beta_dirs = get_dir_names(f'{args.data_path}/{lattice}')
        for beta in beta_dirs:
            additional_params_dirs = get_dir_names(f'{args.data_path}/{lattice}/{beta}')
            for add_params in additional_params_dirs:
                rho, stout_steps, ma, lamb, calc_type = parse_add_params(add_params)
                mu_dirs = get_dir_names(f'{args.data_path}/{lattice}/{beta}/{add_params}')
                for mu_name in mu_dirs:
                    mu, chain = parse_mu_value(mu_name)
                    if (os.path.isfile(f'{args.data_path}/{lattice}/{beta}/{add_params}/{mu_name}/{args.observable}.txt')):
                        if args.observable == 'PL':
                            df_tmp = read_data_PL(f'{args.data_path}/{lattice}/{beta}/{add_params}/{mu_name}/{args.observable}.txt')
                        if args.observable == 'QuarkDens':
                            df_tmp = read_data_quark_density(f'{args.data_path}/{lattice}/{beta}/{add_params}/{mu_name}/{args.observable}.txt')
                        if args.observable == 'Condensates':
                            df_tmp = read_data_condensates(f'{args.data_path}/{lattice}/{beta}/{add_params}/{mu_name}/{args.observable}.txt')
                        if chain == 's0':
                            df_tmp = df_tmp[df_tmp['conf'] > 200]
                        df_tmp['Nt'] = Nt
                        df_tmp['Ns'] = Ns
                        df_tmp['mu'] = mu
                        df_tmp['chain'] = chain
                        df_tmp['rho'] = rho
                        df_tmp['ma'] = ma
                        df_tmp['lambda'] = lamb
                        df_tmp['stout_steps'] = stout_steps
                        df_tmp['calc_type'] = calc_type
                        df.append(df_tmp)

df = pd.concat(df)
# print(df[(df['Nt'] == 10) & (df['Ns'] == 32) & (df['mu'] == 0.15) & (df['chain'] == 's3') & (df['conf'] == 80)])
print(df)
if args.observable == 'PL':
    path_output = '/home/ilya/soft/lattice/observables/result/vladimir_data/merged/polyakov_loop'
if args.observable == 'QuarkDens':
    path_output = '/home/ilya/soft/lattice/observables/result/vladimir_data/merged/quark_density'
if args.observable == 'Condensates':
    path_output = '/home/ilya/soft/lattice/observables/result/vladimir_data/merged/condensates'
try:
    os.makedirs(f'{path_output}')
except:
    pass
if args.observable == 'PL':
    df.to_csv(f"{path_output}/polyakov_loops.csv", index=False)
if args.observable == 'QuarkDens':
    df.to_csv(f"{path_output}/quark_density.csv", index=False)
if args.observable == 'Condensates':
    df.to_csv(f"{path_output}/condensates.csv", index=False)
