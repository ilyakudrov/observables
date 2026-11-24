import os
import pandas as pd
import argparse

def get_dir_names(path: str) -> list[str]:
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

def read_PL(file_path):
    with open(file_path, "r") as file:
        HYP_arr = []
        conf_arr = []
        PL_arr = []
        for line in file:
            if (line.startswith("# TASKINFO:") or
            line.startswith("# PARAM:") or
            line.startswith("# IFILE:") or
            line.startswith("# PL")):
                pass
            elif line.startswith("# CPARAM: nameHYPT"):
                HYP_step = 0
            elif line.startswith("# IPARAM: itrj0"):
                conf_num = 0
            elif line.startswith("# IPARAM:"):
                conf_num = int(line[line.rstrip().rfind('iconf')+5:line.rstrip().rfind('mode')].rstrip())
            elif line.startswith("# CPARAM:"):
                HYP_step = int(line[line.rstrip().rfind('nbs')+3:])
            elif not line.startswith("#"):
                PL_arr.append(float(line.rstrip()))
                conf_arr.append(conf_num)
                HYP_arr.append(HYP_step)
    return pd.DataFrame({"HYP": HYP_arr, "conf": conf_arr, "PL": PL_arr})

def read_condensates(file_path):
    with open(file_path, "r") as file:
        conf_arr = []
        chiral_condensate_arr = []
        diquark_condensate_arr = []
        for line in file:
            if (line.startswith("# TASKINFO:") or
            line.startswith("# LSOBS:") or
            line.startswith("# PARAM:") or
            line.startswith("#ChiralCond")):
                pass
            elif line.startswith("# IPARAM:"):
                conf_num = int(line[line.rstrip().rfind('iconf')+5:line.rstrip().rfind('mode')].rstrip())
            elif not line.startswith("#"):
                chiral_condensate_arr.append(float(line.rstrip().split()[0].rstrip()))
                diquark_condensate_arr.append(float(line.rstrip().split()[1].rstrip()))
                conf_arr.append(conf_num)
    return pd.DataFrame({"conf": conf_arr, "chiral_condensate": chiral_condensate_arr,
                        "diquark_condensate": diquark_condensate_arr})

def read_quark_density(file_path):
    with open(file_path, "r") as file:
        HYP_arr = []
        conf_arr = []
        quark_density_arr = []
        for line in file:
            if (line.startswith("# TASKINFO:") or
            line.startswith("# LSOBS:") or
            line.startswith("# PARAM:") or
            line.startswith("#rho_q")):
                pass
            elif line.startswith("# IPARAM:"):
                conf_num = int(line[line.rstrip().rfind('iconf')+5:line.rstrip().rfind('mode')].rstrip())
            elif not line.startswith("#"):
                quark_density_arr.append(float(line.rstrip()))
                conf_arr.append(conf_num)
                HYP_arr.append(HYP_step)
    return pd.DataFrame({"HYP": HYP_arr, "conf": conf_arr, "quark_density": quark_density_arr})

def parse_lattice(lattice):
    splited = lattice.split('_')
    Nt = int(splited[0][2:])
    Ns = int(splited[1][2:])
    return Nt, Ns

def parse_mu(mu):
    splited = mu.split('_')
    mu_re = float(splited[1][3:])
    if len(splited) == 3:
        chain = splited[2]
    else:
        chain = 's0'
    return mu_re, chain

def parse_additional_parameters(additional_parameters):
    splited = additional_parameters.split('_')
    if len(splited) == 3:
        rho = float(splited[0][3:])
        ma = float(splited[1][2:])
        lamb = float(splited[2][6:])
        stout_steps = int(2)
    if len(splited) == 4:
        rho = float(splited[0][3:])
        stout_steps = int(splited[1])
        ma = float(splited[2][2:])
        lamb = float(splited[3][6:])
    return rho, stout_steps, ma, lamb

parser = argparse.ArgumentParser()
parser.add_argument('--base_path')
parser.add_argument('--observable')
args = parser.parse_args()
print('args: ', args)

data = []
lattice_dirs = get_dir_names(args.base_path)
for lattice in lattice_dirs:
    Nt, Ns = parse_lattice(lattice)
    beta_dirs = get_dir_names(f'{args.base_path}/{lattice}')
    for beta in beta_dirs:
        additional_parameters_dirs = get_dir_names(f'{args.base_path}/{lattice}/{beta}')
        for additional_parameters in additional_parameters_dirs:
            rho, stout_steps, ma, lamb = parse_additional_parameters(additional_parameters)
            mu_dirs = get_dir_names(f'{args.base_path}/{lattice}/{beta}/{additional_parameters}')
            for mu in mu_dirs:
                mu_re, chain = parse_mu(mu)
                if args.observable == 'PL':
                    file_path = f'{args.base_path}/{lattice}/{beta}/{additional_parameters}/{mu}/PL.txt'
                    if os.path.isfile(file_path):
                        df_tmp = read_PL(file_path)
                elif args.observable == 'Condensates':
                    file_path = f'{args.base_path}/{lattice}/{beta}/{additional_parameters}/{mu}/Condensates.txt'
                    if os.path.isfile(file_path):
                        df_tmp = read_condensates(file_path)
                elif args.observable == 'QuarkDensity':
                    file_path = f'{args.base_path}/{lattice}/{beta}/{additional_parameters}/{mu}/QuarkDensity.txt'
                    if os.path.isfile(file_path):
                        df_tmp = read_quark_density(file_path)
                else:
                    raise ValueError(f'Unknown observable: {args.observable}')
                df_tmp['Nt'] = Nt
                df_tmp['Ns'] = Ns
                df_tmp['mu'] = mu_re
                df_tmp['rho'] = rho
                df_tmp['stout_steps'] = stout_steps
                df_tmp['ma'] = ma
                df_tmp['lambda'] = lamb
                df_tmp['chain'] = chain
                data.append(df_tmp)
data = pd.concat(data)

output_path = f'../../../result/qc2dstag/from_kiae'
try:
    os.makedirs(f'{output_path}')
except:
    pass
if args.observable == 'PL':
    data.to_csv(f'{output_path}/polyakov_loop.csv', index=False)
if args.observable == 'Condensates':
    data.to_csv(f'{output_path}/condensates.csv', index=False)
if args.observable == 'QuarkDensity':
    data.to_csv(f'{output_path}/quark_density.csv', index=False)