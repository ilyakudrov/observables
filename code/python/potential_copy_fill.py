import pandas as pd
import argparse
import os

def read_copy(conf, path, copies):
    data = pd.DataFrame()
    for copy in copies:
        file_path = f'{path}/{chain}/wilson_loop_{conf:04}_{copy}'
        if (os.path.isfile(file_path)):
            df = pd.read_csv(file_path)
            df["conf"] = f'{conf}-{chain}'
            df["copy"] = copy
            data = pd.concat([data, df])
    return data

def fillup_copies(df):
    copy_num = df.groupby(['copy']).ngroups
    if copy_num > 1:
        for i in range(1, copy_num + 1):
            df1 = df[df['copy'] == i - 1]
            df2 = df[df['copy'] == i]
            df3 = df1[~df1['conf'].isin(df2['conf'])]
            df3.loc[:, 'copy'] = i
            df = pd.concat([df, df3])
    return df

parser = argparse.ArgumentParser()
parser.add_argument('--axis')
parser.add_argument('--base_path')
parser.add_argument('--conf_type')
parser.add_argument('--conf_size')
parser.add_argument('--theory_type')
parser.add_argument('--operator_type')
parser.add_argument('--representation')
parser.add_argument('--smearing')
parser.add_argument('--mu')
parser.add_argument('--matrix_type')
parser.add_argument('--smearing_param')
parser.add_argument('--additional_parameters')
parser.add_argument('--beta')
parser.add_argument('--path_output_base')
parser.add_argument('--copies', type=int)
args = parser.parse_args()
print('args: ', args)

chains = ['/', 's0', 's1', 's2', 's3',
           's4', 's5', 's6', 's7', 's8']
conf_max = 9999
for chain in chains:
    for conf in range(0, conf_max + 1):
        path = f'{args.base_path}/{args.smearing}/{args.operator_type}/{args.representation}/{args.axis}\
            /{args.theory_type}/{args.conf_type}/{args.conf_size}/{args.beta}/{args.mu}/{args.matrix_type}\
            /{args.smearing_param}/{args.additional_parameters}/{chain}'
        df = read_copy(conf, path, args.copies)
        if not df.empty:
            df = fillup_copies(df)
            for copy in df['copy'].unique():
                path_out = f'{args.path_output_base}/{args.smearing}/{args.operator_type}/{args.representation}/{args.axis}\
                    /{args.theory_type}/{args.conf_type}/{args.conf_size}/{args.beta}/{args.mu}/{args.matrix_type}\
                    /{args.smearing_param}/{args.additional_parameters}/{chain}/wilson_loop_{conf:04}_{copy}'
                try:
                    os.makedirs(f'{args.path_output_base}/{args.smearing}/{args.operator_type}/{args.representation}/{args.axis}\
                        /{args.theory_type}/{args.conf_type}/{args.conf_size}/{args.beta}/{args.mu}/{args.matrix_type}\
                        /{args.smearing_param}/{args.additional_parameters}/{chain}')
                except:
                    pass
                df.to_csv(path_out)
