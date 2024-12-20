import pandas as pd
import argparse
import os

def read_copy(conf, path, copies):
    data = pd.DataFrame()
    for copy in range(1, copies + 1):
        file_path = f'{path}/{chain}/wilson_loop_{conf:04}_{copy}'
        #print(os.path.isfile(file_path))
        if (os.path.isfile(file_path)):
            #print(os.path.isfile(file_path))
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
parser.add_argument('--base_path')
parser.add_argument('--path_output_base')
parser.add_argument('--path_params')
parser.add_argument('--copies', type=int)
args = parser.parse_args()
print('args: ', args)

chains = ['/', 's0', 's1', 's2', 's3',
           's4', 's5', 's6', 's7', 's8']
conf_max = 1001
for chain in chains:
    #print(chain)
    #print(f'{args.base_path}/{args.path_params}/{chain}')
    if os.path.isdir(f'{args.base_path}/{args.path_params}/{chain}'):
        try:
            os.makedirs(f'{args.path_output_base}/{args.path_params}/{chain}')
        except:
            pass
        for conf in range(1001, conf_max + 1):
            path = f'{args.base_path}/{args.path_params}/{chain}'
            df = read_copy(conf, path, args.copies)
            if not df.empty:
                df = fillup_copies(df)
                df = df.drop('conf', axis = 1)
                for copy in df['copy'].unique():
                    df1 = df[df['copy'] == copy]
                    path_out = f'{args.path_output_base}/{args.path_params}/{chain}/wilson_loop_{conf:04}_{copy}'
                    df1.to_csv(path_out, index = False)
