import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from tsseg.omslr import *


def try_int(s):
    try:
        return int(s)
    except:
        return None

def log_parser(expname):
    lines = list(filter(None, open(f'result/{expname}/log.txt').read().split('\n')))
    # print(lines)
    cno = None
    exp_results = []
    exp_result = {}
    for line in lines:
        exp_no = try_int(line)
        if exp_no != None:
            if exp_result != {}:
                exp_results.append(exp_result)
            cno = exp_no
            seg = 1
            continue
        else:
            err = float(line[15:])
            exp_result[seg] = err
            seg += 1
    exp_results.append(exp_result)
    return exp_results

def naive_parse(expname, exp_no):
    lines = filter(None, open(f'result/{expname}/exp/{exp_no}/naive.txt').read().split('\n'))
    errors = []
    for line in lines:
        part = line.split('\t')
        error = float(part[0])
        pvts = eval(part[1])
        errors.append(error)
    return errors

def migrate(expname):
    df = pd.read_csv(f'result/{expname}/exp/stat.csv', index_col='exp_no')
    df['es_time'] = np.nan
    df['es_k'] = np.nan
    df['es_err'] = np.nan
    # print(df)
    spend_result = log_parser(expname)
    for i in range(10):
        error_result = naive_parse(expname, i)
        spend = spend_result[i][4]
        err = error_result[-1]
        df.iloc[i, df.columns.get_loc('es_time')] = spend
        df.iloc[i, df.columns.get_loc('es_k')] = 4
        df.iloc[i, df.columns.get_loc('es_err')] = err
        # print(i, spend, err)
    df2 = df[['data_length','op_time','es_time','td_time','bu_time','es_err','op_err','td_err','bu_err','es_k','op_k','td_k','bu_k']]
    df.to_csv(f'result/{expname}/stat_migrate.csv', float_format='%.4f')
    df2.to_csv(f'result/{expname}/stat_migrate_sort_column.csv', float_format='%.4f')

migrate('simun')
migrate('simup')