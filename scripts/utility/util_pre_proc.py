#======================================================
# Data Pre-processing Utility Functions
#======================================================
'''
Info:       Utility functions for pre-processing data.
Version:    2.0
Author:     Young Lee
Created:    Saturday, 13 April 2019
'''
# Import modules
import numpy as np
import pandas as pd
import sys
#import dill as pickle
import pickle
main_dir = '/Users/younle/Documents/projects/promo_insights'
sys.path.append(main_dir)
from utility.util_general import set_section


#------------------------------
# Utility Functions
#------------------------------
# Check primary key uniqueness
def check_pk(df, pk_list, verbose=1):
    pk_rows = len(df[pk_list].drop_duplicates(keep='first'))
    total_rows = len(df)
    print('Primary Key Rows:\t', pk_rows)
    print('Total Numebr of Rows:\t', total_rows)
    if pk_rows == total_rows:
        return 1
    else:
        if verbose:
            return df[df.duplicated(pk_list, keep=False)].sort_values(pk_list)
        return 0

# Drop useless columns
def drop_useless_columns(df):
    useless_columns = list((df.nunique()[df.nunique() == 1]).index)
    set_section('Dropping columns with only 1 value')
    for col in useless_columns:
        print('Dropped:\t', col)
    df = df.drop(useless_columns, axis=1)
    print('\n\n')
    return df

# Find nulls
def find_nulls(df):
    set_section('Null columns')
    null_count = np.sum(df.isnull(), axis=0).reset_index()
    null_count = null_count[null_count[0]>0]
    null_count.loc[:, '%_null'] = np.round(null_count.loc[:, 0]/len(df) * 100.0, 1)
    null_count.columns = ['col', 'n_nulls', '%_null']
    return null_count

# Kill columns with large num nulls
def kill_bad_nulls(df, null_perc_thr=50):
    set_section('Killing columns with >' + str(null_perc_thr) + '% nulls')
    null_count = np.sum(df.isnull(), axis=0).reset_index()
    null_count = null_count[null_count[0]>0]
    null_count.loc[:, '%_null'] = np.round(null_count.loc[:, 0]/len(df) * 100.0, 1)
    null_count.columns = ['col', 'n_nulls', '%_null']
    kill_me_cols = list(null_count[null_count['%_null']>null_perc_thr].col)
    for col in kill_me_cols:
        print('Killed:\t', col)
    df = df.drop(kill_me_cols, axis=1)
    print('\n\n')
    return df

# Get high frequency field values
def get_hfv(df, col, thr=0.01):
    temp_df = df[col].value_counts(dropna=False) / len(df)
    return list(temp_df[temp_df>thr].index.values)