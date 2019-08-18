#======================================================
# EDA Utility Functions
#======================================================
'''
Info:       Utility functions for exploratory data 
            analysis.
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
from datetime import datetime


#------------------------------
# Utility Functions
#------------------------------
def get_crosstab(df, dependent_col, independent_col):
    print('\nCross tabulation of:')
    print('\t- '+independent_col)
    print('\t- '+dependent_col + '\n')
    ct = pd.crosstab(df[independent_col], df[dependent_col],dropna=False)
    ct_perc = np.round(ct.div(np.sum(ct, axis=1), axis=0) * 100, 1)
    ct_perc = ct_perc.sort_values(ct_perc.columns[1], ascending=False)
    ct_perc = ct_perc.applymap(lambda x: str(x) + '%')
    ct_perc.loc[:, 'Support'] = np.sum(ct, axis=1).astype(int)
    print(ct_perc)
    return ct, ct_perc