#======================================================
# General Utility Functions
#======================================================
'''
Info:       Utility functions for general applications.
Version:    2.0
Author:     Young Lee
Created:    Saturday, 13 April 2019
'''
# Import modules
import os
import numpy as np
import pandas as pd
import sys
#import dill as pickle
import pickle
from datetime import datetime


#------------------------------
# Utility Functions
#------------------------------
# Set title
def set_title(string):
    # Check if string is too long
    string_size = len(string)
    max_length  = 57
    if string_size > max_length:
        print('TITLE TOO LONG')
    else:
        lr_buffer_len   = int((max_length - string_size) / 2)
        full_buffer_len = lr_buffer_len * 2 + string_size
        print('\n')
        print(full_buffer_len * '=')
        print(full_buffer_len * ' ')
        print(lr_buffer_len * ' ' + string + lr_buffer_len * ' ')
        print(full_buffer_len * ' ')
        print(full_buffer_len * '='+'\n\n')

# Set section
def set_section(string):
    # Check if string is too long
    string_size = len(string)
    max_length  = 100
    if string_size > max_length:
        print('TITLE TOO LONG')
    else:
        full_buffer_len = string_size
        print('\n')
        print(full_buffer_len * '-')
        print(string)
        print(full_buffer_len * '-'+'\n')

# Print time taken
def print_dur(string, st):
    print(string, datetime.now() - st)

# Date conversion
def pdf_cast_date(df, date_field):
    #df.loc[:, date_field] = list(map(lambda x: pd.to_datetime(x, format='%m/%d/%Y %H:%M'), list(df.loc[:, date_field])))
    #df.loc[:, date_field] = list(map(lambda x: pd.to_datetime(x, format='%Y/%m/%d %H:%M:%S'), list(df.loc[:, date_field])))
    df.loc[:, date_field] = list(map(lambda x: pd.to_datetime(x, format='%d/%m/%Y %H:%M'), list(df.loc[:, date_field])))
    return df

# Create date table
def create_date_table(start='2000-01-01', end='2050-12-31'):
    start_ts = pd.to_datetime(start).date()

    end_ts = pd.to_datetime(end).date()

    # Record timetsamp is empty for now
    dates =  pd.DataFrame(columns=['Record_timestamp'],
        index=pd.date_range(start_ts, end_ts))
    dates.index.name = 'date'

    days_names = {
        i: name
        for i, name
        in enumerate(['Monday', 'Tuesday', 'Wednesday',
                      'Thursday', 'Friday', 'Saturday', 
                      'Sunday'])
    }

    dates['day'] = dates.index.dayofweek.map(days_names.get)
    dates['week'] = dates.index.week
    dates['month'] = dates.index.month
    dates['quarter'] = dates.index.quarter
    dates['year_half'] = dates.index.month.map(lambda mth: 1 if mth <7 else 2)
    dates['year'] = dates.index.year
    dates.reset_index(inplace=True)
    dates = dates.drop('Record_timestamp', axis=1)
    return dates

# Ensure dir
def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

#------------------------------
# Save data
#------------------------------
def save(data, file_path):
    # Define output file
    pickle.dump(data, open(file_path, 'wb')) 

def load(file_path):
    return pickle.load(open(file_path, 'rb'))


#------------------------------
# Set up folder structure
#------------------------------
def setup(main_dir):
    # Create directories
    config_dir  = os.path.join(main_dir, 'config'); ensure_dir(config_dir)
    data_dir    = os.path.join(main_dir, 'data'); ensure_dir(data_dir)
    model_dir   = os.path.join(main_dir, 'model'); ensure_dir(model_dir)
    doc_dir     = os.path.join(main_dir, 'doc'); ensure_dir(doc_dir)
    results_dir = os.path.join(main_dir, 'results'); ensure_dir(results_dir)
    plots_dir   = os.path.join(main_dir, 'plots'); ensure_dir(plots_dir)
    scripts_dir = os.path.join(main_dir, 'scripts'); ensure_dir(scripts_dir)
    utility_dir = os.path.join(main_dir, 'scripts', 'utility'); ensure_dir(utility_dir)
    temp_dir    = os.path.join(main_dir, 'temp'); ensure_dir(temp_dir)
    # Create .gitignore
    pass
