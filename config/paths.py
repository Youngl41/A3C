#======================================================
# Config
#======================================================
'''
Info:       Append main working directory and scripts 
            folders to system path.
Version:    1.0
Author:     Young Lee
Created:    Saturday, 13 April 2019

'''
# Import modules
import os
import sys


#------------------------------
# Set up working dir
#------------------------------
if 'scripts' in os.getcwd():
    main_dir    = os.getcwd().split('scripts')[0]
elif os.path.exists(os.path.join(os.getcwd(), 'scripts')):
    main_dir    = os.getcwd()
else:
    raise Exception('\n\nCannot find main directory.\nMake sure to execute the script from project folder.\nE.g. from folder that contains scripts, or the .py file.\n')

# Append main dir and subpaths
scripts_dir     = os.path.join(main_dir, 'scripts')
sub_scripts_dir = [dir[0] for dir in os.walk(scripts_dir)]
sys.path.append(main_dir)
for dir in sub_scripts_dir:
    sys.path.append(dir)