#======================================================
# A3C main execution file
#======================================================
'''
Info:       
Version:    2.0
Author:     Young Lee
Created:    Saturday, 7 September 2019
'''

# Import modules
import argparse
import os
import sys
import threading
import time
from glob import glob
from queue import Queue

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from gym.envs.classic_control import rendering
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tqdm import tqdm

from baselines.common.atari_wrappers import WarpFrame

# Import custom modules
try:
    sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))) # 1 level upper dir
    sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, '..', '..')))) # 2 levels upper dir
except NameError:
    sys.path.append('.') # current dir
    sys.path.append('..') # 1 level upper dir
    sys.path.append(os.path.join(os.getcwd(), '..')) # 1 levels upper dir
    sys.path.append(os.path.join(os.getcwd(), '..', '..')) # 2 levels upper dir

from config.paths import main_dir
import utility.util_dqn as dqn
import utility.util_general as gen
from a3c_agents import RandomAgent
from a3c_util import record, Memory
# from a3c_cnn_main_windows import repeat_upsample, A3C, MasterAgent, Worker
from a3c_cnn_main_windows_v2 import MasterAgent as MasterAgent2

os.environ['CUDA_VISIBLE_DEVICES']                  = '-1'

master                                          = MasterAgent2()
master.train()


# Train model
'''
# Windows train
>>> python scripts\\a3c_cnn_main_windows.py --game-name BreakoutNoFrameskip-v4 --algorithm a3c --max-eps=1000000 --save-dir C:\\genesis_ai\\A3C\\models --train --update-freq 5 --memory-size 5 --framestack 2 --lr 0.00025 --gamma 0.99 --skip-frames 4 --save-freq 50 --time-limit 300
>>> python scripts\\a3c_cnn_main_windows.py --game-name BreakoutNoFrameskip-v4 --algorithm a3c --max-eps=1000000 --save-dir C:\\genesis_ai\\A3C\\models --train --update-freq 5 --memory-size 5 --framestack 2 --lr 0.00025 --gamma 0.99 --skip-frames 4 --save-freq 50 --trained-model C:\\genesis_ai\\A3C\\models\\model_BreakoutNoFrameskip-v4.h5

# Windows play
>>> python scripts\\a3c_cnn_main_windows.py --game-name BreakoutNoFrameskip-v4 --save-dir C:\\genesis_ai\\A3C\\models --framestack 2 --skip-frames 4 --periodic-save 1
'''

