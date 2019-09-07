# Windows gpu supported tensorflow environment build

# Install conda environment
conda update -n base -c defaults conda
conda remove -n gym --all
conda create -n gym python=3.6
conda activate gym

# conda remove -n rl --all
# conda create -n rl python=3.7
# conda activate rl

# Upgrade pip
python -m pip install --upgrade pip

# Install tensorflow
# pip install tensorflow==2.0.0-rc0
pip install tensorflow-gpu==2.0.0-rc0

# Install gym
conda install git
conda install -c conda-forge ffmpeg
conda update --all
conda clean -a
pip install git+https://github.com/Kojoley/atari-py.git
conda install swig
pip install Box2D
pip install gym[atari]

# Install pre-requisites
pip install matplotlib
pip install dropbox
pip install tqdm
pip install pandas

# Install baselines
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .


conda activate gym
python scripts\\a3c_cnn_main.py --game-name DemonAttackNoFrameskip-v4 --algorithm a3c --max-eps=30000 --save-dir C:\\genesis_ai\\A3C\\models --train --update-freq 30 --memory-size 30 --framestack 1 --lr 0.00025 --gamma 0.99 --time-limit 5000 --skip-frames 3 --trained-model C:\\Users\\Young\\Dropbox\\Apps\\reinforcement_learning\\a3c\\models\\model_DemonAttackNoFrameskip-v4_periodic_save.h5

python scripts\\a3c_cnn_main.py --game-name DemonAttackNoFrameskip-v4 --algorithm a3c --max-eps=10000 --save-dir C:\\genesis_ai\\A3C\\models --train --update-freq 30 --memory-size 30 --framestack 1 --lr 0.00025 --gamma 0.99 --time-limit 300 --skip-frames 3

python scripts\\a3c_cnn_main.py --game-name DemonAttackNoFrameskip-v4 --save-dir C:\\genesis_ai\\A3C\\models --framestack 1 --skip-frames 3 --periodic-save 1



set DISPLAY=:0



# Test
python
import tensorflow as tf
import numpy as np

def my_func(arg):
  arg = tf.compat.v2.convert_to_tensor(arg, dtype=tf.float32)
  return tf.compat.v2.matmul(arg, arg) + arg

# The following calls are equivalent.
value_1 = my_func(tf.compat.v2.constant([[1.0, 2.0], [3.0, 4.0]]))
value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])
value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))



# set DISPLAY=:0

import gym

env = gym.make('CartPole-v0')
env = gym.make('DemonAttackNoFrameskip-v4')
env.reset()

# for _ in range(100):
#     env.render()
#     env.step(env.action_space.sample())
    
# env.close()






# env                                       = gym.make('DemonAttackNoFrameskip-v4')
# env                                       = WarpFrame(env)
# env                                       = dqn.MaxAndSkipEnv(env,3)
# env                                       = dqn.ScaledFloatFrame(env)
# env                                       = dqn.EpisodicLifeEnv(env)
# env                                       = dqn.FrameStack(env, 1)
# state_size                           = (84,84,3)#env.observation_space.shape[0]
# action_size                          = env.action_space.n
# # Define global model



# # Import modules
# import argparse
# import os
# import sys
# import time
# import threading
# from glob import glob
# from queue import Queue

# import gym
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# # from tensorflow.keras.optimizers import Adam
# from tensorflow.python import keras
# from tensorflow.python.keras import layers
# from tqdm import tqdm

# try:
#     from stable_baselines.common.atari_wrappers import WarpFrame
# except ModuleNotFoundError:
#     try: 
#         from stable_baselines.common.atari_wrappers import WarpFrame
#     except ModuleNotFoundError:
#         from baselines.common.atari_wrappers import WarpFrame

# # Import custom modules
# try:
#     sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))) # 1 level upper dir
#     sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, '..', '..')))) # 2 levels upper dir
# except NameError:
#     sys.path.append('.') # current dir
#     sys.path.append('..') # 1 level upper dir
#     sys.path.append(os.path.join(os.getcwd(), '..')) # 1 levels upper dir
#     sys.path.append(os.path.join(os.getcwd(), '..', '..')) # 2 levels upper dir

# from config.paths import main_dir
# import utility.util_dqn as dqn
# from a3c_agents import RandomAgent
# from a3c_util import record, Memory


# # # _                                         = global_model(tf.convert_to_tensor(np.random.random((1, 84,84,3,1)), dtype=tf.float32)) # Initialise model (get weights)

# # import tensorflow as tf
# # import numpy as np
# # global_model                         = A3C((84,84,3),6)
# tf.enable_eager_execution()
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.convert_to_tensor(np.random.random((1, 84,84,3,1)))
# global_model(tf.convert_to_tensor(np.random.random((1, 84,84,3,1)), dtype=tf.float32))


