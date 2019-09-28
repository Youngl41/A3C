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
from tensorflow.python.keras.layers.wrappers import TimeDistributed
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

# Define model
model_input                                 = keras.models.Input(shape=(84,84,3,1))
# CNN
x                                           = layers.Conv3D(32, kernel_size=(8,8,1), strides=(4,4,1), input_shape=state_size, activation='relu', padding='valid', kernel_initializer='random_uniform', bias_initializer='random_uniform')(model_input)
x                                           = layers.MaxPooling3D(pool_size=(2,2,1), strides=(1,1,1), padding='valid')(x)
x                                           = layers.Conv3D(64, kernel_size=(4,4,1), strides=(2,2,1), activation='relu', padding='valid', kernel_initializer='random_uniform', bias_initializer='random_uniform')(x)
x                                           = layers.MaxPooling3D(pool_size=(2,2,1), strides=(1,1,1), padding='valid')(x)
x                                           = layers.Conv3D(128, kernel_size=(3,3,1), strides=(1,1,1), activation='relu', padding='valid', kernel_initializer='random_uniform', bias_initializer='random_uniform')(x)
x                                           = layers.MaxPooling3D(pool_size=(2,2,1), strides=(1,1,1), padding='valid')(x)
# x                                         = self.dropout(x)
# Policy model
x1                                          = layers.Flatten()(x)
x1                                          = layers.Dense(256, activation='tanh')(x1)
logits                                      = layers.Dense(action_size, activation='linear')(x1)
# Value model
x2                                          = tf.stack([layers.Flatten()((x[:, :, :, i, :])) for i in range(3)], axis=1)
x2                                          = layers.Bidirectional(layers.LSTM(128, activation='relu', return_sequences=True))(x2)
x2                                          = layers.Bidirectional(layers.LSTM(128, activation='relu', return_sequences=True))(x2)
x2                                          = layers.LSTM(50, activation='relu')(x2)
values                                      = layers.Dense(1, activation='linear')(x2)

output                                      = logits, values
model_name                                  = 'A3C'
model                                       = keras.models.Model(inputs=model_input, outputs=output, name=model_name)
model.compile(optimizer='adam', loss='mse')
model.summary()



env                                         = gym.make('BreakoutNoFrameskip-v4')
env                                         = WarpFrame(env)
env                                         = dqn.MaxAndSkipEnv(env,4)
env                                         = dqn.ScaledFloatFrame(env)
env                                         = dqn.EpisodicLifeEnv(env)
env                                         = dqn.FireResetEnv(env)
env                                         = dqn.FrameStack(env, 3)
state_size                                  = (84,84,3)#env.observation_space.shape[0]
action_size                                 = env.action_space.n
print('\nState size:', state_size, 'Action size:', action_size, '\n')
# Define global model


start                                       = tf.convert_to_tensor(np.random.random((1, 84,84,3,1)), dtype=tf.float32)
a,b= model(start) # Initialise model (get weights)

a
np.array(b)


model.summary()

layer_name                                  = 'max_pooling3d_8'
intermediate_layer_model                    = keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
intermediate_output                         = intermediate_layer_model.predict(start)

intermediate_output.shape
