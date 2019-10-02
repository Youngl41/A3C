#======================================================
# Test transfer learning
#======================================================
'''
Info:       
Version:    1.0
Author:     Young Lee
Created:    Sunday, 29 September 2019
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

os.environ['CUDA_VISIBLE_DEVICES']                  = '-1'


#------------------------------
# Utility functions
#------------------------------
# New A3C
def a3c(state_size, action_size):
    # Initialisation parameters
    framestack = 5#args.framestack
    # Define model
    model_input                                 = keras.models.Input(shape=(84,84,framestack,1))
    # CNN
    x                                           = layers.Conv3D(32, kernel_size=(8,8,1), strides=(4,4,1), input_shape=state_size, activation='relu', padding='valid', kernel_initializer='random_uniform', bias_initializer='random_uniform')(model_input)
    x                                           = layers.MaxPooling3D(pool_size=(2,2,1), strides=(1,1,1), padding='valid')(x)
    x                                           = layers.Conv3D(64, kernel_size=(4,4,1), strides=(2,2,1), activation='relu', padding='valid', kernel_initializer='random_uniform', bias_initializer='random_uniform')(x)
    x                                           = layers.MaxPooling3D(pool_size=(2,2,1), strides=(1,1,1), padding='valid')(x)
    x                                           = layers.Conv3D(128, kernel_size=(3,3,1), strides=(1,1,1), activation='relu', padding='valid', kernel_initializer='random_uniform', bias_initializer='random_uniform')(x)
    x                                           = layers.MaxPooling3D(pool_size=(2,2,1), strides=(1,1,1), padding='valid')(x)
    x                                           = tf.stack([layers.Flatten()((x[:, :, :, i, :])) for i in range(framestack)], axis=1)
    # x                                         = layers.Dropout(x)
    # Policy model
    x1                                          = layers.LSTM(256, activation='tanh', return_sequences=False)(x)
    # x1                                          = layers.Dense(64, activation='tanh')(x1)
    logits                                      = layers.Dense(action_size, activation='linear')(x1)
    # Value model
    x2                                          = layers.LSTM(256, activation='tanh', return_sequences=False)(x)
    # x2                                          = layers.Dense(64, activation='tanh')(x2)
    values                                      = layers.Dense(1, activation='linear')(x2)
    output                                      = logits, values
    model_name                                  = 'A3C'
    model                                       = keras.models.Model(inputs=model_input, outputs=output, name=model_name)
    model.compile(optimizer='adam', loss='mse')
    return model

# Old A3C
class A3C(keras.Model):
    def __init__(self, state_size, action_size):
        super(A3C, self).__init__()
        self.state_size                             = state_size
        self.action_size                            = action_size
        # keras.initializers.RandomUniform(minval=-1.0, maxval=1.0, seed=None)
        # Policy model
        self.conv1                                  = layers.Conv3D(32, kernel_size=(8,8,1), strides=(4,4,1), activation='relu', padding='valid', kernel_initializer='random_uniform', bias_initializer='random_uniform')#kernel_initializer=VarianceScaling(scale=2.0))
        self.pool1                                  = layers.MaxPooling3D(pool_size=(2,2,1), strides=(1,1,1), padding='valid')
        self.conv2                                  = layers.Conv3D(64, kernel_size=(4,4,1), strides=(2,2,1), activation='relu', padding='valid', kernel_initializer='random_uniform', bias_initializer='random_uniform')#kernel_initializer=VarianceScaling(scale=2.0))
        self.pool2                                  = layers.MaxPooling3D(pool_size=(2,2,1), strides=(1,1,1), padding='valid')
        self.conv3                                  = layers.Conv3D(128, kernel_size=(3,3,1), strides=(1,1,1), activation='relu', padding='valid', kernel_initializer='random_uniform', bias_initializer='random_uniform')#kernel_initializer=VarianceScaling(scale=2.0))
        self.pool3                                  = layers.MaxPooling3D(pool_size=(2,2,1), strides=(1,1,1), padding='valid')
        # self.dropout                              = layers.Dropout(0.5)
        self.flatten                                = layers.Flatten()
        # self.lstm = layers.LSTM(128, activation='tanh', bias_initializer='uniform')
        self.dense1                                 = layers.Dense(256, activation='tanh')
        self.policy_logits                          = layers.Dense(action_size, activation='linear')
        # Value model
        self.dense2                                 = layers.Dense(256, activation='tanh')
        self.values                                 = layers.Dense(1, activation='linear')
    # Forward
    def call(self, inputs):
        # Policy model
        x                                           = self.conv1(inputs)
        x                                           = self.pool1(x)
        x                                           = self.conv2(x)
        x                                           = self.pool2(x)
        x                                           = self.conv3(x)
        x                                           = self.pool3(x)
        # x                                         = self.dropout(x)
        x                                           = self.flatten(x)
        x1 = self.dense1(x)
        logits                                      = self.policy_logits(x1)
        # Value model
        x2                                          = self.dense2(x)
        values                                      = self.values(x2)
        return logits, values


#------------------------------
# Load model
#------------------------------
# Initialise old model
model_old = A3C(state_size=(84,84,2), action_size=4)
_ = model_old(tf.convert_to_tensor(np.random.random((1, 84,84,2,1)), dtype=tf.float32))

# Load old weights
weights_path = 'C:\\genesis_ai\\A3C\\models\\breakout\\model_BreakoutNoFrameskip-v4 (3 days).h5'
model_old.load_weights(weights_path)

model_old.layers[:6]
model_old.summary()

model_old

model_old_2 = A3C(state_size=(84,84,2), action_size=4)
_ = model_old_2(tf.convert_to_tensor(np.random.random((1, 84,84,2,1)), dtype=tf.float32))


def copy_layer_weights(model_from, model_to, layer_idx_from, layer_idx_to):
    model_to.layers[layer_idx_to].set_weights(model_from.layers[layer_idx_from].get_weights())

for i in range(6):
    copy_layer_weights(model_from=model_old_2, model_to=model_old, layer_idx_from=i, layer_idx_to=i)

model_old_2.get_weights()[6]
model_old.get_weights()[6]


model_new = a3c(state_size=(84,84,2), action_size=4)

model_new.summary()
model_new.layers
[print(l) for l in model_new.layers[:8]]
[print(l) for l in model_old.layers[:8]]

# Copy weights over
[copy_layer_weights(model_from=model_old, model_to=model_new, layer_idx_from=i, layer_idx_to=i+1) for i in range(6)]
model_new.get_weights()[0]
model_old.get_weights()[0]

# m = keras.models.load_model('C:\\genesis_ai\\A3C\\models\\breakout\\model_BreakoutNoFrameskip-v4_periodic_save.h5')


def a3c(state_size, action_size):
    # Initialisation parameters
    framestack = 4
    # Define model
    model_input                                 = keras.models.Input(shape=(84,84,framestack,1))
    # CNN
    x                                           = layers.Conv3D(32, kernel_size=(8,8,1), strides=(4,4,1), input_shape=state_size, activation='relu', padding='valid', kernel_initializer='random_uniform', bias_initializer='random_uniform')(model_input)
    x                                           = layers.MaxPooling3D(pool_size=(2,2,1), strides=(1,1,1), padding='valid')(x)
    x                                           = layers.Conv3D(64, kernel_size=(4,4,1), strides=(2,2,1), activation='relu', padding='valid', kernel_initializer='random_uniform', bias_initializer='random_uniform')(x)
    x                                           = layers.MaxPooling3D(pool_size=(2,2,1), strides=(1,1,1), padding='valid')(x)
    x                                           = layers.Conv3D(128, kernel_size=(3,3,1), strides=(1,1,1), activation='relu', padding='valid', kernel_initializer='random_uniform', bias_initializer='random_uniform')(x)
    x                                           = layers.MaxPooling3D(pool_size=(2,2,1), strides=(1,1,1), padding='valid')(x)
    x                                           = tf.stack([layers.Flatten()((x[:, :, :, i, :])) for i in range(framestack)], axis=1)
    # x                                         = layers.Dropout(x)
    # Policy model
    x1                                          = layers.LSTM(64, activation='tanh', return_sequences=False, bias_initializer='random_uniform')(x)
    x1                                          = layers.Dense(64, activation='tanh')(x1)
    logits                                      = layers.Dense(action_size, activation='linear')(x1)
    # Value model
    x2                                          = layers.LSTM(32, activation='tanh', return_sequences=False, bias_initializer='random_uniform')(x)
    x2                                          = layers.Dense(32, activation='tanh')(x2)
    values                                      = layers.Dense(1, activation='linear')(x2)
    output                                      = logits, values
    model_name                                  = 'A3C'
    model                                       = keras.models.Model(inputs=model_input, outputs=output, name=model_name)
    # Transfer learn
    # Initialise old model
    model_old = A3C(state_size=(84,84,2), action_size=4)
    _ = model_old(tf.convert_to_tensor(np.random.random((1, 84,84,2,1)), dtype=tf.float32))
    # Load old weights
    weights_path = 'C:\\genesis_ai\\A3C\\models\\breakout\\model_BreakoutNoFrameskip-v4 (3 days).h5'
    model_old.load_weights(weights_path)
    [copy_layer_weights(model_from=model_old, model_to=model, layer_idx_from=i, layer_idx_to=i+1) for i in range(6)]
    for layer in model.layers[1:7]:
        layer.trainable = False
    model.compile(optimizer='adam', loss='mse')
    return model

path = 'C:\\genesis_ai\\A3C\\models\\model_BreakoutNoFrameskip-v4_periodic_save.h5'
model_new.load_weights(path)

m.save('C:\\genesis_ai\\A3C\\models\\BreakoutNoFrameskip-v4_periodic_save.h5')

m = keras.models.load_model('C:\\genesis_ai\\A3C\\models\\BreakoutNoFrameskip-v4_periodic_save.h5')
m.summary()
[print(x.trainable) for x in m.layers]
for layer in m.layers:
    layer.trainable = True


import numpy as np
import tensorflow as tf
from datetime import datetime
tf.convert_to_tensor(np.random.random((1, 84,84,4,1)), dtype=tf.float32)
model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(84,84,3), include_top=False, weights='imagenet')

st = datetime.now()
model.predict((tf.convert_to_tensor(np.random.random((1, 84,84,3)), dtype=tf.float32)))
print(datetime.now()-st)
model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(84,84,3), weights='imagenet')
model.summary()
model.predict((tf.convert_to_tensor(np.random.random((1, 84,84,3)), dtype=tf.float32))).shape


env                                         = gym.make('BreakoutNoFrameskip-v4')
env                                         = WarpFrame(env, width=160, height=160)
env                                         = dqn.MaxAndSkipEnv(env,4)
env                                         = dqn.ScaledFloatFrame(env)
env                                         = dqn.EpisodicLifeEnv(env)
env                                         = dqn.FireResetEnv(env)
env                                         = dqn.FrameStack(env, args.framestack)
env.observation_space.shape
model = keras.applications.nasnet.NASNetMobile(input_shape=(150,84,1), include_top=False, weights='imagenet', input_tensor=None, pooling=None, classes=1000)

model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(160,160,3), weights='imagenet')
model.summary()