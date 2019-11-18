#======================================================
# CNN pre-train model
#======================================================
'''
Info:       
Version:    1.0
Author:     Young Lee
Created:    Wednesday, 16 October 2019
'''

# Import modules
import argparse
import os
import sys
import threading
import time
from glob import glob
from queue import Queue

from collections import deque

import cv2
import gym
from PIL import Image
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

# parser                                              = argparse.ArgumentParser(description='Run A3C algorithm on the game Cartpole.')
# parser.add_argument('--game-name', default='BreakoutNoFrameskip-v4', type=str, help='Game name.')
# parser.add_argument('--algorithm', default='a3c', type=str, help='Choose between \'a3c\' and \'random\'.')
# args                                                = parser.parse_args()


#------------------------------
# Utility functions
#------------------------------
# Unpickle cifar data
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def pp_cifar_img(array, greyscale=True, enlarge_dim=(84,84), verbose=False):
    # Take image from array
    image = Image.fromarray(array)
    # Convert to greyscale or RGB
    if greyscale:
        image = image.convert('L')
    else:
        image = image.convert('RGB')
    image_array = np.array(image)
    # Enlarge
    width, height = enlarge_dim
    image_array = cv2.resize(image_array, (width, height), interpolation=cv2.INTER_CUBIC)
    # Show image
    if verbose:
        plt.imshow(image_array)
        plt.show()
    return image_array


#------------------------------
# CIFAR data
#------------------------------
# Load CIFAR 100 data
cifar_dir = os.path.join(main_dir, 'data', 'cifar-100-python')
meta_data = unpickle(os.path.join(cifar_dir, 'meta'))
train_data = unpickle(os.path.join(cifar_dir, 'train'))
test_data = unpickle(os.path.join(cifar_dir, 'test'))
train_data.keys()
train_data[b'fine_labels'][0]
train_data[b'data'][0]

meta_data[b'fine_label_names']
b'batch_label', b'fine_labels', b'coarse_labels', b'data'


# Image dimensions
 
# Rotate the image by right angles


# Reshape raw dataset
train_arr = train_data[b'data'].reshape(train_data[b'data'].shape[0], 3, 32, 32)
train_arr = train_arr.transpose(0, 2, 3, 1).astype('uint8')

# Convert to greyscale
train_arr_sc = np.array(list(map(lambda x: pp_cifar_img(x, greyscale=True, enlarge_dim=(84,84)), list(train_arr))))

a = train_df[0]
plt.imshow(np.rot90(a))
plt.show()
def augment_rot(image_array):
    images = [image_array]
    for i in range(3):
        image_array = np.rot90(image_array)
        images.append(image_array)
    return images
    
train_pp = [item for sublist in list(map(lambda x: augment_rot(x), train_arr_sc)) for item in sublist]
labels = [0,1,2,3]*len(train_arr_sc)






#------------------------------
# CNN model
#------------------------------
game_name                              = 'BreakoutNoFrameskip-v4'
skip_frames = 4
framestack = 3
state_size                             = (84,84,framestack)
action_size                            = env.action_space.n

# Define model
model_input                                 = keras.models.Input(shape=(84,84,framestack,1))
x                                           = layers.Conv3D(32, kernel_size=(8,8,1), strides=(4,4,1), input_shape=state_size, activation='relu', padding='valid', kernel_initializer='random_uniform', bias_initializer=keras.initializers.Constant(0.1))(model_input)
x                                           = layers.MaxPooling3D(pool_size=(2,2,1), strides=(1,1,1), padding='valid')(x)
x                                           = layers.Conv3D(64, kernel_size=(4,4,1), strides=(2,2,1), activation='relu', padding='valid', kernel_initializer='random_uniform', bias_initializer=keras.initializers.Constant(0.1))(x)
x                                           = layers.MaxPooling3D(pool_size=(2,2,1), strides=(1,1,1), padding='valid')(x)
x                                           = layers.Conv3D(128, kernel_size=(3,3,1), strides=(1,1,1), activation='relu', padding='valid', kernel_initializer='random_uniform', bias_initializer=keras.initializers.Constant(0.1))(x)
x                                           = layers.MaxPooling3D(pool_size=(2,2,1), strides=(1,1,1), padding='valid')(x)
x                                           = tf.stack([layers.Flatten()((x[:, :, :, i, :])) for i in range(framestack)], axis=1)
# x                                         = layers.Dropout(x)
# Reward model
# x                                           = layers.BatchNormalization(center=True, scale=True)(x)
# reward                                      = layers.Dense(1, activation='linear')(x)
# Imagination state model
x2                                          = layers.LSTM(128, activation='tanh', return_sequences=False, bias_initializer='random_uniform')(x)
istate                                 = layers.Dense(84*84, activation='linear')(x2)
# output                                      = reward, istate
output = istate
model_name                                  = 'environment_awareness'
model                                       = keras.models.Model(inputs=model_input, outputs=output, name=model_name)
model.compile(optimizer='adam', loss='mse')


#------------------------------
# Memory
#------------------------------
class Memory:
    def __init__(self, memory_size=10**5):
        # Memory is cleared automatically to hold only up to memory size records to train
        self.memory_size = memory_size
        self.states      = deque(maxlen=self.memory_size)
        self.actions     = deque(maxlen=self.memory_size)
        self.rewards     = deque(maxlen=self.memory_size)
    def add(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    def clear(self):
        self.states      = deque(maxlen=self.memory_size)
        self.actions     = deque(maxlen=self.memory_size)
        self.rewards     = deque(maxlen=self.memory_size)


#------------------------------
# Train
#------------------------------
# Environment
env                                         = gym.make(game_name)
env                                         = WarpFrame(env)
env                                         = dqn.MaxAndSkipEnv(env,skip_frames)
env                                         = dqn.ScaledFloatFrame(env)
env                                         = dqn.EpisodicLifeEnv(env)
env                                         = dqn.FireResetEnv(env)
env                                         = dqn.FrameStack(env, framestack)

random_seed                                 = (int(time.time()*100) % 1999)**2 % 199999
np.random.seed(random_seed+1)
env.seed(random_seed)
print('\nRandom seed:\t', random_seed,'\n')
print('\nAction meanings:\n', env.get_action_meanings(),'\n')

state                                       = env.reset()
state                                       = np.reshape(state, [1,84,84,framestack,1])

memory = Memory()
done                                        = False
step_counter                                = 0
reward_sum                                  = 0



model_primary.fit(np.reshape(np.array(states), [self.batch_size,self.state_size[0],self.state_size[1],self.state_size[2],1]), np.reshape(np.array(q_valuess), [self.batch_size, self.action_size]), epochs=1, verbose=0, use_multiprocessing=True)


# Simulate games
while not done:
    # env.render()
    istate                           = model(tf.convert_to_tensor(state, dtype=tf.float32))
    action                          = env.action_space.sample()
    state, reward, done, _              = env.step(action)
    state                               = np.reshape(state, [1,84,84,args.framestack,1])
    reward_sum                          = reward_sum + reward
    print('{}.\tReward: {}\t Action #: {}\t Action: {}'.format(step_counter, reward_sum, action, env.get_action_meanings()[action]))
    step_counter                        = step_counter + 1

state[:,:,-1].shape

action                          = env.action_space.sample()


state, reward, done, _              = env.step(action)

state                                       = np.array(env.reset())
plt.imshow(state[:,:,-1])
plt.show()