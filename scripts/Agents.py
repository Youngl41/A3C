# env.unwrapped.get_action_meanings()

#======================================================
# Agent classes
#======================================================
'''
Info:       
Version:    1.0
Author:     Young Lee
Created:    Friday, 16 August 2019
'''
# Import modules
import os
import re
import sys
try:
    get_ipython().system('pip install gym')
    get_ipython().system('pip install tqdm')
    get_ipython().system('pip install dropbox')
    get_ipython().system('pip install gym[atari]')
except NameError:
    pass
# get_ipython().system('apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev')
# get_ipython().system('apt-get install -y python-mpi4py')
# get_ipython().system('pip install stable-baselines')

# get_ipython().system('brew install cmake openmpi')

# !pip install pandas
# !pip install keras
# !pip install matplotlib
# !pip install gym[atari]
try:
    from stable_baselines.common.atari_wrappers import WarpFrame
except ModuleNotFoundError:
    try: 
        from stable_baselines.common.atari_wrappers import WarpFrame
    except ModuleNotFoundError:
        from baselines.common.atari_wrappers import WarpFrame

import gym
from gym import spaces
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym import envs
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import dropbox
from datetime import datetime
from scipy.special import softmax
import matplotlib.pyplot as plt
from copy import deepcopy
# %matplotlib inline

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
import utility.util_general as gen
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Conv3D, MaxPooling3D, AveragePooling3D
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, SGD
from tensorflow.keras.models import clone_model
from tensorflow.keras import backend as K
import tensorflow as tf
# dtype = 'float16'
# K.set_floatx(dtype)
# K.set_epsilon(1e-4)
# print(tf.__version__)

# config                                  = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=12, device_count = {'CPU': 12 })
# session                                 = tf.compat.v1.Session(config=config)
# K.set_session(session)

# Suppress warnings
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 


#------------------------------
# DQN Agent
#------------------------------
# Define agent
class DQNAgent:
    # Initialise
    def __init__(self, state_size, action_size):
        self.state_size                                = state_size
        self.action_size                               = action_size
        self.memory                                    = deque(maxlen=500)
        self.train_interval                            = 5
        self.memory_size                               = 0
        self.gamma                                     = 0.95
        self.learning_rate                             = 0.001
        self.model_primary                             = self._build_model_primary() # primary network
        self.update_target_network_freq                = 10000
        self.polyak_weight                             = 0.95
        
        # Epsilon - greedy algorithm
        self.policy_method                             = 'epsilon-greedy'
        self.epsilon_start                             = 1.0
        self.epsilon_decay_steps                       = 100000
        self.epsilon_min                               = 0.1
        
    # Model predicts the action values (Q-values)
    def _build_model_primary(self):
        pass
    # Target network
    def _build_model_target(self):
        self.model_target                              = clone_model(self.model_primary)
        self.model_target.set_weights(self.model_primary.get_weights())
    # Update target model
    def update_target_network(self):
        # Number of layers
        n_layers = len(self.model_primary.get_weights())
        # Polyak averaging weights
        weights = [self.polyak_weight, 1-self.polyak_weight]
        # Allocate models
        models = [self.model_primary, self.model_target]
        avg_model_weights = []
        # For each layer get Polyak avg weights
        for layer in range(n_layers):
            # Get layer weights
            layer_weights = np.array([model_.get_weights()[layer] for model_ in models])
            # Weighted average of weights for the layer
            avg_layer_weights = np.average(layer_weights, axis=0, weights=weights)
            avg_model_weights.append(avg_layer_weights)
        # Update target model
        self.model_target = clone_model(self.model_primary)
        self.model_target.set_weights(avg_model_weights)
    def _initialise_decay(self):
        self.epsilon                                   = deepcopy(self.epsilon_start)
        self.lambda_ = -1*np.log(self.epsilon_min)/(self.epsilon_decay_steps)
    # Story in memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.memory_size                               = self.memory_size+1
    # Epsilon greedy or Boltzmann action
    def act(self, state):
        if self.policy_method.lower()=='epsilon-greedy':
            # Random action
            if np.random.rand() <= self.epsilon:
                action                          = random.sample(list(np.arange(self.action_size)), 1)[0]
            # Best action w.r.t. q-values
            else:
                act_values                             = self.model_primary.predict(state)
                action = np.nanargmax(act_values[0])
            # Decay epsilon (exponential)
            if self.epsilon>=self.epsilon_min:
                self.epsilon = max(self.epsilon * np.exp(-1*self.lambda_), self.epsilon_min)
            # Return action
            return action
        elif self.policy_method.lower()=='boltzmann':
            act_values                                 = self.model_primary.predict(state)[0]
            # Softmax
            softmax_val                                = softmax(act_values)
#             softmax_val                                = np.around(softmax_val, 3)
            try:
                random_choice                          = np.random.choice(np.arange(len(softmax_val)), p=softmax_val)
                return random_choice
            except ValueError as e:
    #                 print(e, '\n', softmax_val)
                softmax_val                            = np.array(softmax_val) * (1./ np.array(softmax_val).sum())
                random_choice                          = np.random.choice(np.arange(len(softmax_val)), p=softmax_val)
                return random_choice
    # Replay memory
    def replay(self, batch_size):
        random_idx = np.random.choice(range(len(self.memory)), size=batch_size, replace=True)
#         minibatch                                      = random.sample(self.memory, batch_size) + list(self.transition)
        minibatch                                      = [self.memory[idx] for idx in random_idx] + list(self.memory)[-20000:-20000+self.train_interval]
        states, q_valuess = [], []
        for state, action, reward, next_state, done in minibatch:
            q_update                                   = reward
            if not done:
                best_action = np.argmax(self.model_primary.predict(next_state)[0])
                q_update                               = (reward + self.gamma * self.model_target.predict(next_state)[0][best_action])
            q_values                                   = self.model_primary.predict(state)
            q_values[0][action]                        = q_update
            states.append(state)
            q_valuess.append(q_values)
        self.model_primary.fit(np.reshape(np.array(states), [self.train_interval+self.batch_size,self.state_size[0],self.state_size[1],self.state_size[2],1]), np.reshape(np.array(q_valuess), [self.train_interval+self.batch_size, self.action_size]), epochs=1, verbose=0, use_multiprocessing=True)
    # Replay memory
    def fast_replay(self, batch_size):
        random_idx = np.random.choice(range(len(self.memory)), size=batch_size, replace=True)
#         minibatch                                      = random.sample(self.memory, batch_size) + list(self.transition)
        minibatch                                      = [self.memory[idx] for idx in random_idx]# + list(self.memory)[-20000:-20000+self.train_interval]
        states, q_valuess = [], []
        for state, action, reward, next_state, done in minibatch:
            q_update                                   = reward
            if not done:
                best_action = np.argmax(self.model_primary.predict(next_state)[0])
                q_update                               = (reward + self.gamma * self.model_target.predict(next_state)[0][best_action])
            q_values                                   = self.model_primary.predict(state)
            q_values[0][action]                        = q_update
            states.append(state)
            q_valuess.append(q_values)
        self.model_primary.fit(np.reshape(np.array(states), [self.batch_size,self.state_size[0],self.state_size[1],self.state_size[2],1]), np.reshape(np.array(q_valuess), [self.batch_size, self.action_size]), epochs=1, verbose=0, use_multiprocessing=True)
    # Load
    def load(self, name):
        self.model_primary.load_weights(name)
    # Save
    def save(self, name):
        self.model_primary.save_weights(name)