#======================================================
# Pong RL
#======================================================
'''
Info:       
Version:    1.0
Author:     
Created:    Saturday, 17 August 2019
'''
# Import modules
# get_ipython().system('pip install gym')
# get_ipython().system('pip install tqdm')
# get_ipython().system('pip install dropbox')
# get_ipython().system('pip install gym[atari]')
# get_ipython().system('apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev')
# get_ipython().system('apt-get install -y python-mpi4py')
# get_ipython().system('pip install stable-baselines')

# get_ipython().system('brew install cmake openmpi')

# !pip install pandas
# !pip install keras
# !pip install matplotlib
# !pip install gym[atari]
import os
import re
import sys

try:
    from stable_baselines.common.atari_wrappers import WarpFrame
except ModuleNotFoundError:
    try: 
        from stable_baselines.common.atari_wrappers import WarpFrame
    except ModuleNotFoundError:
        from baselines.common.atari_wrappers import WarpFrame

import gym
# from gym.wrappers.atari_preprocessing import AtariPreprocessing
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import threading
import multiprocessing
from datetime import datetime
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
import utility.util_dqn as dqn
from Agents import DQNAgent
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Conv3D, MaxPooling3D, AveragePooling3D
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, SGD
from tensorflow.keras.models import clone_model
from tensorflow.keras import backend as K
import tensorflow as tf

# Suppress warnings
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"]      = "3" 


#------------------------------
# Set up DQN
#------------------------------
# Save path
model_name                              = 'test_v1.2'
save_dir                                = os.path.join(main_dir, 'model', 'pong', 'ddcqn', model_name)
acc_save_path                           = os.path.join(save_dir, 'summary.csv')
gen.ensure_dir(save_dir)

# Parameters (1/3) - policy
policy_method                           = 'boltzmann'#'epsilon-greedy'
epsilon_start                           = 1.0
epsilon_decay_steps                     = 1000000
epsilon_min                             = 0.07

# Parameters (2/3) - learning parameters
n_games                                 = 10**4
max_memory                              = 100000
replay_start_size                       = 10*1000
update_target_network_freq              = 1000
train_interval                          = 4
batch_size                              = 32
gamma                                   = 0.90
learning_rate                           = 0.001
polyak_weight                           = 0.95

# Parameters (3/3) - save
save_interval                           = 5000
sync_summary_to_dropbox                 = 0
sync_weights_to_dropbox                 = 0

# Environment
framestack                              = 1
stacked_frames                          = 1
skip_frames                             = 4
env_scale_size                          = 84
state_size                              = (env_scale_size,env_scale_size,int(framestack/stacked_frames), 1)
action_size                             = 6

# Define agent
agent_1                                 = DQNAgent(state_size=state_size, action_size=action_size)
agent_1.batch_size                      = batch_size
agent_1.train_interval                  = train_interval
agent_1.memory                          = deque(maxlen=max_memory)
agent_1.learning_rate                   = learning_rate
agent_1.gamma                           = gamma
agent_1.update_target_network_freq      = update_target_network_freq
agent_1.policy_method                   = policy_method
agent_1.epsilon_start                   = epsilon_start
agent_1.epsilon_decay_steps             = epsilon_decay_steps
agent_1.epsilon_min                     = epsilon_min
agent_1._initialise_decay()
agent_1.polyak_weight                   = polyak_weight

# Define (primary) model
agent_1.model_primary                   = Sequential()
agent_1.model_primary.add(Conv3D(32, kernel_size=(8,8, 1), strides=(4,4,1), activation='relu', padding='valid', input_shape=agent_1.state_size, kernel_initializer=VarianceScaling(scale=2.0)))
agent_1.model_primary.add(Conv3D(64, kernel_size=(4,4, 1), strides=(2,2,1), activation='relu', padding='valid', input_shape=agent_1.state_size, kernel_initializer=VarianceScaling(scale=2.0)))
agent_1.model_primary.add(Conv3D(64, kernel_size=(3,3, 1), strides=(1,1,1), activation='relu', padding='valid', input_shape=agent_1.state_size, kernel_initializer=VarianceScaling(scale=2.0)))
# agent_1.model_primary.add(MaxPooling3D(pool_size=(2,2,1), strides=(1,1,1), padding='valid'))
agent_1.model_primary.add(Flatten())
# agent_1.model_primary.add(Dropout(0.5))
agent_1.model_primary.add(Dense(512,activation='relu', kernel_initializer=VarianceScaling(scale=2.0)))
agent_1.model_primary.add(Dense(512,activation='relu', kernel_initializer=VarianceScaling(scale=2.0)))
agent_1.model_primary.add(Dense(action_size, activation='linear'))
agent_1.model_primary.compile(loss='MSE', optimizer=Adam(lr=learning_rate))

# Initiate (target) model
agent_1._build_model_target()

# Model summary
print(agent_1.model_primary.summary())


#------------------------------
# Define environment
#------------------------------
# Environment name
env_name                                = 'PongNoFrameskip-v4'
env_raw                                 = gym.make(env_name)
env                                     = WarpFrame(env_raw)
env                                     = dqn.MaxAndSkipEnv(env,skip_frames)
env                                     = dqn.ScaledFloatFrame(env)
env                                     = dqn.EpisodicLifeEnv(env)
# env                                   = NoopResetEnv(env,3)
# env                                   = FrameStack(env, framestack)
# env                                   = ClipRewardEnv(env)
random.seed(42)
env.seed(1995)
state_size_                             = env.observation_space.shape[0]
action_size_                            = env.action_space.n

print('State size:\t',state_size_)
print('Action size:\t',action_size_)

# Print environment status
state                                   = env.reset();# print('\n',env.env.step(2))

# Plot state
_                                       = env.step(random.sample(range(6),1)[0])[0]
_                                       = env.step(random.sample(range(6),1)[0])[0]
state                                   = env.step(random.sample(range(6),1)[0])[0]
state                                   = np.reshape(state, [env_scale_size,env_scale_size, framestack, 1], order='F')
# plt.imshow(state[:,:,0,0])
# plt.show()

# Play game
# done                                  = False
# while not done:
#     env.render()
#     state, _, done, _                 = env.step(random.sample(range(6),1)[0])

# Play 6 rounds randomly
rounds                                  = 6
for i in range(rounds):
    state                               = env.step(random.sample(range(6),1)[0])[0]

state                                   = np.reshape(state, [1, env_scale_size, env_scale_size, int(framestack/stacked_frames),1])

# Copy state for future reference
state_x                                 = deepcopy(state)

# Plot layers
layer_outputs                           = [layer.output for layer in agent_1.model_primary.layers][1:] # Extracts the outputs of the top 12 layers
activation_model                        = Model(inputs=agent_1.model_primary.input, outputs=layer_outputs) 
activations                             = activation_model.predict(state)

# plt.imshow(state[0,:,:,0,0])
# plt.show()
# plt.imshow(activations[0][0, :,:,0,2])
# plt.show()
# plt.imshow(activations[1][0, :,:,0,2])
# plt.show()
# plt.imshow(activations[2][0, :,:,0,2])
# plt.show()

# Layer info
print(layer_outputs[2])


#------------------------------
# Fit
#------------------------------
# Interact with environment
done                                    = False
started_training                        = 0
# for game_count in tqdm(range(n_games)):
game_count                              = 0
while game_count < n_games:
    st                                  = datetime.now()
    state                               = env.reset()
    state                               = np.reshape(state, [1, env_scale_size, env_scale_size, int(framestack/stacked_frames),1])
    total_reward                        = 0
    total_adjusted_reward               = 0
    turn                                = 0
    target_net_updated                  = 0
    current_lives                       = 4
    while True:
#         env.render()
        # Predict action
        action                          = agent_1.act(state)
        # Observe environment
        next_state, reward, done, lives = env.step(action)
        lives                           = lives['ale.lives']
        # Adjust reward
        if current_lives > lives:
            adjusted_reward             = -20
            current_lives               = current_lives - 1
        else:
            adjusted_reward             = reward#np.sign(reward)
#         adjusted_reward               = reward
        adjusted_reward                 = dqn.scale_reward(adjusted_reward, scale=20)
        total_adjusted_reward           = total_adjusted_reward + adjusted_reward
        # Update turn
        turn                            = turn + 1
        next_state                      = np.reshape(next_state, [1, env_scale_size, env_scale_size, int(framestack/stacked_frames),1])
        # Update memory
        agent_1.remember(state, action, adjusted_reward, next_state, done)
        state                           = next_state
        total_reward                    = total_reward + reward
        # Print termination scores
        if done:
            print('Game: {}/{}\t score: {},\t adjusted score: {},\t turns: {},\t time: {}'.format(game_count, n_games, total_reward, np.around(total_adjusted_reward,3), turn, str(datetime.now()-st).replace('(\.\d{2})\d+', r'\1')))
            # Save
            acc                         = pd.DataFrame([[datetime.now(), game_count, total_reward, total_adjusted_reward, turn, started_training, target_net_updated, datetime.now()-st]], columns = ['current_time', 'game_count', 'total_reward', 'total_adjusted_reward', 'turn', 'started_training', 'target_net_updated', 'time_taken'])
            # Append if file exists
            if os.path.isfile(acc_save_path):
                acc.to_csv(acc_save_path, index=False, mode='a', header=False)
            elif not os.path.isfile(acc_save_path):
                acc.to_csv(acc_save_path, index=False)
            if sync_summary_to_dropbox==1:
                dqn.to_dropbox(acc_save_path, os.path.join(re.sub(main_dir, '', save_dir), os.path.basename(acc_save_path)))
            if game_count % save_interval==0:
                print('\nSaving weights\n')
                weights_name            = model_name+ '_['+str(game_count)+'_games]_['+str(total_reward)+'_reward]_['+str(turn)+'_turns]'+'.hdf5'
                weights_save_path       = os.path.join(save_dir, weights_name)
                agent_1.save(weights_save_path)
                if sync_weights_to_dropbox==1:
                    dropbox_path        = os.path.join(re.sub(main_dir, '', save_dir), weights_name)
                    dqn.to_dropbox(weights_save_path, dropbox_path)
            game_count                  = game_count + 1
            # print(agent_1.model_primary.predict(state_x)[0], np.argmax(agent_1.model_primary.predict(state_x)[0]))
            # print(agent_1.model_target.predict(state_x)[0], np.argmax(agent_1.model_target.predict(state_x)[0]))
            break
        # Fit
        if (agent_1.memory_size % agent_1.train_interval==0) & (agent_1.memory_size > agent_1.batch_size) & (agent_1.memory_size > 5000+replay_start_size):
            agent_1.fast_replay(agent_1.batch_size)
            if started_training==0:
                started_training        = 1
                print('\nStarted training\n')
        # Update target network
        if ((agent_1.memory_size-replay_start_size) % agent_1.update_target_network_freq==0) & (agent_1.memory_size > 5000+replay_start_size):
            agent_1.update_target_network()
            print('\nUpdated target network\n')
            target_net_updated          = target_net_updated+1

