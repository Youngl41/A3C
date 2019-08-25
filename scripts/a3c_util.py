#======================================================
# Utility Functions for A3C
#======================================================
'''
Info:       
Version:    1.0
Author:     Young Lee
Created:    Saturday, 17 August 2019
'''
# Import modules
import os
import sys
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import gym
from queue import Queue
from collections import deque


#------------------------------
# Utility functions
#------------------------------
def record(episode,
           episode_reward,
           adjusted_reward,
           worker_idx,
           global_ep_reward,
           global_ep_adjusted_reward,
           result_queue,
           avg_min_prob,
           total_loss,
           num_steps):
    '''Helper function to store score and print statistics.

    Arguments:
        episode: Current episode
        episode_reward: Reward accumulated over the current episode
        worker_idx: Which thread (worker)
        global_ep_reward: The moving average of the global reward
        result_queue: Queue storing the moving average of the scores
        total_loss: The total loss accumualted over the current episode
        num_steps: The number of steps the episode took to complete
    '''
    if global_ep_reward==0:
        global_ep_reward = episode_reward
    else:
        global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
    if global_ep_adjusted_reward==0:
        global_ep_adjusted_reward = adjusted_reward
    else:
        global_ep_adjusted_reward = global_ep_adjusted_reward * 0.99 + adjusted_reward * 0.01
    print(
            f'Episode: {episode} | '
            f'Moving Average Reward: {int(global_ep_reward)} ({int(global_ep_adjusted_reward)}) | '
            f'Episode Reward: {int(episode_reward)} ({int(adjusted_reward)}) | '
            'Avg Min Probability: '+str(int(avg_min_prob*100))+'% | '
            f'Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | '
            f'Steps: {num_steps} | '
            f'Worker: {worker_idx}'
    )
    result_queue.put(global_ep_reward)
    return global_ep_reward, global_ep_adjusted_reward


class Memory:
    def __init__(self, memory_size=20):
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

