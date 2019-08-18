#======================================================
# Agent classes for A3C
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

# Import custom modules
try:
    sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))) # 1 level upper dir
    sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, '..', '..')))) # 2 levels upper dir
except NameError:
    sys.path.append('.') # current dir
    sys.path.append('..') # 1 level upper dir
    sys.path.append(os.path.join(os.getcwd(), '..')) # 1 levels upper dir
    sys.path.append(os.path.join(os.getcwd(), '..', '..')) # 2 levels upper dir

from a3c_util import record, Memory


#------------------------------
# Random agent
#------------------------------
class RandomAgent:
    """Random Agent that will play the specified game

        Arguments:
            env_name: Name of the environment to be played
            max_eps: Maximum number of episodes to run agent for.
    """
    def __init__(self, env_name, max_eps):
        self.env                              = gym.make(env_name)
        self.max_episodes                     = max_eps
        self.global_moving_average_reward     = 0
        self.res_queue                        = Queue()

    def run(self):
        reward_avg                            = 0
        for episode in range(self.max_episodes):
            done                              = False
            self.env.reset()
            reward_sum                        = 0.0
            steps                             = 0
            while not done:
                # Sample randomly from the action space and step
                _, reward, done, _            = self.env.step(self.env.action_space.sample())
                steps += 1
                reward_sum += reward
            # Record statistics
            self.global_moving_average_reward = record(episode,
                                                    reward_sum,
                                                    0,
                                                    self.global_moving_average_reward,
                                                    self.res_queue, 0, steps)

            reward_avg += reward_sum
        final_avg                             = reward_avg / float(self.max_episodes)
        print("Average score across {} episodes: {}".format(self.max_episodes, final_avg))
        return final_avg
