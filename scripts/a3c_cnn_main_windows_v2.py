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

os.environ['CUDA_VISIBLE_DEVICES']                  = '-1'

parser                                              = argparse.ArgumentParser(description='Run A3C algorithm on the game Cartpole.')
parser.add_argument('--game-name', default='BreakoutNoFrameskip-v4', type=str, help='Game name.')
parser.add_argument('--algorithm', default='a3c', type=str, help='Choose between \'a3c\' and \'random\'.')
parser.add_argument('--train', dest='train', action='store_true', help='Train our model.')
parser.add_argument('--lr', default=0.00025, type=float, help='Learning rate for the shared optimizer.')
parser.add_argument('--skip-frames', default=4, type=int, help='Learning rate for the shared optimizer.')
parser.add_argument('--framestack', default=2, type=int, help='Number of frames + n-1 past frames to pass to the agent.')
parser.add_argument('--update-freq', default=5, type=int, help='How often to update the global model.')
parser.add_argument('--memory-size', default=5, type=int, help='Max memory size before forgetting.')
parser.add_argument('--time-limit', type=int, help='Max memory size before forgetting.')
# parser.add_argument('--batch-size', default=32, type=int, help='How many turns to train global model per update.')
parser.add_argument('--max-eps', default=40, type=int, help='Global maximum number of episodes to run.')
parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor of rewards.')
parser.add_argument('--trained-model', default='', type=str, help='Load trained model for continued training.')
parser.add_argument('--periodic-save', default=0, type=int, help='Load periodic save model.')
parser.add_argument('--save-dir', default=os.path.join(main_dir, 'models'), type=str, help='Directory in which you desire to save the model.')
parser.add_argument('--save-freq', default=10, type=int, help='How often the model saves periodically.')
args                                                = parser.parse_args()

# # Unit test
# class Args():
#     def __init__(self, mode='train'):
#         if mode.lower()=='train':
#             self.game_name = 'BreakoutNoFrameskip-v4'
#             self.algorithm = 'a3c'
#             self.max_eps = 40
#             self.save_dir = 'C:\\genesis_ai\\A3C\\temp'
#             self.train = True
#             self.update_freq = 5
#             self.memory_size = 5
#             self.framestack = 2
#             self.lr = 0.00025
#             self.gamma = 0.99
#             self.skip_frames = 4
#             self.trained_model = 'C:\\genesis_ai\\A3C\\models\\model_BreakoutNoFrameskip-v4.h5'
#             self.time_limit = None
#             self.save_freq = 10
#         elif model.lower()=='play':
#             self.game_name = 'BreakoutNoFrameskip-v4'
#             self.save_dir = 'C:\\genesis_ai\\A3C\\temp'
#             self.framestack = 2
#             self.skip_frames = 4
#             self.periodic_save = 1
#             self.time_limit = None
#             self.save_freq = 10

# args = Args(mode='train')


#------------------------------
# Utility functions
#------------------------------
def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0: 
        if not err: 
            print("Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l))
            err.append('logged')
        return rgb_array
    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)
    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)


#------------------------------
# A3C model
#------------------------------
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
# Master agent
#------------------------------
class MasterAgent():
    def __init__(self):
        save_dir                                    = args.save_dir
        # Ensure save directory
        self.save_dir                               = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Create game
        self.game_name                              = args.game_name
        env                                         = gym.make(self.game_name)
        env                                         = WarpFrame(env)
        env                                         = dqn.MaxAndSkipEnv(env,args.skip_frames)
        env                                         = dqn.ScaledFloatFrame(env)
        env                                         = dqn.EpisodicLifeEnv(env)
        env                                         = dqn.FireResetEnv(env)
        env                                         = dqn.FrameStack(env, args.framestack)
        if args.time_limit is not None:
            env                                     = dqn.TimeLimit(env, max_episode_steps=args.time_limit)
        self.state_size                             = (84,84,args.framestack)#env.observation_space.shape[0]
        self.action_size                            = env.action_space.n
        print('\nState size:', self.state_size, 'Action size:', self.action_size, '\n')
        # Define global model
        self.global_model                           = A3C(self.state_size, self.action_size)
        _                                           = self.global_model(tf.convert_to_tensor(np.random.random((1, 84,84,args.framestack,1)), dtype=tf.float32)) # Initialise model (get weights)
        print(self.global_model.summary())
        # Load pre-trained model
        if args.train:
            try:
                self.global_model.load_weights(args.trained_model)
                gen.set_section('Loaded pre-trained model')
                time.sleep(1.0)
            except:
                gen.set_section('NOT loaded pre-trained model')
                time.sleep(1.0)
                pass
        # Define optimiser
        self.opt                                    = tf.optimizers.Adam(args.lr)#, use_locking=True)
        # self.opt                                  = tf.train.AdamOptimizer(args.lr, use_locking=True)
    def train(self):
        # Random agent for benchmark
        if args.algorithm=='random':
            random_agent                            = RandomAgent(self.game_name, args.max_eps)
            random_agent.run()
            return
        # If not random algorithm
        res_queue                                   = Queue()
        workers                                     = [Worker(self.state_size,
                                                            self.action_size,
                                                            self.global_model,
                                                            self.opt, res_queue,
                                                            i, 
                                                            game_name=self.game_name,
                                                            save_dir=self.save_dir) \
                                                    for i in range(os.cpu_count())]
        for i, worker in enumerate(workers):
            print('Starting worker {}'.format(i))
            worker.start()
        # Record episode reward
        moving_average_rewards                      = []
        while True:
            reward                                  = res_queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
            else:
                break
        [w.join() for w in workers]
        plt.plot(moving_average_rewards)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.savefig(os.path.join(self.save_dir,'{} Moving Average.png'.format(self.game_name)))
        plt.show()

    def play(self):
        env                                         = gym.make(self.game_name)#.unwrapped
        env                                         = WarpFrame(env)
        env                                         = dqn.MaxAndSkipEnv(env,args.skip_frames)
        env                                         = dqn.ScaledFloatFrame(env)
        env                                         = dqn.EpisodicLifeEnv(env)
        env                                         = dqn.FireResetEnv(env)
        env                                         = dqn.FrameStack(env, args.framestack)
        random_seed                                 = (int(time.time()*100) % 1999)**2 % 199999
        np.random.seed(random_seed+1)
        env.seed(random_seed)
        print('\nRandom seed:\t', random_seed,'\n')
        print('\nAction meanings:\n', env.get_action_meanings(),'\n')
        if args.time_limit is not None:
            env                                     = dqn.TimeLimit(env, max_episode_steps=args.time_limit)
        state                                       = env.reset()
        state                                       = np.reshape(state, [1,84,84,args.framestack,1])
        model                                       = self.global_model
        if args.periodic_save==1:
            model_path                              = os.path.join(self.save_dir, 'model_{}_periodic_save.h5'.format(self.game_name))
        else:
            model_path                              = os.path.join(self.save_dir, 'model_{}.h5'.format(self.game_name))
        print('Loading model from: {}'.format(model_path))
        try:
            model.load_weights(model_path)
            # print(model.get_weights()[0][-1][-1][-1][-1][-1])
        except:
            print('\n\nNo model loaded.\n\n')
            pass
        done                                        = False
        step_counter                                = 0
        reward_sum                                  = 0

        viewer                                      = rendering.SimpleImageViewer()
        try:
            while not done:
                rgb                                 = env.render('rgb_array')
                upscaled=repeat_upsample(rgb,2, 2)
                viewer.imshow(upscaled)
                # env.render()#mode='rgb_array')
                policy, _                           = model(tf.convert_to_tensor(state, dtype=tf.float32))
                policy                              = tf.nn.softmax(policy)[0,:]
                print(policy)
                if args.algorithm=='random':
                    action                          = env.action_space.sample()
                else:
                    action                          = np.argmax(policy)
                state, reward, done, _              = env.step(action)
                state                               = np.reshape(state, [1,84,84,args.framestack,1])
                reward_sum                          = reward_sum + reward
                print('{}.\tReward: {}\t Action #: {}\t Action: {}'.format(step_counter, reward_sum, action, env.get_action_meanings()[action]))
                step_counter                        = step_counter + 1
        except KeyboardInterrupt:
            print('Received Keyboard Interrupt. Shutting down.')
        finally:
            env.close()


#------------------------------
# Worker
#------------------------------
class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode                                  = 0
    global_moving_average_reward                    = 0
    global_moving_average_adjusted_reward           = 0
    best_score                                      = 0
    save_lock                                       = threading.Lock()
    save_periodic_lock                              = threading.Lock()
    update_lock                                     = threading.Lock()
    def __init__(self,
                state_size,
                action_size,
                global_model,
                opt,
                result_queue,
                idx,
                game_name=args.game_name,
                save_dir=os.path.join(main_dir, 'models')):
        super(Worker, self).__init__()
        self.state_size                             = state_size
        self.action_size                            = action_size
        self.result_queue                           = result_queue
        self.global_model                           = global_model
        self.opt                                    = opt
        self.local_model                            = A3C(self.state_size, self.action_size)
        self.worker_idx                             = idx
        self.game_name                              = game_name
        self.env                                    = gym.make(self.game_name).unwrapped
        self.env                                    = WarpFrame(self.env)
        self.env                                    = dqn.MaxAndSkipEnv(self.env, args.skip_frames)
        self.env                                    = dqn.ScaledFloatFrame(self.env)
        self.env                                    = dqn.EpisodicLifeEnv(self.env)
        self.env                                    = dqn.FireResetEnv(self.env)
        self.env                                    = dqn.FrameStack(self.env, args.framestack)
        
        if args.time_limit is not None:
            self.env                                = dqn.TimeLimit(self.env, max_episode_steps=args.time_limit)
        self.save_dir                               = save_dir
        self.epi_loss                               = 0.0

    def run(self):
        total_step                                  = 1
        mem                                         = Memory(args.memory_size)
        no_action_idx = np.where(np.array(self.env.get_action_meanings())=='NOOP')[0][0]
        while Worker.global_episode < args.max_eps:
            random_seed                             = (int(time.time()*100) % 1999)**2 % 199999
            np.random.seed(random_seed+1)
            self.env.seed(random_seed)
            current_state                           = self.env.reset()
            current_state                           = np.reshape(current_state, [1,84,84,args.framestack,1])
            mem.clear()
            epi_reward                              = 0.
            epi_adjusted_reward                     = 0.
            epi_steps                               = 1
            self.epi_loss                           = 0

            time_count                              = 0
            done                                    = False
            min_probs                               = []
            while not done:
                # Boltzmann action selection
                logits, _                           = self.local_model(tf.convert_to_tensor(current_state,dtype=tf.float32))
                probs                               = tf.nn.softmax(logits[0,:])
                min_probs.append(min(probs))
                action                              = np.random.choice(self.action_size, p=probs.numpy())
                # Play action
                new_state, reward, done, _          = self.env.step(action)
                new_state                           = np.reshape(new_state, [1,84,84,args.framestack,1])
                if done:
                    adjusted_reward                 = -1.
                elif (abs(reward)<0.0001) & (action!=no_action_idx):
                    adjusted_reward                 = -0.01#-.02
                elif (abs(reward)<0.0001):
                    adjusted_reward                 = 0.
                elif (abs(reward)>=0.0001):
                    adjusted_reward                 = 1.0#np.sign(reward)# - 0.01
                epi_reward                          = epi_reward + reward
                epi_adjusted_reward                 = epi_adjusted_reward + adjusted_reward
                # Update memory
                mem.add(current_state, action, adjusted_reward)

                if ((time_count % args.update_freq)==0) or done:
                # if (abs(adjusted_reward)>0.1) or done:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape
                    with tf.GradientTape() as tape:
                        total_loss                  = self.compute_loss(done, new_state, mem, args.gamma)
                    self.epi_loss                   = self.epi_loss + total_loss
                    # Clear memory
                    # mem.clear()
                    # Calculate local gradients
                    grads                           = tape.gradient(total_loss, self.local_model.trainable_weights)
                    grads, _ = tf.clip_by_global_norm(grads, 40.0)
                    while True:
                        try:
                            with Worker.update_lock:
                                # Push local gradients to global model
                                self.opt.apply_gradients(zip(grads, self.global_model.trainable_weights))
                                # Update local model with new weights
                                self.local_model.set_weights(self.global_model.get_weights())
                                break
                        except OSError:
                            time.sleep(0.01)
                            pass
                    # Print results at end of game
                    if done:
                        # print(self.global_model.get_weights()[0][-1][-1][-1][-1][-1])
                        # Update global moving avg and also print results
                        while True:
                            try:
                                with Worker.update_lock:
                                    Worker.global_moving_average_reward, Worker.global_moving_average_adjusted_reward= \
                                        record(Worker.global_episode, epi_reward, epi_adjusted_reward,
                                            self.worker_idx, Worker.global_moving_average_reward, 
                                            Worker.global_moving_average_adjusted_reward,
                                            self.result_queue, np.mean(min_probs),
                                            self.epi_loss, num_steps=epi_steps)
                                    Worker.global_episode=Worker.global_episode + 1
                                    break
                            except OSError:
                                time.sleep(0.01)
                                pass
                        # Lock to save model and prevent data races
                        if epi_reward > Worker.best_score:
                            while True:
                                try:
                                    with Worker.save_lock:
                                        print('Saving best model to {}, '
                                                    'episode score: {}'.format(self.save_dir, epi_reward))
                                        self.global_model.save_weights(os.path.join(self.save_dir,'model_{}.h5'.format(self.game_name)))
                                        Worker.best_score=epi_reward
                                        break
                                except OSError:
                                    time.sleep(0.01)
                                    pass
                        # Save periodically
                        if int(Worker.global_episode-1) % args.save_freq==0:
                            while True:
                                try:
                                    with Worker.save_periodic_lock:
                                        self.global_model.save_weights(os.path.join(self.save_dir,'model_{}_periodic_save.h5'.format(self.game_name)))
                                        print('Model periodic save')
                                        break
                                except OSError:
                                    time.sleep(0.01)
                                    pass
                epi_steps                           = epi_steps + 1
                time_count                          = time_count + 1
                current_state                       = new_state
                total_step                          = total_step + 1
        self.result_queue.put(None)

    def compute_loss(self,done,new_state,memory,gamma=0.99):#,sample=args.batch_size):
        # random_idx                                = np.random.choice(len(memory.states),sample,replace=True)
        # sample_states                             = [memory.states[idx] for idx in random_idx]
        # sample_actions                            = [memory.actions[idx] for idx in random_idx]
        # sample_rewards                            = [memory.rewards[idx] for idx in random_idx]
        if done:
            reward_sum                              = 0.
        else:
            reward_sum                              = self.local_model(tf.convert_to_tensor(new_state,dtype=tf.float32))[-1].numpy()[0,0]

        # Get discounted rewards
        discounted_rewards                          = []
        for reward in list(memory.rewards)[::-1]: # reverse
            reward_sum                              = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        logits, values                              = self.local_model(tf.convert_to_tensor(np.vstack(memory.states),dtype=tf.float32))
        # Get advantages
        advantage                                   = tf.convert_to_tensor(np.array(discounted_rewards),dtype=tf.float32) - values[:,0]
        # Value loss
        value_loss                                  = advantage ** 2
        # Policy loss
        policy                                      = tf.nn.softmax(logits[:,:])
        # entropy                                   = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy,logits=logits[:,:])
        entropy                                     = tf.nn.softmax_cross_entropy_with_logits(labels=policy,logits=logits[:,:])
        policy_loss                                 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions,logits=logits[:,:])
        policy_loss                                 = policy_loss * tf.stop_gradient(advantage)
        policy_loss                                 = policy_loss - 0.01 * entropy
        total_loss                                  = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss


#------------------------------
# Main
#------------------------------
if __name__=='__main__':
    print(args)
    master                                          = MasterAgent()
    if args.train:
        master.train()
    else:
        master.play()

# Train model
'''
# Windows train
>>> python scripts\\a3c_cnn_main_windows_v2.py --game-name BreakoutNoFrameskip-v4 --algorithm a3c --max-eps=10000 --save-dir C:\\genesis_ai\\A3C\\models --train --update-freq 5 --memory-size 5 --framestack 2 --lr 0.00025 --gamma 0.99 --skip-frames 4 --save-freq 50 --trained-model C:\\genesis_ai\\A3C\\models\\model_BreakoutNoFrameskip-v4.h5
>>> python scripts\\a3c_cnn_main_windows_v2.py --game-name BreakoutNoFrameskip-v4 --algorithm a3c --max-eps=1000000 --save-dir C:\\genesis_ai\\A3C\\models --train --update-freq 5 --memory-size 5 --framestack 2 --lr 0.00025 --gamma 0.99 --skip-frames 4 --save-freq 50 --trained-model C:\\genesis_ai\\A3C\\models\\model_BreakoutNoFrameskip-v4.h5

# Windows play
>>> python scripts\\a3c_cnn_main_windows_v2.py --game-name BreakoutNoFrameskip-v4 --save-dir C:\\genesis_ai\\A3C\\models --framestack 2 --skip-frames 4 --periodic-save 1
>>> python scripts\\a3c_cnn_main_windows_v2.py --game-name BreakoutNoFrameskip-v4 --save-dir C:\\genesis_ai\\A3C\\models\\breakout --framestack 2 --skip-frames 4 --periodic-save 1
'''

