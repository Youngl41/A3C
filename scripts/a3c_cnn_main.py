#======================================================
# A3C main execution file
#======================================================
'''
Info:       
Version:    1.0
Author:     Young Lee
Created:    Saturday, 17 August 2019
'''

# Import modules
import argparse
import os
import sys
import threading
from glob import glob
from queue import Queue

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.train import AdamOptimizer
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tqdm import tqdm

try:
    from stable_baselines.common.atari_wrappers import WarpFrame
except ModuleNotFoundError:
    try: 
        from stable_baselines.common.atari_wrappers import WarpFrame
    except ModuleNotFoundError:
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
from a3c_agents import RandomAgent
from a3c_util import record, Memory

os.environ['CUDA_VISIBLE_DEVICES']                = ''
tf.enable_eager_execution()

parser                                            = argparse.ArgumentParser(description='Run A3C algorithm on the game Cartpole.')
parser.add_argument('--game-name', default='PongNoFrameskip-v4', type=str, help='Game name.')
parser.add_argument('--algorithm', default='a3c', type=str, help='Choose between \'a3c\' and \'random\'.')
parser.add_argument('--train', dest='train', action='store_true', help='Train our model.')
parser.add_argument('--lr', default=0.00025, type=float, help='Learning rate for the shared optimizer.')
parser.add_argument('--skip-frames', default=4, type=int, help='Learning rate for the shared optimizer.')
parser.add_argument('--framestack', default=4, type=int, help='Number of frames + n-1 past frames to pass to the agent.')
parser.add_argument('--update-freq', default=300, type=int, help='How often to update the global model.')
parser.add_argument('--memory-size', default=300, type=int, help='Max memory size before forgetting.')
# parser.add_argument('--batch-size', default=32, type=int, help='How many turns to train global model per update.')
parser.add_argument('--max-eps', default=1000, type=int, help='Global maximum number of episodes to run.')
parser.add_argument('--gamma', default=0.99, help='Discount factor of rewards.')
parser.add_argument('--save-dir', default=os.path.join(main_dir, 'models'), type=str, help='Directory in which you desire to save the model.')
args                                              = parser.parse_args()

# env                                             = gym.make('PongNoFrameskip-v4').unwrapped
# env                                             = WarpFrame(env)
# env                                             = dqn.MaxAndSkipEnv(env,4)#args.skip_frames)
# env                                             = dqn.ScaledFloatFrame(env)
# env                                             = dqn.EpisodicLifeEnv(env)
# env.observation_space.shape
# action_size                                     = env.action_space.n
# model                                           = A3C([84,84,1],6)
# a,b                                             = model(tf.convert_to_tensor(np.random.random((1, 84,84,1,1)), dtype=tf.float32))
# a[0].shape
# a[0,0,0,0,:].shape
# tf.nn.softmax(a).shape
# np.array(tf.nn.softmax(a))[0,0,0,0,:].shape
# b.shape
# tf.nn.softmax(policy)

# logits, _                                       = model(tf.convert_to_tensor(current_state,dtype=tf.float32))
# probs                                           = tf.nn.softmax(logits[0,0,0,0,:])
# action                                          = np.random.choice(action_size, p=probs.numpy())
#------------------------------
# A3C model
#------------------------------
class A3C(keras.Model):
    def __init__(self, state_size, action_size):
        super(A3C, self).__init__()
        self.state_size                           = state_size
        self.action_size                          = action_size
        # Policy model
        self.conv1                                = layers.Conv3D(16, kernel_size=(8,8, 1), strides=(4,4,1), activation='relu', padding='valid')#, kernel_initializer=VarianceScaling(scale=2.0))
        self.conv2                                = layers.Conv3D(32, kernel_size=(4,4, 1), strides=(2,2,1), activation='relu', padding='valid')#, kernel_initializer=VarianceScaling(scale=2.0))
        self.dense1                               = layers.Dense(256, activation='relu')
        # self.dense1b                            = layers.Dense(100, activation='relu')
        self.policy_logits                        = layers.Dense(action_size)
        # Value model
        self.dense2                               = layers.Dense(256, activation='relu')
        self.values                               = layers.Dense(1)

    # Forward
    def call(self, inputs):
        # Policy
        x                                         = self.conv1(inputs)
        x                                         = self.conv2(x)
        x                                         = self.dense1(x)
        # x                                       = self.dense1b(x)
        logits                                    = self.policy_logits(x)
        # Value
        v1                                        = self.dense2(inputs)
        values                                    = self.values(v1)
        return logits, values


#------------------------------
# Master agent
#------------------------------
class MasterAgent():
    def __init__(self):
        # Ensure save directory
        save_dir                                  = args.save_dir
        self.save_dir                             = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Create game
        self.game_name                            = args.game_name
        env                                       = gym.make(self.game_name)
        env                                       = WarpFrame(env)
        env                                       = dqn.MaxAndSkipEnv(env,args.skip_frames)
        env                                       = dqn.ScaledFloatFrame(env)
        env                                       = dqn.EpisodicLifeEnv(env)
        env                                       = dqn.FrameStack(env, args.framestack)
        self.state_size                           = (84,84,args.framestack)#env.observation_space.shape[0]
        self.action_size                          = env.action_space.n
        print('State size:', self.state_size, 'Action size:', self.action_size)
        # Define global model
        self.global_model                         = A3C(self.state_size, self.action_size)
        _                                         = self.global_model(tf.convert_to_tensor(np.random.random((1, 84,84,args.framestack,1)), dtype=tf.float32)) # Initialise model (get weights)
        # Define optimiser
        self.opt                                  = AdamOptimizer(args.lr, use_locking=True)
    def train(self):
        # Random agent for benchmark
        if args.algorithm=='random':
            random_agent                          = RandomAgent(self.game_name, args.max_eps)
            random_agent.run()
            return
        # If not random algorithm
        res_queue                                 = Queue()
        workers                                   = [Worker(self.state_size,
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
        moving_average_rewards                    = []
        while True:
            reward                                = res_queue.get()
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
        env                                       = gym.make(self.game_name).unwrapped
        env                                       = WarpFrame(env)
        env                                       = dqn.MaxAndSkipEnv(env,args.skip_frames)
        env                                       = dqn.ScaledFloatFrame(env)
        env                                       = dqn.EpisodicLifeEnv(env)
        env                                       = dqn.FrameStack(env, args.framestack)
        state                                     = env.reset()
        state                                     = np.reshape(state, [1,84,84,args.framestack,1])
        model                                     = self.global_model
        model_path                                = os.path.join(self.save_dir, 'model_{}.h5'.format(self.game_name))
        print('Loading model from: {}'.format(model_path))
        model.load_weights(model_path)
        done                                      = False
        step_counter                              = 0
        reward_sum                                = 0

        try:
            while not done:
                env.render()#mode='rgb_array')
                policy, _                         = model(tf.convert_to_tensor(state, dtype=tf.float32))
                policy                            = tf.nn.softmax(policy)[0,0,0,0,:]
                action                            = np.argmax(policy)
                state, reward, done, _            = env.step(action)
                state                             = np.reshape(state, [1,84,84,args.framestack,1])
                reward_sum                        = reward_sum + reward
                print('{}. Reward: {}, action: {}'.format(step_counter, reward_sum, action))
                step_counter                      = step_counter + 1
        except KeyboardInterrupt:
            print('Received Keyboard Interrupt. Shutting down.')
        finally:
            env.close()


#------------------------------
# Worker
#------------------------------
class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode                                = 0
    global_moving_average_reward                  = 0
    best_score                                    = 0
    save_lock                                     = threading.Lock()

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
        self.state_size                           = state_size
        self.action_size                          = action_size
        self.result_queue                         = result_queue
        self.global_model                         = global_model
        self.opt                                  = opt
        self.local_model                          = A3C(self.state_size, self.action_size)
        self.worker_idx                           = idx
        self.game_name                            = game_name
        self.env                                  = gym.make(self.game_name).unwrapped
        self.env                                  = WarpFrame(self.env)
        self.env                                  = dqn.MaxAndSkipEnv(self.env, args.skip_frames)
        self.env                                  = dqn.ScaledFloatFrame(self.env)
        self.env                                  = dqn.EpisodicLifeEnv(self.env)
        self.env                                  = dqn.FrameStack(self.env, args.framestack)
        self.save_dir                             = save_dir
        self.epi_loss                             = 0.0

    def run(self):
        total_step                                = 1
        mem                                       = Memory(args.memory_size)
        while Worker.global_episode < args.max_eps:
            current_state                         = self.env.reset()
            current_state                         = np.reshape(current_state, [1,84,84,args.framestack,1])
            mem.clear()
            epi_reward                            = 0.
            epi_asjusted_reward                   = 0.
            epi_steps                             = 1
            self.epi_loss                         = 0

            time_count                            = 0
            done                                  = False
            while not done:
                # Boltzmann action selection
                logits, _                         = self.local_model(tf.convert_to_tensor(current_state,dtype=tf.float32))
                probs                             = tf.nn.softmax(logits[0,0,0,0,:])
                action                            = np.random.choice(self.action_size, p=probs.numpy())
                # Play action
                new_state, reward, done, _        = self.env.step(action)
                new_state                         = np.reshape(new_state, [1,84,84,args.framestack,1])
                if done:
                    adjusted_reward               = -25
                else:
                    adjusted_reward               = reward
                epi_reward                        = epi_reward + reward
                epi_asjusted_reward               = epi_asjusted_reward + adjusted_reward
                # Update memory
                mem.add(current_state, action, adjusted_reward)

                if ((time_count % args.update_freq)==0) or done:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape
                    with tf.GradientTape() as tape:
                        total_loss                = self.compute_loss(done, new_state, mem, args.gamma)
                    self.epi_loss                 = self.epi_loss + total_loss
                    # Calculate local gradients
                    grads                         = tape.gradient(total_loss, self.local_model.trainable_weights)
                    # Push local gradients to global model
                    self.opt.apply_gradients(zip(grads, self.global_model.trainable_weights))
                    # Update local model with new weights
                    self.local_model.set_weights(self.global_model.get_weights())
                    # Print results at end of game
                    if done:
                        # Update global moving avg and also print results
                        Worker.global_moving_average_reward= \
                            record(Worker.global_episode, epi_reward, epi_asjusted_reward,
                                   self.worker_idx, Worker.global_moving_average_reward, 
                                   self.result_queue,self.epi_loss, epi_steps)
                        # Lock to save model and prevent data races
                        if epi_reward > Worker.best_score:
                            with Worker.save_lock:
                                print('Saving best model to {}, '
                                            'episode score: {}'.format(self.save_dir, epi_reward))
                                self.global_model.save_weights(os.path.join(self.save_dir,'model_{}.h5'.format(self.game_name)))
                                Worker.best_score = epi_reward
                        Worker.global_episode     = Worker.global_episode + 1
                epi_steps                         = epi_steps + 1
                time_count                        = time_count + 1
                current_state                     = new_state
                total_step                        = total_step + 1
        self.result_queue.put(None)

    def compute_loss(self,done,new_state,memory,gamma=0.99):#,sample=args.batch_size):
        # random_idx                              = np.random.choice(len(memory.states),sample,replace=True)
        # sample_states                           = [memory.states[idx] for idx in random_idx]
        # sample_actions                          = [memory.actions[idx] for idx in random_idx]
        # sample_rewards                          = [memory.rewards[idx] for idx in random_idx]

        if done:
            reward_sum                            = 0.
        else:
            reward_sum                            = self.local_model(tf.convert_to_tensor(new_state,dtype=tf.float32))[-1].numpy()[0]

        # Get discounted rewards
        discounted_rewards                        = []
        for reward in list(memory.rewards)[::-1]: # reverse
            reward_sum                            = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        logits, values                            = self.local_model(tf.convert_to_tensor(np.vstack(memory.states),dtype=tf.float32))
        # Get our advantages
        # print(values.shape, np.array(discounted_rewards).shape, np.array(discounted_rewards).shape, np.vstack(memory.states).shape)
        advantage                                 = tf.convert_to_tensor(np.array(discounted_rewards),dtype=tf.float32) - values
        # Value loss
        value_loss                                = advantage ** 2
        # Calculate our policy loss
        policy                                    = tf.nn.softmax(logits[:,0,0,0,:])
        entropy                                   = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy,logits=logits[:,0,0,0,:])
        policy_loss                               = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions,logits=logits[:,0,0,0,:])
        policy_loss                               = policy_loss * tf.stop_gradient(advantage)
        policy_loss                               = policy_loss - 0.01 * entropy
        total_loss                                = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss


#------------------------------
# Main
#------------------------------
if __name__=='__main__':
    print(args)
    master                                        = MasterAgent()
    if args.train:
        master.train()
    else:
        master.play()


# python scripts/a3c_cnn_main.py --algorithm random --max-eps=4000 --save-dir model2 --train
# python scripts/a3c_cnn_main.py --algorithm a3c --max-eps=250 --save-dir model3 --train --update-freq 500 --memory-size 500
# python scripts/a3c_cnn_main.py --save-dir model
# python scripts/a3c_cnn_main.py --algorithm a3c --max-eps=10000 --save-dir model3 --train --update-freq 32 --memory-size 32
# python scripts/a3c_cnn_main.py --game-name DemonAttackNoFrameskip-v4 --algorithm a3c --max-eps=10000 --save-dir models --train --update-freq 5 --memory-size 5 --framestack 4 --lr 0.00025
# python scripts/a3c_cnn_main.py --game-name DemonAttackNoFrameskip-v4 --save-dir models