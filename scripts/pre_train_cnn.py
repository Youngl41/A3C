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

def augment_rot(image_array):
    images = [image_array]
    for i in range(3):
        image_array = np.rot90(image_array)
        images.append(image_array)
    return images


#------------------------------
# CIFAR data
#------------------------------
# Load CIFAR 100 data
cifar_dir = os.path.join(main_dir, 'data', 'cifar-100-python')
meta_data = unpickle(os.path.join(cifar_dir, 'meta'))
train_data = unpickle(os.path.join(cifar_dir, 'train'))
test_data = unpickle(os.path.join(cifar_dir, 'test'))

# Describe data
# train_data.keys()
# train_data[b'fine_labels'][0]
# train_data[b'data'][0]
# meta_data[b'fine_label_names']
# b'batch_label', b'fine_labels', b'coarse_labels', b'data'

# Reshape raw dataset
train_arr = train_data[b'data'].reshape(train_data[b'data'].shape[0], 3, 32, 32)
train_arr = train_arr.transpose(0, 2, 3, 1).astype('uint8')

# Convert to greyscale
train_arr_sc = np.array(list(map(lambda x: pp_cifar_img(x, greyscale=True, enlarge_dim=(84,84)), list(train_arr))))

# Rotate the image by right angles
train_pp = [item for sublist in list(map(lambda x: augment_rot(x), train_arr_sc)) for item in sublist]
np.array(train_pp).shape
labels = [0,1,2,3]*len(train_arr_sc)


#------------------------------
# CNN model
#------------------------------
# Define model
model_input                                 = keras.models.Input(shape=(84,84,1,1))
x                                           = layers.Conv3D(32, kernel_size=(8,8,1), strides=(4,4,1), input_shape=(84,84,1,1), activation='relu', padding='valid', kernel_initializer='random_uniform', bias_initializer=keras.initializers.Constant(0.1))(model_input)
x                                           = layers.MaxPooling3D(pool_size=(2,2,1), strides=(1,1,1), padding='valid')(x)
x                                           = layers.Conv3D(64, kernel_size=(4,4,1), strides=(2,2,1), activation='relu', padding='valid', kernel_initializer='random_uniform', bias_initializer=keras.initializers.Constant(0.1))(x)
x                                           = layers.MaxPooling3D(pool_size=(2,2,1), strides=(1,1,1), padding='valid')(x)
x                                           = layers.Conv3D(128, kernel_size=(3,3,1), strides=(1,1,1), activation='relu', padding='valid', kernel_initializer='random_uniform', bias_initializer=keras.initializers.Constant(0.1))(x)
x                                           = layers.MaxPooling3D(pool_size=(2,2,1), strides=(1,1,1), padding='valid')(x)
x                                           = layers.BatchNormalization(center=True, scale=True)(x)
x                                 = layers.Dense(512, activation='relu')(x)
x                                           = layers.Dropout(0.5)(x)
x                                 = layers.Dense(512, activation='relu')(x)
output                                 = layers.Dense(4, activation='softmax')(x)
model_name                                  = 'cnn_pre_training'
model                                       = keras.models.Model(inputs=model_input, outputs=output, name=model_name)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Model fit
