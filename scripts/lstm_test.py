#======================================================
# LSTM practice script
#======================================================
'''
Info:       
Version:    1.0
Author:     Young Lee
Created:    Sunday, 15 September 2019
Source:     https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
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

# os.environ['CUDA_VISIBLE_DEVICES']  = '-1'


#------------------------------
# Utility functions
#------------------------------
# Split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y                            = [], []
	for i in range(len(sequence)):
		end_idx                      = i + n_steps
		if end_idx > len(sequence)-1:
			break
		elif end_idx <= len(sequence)-1:
			seq_x, seq_y               = sequence[i:end_idx], sequence[end_idx]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

# Split a multivariate sequence into samples
def split_sequences(sequences, n_steps, parallel=False):
	'''
	Parallel series: 		looks at the last n_steps and 
							predicts the time point, t_(n+1).
	Non-parallel series: 	looks at the last n_steps of all
							but last feature and predicts the
							last feature for time point, t_n.
	'''
	X, y                            = [], []
	for i in range(len(sequences)):
		end_idx                      = i + n_steps
		if parallel==False:
			if end_idx > len(sequences):
				break
			elif end_idx <= len(sequences):
				seq_x, seq_y              = sequences[i:end_idx, :-1], sequences[end_idx-1, -1]
		elif parallel==True:
			if end_idx > len(sequences)-1:
				break
			elif end_idx <= len(sequences)-1:
				seq_x, seq_y              = sequences[i:end_idx, :], sequences[end_idx, :]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

# Split a multivariate sequence into samples for multi-step LSTM
def split_seq_multi_step(sequences, n_steps_in, n_steps_out, parallel=True):
	'''
	Parallel series: 		looks at the last n_steps and 
							predicts the time points, t_(n+1)
							to t_(n+m).
	Non-parallel series: 	looks at the last n_steps of all
							but last feature and predicts the
							last feature for time points, t_n
							to t_(n+m-1).
	'''
	X, y                            = [], []
	for i in range(len(sequences)):
		end_input_idx                = i + n_steps_in
		end_output_idx               = end_input_idx + n_steps_out
		if parallel==False:
			if end_output_idx > len(sequences):
				break
			elif end_output_idx <= len(sequences):
				seq_x, seq_y              = sequences[i:end_input_idx, :-1], sequences[end_input_idx-1:end_output_idx-1, -1]
		elif parallel==True:
			if end_output_idx > len(sequences)-1:
				break
			elif end_output_idx <= len(sequences)-1:
				seq_x, seq_y              = sequences[i:end_input_idx, :], sequences[end_input_idx:end_output_idx, :]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

# Reshape multivariate series input data for lstm
def reshape_lstm_input(arrays):
	reshaped_arrays                 = [array.reshape((len(array),1)) for array in arrays]
	data                            = np.hstack(reshaped_arrays)
	return data


#------------------------------
# Pre-process data (univariate)
#------------------------------
# Prepare data
n_features                          = 1
n_steps                             = 5

data                                = np.arange(0,100,1)**1.5 + np.random.normal(0,1,100) * 5
X, y                                = split_sequence(data,n_steps)
X                                   = X.reshape((X.shape[0], X.shape[1], n_features))

for i in range(5):
	print(X[i], y[i])

data_test                           = np.arange(0,200,1)**1.5
X_test, y_test                      = split_sequence(data_test,n_steps)
X_test                              = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))


#------------------------------
# Vanilla LSTM
#------------------------------
model                               = keras.models.Sequential()
model.add(layers.LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mae')
model.summary()

# Fit
model.fit(X,y, validation_data=(X_test, y_test), epochs=100, verbose=1)

# Evaluate
p_test                              = model.predict(X_test)
p_test                              = p_test.reshape((p_test.shape[0],))

mae                                 = np.mean(abs(y_test - p_test))
print('MAE:\t', mae)

plt.plot(y_test, label='true')
plt.plot(p_test, label='pred')
plt.legend()
plt.show()


#------------------------------
# Stacked LSTM
#------------------------------
model                               = keras.models.Sequential()
model.add(layers.LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(layers.LSTM(50, activation='relu', return_sequences=True))
model.add(layers.LSTM(50, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

# Fit
model.fit(X,y, epochs=100, verbose=1)

# Evaluate
p_test                              = model.predict(X_test)
p_test                              = p_test.reshape((p_test.shape[0],))

mae                                 = np.mean(abs(y_test - p_test))
print('MAE:\t', mae)


#------------------------------
# Bidirectional LSTM
#------------------------------
model                               = keras.models.Sequential()
model.add(layers.Bidirectional(layers.LSTM(50, activation='relu', return_sequences=True), input_shape=(n_steps, n_features)))
model.add(layers.Bidirectional(layers.LSTM(50, activation='relu', return_sequences=True)))
model.add(layers.LSTM(50, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

# Fit
model.fit(X,y, epochs=100, verbose=1)

# Evaluate
p_test                              = model.predict(X_test)
p_test                              = p_test.reshape((p_test.shape[0],))

mae                                 = np.mean(abs(y_test - p_test))
print('MAE:\t', mae)


#------------------------------
# Multivariate LSTM - non-parallel
#------------------------------
# Pre-process data (univariate)
n_features                          = 3
n_steps                             = 5

data_1                              = np.arange(0,100,1)**1.5 + np.random.normal(0,1,100) * 5
data_2                              = np.arange(0,100,1) + np.random.normal(0,1,100) * 5
data_3                              = np.random.normal(0,1,100) * 5
data_4                              = data_1 + data_2 * 2

data_1                              = data_1.reshape((len(data_1),1))
data_2                              = data_2.reshape((len(data_2),1))
data_3                              = data_3.reshape((len(data_3),1))
data_4                              = data_4.reshape((len(data_4),1))
data                                = np.hstack((data_1, data_2, data_3, data_4))

X, y                                = split_sequences(data,n_steps)

print(X.shape, y.shape) # Num rows | num of historical time | num features
for i in range(5):
	print(X[i], y[i])

data_1_test                         = np.arange(0,200,1)**1.5 + np.random.normal(0,1,200) * 5
data_2_test                         = np.arange(0,200,1) + np.random.normal(0,1,200) * 5
data_3_test                         = np.random.normal(0,1,200) * 5
data_4_test                         = data_1_test + data_2_test * 2

data_1_test                         = data_1_test.reshape((len(data_1_test),1))
data_2_test                         = data_2_test.reshape((len(data_2_test),1))
data_3_test                         = data_3_test.reshape((len(data_3_test),1))
data_4_test                         = data_4_test.reshape((len(data_4_test),1))
data_test                           = np.hstack((data_1_test, data_2_test, data_3_test, data_4_test))

X_test, y_test                      = split_sequences(data_test,n_steps)

# Bidirectional stacked LSTM
model                               = keras.models.Sequential()
model.add(layers.Bidirectional(layers.LSTM(50, activation='relu', return_sequences=True), input_shape=(n_steps, n_features)))
model.add(layers.Dropout(0.05))
model.add(layers.Bidirectional(layers.LSTM(50, activation='relu', return_sequences=True)))
model.add(layers.LSTM(50, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

# Fit
model.fit(X,y, validation_data=(X_test, y_test), epochs=100, verbose=1)

# Evaluate
p_test                              = model.predict(X_test)
p_test                              = p_test.reshape((p_test.shape[0],))
mae                                 = np.mean(abs(y_test - p_test))
print('MAE:\t', mae)

# Plot
plt.plot(y_test, label='true')
plt.plot(p_test, label='pred')
plt.axvline(x=100, color='r', linestyle=':', label='extrapolation to the right')
plt.legend()
plt.show()


#------------------------------
# Multivariate LSTM - parallel
#------------------------------
# Pre-process data (univariate)
n_features                          = 4
n_steps                             = 5
n_points                            = 300

data_1                              = np.sin(np.pi*np.arange(0,n_points,1)*10/360)*10 + np.random.normal(0,1,n_points)
data_2                              = np.cos(np.pi*np.arange(0,n_points,1)*30/360)**3*4 + np.random.normal(0,1,n_points)
data_3                              = np.random.normal(0,1,n_points)
data_4                              = data_1 + data_2 + np.random.normal(0,1,n_points)
data                                = reshape_lstm_input([data_1, data_2, data_3, data_4])
X, y                                = split_sequences(data,n_steps, parallel=True)

print(X.shape, y.shape) # Num rows | num of historical time | num features
for i in range(5):
	print(X[i], y[i])

n_points                            = 500
data_1_test                         = np.sin(np.pi*np.arange(0,n_points,1)*10/360)*10 + np.random.normal(0,1,n_points)
data_2_test                         = np.cos(np.pi*np.arange(0,n_points,1)*30/360)**3*4 + np.random.normal(0,1,n_points)
data_3_test                         = np.random.normal(0,1,n_points)
data_4_test                         = data_1_test + data_2_test + np.random.normal(0,1,n_points)

data_test                           = reshape_lstm_input([data_1_test, data_2_test, data_3_test, data_4_test])
X_test, y_test                      = split_sequences(data_test,n_steps,parallel=True)

# Bidirectional stacked LSTM
model                               = keras.models.Sequential()
model.add(layers.Bidirectional(layers.LSTM(100, activation='relu', return_sequences=True), input_shape=(n_steps, n_features)))
# model.add(layers.Dropout(0.05))
model.add(layers.Bidirectional(layers.LSTM(100, activation='relu', return_sequences=True)))
model.add(layers.LSTM(50, activation='relu'))
model.add(layers.Dense(n_features))
model.compile(optimizer='adam', loss='mse')
model.summary()

# Fit
model.fit(X,y, validation_data=(X_test, y_test), epochs=100, verbose=1)

# Evaluate
p_test                              = model.predict(X_test)
p_test                              = p_test.reshape((p_test.shape[0],))
mae                                 = np.mean(abs(y_test - p_test))
print('MAE:\t', mae)

# Plot
ax                                  = plt.gca()
for i in range(y_test.shape[1]):
	color                           = next(ax._get_lines.prop_cycler)['color']
	plt.plot(y_test[:, i], label='true_'+str(i), color=color)
	plt.plot(p_test[:, i], label='pred_'+str(i), color=color, linestyle='dotted')

plt.axvline(x=300, color='r', linestyle=':', label='extrapolation to the right')
plt.legend()
plt.show()


#------------------------------
# Multi-step LSTM
#------------------------------
'''
Multi-step LSTM looks at the last n_steps and predicts 
the next m_step time points, t_(n+1) to t_(n+m).
'''

# Pre-process data (univariate)
n_features                          = 4
n_steps_in                          = 5
n_steps_out                         = 10
n_points                            = 300

data_1                              = np.sin(np.pi*np.arange(0,n_points,1)*10/360)*10 + np.random.normal(0,1,n_points)
data_2                              = np.cos(np.pi*np.arange(0,n_points,1)*30/360)**3*4 + np.random.normal(0,1,n_points)
data_3                              = np.random.normal(0,1,n_points)
data_4                              = data_1 + data_2 + np.random.normal(0,1,n_points)
data                                = reshape_lstm_input([data_1, data_2, data_3, data_4])

X, y                                = split_seq_multi_step(data, n_steps_in=n_steps_in, n_steps_out=n_steps_out, parallel=True)

n_points                            = 500
data_1_test                         = np.sin(np.pi*np.arange(0,n_points,1)*10/360)*10 + np.random.normal(0,1,n_points)
data_2_test                         = np.cos(np.pi*np.arange(0,n_points,1)*30/360)**3*4 + np.random.normal(0,1,n_points)
data_3_test                         = np.random.normal(0,1,n_points)
data_4_test                         = data_1_test + data_2_test + np.random.normal(0,1,n_points)

data_test                           = reshape_lstm_input([data_1_test, data_2_test, data_3_test, data_4_test])
X_test, y_test                      = split_seq_multi_step(data_test, n_steps_in=n_steps_in, n_steps_out=n_steps_out, parallel=True)

# Bidirectional stacked LSTM
model                               = keras.models.Sequential()
model.add(layers.Bidirectional(layers.LSTM(100, activation='relu'), input_shape=(n_steps_in, n_features)))
model.add(layers.RepeatVector(n_steps_out))
# model.add(layers.Dropout(0.05))
model.add(layers.Bidirectional(layers.LSTM(100, activation='relu', return_sequences=True)))
model.add(layers.LSTM(50, activation='relu', return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(n_features)))
model.compile(optimizer='adam', loss='mse')
model.summary()

# Fit
model.fit(X,y, validation_data=(X_test, y_test), epochs=100, batch_size=1024, verbose=1, use_multiprocessing=True)

# Evaluate
p_test                              = model.predict(X_test)
p_test                              = p_test.reshape((p_test.shape[0],))
mae                                 = np.mean(abs(y_test - p_test))
print('MAE:\t', mae)

for i in range(y_test.shape[1]):
	print('MAE n step '+str(i)+':\t', np.mean(abs(y_test[:, i, -1] - p_test[:, i, -1])))

# Plot
ax                                  = plt.gca()
for i in range(y_test.shape[1]):
	color                           = next(ax._get_lines.prop_cycler)['color']
	plt.plot(y_test[:, i, -1], label='true_n_step_out='+str(i), color=color)
	plt.plot(p_test[:, i, -1], label='pred_n_step_out='+str(i), color=color, linestyle='dotted')

plt.axvline(x=300, color='r', linestyle=':', label='extrapolation to the right')
plt.legend()
plt.show()

# Plot
plt.figure(figsize=(16,9))
plt.plot(y_test[:, -1, -1], label='true_n_step_out='+str(-1))
plt.plot(p_test[:, -1, -1], label='pred_n_step_out='+str(-1))

plt.axvline(x=300, color='r', linestyle=':', label='extrapolation to the right')
plt.legend()
plt.show()


#------------------------------
# Multi-step LSTM non seq
#------------------------------
'''
Multi-step LSTM looks at the last n_steps and predicts 
the next m_step time points, t_(n+1) to t_(n+m).
'''

# Pre-process data (univariate)
n_features                          = 4
n_steps_in                          = 5
n_steps_out                         = 10
n_points                            = 300

data_1                              = np.sin(np.pi*np.arange(0,n_points,1)*10/360)*10 + np.random.normal(0,1,n_points)
data_2                              = np.cos(np.pi*np.arange(0,n_points,1)*30/360)**3*4 + np.random.normal(0,1,n_points)
data_3                              = np.random.normal(0,1,n_points)
data_4                              = data_1 + data_2 + np.random.normal(0,1,n_points)
data                                = reshape_lstm_input([data_1, data_2, data_3, data_4])

X, y                                = split_seq_multi_step(data, n_steps_in=n_steps_in, n_steps_out=n_steps_out, parallel=True)

n_points                            = 500
data_1_test                         = np.sin(np.pi*np.arange(0,n_points,1)*10/360)*10 + np.random.normal(0,1,n_points)
data_2_test                         = np.cos(np.pi*np.arange(0,n_points,1)*30/360)**3*4 + np.random.normal(0,1,n_points)
data_3_test                         = np.random.normal(0,1,n_points)
data_4_test                         = data_1_test + data_2_test + np.random.normal(0,1,n_points)

data_test                           = reshape_lstm_input([data_1_test, data_2_test, data_3_test, data_4_test])
X_test, y_test                      = split_seq_multi_step(data_test, n_steps_in=n_steps_in, n_steps_out=n_steps_out, parallel=True)

# Bidirectional stacked LSTM
state_size = (n_steps_in, n_features)
model_name = 'Bidirectional_LSTM'
model_input = keras.models.Input(shape=state_size)
x = layers.Bidirectional(layers.LSTM(100, activation='relu'))(model_input)
x = layers.RepeatVector(n_steps_out)(x)
x = layers.Bidirectional(layers.LSTM(100, activation='relu', return_sequences=True))(x)
x = layers.LSTM(50, activation='relu', return_sequences=True)(x)
output = layers.TimeDistributed(layers.Dense(n_features))(x)
model = keras.models.Model(inputs=model_input, outputs=output, name=model_name)
model.compile(optimizer='adam', loss='mse')
model.summary()

# Fit
model.fit(X,y, validation_data=(X_test, y_test), epochs=100, batch_size=1024, verbose=1, use_multiprocessing=True)

# Evaluate
p_test                              = model.predict(X_test)
p_test                              = p_test.reshape((p_test.shape[0],))
mae                                 = np.mean(abs(y_test - p_test))
print('MAE:\t', mae)

for i in range(y_test.shape[1]):
	print('MAE n step '+str(i)+':\t', np.mean(abs(y_test[:, i, -1] - p_test[:, i, -1])))

# Plot
ax                                  = plt.gca()
for i in range(y_test.shape[1]):
	color                           = next(ax._get_lines.prop_cycler)['color']
	plt.plot(y_test[:, i, -1], label='true_n_step_out='+str(i), color=color)
	plt.plot(p_test[:, i, -1], label='pred_n_step_out='+str(i), color=color, linestyle='dotted')

plt.axvline(x=300, color='r', linestyle=':', label='extrapolation to the right')
plt.legend()
plt.show()

# Plot
plt.figure(figsize=(16,9))
plt.plot(y_test[:, -1, -1], label='true_n_step_out='+str(-1))
plt.plot(p_test[:, -1, -1], label='pred_n_step_out='+str(-1))

plt.axvline(x=300, color='r', linestyle=':', label='extrapolation to the right')
plt.legend()
plt.show()


#------------------------------
# Single-step LSTM non seq
#------------------------------
'''
Single-step LSTM looks at the last n_steps and predicts 
the next m_step time points, t_(n+1) to t_(n+m).
'''

# Pre-process data (univariate)
n_features                          = 4
n_steps_in                          = 5
n_steps_out                         = 1
n_points                            = 300

data_1                              = np.sin(np.pi*np.arange(0,n_points,1)*10/360)*10 + np.random.normal(0,1,n_points)
data_2                              = np.cos(np.pi*np.arange(0,n_points,1)*30/360)**3*4 + np.random.normal(0,1,n_points)
data_3                              = np.random.normal(0,1,n_points)
data_4                              = data_1 + data_2 + np.random.normal(0,1,n_points)
data                                = reshape_lstm_input([data_1, data_2, data_3, data_4])

X, y                                = split_seq_multi_step(data, n_steps_in=n_steps_in, n_steps_out=n_steps_out, parallel=False)
X[0]
y[0]
n_points                            = 500
data_1_test                         = np.sin(np.pi*np.arange(0,n_points,1)*10/360)*10 + np.random.normal(0,1,n_points)
data_2_test                         = np.cos(np.pi*np.arange(0,n_points,1)*30/360)**3*4 + np.random.normal(0,1,n_points)
data_3_test                         = np.random.normal(0,1,n_points)
data_4_test                         = data_1_test + data_2_test + np.random.normal(0,1,n_points)

data_test                           = reshape_lstm_input([data_1_test, data_2_test, data_3_test, data_4_test])
X_test, y_test                      = split_seq_multi_step(data_test, n_steps_in=n_steps_in, n_steps_out=n_steps_out, parallel=True)

# Bidirectional stacked LSTM
state_size = (n_steps_in, n_features)
model_name = 'Bidirectional_LSTM'
model_input = keras.models.Input(shape=state_size)
x = layers.Bidirectional(layers.LSTM(100, activation='relu'))(model_input)
x = layers.RepeatVector(n_steps_out)(x)
x = layers.Bidirectional(layers.LSTM(100, activation='relu', return_sequences=True))(x)
x = layers.LSTM(50, activation='relu', return_sequences=True)(x)
output = layers.TimeDistributed(layers.Dense(n_features))(x)
model = keras.models.Model(inputs=model_input, outputs=output, name=model_name)
model.compile(optimizer='adam', loss='mse')
model.summary()

# Fit
model.fit(X,y, validation_data=(X_test, y_test), epochs=100, batch_size=1024, verbose=1, use_multiprocessing=True)

# Evaluate
p_test                              = model.predict(X_test)
p_test                              = p_test.reshape((p_test.shape[0],))
mae                                 = np.mean(abs(y_test - p_test))
print('MAE:\t', mae)

for i in range(y_test.shape[1]):
	print('MAE n step '+str(i)+':\t', np.mean(abs(y_test[:, i, -1] - p_test[:, i, -1])))

# Plot
ax                                  = plt.gca()
for i in range(y_test.shape[1]):
	color                           = next(ax._get_lines.prop_cycler)['color']
	plt.plot(y_test[:, i, -1], label='true_n_step_out='+str(i), color=color)
	plt.plot(p_test[:, i, -1], label='pred_n_step_out='+str(i), color=color, linestyle='dotted')

plt.axvline(x=300, color='r', linestyle=':', label='extrapolation to the right')
plt.legend()
plt.show()

# Plot
plt.figure(figsize=(16,9))
plt.plot(y_test[:, -1, -1], label='true_n_step_out='+str(-1))
plt.plot(p_test[:, -1, -1], label='pred_n_step_out='+str(-1))

plt.axvline(x=300, color='r', linestyle=':', label='extrapolation to the right')
plt.legend()
plt.show()
