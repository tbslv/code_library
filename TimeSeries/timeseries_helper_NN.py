
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def moving_average_forecast(series, window_size):
	"""Generates a moving average forecast

	Args:
	  series (array of float) - contains the values of the time series
	  window_size (int) - the number of time steps to compute the average for

	Returns:
	  forecast (array of float) - the moving average forecast
	"""

	# Initialize a list
	forecast = []
	
	# Compute the moving average based on the window size
	for time in range(len(series) - window_size):
	  forecast.append(series[time:time + window_size].mean())
	
	# Convert to a numpy array
	forecast = np.array(forecast)

	return forecast

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
	"""Generates dataset windows

	Args:
	  series (array of float) - contains the values of the time series
	  window_size (int) - the number of time steps to include in the feature
	  batch_size (int) - the batch size
	  shuffle_buffer(int) - buffer size to use for the shuffle method

	Returns:
	  dataset (TF Dataset) - TF Dataset containing time windows
	"""
  
	# Generate a TF Dataset from the series values
	dataset = tf.data.Dataset.from_tensor_slices(series)
	
	# Window the data but only take those with the specified size
	dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
	
	# Flatten the windows by putting its elements in a single batch
	dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

	# Create tuples with features and labels 
	dataset = dataset.map(lambda window: (window[:-1], window[-1]))

	# Shuffle the windows
	dataset = dataset.shuffle(shuffle_buffer)
	
	# Create batches of windows
	dataset = dataset.batch(batch_size)
	
	# Optimize the dataset for training
	dataset = dataset.cache().prefetch(1)
	
	return dataset

def model_forecast(model, series, window_size, batch_size):
	"""Uses an input model to generate predictions on data windows

	Args:
	  model (TF Keras Model) - model that accepts data windows
	  series (array of float) - contains the values of the time series
	  window_size (int) - the number of time steps to include in the window
	  batch_size (int) - the batch size

	Returns:
	  forecast (numpy array) - array containing predictions
	"""

	# Add an axis for the feature dimension of RNN layers
	series = tf.expand_dims(series, axis=-1)
	
	# Generate a TF Dataset from the series values
	dataset = tf.data.Dataset.from_tensor_slices(series)

	# Window the data but only take those with the specified size
	dataset = dataset.window(window_size, shift=1, drop_remainder=True)

	# Flatten the windows by putting its elements in a single batch
	dataset = dataset.flat_map(lambda w: w.batch(window_size))
	
	# Create batches of windows
	dataset = dataset.batch(batch_size).prefetch(1)
	
	# Get predictions on the entire dataset
	forecast = model.predict(dataset, verbose=0)
	
	return forecast


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
	'''
	Halts the training when a certain metric is met

	Args:
	  epoch (integer) - index of epoch (required but unused in the function definition below)
	  logs (dict) - metric results from the training epoch
	'''

	# Check the validation set MAE
	if(logs.get('val_mae') < 5.7):

	  # Stop if threshold is met
	  print("\nRequired val MAE is met so cancelling training!")
	  self.model.stop_training = True


def optimize_lr(model,train_set, window_size, epochs=100,
				optimizer=tf.keras.optimizers.SGD(momentum=0.9),
				loss=tf.keras.losses.Huber(),
				metrics=["mae"]):


	# Set the learning rate scheduler
	lr_schedule = tf.keras.callbacks.LearningRateScheduler(
		lambda epoch: 1e-8 * 10**(epoch / 20))


	# Set the training parameters
	model.compile(loss=loss, optimizer=optimizer,metrics=metrics)

	# Train the model
	history = model.fit(train_set, epochs=epochs, callbacks=[lr_schedule])

	# Define the learning rate array
	lrs = 1e-8 * (10 ** (np.arange(epochs) / 20))

	# Set the figure size
	plt.figure(figsize=(10, 6))

	# Set the grid
	plt.grid(True)

	# Plot the loss in log scale
	plt.semilogx(lrs, history.history["loss"])

	# Increase the tickmarks size
	plt.tick_params('both', length=10, width=1, which='both')

	# Set the plot boundaries
	plt.axis([1e-8, 1e-3, 0, 100])
	plt.xlabel("Learning Rate")
	plt.ylabel("Loss")
	plt.title("Learning Rate Optimization")

	# Show the plot
	plt.show()

def train_val_split(time, series):
	""" Splits time series into train and validations sets"""
	time_train = time[:SPLIT_TIME]
	series_train = series[:SPLIT_TIME]
	time_valid = time[SPLIT_TIME:]
	series_valid = series[SPLIT_TIME:]

	return time_train, series_train, time_valid, series_valid

import tensorflow as tf

def create_uncompiled_model(window_size):
	"""Define uncompiled model

	Returns:
		tf.keras.Model: uncompiled model
	"""
	model = tf.keras.models.Sequential([
		# Input layer with placeholder for time steps and number of features
		tf.keras.Input(shape=(window_size, 1)),  # Assuming 1 feature per time step
		
		# 1D Convolutional Layer
		tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
		
		# LSTM layers to learn sequential dependencies
		tf.keras.layers.LSTM(units=64, return_sequences=True),
		tf.keras.layers.LSTM(units=32, return_sequences=False),
		
		# Dense layers for final output processing
		tf.keras.layers.Dense(units=16),
		tf.keras.layers.Dense(units=1)  # Assuming a single output prediction
	])
	return model

# Example of how to compile, fit, and train the model
def compile_and_train_model(train_set, callbacks,epochs=10,
							optimizer=tf.keras.optimizers.Adam(),  # You can try different optimizers like SGD, RMSprop, etc.
							loss='mse',  # Using Mean Squared Error for regression tasks
							metrics=['mae']
							):
	# Step 1: Create the model
	model = create_uncompiled_model()

	# Step 2: Compile the model
	model.compile(optimizer=optimizer,  # You can try different optimizers like SGD, RMSprop, etc.
				  loss=loss,# Using Mean Squared Error for regression tasks
				  metrics=metrics)  # Mean Absolute Error as additional metric

	# Step 3: Fit the model to the training data
	history = model.fit(train_set, epochs=epochs,callbacks=[callbacks])

	# Step 4: Return the training history and model
	return model, history

def compute_metrics(true_series, forecast):
	"""Computes MSE and MAE metrics for the forecast"""
	mse = tf.keras.losses.MSE(true_series, forecast)
	mae = tf.keras.losses.MAE(true_series, forecast)
	return mse, mae

def differentiator(series, period):
	return (series[period:] - series[:-period])