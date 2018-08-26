# !/user/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import math
import os

from IPython import display
from matplotlib import cm  #color map
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# current working directory
cwd = os.getcwd()

training_file_path = cwd + "//california_housing_train.csv"
Testing_file_path = cwd +  "//california_housing_test.csv"


california_housing_dataframe = pd.read_csv(training_file_path, sep = ",")
california_housing_dataframe = california_housing_dataframe.reindex(
	np.random.permutation(california_housing_dataframe.index))

def preprocess_features(california_housing_dataframe):
	ColNums = california_housing_dataframe.shape[1]
	selected_features = california_housing_dataframe.iloc[:, 0:ColNums-1]  #select the first ColNums-1 features
	processed_features = selected_features.copy()
	
	# Create a synthetic feature
	processed_features["rooms_per_person"] = (
		california_housing_dataframe["total_rooms"]/
		california_housing_dataframe["population"])
		
	return processed_features
	

def preprocess_targets(california_housing_dataframe):
	#output_targets = california_housing_dataframe.copy()
	output_targets = pd.DataFrame()
	output_targets["median_house_value"] = (california_housing_dataframe["median_house_value"]/1000.0)
	return output_targets
	
	
training_examples = preprocess_features(california_housing_dataframe.head(12000))
#print(training_examples.describe())

training_targets = preprocess_targets(california_housing_dataframe.head(12000))
#print(training_targets.describe())

validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
#print(validation_examples.describe())

validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
#print(validation_targets.describe())


# plot the input feature and targets
plt.figure(figsize = (13, 8)) # a figure instance with size (13, 8) in inch
ax = plt.subplot(1,2,1)
ax.set_title("Validation Data")

ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])

plt.scatter(validation_examples["longitude"], 
			validation_examples["latitude"],
			cmap = "coolwarm",
			c = validation_targets["median_house_value"]/ validation_targets["median_house_value"].max())

ax = plt.subplot(1,2,2)
ax.set_title("Training Data")

ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
plt.scatter(training_examples["longitude"],
			training_examples["latitude"],
			cmap = "coolwarm",
			c = training_targets["median_house_value"] / training_targets["median_house_value"].max())

_ = plt.plot()
plt.show()


def my_input_fn(features, targets, batch_size = 1, shuffle = True, num_epochs = None):
	# convert pandas data into a dict of np arrays
	features = {key: np.array(value) for key, value in dict(features).items()}

	# Construct a dataset, and configure batching/repeating
	ds = Dataset.from_tensor_slices((features, targets))
	ds = ds.batch(batch_size).repeat(num_epochs)

	if shuffle:
		ds = ds.shuffle(100000)

	# return the next batch of data
	features, labels = ds.make_one_shot_iterator().get_next()
	return features, labels

def construct_feature_columns(input_features):
	return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])


def train_model(
	learning_rate,
	steps,
	batch_size,
	training_examples,
	training_targets,
	validation_examples,
	validation_targets):
	
	periods = 10
	steps_per_period = steps/periods

	muy_optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
	muy_optimizer = tf.contrib.estimator.clip_gradients_by_norm(muy_optimizer, 5.0)

	linear_regressor = tf.estimator.LinearRegressor(
		feature_columns = construct_feature_columns(training_examples),
		optimizer = muy_optimizer)

	# create input function
	trainging_input_fn = lambda: my_input_fn(training_examples, training_targets["median_house_value"], batch_size = batch_size)

	predict_training_input_fn = lambda: my_input_fn(training_examples, training_targets["median_house_value"], num_epochs = 1, shuffle = False)

	predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets["median_house_value"], num_epochs = 1, shuffle = False)

	print("Training model...")
	print("RMSE (on training data): ")

	training_rmse = []
	validation_rmse = []

	for period in range(0, periods):
		linear_regressor.train(input_fn = trainging_input_fn, steps = steps_per_period)

		# Take a break and compute predictions
		training_predictions = linear_regressor.predict(input_fn = predict_training_input_fn)
		training_predictions = np.array([item['predictions'][0] for item in training_predictions])

		validation_predictions = linear_regressor.predict(input_fn = predict_validation_input_fn)
		validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

		#print("Trainig Prediction : ")
		#print(training_predictions)

		#print("Training Targets : ")
		#print(training_targets)

		# compute training and validation loss
		training_root_mean_squared_error = math.sqrt(
				metrics.mean_squared_error(training_predictions, training_targets))

		validation_root_mean_squared_error = math.sqrt(
				metrics.mean_squared_error(validation_predictions,validation_targets))

		print("  period %02d  :  %02f"  % (period, training_root_mean_squared_error))

		training_rmse.append(training_root_mean_squared_error)
		validation_rmse.append(validation_root_mean_squared_error)

	print("Model training finished.")

	plt.ylabel("RMSE")
	plt.xlabel("Periods")
	plt.title("Root Mean Squared Error vs . Periods")
	plt.tight_layout()
	plt.plot(training_rmse, label = "training")
	plt.plot(validation_rmse, label = "validation")
	plt.legend()
	plt.show()
	return linear_regressor


linear_regressor = train_model(
	learning_rate = 0.00003,
	steps = 500,
	batch_size = 5,
	training_examples = training_examples,
	training_targets = training_targets,
	validation_examples = validation_examples,
	validation_targets = validation_targets)

califonia_housing_test_data = pd.read_csv(Testing_file_path)

test_example = preprocess_features(califonia_housing_test_data)
test_targets = preprocess_targets(califonia_housing_test_data)

predict_test_input_fn = lambda: my_input_fn(test_example, test_features("median_house_value"),
	num_epochs = 1, shuffle = False)

test_predictions = linear_regressor.predict(input_fn = predict_test_input_fn)
test_predictions = np.array([item("predictions")[0] for item in test_predictions])

root_mean_squared_error = math.sqrt(
	metrics.mean_squared_error(test_predictions, test_targets))

print("Final RMSE (on the data): %0.2f"  % root_mean_squared_error)

