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

training_file_path = cwd + "\\california_housing_train.csv"

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
	output_targets = california_housing_dataframe.copy()
	output_targets["median_house_value"] = (output_targets["median_house_value"]/1000.0)
	return output_targets
	
	
training_examples = preprocess_features(california_housing_dataframe.head(12000))
print(training_examples.describe())

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