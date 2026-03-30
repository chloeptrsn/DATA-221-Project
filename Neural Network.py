# github repo link for my own reference: https://github.com/rrri01/DATA-221-Group-Project-Raines-Stuff
# group repository link for my own reference: https://github.com/chloeptrsn/DATA-221-Project.git

import pandas as pd
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import numpy as np

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer


from sklearn.metrics import root_mean_squared_error, r2_score
california_house_prices = pd.read_csv('housing.csv', delimiter=',')

#fill in the missing values (the missing values are usually present in total_bedrooms)
california_house_prices["total_bedrooms"] = california_house_prices["total_bedrooms"].fillna(california_house_prices["total_bedrooms"].median())

# mapping (converting categorical to numerical)
mapping = {
    "INLAND": 0,
    "NEAR OCEAN": 1,
    "NEAR BAY": 2,
    "ISLAND": 3,
    "<1H OCEAN": 4
}

california_house_prices["ocean_proximity"] = california_house_prices["ocean_proximity"].map(mapping)

# creates feature matrix X of all columns except "median_house_value" and create label vector y as "median_house_value"
feature_matrix = california_house_prices.drop("median_house_value", axis=1)
target_prices = california_house_prices["median_house_value"]

# splits the data into training data and testing data
features_train, features_test, labels_train, labels_test = train_test_split(feature_matrix, target_prices, test_size=0.3, random_state=42)

# scale data:
# scaling = StandardScaler()
# features_train = scaling.fit_transform(features_train)
# features_test = scaling.transform(features_test)

# neural network code and then print results
tf.random.set_seed(1)
neural_network_model = Sequential()

input_layer = InputLayer(input_shape=(9,))
neural_network_model.add(input_layer)
hidden_layer = Dense(3)
neural_network_model.add(hidden_layer)

second_hidden_layer = Dense(5)
neural_network_model.add(second_hidden_layer)
third_hidden_layer = Dense(2)
neural_network_model.add(third_hidden_layer)

output_layer = Dense(1)
neural_network_model.add(output_layer)

# compile the model
neural_network_model.compile(loss='mean_absolute_error')

# training and implementation
neural_network_model.fit(feature_matrix, target_prices, epochs=10)

predicted_labels = neural_network_model.predict(features_test, verbose = 1)

test_mae = neural_network_model.evaluate(features_test, labels_test)
train_mae = neural_network_model.evaluate(features_train, labels_train)

print(f"Testing Mean Absolute Error: {test_mae}")
print(f"Training Mean Absolute Error: {train_mae}")

# root mean squared error
rmse = root_mean_squared_error(labels_test, predicted_labels)
r2 = r2_score(labels_test, predicted_labels)
print(f"RMSE: {rmse}")
print(f"R2 SCORE: {r2}")
