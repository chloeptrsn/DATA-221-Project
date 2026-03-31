import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import root_mean_squared_error, r2_score

california_house_prices = pd.read_csv('housing.csv', delimiter=',')

#fill in the missing values (the missing values are usually present in total_bedrooms)
california_house_prices["total_bedrooms"] = california_house_prices["total_bedrooms"].fillna(california_house_prices["total_bedrooms"].median())

# converting categorical to numerical instead of mapping: one-hot encoding
california_house_prices = pd.get_dummies(california_house_prices, columns=["ocean_proximity"])

# creates feature matrix X of all columns except "median_house_value" and create label vector y as "median_house_value"
feature_matrix = california_house_prices.drop("median_house_value", axis=1)
target_prices = california_house_prices["median_house_value"]

# splits the data into training data and testing data
features_train, features_test, labels_train, labels_test = train_test_split(feature_matrix, target_prices, test_size=0.3, random_state=42)

# constructing the model
tf.random.set_seed(1)
neural_network_model = Sequential()

input_layer = InputLayer(input_shape=(13,))
neural_network_model.add(input_layer)
hidden_layer = Dense(3)
neural_network_model.add(hidden_layer)

second_hidden_layer = Dense(5)
neural_network_model.add(second_hidden_layer)
third_hidden_layer = Dense(2)
neural_network_model.add(third_hidden_layer)

output_layer = Dense(1)
neural_network_model.add(output_layer)

# compiling the model
neural_network_model.compile(loss='mean_absolute_error')

# training and implementing the model
neural_network_model.fit(feature_matrix, target_prices, epochs=10)

predicted_labels = neural_network_model.predict(features_test)

# finding the mean absolute error
test_mae = neural_network_model.evaluate(features_test, labels_test)
train_mae = neural_network_model.evaluate(features_train, labels_train)

# finding the root mean squared error and r2 score
rmse = root_mean_squared_error(labels_test, predicted_labels)
r2 = r2_score(labels_test, predicted_labels)

print("NEURAL NETWORK RESULTS!")
print(f"Testing Mean Absolute Error: {test_mae}")
print(f"Training Mean Absolute Error: {train_mae}")

print(f"RMSE: {rmse}")
print(f"R2 SCORE: {r2}")
