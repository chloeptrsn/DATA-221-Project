import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# loading csv file
california_house_prices = pd.read_csv("housing.csv")

# PRE-PROCESSING
# find the median and fill in the missing values which can be found in the total_bedrooms
california_house_prices["total_bedrooms"] = (california_house_prices["total_bedrooms"].fillna(california_house_prices["total_bedrooms"].median()))

# converting categorical to numerical instead of mapping
california_house_prices = pd.get_dummies(california_house_prices, columns=["ocean_proximity"])

# converting categorical data which is the ocean_proximity column
california_house_prices = pd.get_dummies(california_house_prices, columns=["ocean_proximity"])

#splitting the features and target (x and y)
feature_matrix = california_house_prices.drop("median_house_value", axis=1)
target_prices = california_house_prices["median_house_value"]

#training data, test split 70/30
features_train, features_test, prices_train, prices_test = train_test_split(feature_matrix, target_prices, test_size=0.33, random_state=42)

# scaling the values
scaled_values = StandardScaler()
features_train_scaled = scaled_values.fit_transform(features_train)
features_test_scaled = scaled_values.transform(features_test)

# Training Linear Regression Model
model = LinearRegression()
model.fit(features_train_scaled, target_prices)
#git push -u origin main