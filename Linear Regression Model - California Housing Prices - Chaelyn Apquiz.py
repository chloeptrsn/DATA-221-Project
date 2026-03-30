import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

california_house_prices = pd.read_csv("housing.csv")

# find the median and fill in the missing values
california_house_prices["total_bedrooms"] = (california_house_prices["total_bedrooms"].fillna(california_house_prices["total_bedrooms"].median()))

# converting categorical data which is the ocean_proximity column
california_house_prices = pd.get_dummies(california_house_prices, columns=["ocean_proximity"])

#splitting the features and target (x and y)
feature_matrix = california_house_prices.drop("median_house_value", axis=1)
target_prices = california_house_prices["median_house_value"]

#training data, test split 70/30
features_train, features_test, prices_train, prices_test = train_test_split(feature_matrix, target_prices, test_size=0.33, random_state=42)

# scaling the values
scaled_values = StandardScaler()


#git push -u origin main