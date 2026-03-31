import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# loading csv file
california_house_prices = pd.read_csv("housing.csv")

# PRE-PROCESSING
# find the median and fill in the missing values which can be found in the total_bedrooms
california_house_prices["total_bedrooms"] = (california_house_prices["total_bedrooms"].fillna(california_house_prices["total_bedrooms"].median()))

# converting categorical to numerical instead of mapping: one-hot encoding
california_house_prices = pd.get_dummies(california_house_prices, columns=["ocean_proximity"])

#splitting the features and target (x and y)
feature_matrix = california_house_prices.drop("median_house_value", axis=1)
target_prices = california_house_prices["median_house_value"]

#training data, test split 70/30
feature_train, feature_test, target_train, target_test = train_test_split(feature_matrix, target_prices, test_size=0.3, random_state=42)

# training linear model
linear_regression_model = LinearRegression()
linear_regression_model.fit(feature_train, target_train)

# predictions
predicted_values = linear_regression_model.predict(feature_test)

# evaluation
mae = mean_absolute_error(target_test, predicted_values)
rmse = root_mean_squared_error(target_test, predicted_values)
r2 = r2_score(target_test, predicted_values)

print("LINEAR REGRESSION RESULTS!")
print("Mean Absolute Error: ", round(mae, 2))
print("Root Mean Squared Error: ", round(rmse, 2))
print("R2 Score: ", round(r2, 4))
