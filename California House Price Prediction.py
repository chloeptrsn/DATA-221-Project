import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# load the dataset/ csv file
california_house_prices = pd.read_csv("C:/Codefile/housing file/housing.csv")

# --- Preprocessing data ---
# fill in the missing values (the missing values are usually present in total_bedrooms)
california_house_prices["total_bedrooms"] = california_house_prices["total_bedrooms"].fillna(california_house_prices["total_bedrooms"].median())

# converting categorical to numerical instead of mapping: one-hot encoding
california_house_prices = pd.get_dummies(california_house_prices, columns=["ocean_proximity"])

# constructing features & target
features_matrix = california_house_prices.drop("median_house_value", axis = 1)
target_values = california_house_prices["median_house_value"]

# train test split 70/30
features_train, features_test, labels_train, labels_test = train_test_split(features_matrix, target_values, test_size = 0.3, random_state = 42)

# data scaling
scaling = StandardScaler()
features_train = scaling.fit_transform(features_train)
features_test = scaling.transform(features_test)

# various k-values
k_values = [1, 3, 5, 7, 9, 15]

results_list = []

# --- Constructing model ---
for k in k_values:
    knn_model = KNeighborsRegressor(n_neighbors = k)
    knn_model.fit(features_train, labels_train)

    predicted_values = knn_model.predict(features_test)

    mae = mean_absolute_error(labels_test, predicted_values)
    rmse = root_mean_squared_error(labels_test, predicted_values)
    r2 = r2_score(labels_test, predicted_values)

    results_list.append([k, mae, rmse, r2])

# results table
results_table = pd.DataFrame(results_list, columns = ["K Values", "MAE", "RMSE", "R2 Score"])
results_table = results_table.sort_values(by = "RMSE")

print("California House Price Prediction Results")
print(results_table)