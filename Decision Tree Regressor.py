from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pandas as pd

# Load dataset
housing = pd.read_csv("housing.csv")

# Encoding
housing = pd.get_dummies(housing)
housing = housing.dropna()

# Split into feature and target variables
feature_matrix = housing.drop("median_house_value", axis=10)
target_values = housing["median_house_value"]

# Train test split-- 70% train, 30% test
features_train, features_test, labels_train, labels_test = train_test_split(
    feature_matrix, target_values, test_size=0.3, random_state=42)

# Data scaling
scaling = StandardScaler()
features_train = scaling.fit_transform(features_train)
features_test = scaling.transform(features_test)

