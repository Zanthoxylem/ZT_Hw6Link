# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the Boston Housing dataset
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target

# Split the data into training and testing sets
X = data.drop('PRICE', axis=1)
y = data['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_reg_preds = lin_reg.predict(X_test)
lin_reg_mse = mean_squared_error(y_test, lin_reg_preds)
print(f"Linear Regression Mean Squared Error: {lin_reg_mse}")

# Random Forest Model
rand_forest = RandomForestRegressor(n_estimators=100, random_state=42)
rand_forest.fit(X_train, y_train)
rand_forest_preds = rand_forest.predict(X_test)
rand_forest_mse = mean_squared_error(y_test, rand_forest_preds)
print(f"Random Forest Mean Squared Error: {rand_forest_mse}")

# Compare performance
if lin_reg_mse < rand_forest_mse:
    print("Linear Regression performed better.")
else:
    print("Random Forest performed better.")
