import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load the data
data = pd.read_csv('../data/cleaned_sleep_data2.csv')

# Prepare the data
X = data[['Resting Heart Rate (bpm)', 'Step Count (steps)']]
y = data['Sleep_Efficiency']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeRegressor

# Train a Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions
dt_predictions = dt_model.predict(X_test)

# Evaluate the model
mse_dt = mean_squared_error(y_test, dt_predictions)
r2_dt = r2_score(y_test, dt_predictions)

print(f"Decision Tree - Mean Squared Error: {mse_dt}")
print(f"Decision Tree - R^2 Score: {r2_dt}")
