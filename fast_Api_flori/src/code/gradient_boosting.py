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

from sklearn.ensemble import GradientBoostingRegressor

# Train a Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# Make predictions
gb_predictions = gb_model.predict(X_test)

# Evaluate the model
mse_gb = mean_squared_error(y_test, gb_predictions)
r2_gb = r2_score(y_test, gb_predictions)

print(f"Gradient Boosting - Mean Squared Error: {mse_gb}")
print(f"Gradient Boosting - R^2 Score: {r2_gb}")
