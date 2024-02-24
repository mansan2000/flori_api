import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the data
data = pd.read_csv('../data/cleaned_sleep_data2.csv')

# Prepare the independent and dependent variables
X = data[['Resting Heart Rate (bpm)', 'Step Count (steps)']]
y = data['Sleep_Efficiency']
# X = data[['Step Count (steps)', 'Sleep_Efficiency']]
# y = data['Resting Heart Rate (bpm)']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model on the training set
model = LinearRegression()
model.fit(X_train, y_train)

# Saving the model
filename = '../models/linear_regression_model.pkl'
pickle.dump(model, open(filename, 'wb'))

# Loading the model
loaded_model = pickle.load(open(filename, 'rb'))

# Using the loaded model to make predictions on new data
# Example: predicting for a data point
example_data = np.array([[62, 2183]])
prediction = loaded_model.predict(example_data)

# Output the prediction
print(prediction)

# Optionally, evaluate the model on the testing set to see how well it performs
from sklearn.metrics import mean_squared_error, r2_score

# Predict on the testing set
y_pred = loaded_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
