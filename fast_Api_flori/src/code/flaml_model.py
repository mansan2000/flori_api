import pandas as pd
# from sklearn.model_selection import train_test_split
#
# from flaml import AutoML
#
# # Load the data
data = pd.read_csv('../data/cleaned_sleep_data2.csv')
#
# # Prepare the independent and dependent variables
# X = data[['Resting Heart Rate (bpm)', 'Step Count (steps)']]
# y = data['Sleep_Efficiency']
X_train = data[['Step Count (steps)', 'Sleep_Efficiency']]
y_train = data['Resting Heart Rate (bpm)']

from flaml import AutoML
from sklearn.datasets import fetch_california_housing

# Initialize an AutoML instance
automl = AutoML()
# Specify automl goal and constraint
automl_settings = {
    "time_budget": 1,  # in seconds
    "metric": 'r2',
    "task": 'regression',
    "log_file_name": "california.log",
}
# X_train, y_train = fetch_california_housing(return_X_y=True)
# Train with labeled input data
automl.fit(X_train=X_train, y_train=y_train,
           **automl_settings)
# Predict
print(automl.predict(X_train))
# Print the best model
print(automl.model.estimator)